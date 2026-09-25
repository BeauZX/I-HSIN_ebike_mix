#!/usr/bin/env python3
"""寸動工具：每次只轉一小段時間，用來把避震器手動轉回全緊、把位置存檔重設（狀態歸位）。

不透過 src/motor.py（沒有堵轉偵測、不動態歸零），直接用 lgpio 驅動 M+/M-：
    f [秒]  M+ 方向（程式定義的「正轉」＝往全緊）轉一小段，預設 0.1 秒
    r [秒]  M- 方向（程式定義的「反轉」＝往全鬆）轉一小段，預設 0.1 秒
    p       印累積脈衝數與有效脈衝累計
    z       確認已經在全緊了：把 motor.yaml 的 position_store 寫成 pos_tight，有效脈衝累計歸零
    q       結束

單次最多 MAX_STEP_S 秒，避免手誤輸入太大的數字一直頂著死點通電。

「有效脈衝累計」用跟 src/motor.py 一樣的 tick 濾波（間隔小於 pulse_filter_us 的邊緣不算），
f 加、r 減，等於程式眼中的 pulse 位置：從全緊按 z 歸零後一路按 r 到全鬆，最後的數字就是 pos_loose 該填的值。

每轉一段會順便印脈衝診斷：lgpio 的 tick（核心記錄的邊緣時間，64-bit 奈秒）跟 Python 回呼被
呼叫的時間（src/motor.py 目前用這個）各有幾個間隔小於 pulse_filter_us。後者明顯比前者多，
代表回呼是一批一批送進來的，src/motor.py 的濾波會把真的脈衝當雜訊丟掉。

執行：uv run tests/manual_jog.py
"""

import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MOTOR_YAML = PROJECT_ROOT / "configs" / "motor.yaml"
DEFAULT_STEP_S = 0.1
MAX_STEP_S = 0.5

cfg = yaml.safe_load(MOTOR_YAML.read_text(encoding="utf-8"))
PIN_PLUS = int(cfg["pin_motor_plus"])
PIN_MINUS = int(cfg["pin_motor_minus"])
PIN_PULSE = int(cfg["pin_pulse"])
FILTER_S = int(cfg["pulse_filter_us"]) / 1e6
POS_TIGHT = int(cfg["pos_tight"])
store = Path(cfg["position_store"])
STORE = store if store.is_absolute() else PROJECT_ROOT / store

print(f"腳位（來自 {MOTOR_YAML}）：M+ = GPIO{PIN_PLUS}, M- = GPIO{PIN_MINUS}, 脈衝 = GPIO{PIN_PULSE}")

import lgpio  # noqa: E402

chip = lgpio.gpiochip_open(0)
lgpio.gpio_claim_output(chip, PIN_PLUS, 0)
lgpio.gpio_claim_output(chip, PIN_MINUS, 0)
lgpio.gpio_claim_alert(chip, PIN_PULSE, lgpio.BOTH_EDGES)

_lock = threading.Lock()
pulse_count = 0
valid_total = 0      # tick 濾波後的有效脈衝，f 加、r 減
events: list[tuple[int, float]] = []   # (lgpio tick 奈秒, 回呼被呼叫時的 perf_counter 秒)


def _on_pulse(chip_, gpio_, level, tick):
    global pulse_count
    now = time.perf_counter()
    with _lock:
        pulse_count += 1
        events.append((tick, now))


cb = lgpio.callback(chip, PIN_PULSE, lgpio.BOTH_EDGES, _on_pulse)


def coast():
    lgpio.gpio_write(chip, PIN_PLUS, 0)
    lgpio.gpio_write(chip, PIN_MINUS, 0)


def brake():
    lgpio.gpio_write(chip, PIN_PLUS, 1)
    lgpio.gpio_write(chip, PIN_MINUS, 1)


def count_valid(ticks: list[int]) -> int:
    """跟 src/motor.py 的 _on_pulse 同一套濾波：跟上一個有效脈衝的 tick 間隔太短就不算。"""
    n, last = 0, None
    for t in ticks:
        if last is None or (t - last) / 1e9 >= FILTER_S:
            n, last = n + 1, t
    return n


def step(plus: int, minus: int, seconds: float, label: str) -> None:
    global valid_total
    with _lock:
        before = pulse_count
        events.clear()
    t_start = time.perf_counter()
    lgpio.gpio_write(chip, PIN_PLUS, plus)
    lgpio.gpio_write(chip, PIN_MINUS, minus)
    time.sleep(seconds)
    brake()
    time.sleep(0.1)
    coast()
    time.sleep(0.2)   # 等煞車後的慣性脈衝與還在排隊的回呼都進來
    with _lock:
        n = pulse_count - before
        ev = list(events)
    valid = count_valid([t for t, _ in ev])
    valid_total += valid if plus else -valid
    print(f"{label} {seconds:g} 秒：{n} 個脈衝（含煞車後慣性），有效 {valid} 個"
          f"　→　有效脈衝累計 = {valid_total}", flush=True)
    if ev:
        gaps = [(b[0] - a[0]) / 1e6 for a, b in zip(ev, ev[1:])]
        print(f"  起步：第一個脈衝約在通電後 {(ev[0][1] - t_start) * 1000:.0f} ms"
              + (f"；脈衝之間最長間隔 {max(gaps):.0f} ms" if gaps else ""), flush=True)
    if len(ev) >= 2:
        tick_iv = [(b[0] - a[0]) / 1e9 for a, b in zip(ev, ev[1:])]
        cb_iv = [b[1] - a[1] for a, b in zip(ev, ev[1:])]
        print(f"  診斷：間隔 < {FILTER_S * 1e6:.0f} µs 的個數　tick 算 = {sum(i < FILTER_S for i in tick_iv)}"
              f"　回呼時間算 = {sum(i < FILTER_S for i in cb_iv)}（共 {len(tick_iv)} 個間隔）；"
              f"tick 最短間隔 {min(tick_iv) * 1e6:.0f} µs、中位數 {sorted(tick_iv)[len(tick_iv) // 2] * 1e6:.0f} µs",
              flush=True)


def parse_seconds(parts: list[str]) -> float | None:
    if len(parts) == 1:
        return DEFAULT_STEP_S
    try:
        s = float(parts[1])
    except ValueError:
        print(f"看不懂秒數：{parts[1]}")
        return None
    if not 0 < s <= MAX_STEP_S:
        print(f"秒數要在 0 ~ {MAX_STEP_S:g} 之間")
        return None
    return s


def reset_store() -> None:
    STORE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STORE.with_suffix(STORE.suffix + ".tmp")
    tmp.write_text(json.dumps({"pulse_position": POS_TIGHT,
                               "last_updated": datetime.now().isoformat(timespec="seconds")}), encoding="utf-8")
    tmp.replace(STORE)
    print(f"已把 {STORE} 重設為 pulse_position = {POS_TIGHT}（全緊）")


print(__doc__)
print(f"目前位置存檔：{STORE.read_text(encoding='utf-8').strip() if STORE.exists() else '（不存在）'}")

try:
    while True:
        parts = input("> ").strip().lower().split()
        if not parts:
            continue
        cmd = parts[0]
        if cmd in ("f", "r"):
            s = parse_seconds(parts)
            if s is None:
                continue
            if cmd == "f":
                step(1, 0, s, "f（M+，程式的往全緊）")
            else:
                step(0, 1, s, "r（M-，程式的往全鬆）")
        elif cmd == "p":
            with _lock:
                print(f"累積脈衝總數 = {pulse_count}，有效脈衝累計 = {valid_total}")
        elif cmd == "z":
            if input("確定避震器現在在全緊？(y/N) ").strip().lower() == "y":
                reset_store()
                valid_total = 0
                print("有效脈衝累計歸零")
        elif cmd == "q":
            break
        else:
            print("指令是 f [秒] / r [秒] / p / z / q")
except KeyboardInterrupt:
    print("\n中斷，收尾中...")
finally:
    coast()
    try:
        cb.cancel()
    except Exception:
        pass
    lgpio.gpiochip_close(chip)
    print("已釋放 GPIO。")
sys.exit(0)
