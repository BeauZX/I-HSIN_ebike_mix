#!/usr/bin/env python3
"""階段 1：最小化的馬達方向控制測試，空轉（不掛避震器）用。

不透過 src/motor.py 的完整邏輯（堵轉偵測、動態歸零都先不管），只直接用 lgpio 驅動 M+/M-、
讀編碼器脈衝，確認：
    1. 正轉/反轉腳位接對了、馬達真的會依指令轉動方向
    2. 編碼器脈衝腳位接對了、轉動時脈衝數真的會跟著變化（正轉遞增、反轉遞減）

腳位讀自 configs/motor.yaml（跟正式程式共用同一份設定，不用兩邊維護），但不透過
src/settings.py 的 load_settings()（那會連 camera/road_type 等其他 HEF 路徑都一起驗證，
這裡只想單獨測馬達接線，故意繞開）。

執行：uv run tests/manual_gpio_test.py

⚠️ 這支腳本沒有堵轉保護，空轉時如果不小心让馬達卡住（例如手誤觸），會持續通電，
測試時人不要離開、看到不對勁立刻 Ctrl+C 或直接拔電源。
"""

import sys
import threading
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MOTOR_YAML = PROJECT_ROOT / "configs" / "motor.yaml"

cfg = yaml.safe_load(MOTOR_YAML.read_text(encoding="utf-8"))
PIN_PLUS = int(cfg["pin_motor_plus"])
PIN_MINUS = int(cfg["pin_motor_minus"])
PIN_PULSE = int(cfg["pin_pulse"])

print(f"讀到腳位設定（來自 {MOTOR_YAML}）：M+ = GPIO{PIN_PLUS}, M- = GPIO{PIN_MINUS}, 脈衝 = GPIO{PIN_PULSE}")

import lgpio  # noqa: E402  # 確定在真正的樹莓派上才 import，本機模擬測試不會跑到這裡

chip = lgpio.gpiochip_open(0)
lgpio.gpio_claim_output(chip, PIN_PLUS, 0)
lgpio.gpio_claim_output(chip, PIN_MINUS, 0)
lgpio.gpio_claim_alert(chip, PIN_PULSE, lgpio.BOTH_EDGES)

pulse_count = 0
_lock = threading.Lock()


def _on_pulse(chip_, gpio_, level, tick):
    global pulse_count
    with _lock:
        pulse_count += 1


cb = lgpio.callback(chip, PIN_PULSE, lgpio.BOTH_EDGES, _on_pulse)


def coast():
    lgpio.gpio_write(chip, PIN_PLUS, 0)
    lgpio.gpio_write(chip, PIN_MINUS, 0)


def forward():
    lgpio.gpio_write(chip, PIN_PLUS, 1)
    lgpio.gpio_write(chip, PIN_MINUS, 0)


def reverse():
    lgpio.gpio_write(chip, PIN_PLUS, 0)
    lgpio.gpio_write(chip, PIN_MINUS, 1)


def brake():
    lgpio.gpio_write(chip, PIN_PLUS, 1)
    lgpio.gpio_write(chip, PIN_MINUS, 1)


def run_for(direction_fn, seconds: float, label: str):
    with _lock:
        before = pulse_count
    print(f"\n{label}中，{seconds:g} 秒...", flush=True)
    direction_fn()
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        time.sleep(0.2)
        with _lock:
            print(f"  已轉 {time.perf_counter() - t0:4.1f}s，脈衝累積 = {pulse_count - before}", flush=True)
    brake()
    time.sleep(0.1)
    coast()
    with _lock:
        after = pulse_count
    print(f"{label}結束，共 {after - before} 個脈衝", flush=True)


print(__doc__)
print("指令：f=正轉2秒　r=反轉2秒　p=印目前累積脈衝數　q=結束")

try:
    while True:
        cmd = input("> ").strip().lower()
        if cmd == "f":
            run_for(forward, 2.0, "正轉")
        elif cmd == "r":
            run_for(reverse, 2.0, "反轉")
        elif cmd == "p":
            with _lock:
                print(f"目前累積脈衝總數 = {pulse_count}")
        elif cmd == "q":
            break
        else:
            print("看不懂，指令是 f / r / p / q")
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

print("\n檢查重點：")
print("  1. f 指令時馬達真的往你預期的方向轉了嗎？（如果方向反了，交換 motor.yaml 的")
print("     pin_motor_plus / pin_motor_minus 或直接對調馬達接線的兩條線）")
print("  2. r 指令時方向有沒有確實反過來？")
print("  3. 轉動的時候「脈衝累積」數字有沒有持續往上跳？完全不動代表編碼器接線有問題")
print("  4. 正轉跟反轉造成的脈衝數量，同樣轉 2 秒應該要差不多（如果差很多，可能是負載不均或接觸不良）")
sys.exit(0)
