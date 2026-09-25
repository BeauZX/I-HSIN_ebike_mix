# /// script
# requires-python = ">=3.11"
# dependencies = ["pyyaml"]
# ///
"""模擬 MotorController + motor_policy 的完整流程，不需要真實硬體、不需要 Hailo/相機。

執行：uv run tests/simulate_motor.py

用一個假的 lgpio 模組取代真正的 GPIO：背景執行緒模擬「正在轉動的馬達」，依照目前輸出腳位的
方向持續呼叫真正的 _on_pulse() 回呼（時間都是真的 time.perf_counter()，不是加速的假時間），可以
設定在某個位置「卡住」來測試堵轉偵測 + 動態歸零，藉此驗證 src/motor.py、src/motor_policy.py 的
邏輯流程對不對，跟 main.py 會怎麼呼叫它們一致。

⚠️ 這裡驗證的是純邏輯（狀態機、堵轉判斷、防抖動、位置持久化），不驗證真正的 lgpio API 呼叫方式
對不對（callback 參數順序、gpio_claim_alert/.cancel() 用法）——lgpio 被整個換成假的了，這部分
只能在真正的樹莓派上跑過一次才能確認，改 motor.py 前建議先跑這支腳本，改完 GPIO 相關的地方
還是要找機會上機測。
"""

import sys
import tempfile
import threading
import time
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ========== 假的 lgpio：背景執行緒模擬馬達轉動 ==========

class MotorSim:
    def __init__(self, pulse_interval_s: float = 0.003):
        self.pin_state: dict[int, int] = {}
        self.plus_pin = self.minus_pin = None
        self.callback_fn = None
        self.pulse_interval_s = pulse_interval_s
        # 機構的真實位置與兩端死點（跟程式算的 pulse 位置分開，才能模擬計數漂移）
        self.real = 0
        self.tight_end = 0
        self.loose_end = -40
        self.batch = 1             # >1 時模擬 lgpio 通知緒一次送一批回呼（實機上會這樣）
        self.emitted = 0            # 馬達通電轉動時實際送出的脈衝數，用來比對有沒有漏算
        self._queue: list[int] = []
        self._level = 0
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _direction(self) -> int:
        if self.plus_pin is None:
            return 0
        p, m = self.pin_state.get(self.plus_pin, 0), self.pin_state.get(self.minus_pin, 0)
        if p == 1 and m == 0:
            return 1
        if p == 0 and m == 1:
            return -1
        return 0   # 全 0（coast）或全 1（brake）都不算在轉

    def _run(self):
        while self._running:
            time.sleep(self.pulse_interval_s)
            direction = self._direction()
            if direction == 0 or self.callback_fn is None:
                self._flush()
                continue
            if (direction == 1 and self.real >= self.tight_end) or (direction == -1 and self.real <= self.loose_end):
                continue   # 頂到死點：不再送脈衝，模擬馬達真的轉不動
            self.real += direction
            # tick 跟 lgpio 一樣是邊緣發生當下的奈秒時間戳，回呼可能晚一點才成批送達
            self._queue.append(time.monotonic_ns())
            self.emitted += 1
            if len(self._queue) >= self.batch:
                self._flush()

    def _flush(self):
        queued, self._queue = self._queue, []
        for tick in queued:
            self._level ^= 1
            self.callback_fn(0, 13, self._level, tick)   # (chip, gpio, level, tick)

    def stop(self):
        self._running = False


def make_fake_lgpio(sim: MotorSim) -> types.ModuleType:
    mod = types.ModuleType("lgpio")
    mod.BOTH_EDGES = "BOTH_EDGES"

    def gpiochip_open(n):
        return "fake_chip"

    def gpio_claim_output(chip, pin, level=0):
        sim.pin_state[pin] = level
        if sim.plus_pin is None:
            sim.plus_pin = pin
        elif sim.minus_pin is None:
            sim.minus_pin = pin

    def gpio_claim_alert(chip, pin, edge):
        pass

    class _Cb:
        def cancel(self):
            pass

    def callback(chip, pin, edge, fn):
        sim.callback_fn = fn
        return _Cb()

    def gpio_write(chip, pin, level):
        sim.pin_state[pin] = level

    def gpiochip_close(chip):
        pass

    mod.gpiochip_open = gpiochip_open
    mod.gpio_claim_output = gpio_claim_output
    mod.gpio_claim_alert = gpio_claim_alert
    mod.callback = callback
    mod.gpio_write = gpio_write
    mod.gpiochip_close = gpiochip_close
    return mod


sim = MotorSim()
sys.modules["lgpio"] = make_fake_lgpio(sim)

from src.motor import MotorController          # noqa: E402
from src.motor_policy import TargetDebouncer, decide_target  # noqa: E402
from src.settings import MotorSettings          # noqa: E402

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"{'[PASS]' if cond else '[FAIL]'} {name}" + (f" — {detail}" if detail else ""))


STORE = Path(tempfile.gettempdir()) / "system_integration_motor_sim_position.json"
STORE.unlink(missing_ok=True)

cfg = MotorSettings(
    pin_motor_plus=5, pin_motor_minus=6, pin_pulse=13,
    pulses_per_rev=100, pulse_filter_us=100,
    pos_tight=0, pos_mid=-20, pos_loose=-40,
    stop_margin_pulses=1, position_tolerance=1,
    brake_time_ms=20, move_timeout_ms=2000,
    stall_ratio_num=17, stall_ratio_den=10, min_stall_floor_us=50_000,
    start_timeout_ms=300, stall_timeout_ms=100,
    pothole_severe_cell_threshold=2,
    position_store=STORE,
    confirm_count=3, min_switch_interval_sec=0.5,
)

sim.tight_end, sim.loose_end = cfg.pos_tight, cfg.pos_loose


def wait_idle(timeout: float = 3.0) -> None:
    t0 = time.perf_counter()
    time.sleep(0.05)
    while motor.latest().busy and time.perf_counter() - t0 < timeout:
        time.sleep(0.02)


print("=" * 70)
print("情境 1：全新開機（沒有存檔）應該假設在全緊 (pos_tight)")
print("=" * 70)
motor = MotorController(cfg)
check("開機初始位置 = pos_tight", motor._get_position() == cfg.pos_tight, f"實際={motor._get_position()}")
motor.start()

print()
print("=" * 70)
print("情境 2：正常移動到 mid（中間沒有死點），走到 -20 附近就該停，理由是 target")
print("=" * 70)
motor.move_to("mid")
wait_idle()
st = motor.latest()
check("移動到 mid 後 busy=False（已完成）", st.busy is False, f"狀態={st}")
check("停止原因是 target（正常到位）", st.last_stop_reason == "target", f"實際={st.last_stop_reason}")
check("最終位置在 tolerance 範圍內", abs(st.position_pulses - cfg.pos_mid) <= cfg.position_tolerance,
      f"位置={st.position_pulses}, 目標={cfg.pos_mid}")

print()
print("=" * 70)
print("情境 3：往全緊移動，程式算的位置比實際多鬆 5 格（漂移）→ 提早撞到死點，堵轉 + 動態歸零到 pos_tight")
print("=" * 70)
sim.real += 5   # 機構實際比程式以為的更靠近全緊 5 格
motor.move_to("tight")
wait_idle()
st = motor.latest()
check("堵轉後停止原因是 stall", st.last_stop_reason == "stall", f"實際={st.last_stop_reason}")
check("動態歸零：位置被強制校正回 pos_tight（計數只走到 -5 左右就撞到）",
      st.position_pulses == cfg.pos_tight and sim.real == sim.tight_end,
      f"程式位置={st.position_pulses}, 實際={sim.real}")

print()
print("=" * 70)
print("情境 4：往全鬆移動，忙碌中第二個 move_to() 應該被拒絕；走到全鬆死點 → 堵轉 + 歸零到 pos_loose")
print("=" * 70)
motor.move_to("loose")
time.sleep(0.01)   # 給執行緒一點時間真的進入忙碌狀態
accepted = motor.move_to("mid")
check("忙碌中第二個命令被拒絕", accepted is False, f"move_to() 回傳={accepted}")
wait_idle()
st = motor.latest()
check("全鬆一律轉到堵轉才停（停止原因 stall）", st.last_stop_reason == "stall", f"實際={st.last_stop_reason}")
check("全鬆停在死點、位置 = pos_loose", st.position_pulses == cfg.pos_loose and sim.real == sim.loose_end,
      f"程式位置={st.position_pulses}, 實際={sim.real}")

print()
print("=" * 70)
print("情境 4a：往全鬆移動，程式算的位置比實際多緊 5 格（漂移）→ 撞到全鬆死點後歸零到 pos_loose")
print("=" * 70)
motor.move_to("mid")
wait_idle()
sim.real -= 5   # 機構實際比程式以為的更靠近全鬆 5 格
motor.move_to("loose")
wait_idle()
st = motor.latest()
check("全鬆動態歸零：位置被強制校正回 pos_loose",
      st.last_stop_reason == "stall" and st.position_pulses == cfg.pos_loose and sim.real == sim.loose_end,
      f"停止原因={st.last_stop_reason}, 程式位置={st.position_pulses}, 實際={sim.real}")

print()
print("=" * 70)
print("情境 4c：鎖不緊的回歸測試 — 程式算的位置比實際多緊 5 格，計數到 0 時機構還沒到全緊，")
print("        不能因為計數到了就停，要繼續轉到頂住全緊死點")
print("=" * 70)
with motor._pulse_lock:
    motor._pulse_position += 5   # 程式以為 -35，實際在 -40
motor.move_to("tight")
wait_idle()
st = motor.latest()
check("全緊一律轉到頂住死點（實際位置 = 全緊）", sim.real == sim.tight_end,
      f"停止原因={st.last_stop_reason}, 程式位置={st.position_pulses}, 實際={sim.real}")
check("頂到後歸零，程式位置 = pos_tight", st.position_pulses == cfg.pos_tight, f"程式位置={st.position_pulses}")

print()
print("=" * 70)
print("情境 4b：回呼一次送 5 個（模擬 lgpio 通知緒成批送達），脈衝不能被濾波當雜訊丟掉")
print("=" * 70)
start_pos = motor.latest().position_pulses
sim.batch, sim.emitted = 5, 0
motor.move_to("mid")   # 從 tight 往 mid，停在範圍內，後面的持久化測試才讀得回同一個值
wait_idle()
st = motor.latest()
counted = start_pos - st.position_pulses
check("成批送達時，程式算到的脈衝數 = 馬達實際送出的脈衝數", counted == sim.emitted,
      f"送出={sim.emitted}, 算到={counted}, 停止原因={st.last_stop_reason}")
check("成批送達時仍正常到位（不是被誤判堵轉）", st.last_stop_reason == "target", f"實際={st.last_stop_reason}")
sim.batch = 1

print()
print("=" * 70)
print("情境 5：位置持久化 — 存檔內容應該可以被新的 MotorController 讀回來")
print("=" * 70)
saved_pos = motor.latest().position_pulses
motor.stop()
motor.join(timeout=3)
motor.close()

motor2 = MotorController(cfg)
check("重新建立 MotorController 讀回上次存檔的位置", motor2._get_position() == saved_pos,
      f"上次存的={saved_pos}, 這次讀到={motor2._get_position()}")
motor2.close()
STORE.unlink(missing_ok=True)

print()
print("=" * 70)
print("情境 6：motor_policy 的決策規則 + TargetDebouncer 防抖動")
print("=" * 70)


class FakeRoad:
    def __init__(self, mode, label, result=None, grader=None):
        self.mode, self.label, self.result, self.grader = mode, label, result, grader


class FakeGrader:
    def __init__(self, severe):
        self._severe = severe

    def grade_counts(self, result):
        return {"severe": self._severe, "slight": 0, "smooth": 0}


class FakeDet:
    def __init__(self, has_person):
        self.detections = [(0, 0, "person", 0.9, 0, 0, 1, 1)] if has_person else []


class FakeDoor:
    def __init__(self, state):
        self.state = state


t1 = decide_target(FakeRoad("asphalt", "Asphalt Road"), None, None, cfg)
check("柏油路 -> tight", t1 == "tight", f"實際={t1}")

t2 = decide_target(FakeRoad("none", "Forest Road"), None, None, cfg)
check("森林路 -> loose", t2 == "loose", f"實際={t2}")

t3 = decide_target(FakeRoad("none", "Belgian Block"), None, None, cfg)
check("比利時路 -> mid", t3 == "mid", f"實際={t3}")

t4 = decide_target(FakeRoad("asphalt", "Asphalt Road", result=object(), grader=FakeGrader(3)), None, None, cfg)
check("柏油路但 severe 格數(3) >= 門檻(2) -> loose 覆蓋", t4 == "loose", f"實際={t4}")

t5 = decide_target(FakeRoad("none", "Forest Road"), FakeDet(True), None, cfg)
check("森林路但畫面有人 -> tight 覆蓋（安全最後贏）", t5 == "tight", f"實際={t5}")

t6 = decide_target(FakeRoad("asphalt", "Asphalt Road"), None, FakeDoor("open"), cfg)
check("柏油路但車門開啟 -> tight（本來就是 tight，驗證不會被錯誤覆蓋成別的）", t6 == "tight", f"實際={t6}")

debouncer = TargetDebouncer(confirm_count=3)
seq = ["tight", "loose", "tight", "loose", "tight"]   # 純抖動，永遠不該確認
results = [debouncer.update(x) for x in seq]
check("純抖動（confirm_count=3，每次都變）不會被確認", all(r is None for r in results),
      f"每一步結果={results}")

debouncer2 = TargetDebouncer(confirm_count=3)
seq2 = ["loose", "loose", "loose", "tight", "tight", "tight"]
results2 = [debouncer2.update(x) for x in seq2]
check("連續 3 次穩定才確認，且能正確切到下一個穩定值",
      results2 == [None, None, "loose", "loose", "loose", "tight"], f"每一步結果={results2}")

print()
print("=" * 70)
summary = f"通過 {len(PASS)} / 失敗 {len(FAIL)}"
print(summary)
if FAIL:
    print("失敗項目：", FAIL)
print("=" * 70)

sim.stop()
sys.exit(1 if FAIL else 0)
