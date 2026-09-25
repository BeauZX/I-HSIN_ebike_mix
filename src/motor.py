"""避震器鎖緊/放鬆馬達控制（改寫自 Arduino 0709_3btn_edge.ino，原本跑在 ESP32 上）。

用 lgpio 驅動 M+/M- 兩個輸出腳位，讀編碼器脈衝腳位做雙邊緣(BOTH_EDGES)計數與時間差堵轉偵測，
邏輯逐項對應 0709_3btn_edge.ino：

    pulseISR()（CHANGE 中斷，維護 baseline）      -> _on_pulse()
    moveToPosition()                              -> _execute_move()
    runUntilTarget()（回傳 STOP_REASON_*）         -> _run_until_target()（回傳字串）
    全緊方向：時間差比例法 + 動態歸零              -> 全緊、全鬆兩端都用，且一律轉到堵轉才停（見 _execute_move）
    中間：簡單版無脈衝逾時                          -> 原樣保留
    motorForward()/motorReverse()/stopWithBrake()  -> _motor_forward()/_motor_reverse()/_stop_with_brake()

沒有實體按鈕：main.py 依路面/坑洞/人員/車門辨識結果呼叫 move_to()，見 src/motor_policy.py。

位置持久化：每次移動完成後把目前 pulse 位置存進 motor.yaml 的 position_store（JSON，atomic write），
開機時讀回來。樹莓派跑完整 Linux、有真正的檔案系統，不需要像 ESP32 那樣依賴 NVS；讀不到/壞掉/超出
範圍時退回 pos_tight（假設機構在全緊狀態，跟 Arduino 版的開機假設一致）。

⚠️ 時間精度：ESP32 是專用微控制器，中斷延遲是微秒等級、非常穩定；這裡跑在 Linux + Python + GIL
之上，lgpio 把邊緣事件排進 pipe，再由它的通知緒一批一批呼叫 Python 回呼，回呼被呼叫的時間不等於
邊緣發生的時間。所以脈衝之間的間隔（濾波、比例堵轉的基準值）一律用 callback 帶的 tick
（核心記錄的邊緣時間，64-bit 奈秒，不會回繞）；用回呼時間算的話，同一批送進來的真脈衝間隔幾乎是 0，
會被濾波當雜訊丟掉（2026-09-25 實測丟掉約三分之一，見 tests/manual_jog.py 的診斷）。
「距上次脈衝多久」（簡單堵轉、堵轉判斷的經過時間）仍用 time.perf_counter()，因為要跟主迴圈的現在時間比，
回呼延遲只有幾毫秒，影響不大。motor.yaml 裡從 ESP32 版本抄過來的門檻值
（stall_ratio_*、min_stall_floor_us、start_timeout_ms、stall_timeout_ms）幾乎確定需要在 Pi 上
重新實測調整，不要假設能直接沿用。
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime

import lgpio

from .settings import MotorSettings

TARGETS = ("tight", "mid", "loose")


@dataclass
class MotorStatus:
    position_pulses: int
    target: str | None          # 這次（或上一次）執行的目標名稱
    busy: bool
    last_stop_reason: str | None  # "target" / "stall" / "stall_start" / "timeout" / "already_close" / "aborted"
    diff_from_target: int | None  # 停止時跟目標差多少 pulse，診斷用（沿用 Arduino 版加的那行 log）


class MotorController(threading.Thread):
    """命令式，不是逐幀模式：move_to() 餵一個目標位置命令，忙碌時新命令被忽略
    （對應 Arduino 的 motorBusy guard）；latest() 隨時讀目前狀態。"""

    def __init__(self, cfg: MotorSettings):
        super().__init__(daemon=True, name="motor")
        self.cfg = cfg
        self._pulse_filter_s = cfg.pulse_filter_us / 1e6

        self._chip = lgpio.gpiochip_open(0)
        lgpio.gpio_claim_output(self._chip, cfg.pin_motor_plus, 0)
        lgpio.gpio_claim_output(self._chip, cfg.pin_motor_minus, 0)
        lgpio.gpio_claim_alert(self._chip, cfg.pin_pulse, lgpio.BOTH_EDGES)
        self._pulse_cb = lgpio.callback(self._chip, cfg.pin_pulse, lgpio.BOTH_EDGES, self._on_pulse)

        # 脈衝計數/時間差狀態，_on_pulse()（lgpio 的 callback 執行緒）與主迴圈/本執行緒都會存取
        self._pulse_lock = threading.Lock()
        self._pulse_position = self._load_position()
        self._last_pulse_time = time.perf_counter()   # 上次有效脈衝的回呼時間，算「多久沒脈衝」用
        self._last_pulse_tick: int | None = None      # 上次有效脈衝的 lgpio tick（奈秒），算脈衝間隔用
        self._baseline_interval_s = 0.0
        self._motor_direction = 0   # 1 正轉 / -1 反轉 / 0 停止（只有本執行緒寫、_on_pulse 讀，單字賦值不額外上鎖）

        # 命令佇列（同一時間只接受一個待處理命令）
        self._pending: str | None = None
        self._cmd_lock = threading.Lock()
        self._new_cmd = threading.Event()
        self._stop_evt = threading.Event()      # 不能取名 _stop：會蓋掉 Thread 內部方法

        self._status = MotorStatus(self._pulse_position, None, False, None, None)
        self._status_lock = threading.Lock()

    # ── 對外介面（main.py 用，跟其他背景緒的 submit()/latest() 精神一致） ──

    def move_to(self, target: str) -> bool:
        """要求移動到 target（"tight"/"mid"/"loose"）。忙碌中會被忽略，回傳 False。"""
        if target not in TARGETS:
            raise ValueError(f"未知的目標位置: {target}")
        with self._status_lock:
            if self._status.busy:
                return False
            # 在這裡（而不是等 _execute_move() 真正開始跑）就標記忙碌，
            # 避免命令排進佇列但背景執行緒還沒醒來處理前，被下一次 move_to() 呼叫覆寫掉
            self._status.busy = True
        with self._cmd_lock:
            self._pending = target
        self._new_cmd.set()
        return True

    def latest(self) -> MotorStatus | None:
        with self._status_lock:
            return self._status

    def stop(self) -> None:
        self._stop_evt.set()
        self._new_cmd.set()

    def close(self) -> None:
        """釋放 GPIO；main.py 在 join() 之後呼叫。"""
        self._motor_coast()
        try:
            self._pulse_cb.cancel()
        except Exception:
            pass
        lgpio.gpiochip_close(self._chip)

    # ── 位置持久化（開機讀回、每次移動完成後存檔） ──

    def _load_position(self) -> int:
        path = self.cfg.position_store
        if not path.exists():
            print(f"[馬達] {path} 不存在，視為全緊（第一次啟動或從沒存過）", flush=True)
            return self.cfg.pos_tight
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            pos = int(data["pulse_position"])
        except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            print(f"[馬達] 讀取 {path} 失敗（{e}），視為全緊", flush=True)
            return self.cfg.pos_tight
        clamped = max(self.cfg.pos_loose, min(self.cfg.pos_tight, pos))
        if clamped != pos:
            print(f"[馬達] 存檔位置 {pos} 超出 pos_loose~pos_tight 範圍，夾到 {clamped}", flush=True)
        print(f"[馬達] 讀回上次位置：pulse = {clamped}", flush=True)
        return clamped

    def _save_position(self, position: int) -> None:
        """先寫暫存檔、再改名覆蓋（atomic write），避免寫到一半斷電讓存檔壞掉。"""
        path = self.cfg.position_store
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            data = {"pulse_position": position, "last_updated": datetime.now().isoformat(timespec="seconds")}
            tmp.write_text(json.dumps(data), encoding="utf-8")
            os.replace(tmp, path)
        except OSError as e:
            print(f"[馬達] 存檔位置失敗（{e}），這次移動的位置不會保留到下次開機", flush=True)

    # ── 中斷回呼（對應 pulseISR()） ──

    def _on_pulse(self, chip, gpio, level, tick) -> None:
        now = time.perf_counter()
        with self._pulse_lock:
            # 間隔用 tick 算（見模組說明）；每次移動的第一個脈衝沒有前一個可比，直接收下、不設基準值
            if self._last_pulse_tick is not None:
                interval = (tick - self._last_pulse_tick) / 1e9
                if interval < self._pulse_filter_s:
                    return
                if self._baseline_interval_s == 0.0:
                    self._baseline_interval_s = interval
                elif interval * self.cfg.stall_ratio_den < self._baseline_interval_s * self.cfg.stall_ratio_num:
                    self._baseline_interval_s += (interval - self._baseline_interval_s) / 4
            self._last_pulse_tick = tick
            self._last_pulse_time = now
            if self._motor_direction == 1:
                self._pulse_position += 1
            elif self._motor_direction == -1:
                self._pulse_position -= 1

    def _get_position(self) -> int:
        with self._pulse_lock:
            return self._pulse_position

    # ── 執行緒主迴圈 ──

    def run(self) -> None:
        while not self._stop_evt.is_set():
            self._new_cmd.wait(timeout=1.0)
            if self._stop_evt.is_set():
                break
            with self._cmd_lock:
                target = self._pending
                self._pending = None
            self._new_cmd.clear()
            if target is None:
                continue
            try:
                self._execute_move(target)
            except Exception as e:
                print(f"[馬達] 移動失敗: {e}", flush=True)
                self._motor_coast()
                with self._status_lock:
                    self._status.busy = False

    # ── 移動邏輯（對應 moveToPosition()） ──

    def _target_pulses(self, target: str) -> int:
        return {"tight": self.cfg.pos_tight, "mid": self.cfg.pos_mid, "loose": self.cfg.pos_loose}[target]

    def _execute_move(self, target: str) -> None:
        # busy 已經在 move_to() 裡設過了，這裡不用再設一次
        target_pulses = self._target_pulses(target)
        target_pulses = max(self.cfg.pos_loose, min(self.cfg.pos_tight, target_pulses))
        current = self._get_position()

        if abs(target_pulses - current) <= self.cfg.position_tolerance:
            self._finish(target, current, "already_close")
            return

        print(f"[馬達] 移動到 {target}. 目前 pulse = {current} , 目標 pulse = {target_pulses}", flush=True)

        # 重置這次移動的脈衝計時基準，避免沿用上次移動、甚至上次閒置很久的舊時間點
        with self._pulse_lock:
            self._last_pulse_time = time.perf_counter()
            self._last_pulse_tick = None
            self._baseline_interval_s = 0.0

        # 全緊、全鬆兩端都是實測確認過的機構死點（2026-09-25），目標是兩端時不看計數，一律轉到頂住死點
        # （時間差比例堵轉）才停，再動態歸零。計數往兩個方向有約 10% 的漂移，靠計數停會停在死點前
        # （例如鎖不緊）；中間位置沒有死點，照計數停，搭配簡單版堵轉偵測當安全網
        to_end = target_pulses in (self.cfg.pos_tight, self.cfg.pos_loose)

        if target_pulses > current:
            self._motor_forward()
            stop_reason = self._run_until_target(target_pulses, 1, to_end)
        else:
            self._motor_reverse()
            stop_reason = self._run_until_target(target_pulses, -1, to_end)

        self._stop_with_brake()

        # 動態歸零：目標是兩端、且是因為時間差堵轉才停下來（不是一啟動就沒脈衝的弱證據），
        # 代表真的撞到死點了，清掉之前累積的誤差
        if to_end and stop_reason == "stall":
            with self._pulse_lock:
                counted = self._pulse_position
                self._pulse_position = target_pulses
            # 校正前的計數跟死點差多少 = 這一趟累積的計數漂移，車上長時間跑可以從 log 看漂移有沒有變大
            print(f"[馬達] 撞到{'全緊' if target == 'tight' else '全鬆'}死點（時間差堵轉），"
                  f"位置自動校正：{counted} → {target_pulses}（漂移 {counted - target_pulses:+d}）", flush=True)

        final_pos = self._get_position()
        print(f"[馬達] 完成. 最終 pulse = {final_pos} , 停止原因 = {stop_reason} , "
              f"跟目標差 = {final_pos - target_pulses}", flush=True)
        self._finish(target, final_pos, stop_reason)

    def _run_until_target(self, target_pulses: int, direction: int, to_end: bool) -> str:
        """回傳停止原因："target" / "stall" / "stall_start" / "timeout" / "aborted"。
        to_end=True（目標是全緊/全鬆死點）：用時間差比例堵轉、不因計數到了就停，一律轉到堵轉或逾時。"""
        start = time.perf_counter()
        move_timeout_s = self.cfg.move_timeout_ms / 1000.0
        stall_ratio = self.cfg.stall_ratio_num / self.cfg.stall_ratio_den
        min_floor_s = self.cfg.min_stall_floor_us / 1e6
        start_timeout_s = self.cfg.start_timeout_ms / 1000.0
        stall_timeout_s = self.cfg.stall_timeout_ms / 1000.0

        while True:
            if self._stop_evt.is_set():
                return "aborted"

            now = time.perf_counter()
            if now - start > move_timeout_s:
                return "timeout"

            pos = self._get_position()
            with self._pulse_lock:
                last_pulse_time = self._last_pulse_time
                baseline = self._baseline_interval_s
            elapsed_since_pulse = now - last_pulse_time

            if to_end:
                if baseline > 0:
                    # 已經有基準值：距上次脈衝的時間超過基準的 stall_ratio 倍，
                    # 且至少經過 min_floor_s，才代表馬達真的在變慢（或已經停了）
                    if elapsed_since_pulse > baseline * stall_ratio and elapsed_since_pulse > min_floor_s:
                        return "stall"
                else:
                    # 還沒有基準值（剛啟動、第一個脈衝還沒來），證據較弱，不能拿來做動態歸零
                    if elapsed_since_pulse > start_timeout_s:
                        return "stall_start"
            else:
                # 簡單版：連續這麼久完全沒有新脈衝，才判定堵轉
                if elapsed_since_pulse > stall_timeout_s:
                    return "stall"

            # 兩端不看計數，等堵轉；中間位置計數到了就停
            if not to_end:
                if direction == 1 and pos >= target_pulses - self.cfg.stop_margin_pulses:
                    return "target"
                if direction == -1 and pos <= target_pulses + self.cfg.stop_margin_pulses:
                    return "target"

            time.sleep(0.001)

    def _finish(self, target: str, position: int, stop_reason: str) -> None:
        self._save_position(position)
        with self._status_lock:
            self._status = MotorStatus(position, target, False, stop_reason,
                                       position - self._target_pulses(target))

    # ── 馬達輸出（對應 motorForward()/motorReverse()/stopWithBrake()/motorCoast()） ──

    def _motor_forward(self) -> None:
        self._motor_direction = 1
        lgpio.gpio_write(self._chip, self.cfg.pin_motor_plus, 1)
        lgpio.gpio_write(self._chip, self.cfg.pin_motor_minus, 0)

    def _motor_reverse(self) -> None:
        self._motor_direction = -1
        lgpio.gpio_write(self._chip, self.cfg.pin_motor_plus, 0)
        lgpio.gpio_write(self._chip, self.cfg.pin_motor_minus, 1)

    def _stop_with_brake(self) -> None:
        # 保留 motor_direction，讓煞車期間的脈衝還能計算方向
        lgpio.gpio_write(self._chip, self.cfg.pin_motor_plus, 1)
        lgpio.gpio_write(self._chip, self.cfg.pin_motor_minus, 1)
        time.sleep(self.cfg.brake_time_ms / 1000.0)
        self._motor_coast()
        self._motor_direction = 0

    def _motor_coast(self) -> None:
        lgpio.gpio_write(self._chip, self.cfg.pin_motor_plus, 0)
        lgpio.gpio_write(self._chip, self.cfg.pin_motor_minus, 0)
