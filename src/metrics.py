"""系統指標取樣：功耗、頻率、降頻、負載、Hailo 使用率、程式各階段耗時 → CSV + 摘要。

跟整合程式跑在同一個程序裡（背景緒，每 interval 秒取樣一次），原因：
Hailo-8 一次只能被一個程序開啟，晶片溫度與各模型推論耗時都只有這個程序拿得到。

哪些是實測、哪些是估算（這台板子沒有外接功率計時的極限）：
    實測   Pi 5 板上功耗：PMIC 每條電源軌的電流 × 電壓加總（vcgencmd pmic_read_adc）
           ARM 頻率與各檔停留時間（cpufreq）、SoC 溫度、get_throttled 旗標、風扇 PWM/轉速
           CPU 使用率、記憶體、Hailo 裝置/各模型使用率（hailortcli monitor）、Hailo 晶片溫度
    估算   Hailo / 相機 / 風扇的瓦數：板子沒有電流感測器（hailortcli measure-power 回 UNSUPPORTED），
           只能拿 configs/metrics.yaml 的規格值依使用率 / PWM 折算，再除以 DC-DC 效率得到整套。
           摘要會把兩者分開列，報告時請註明估算部分。

Hailo 使用率靠 HailoRT 的 monitor：程序要在建 VDevice 前設 HAILO_MONITOR=1（main.py 負責），
之後 `hailortcli monitor` 會每秒印一份純文字表格，這裡開它當子程序、解析 stdout。

重新產摘要：uv run python -m src.metrics outputs/metrics/<檔名>.csv
"""

from __future__ import annotations

import csv
import re
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from .settings import MetricsSettings

# PMIC 電源軌（vcgencmd pmic_read_adc 的名稱去掉 _A / _V 後綴）。EXT5V 只有電壓、沒有電流，另外記
PMIC_RAILS = ["VDD_CORE", "0V8_SW", "1V1_SYS", "1V8_SYS", "3V3_SYS", "DDR_VDD2", "DDR_VDDQ",
              "3V7_WL_SW", "0V8_AON", "HDMI", "3V3_ADC", "3V3_DAC"]
HAILO_MODELS = ["resnet", "asphalt", "cement", "yolo"]
N_CPU = 4

COLUMNS = (
    ["time", "elapsed_s", "board_w"]
    + [f"w_{r.lower()}" for r in PMIC_RAILS]
    + ["ext5v_v", "vdd_core_v",
       "arm_mhz", "arm_avg_mhz", "arm_max_share", "soc_temp", "throttled", "fan_pwm", "fan_rpm",
       "cpu_pct"] + [f"cpu{i}_pct" for i in range(N_CPU)] + ["mem_used_mb",
       "hailo_util", "hailo_temp"]
    + [f"{m}_{k}" for m in HAILO_MODELS for k in ("util", "fps")]
    + ["fps", "road_mode", "road_label", "road_round_ms", "resnet_infer_ms", "grade_infer_ms", "crack_ms",
       "det_round_ms", "yolo_infer_ms", "draw_ms", "write_ms", "show_ms", "recording",
       "est_hailo_w", "est_camera_w", "est_fan_w", "est_total_w"]
)

# get_throttled 的位元：低 4 位是「現在」，左移 16 位是「開機以來曾發生」
THROTTLE_BITS = {0x1: "低電壓", 0x2: "ARM 頻率被限制", 0x4: "正在降頻", 0x8: "到達軟性溫度上限"}

_CPUFREQ = Path("/sys/devices/system/cpu/cpu0/cpufreq")
_THERMAL = Path("/sys/class/thermal/thermal_zone0/temp")
_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _vcgencmd(*args: str) -> str:
    try:
        return subprocess.run(["vcgencmd", *args], capture_output=True, text=True, timeout=2).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


# ── 系統端 ────────────────────────────────────────────────────
def read_pmic() -> dict[str, float]:
    """每條軌的瓦數（key 為軌名）+ 'board_w' 加總 + 'ext5v_v' + 'vdd_core_v'。讀不到就回空 dict。"""
    out = _vcgencmd("pmic_read_adc")
    amps = {m.group(1): float(m.group(2)) for m in re.finditer(r"(\S+)_A current\(\d+\)=([\d.]+)A", out)}
    volts = {m.group(1): float(m.group(2)) for m in re.finditer(r"(\S+)_V volt\(\d+\)=([\d.]+)V", out)}
    if not amps:
        return {}
    watts = {r: amps.get(r, 0.0) * volts.get(r, 0.0) for r in PMIC_RAILS}
    watts["board_w"] = sum(watts.values())
    watts["ext5v_v"] = volts.get("EXT5V", 0.0)
    watts["vdd_core_v"] = volts.get("VDD_CORE", 0.0)
    return watts


def read_throttled() -> int | None:
    m = re.search(r"0x([0-9a-fA-F]+)", _vcgencmd("get_throttled"))
    return int(m.group(1), 16) if m else None


def describe_throttled(bits: int) -> str:
    now = [n for b, n in THROTTLE_BITS.items() if bits & b]
    ever = [n for b, n in THROTTLE_BITS.items() if bits & (b << 16)]
    parts = []
    if now:
        parts.append("現在：" + "、".join(now))
    if ever:
        parts.append("開機以來曾：" + "、".join(ever))
    return "；".join(parts) if parts else "無"


def _time_in_state() -> dict[int, int]:
    """{頻率 kHz: 累計 10ms tick}。"""
    txt = _read(_CPUFREQ / "stats" / "time_in_state") or ""
    out = {}
    for line in txt.splitlines():
        try:
            f, t = line.split()
            out[int(f)] = int(t)
        except ValueError:
            continue
    return out


def _cpu_times() -> list[tuple[int, int]]:
    """[(busy, total)] 依序為 cpu 總計、cpu0、cpu1…（/proc/stat 的 jiffies）。"""
    res = []
    for line in (_read(Path("/proc/stat")) or "").splitlines():
        if not line.startswith("cpu"):
            break
        v = [int(x) for x in line.split()[1:]]
        idle = v[3] + (v[4] if len(v) > 4 else 0)
        res.append((sum(v) - idle, sum(v)))
    return res


def _mem_used_mb() -> float | None:
    info = {}
    for line in (_read(Path("/proc/meminfo")) or "").splitlines():
        k, _, rest = line.partition(":")
        try:
            info[k] = int(rest.split()[0])
        except (ValueError, IndexError):
            pass
    if "MemTotal" in info and "MemAvailable" in info:
        return (info["MemTotal"] - info["MemAvailable"]) / 1024.0
    return None


def _find_hwmon(name: str) -> Path | None:
    for d in Path("/sys/class/hwmon").glob("hwmon*"):
        if _read(d / "name") == name:
            return d
    return None


# ── Hailo monitor ─────────────────────────────────────────────
class HailoMonitor:
    """把 `hailortcli monitor` 開成子程序，解析它每秒刷新的表格：
    裝置使用率、每個模型的使用率與 fps。應用程序沒設 HAILO_MONITOR=1 時表格會是空的。"""

    def __init__(self):
        self.device_util: float | None = None
        self.models: dict[str, tuple[float, float]] = {}      # name -> (util %, fps)
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> bool:
        exe = shutil.which("hailortcli")
        if exe is None:
            return False
        try:
            self._proc = subprocess.Popen([exe, "monitor"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                          text=True, bufsize=1)
        except OSError:
            return False
        self._thread = threading.Thread(target=self._reader, daemon=True, name="hailo-monitor")
        self._thread.start()
        return True

    def _reader(self) -> None:
        section = None
        assert self._proc and self._proc.stdout
        for raw in self._proc.stdout:
            line = _ANSI.sub("", raw).strip()
            if not line:
                continue
            if line.startswith("Device ID"):
                section = "device"
                continue
            if line.startswith("Model") and "FPS" in line:
                section = "models"
                with self._lock:
                    self.models = {}
                continue
            if line.startswith("Model") and "Stream" in line:
                section = None
                continue
            if line.startswith("-") or line.startswith("Avg"):
                continue
            parts = line.split()
            try:
                if section == "device" and len(parts) >= 3:
                    with self._lock:
                        self.device_util = float(parts[1])
                elif section == "models" and len(parts) >= 4:
                    with self._lock:
                        self.models[parts[0]] = (float(parts[1]), float(parts[2]))
            except ValueError:
                continue

    def snapshot(self) -> tuple[float | None, dict[str, tuple[float, float]]]:
        with self._lock:
            return self.device_util, dict(self.models)

    def stop(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None


# ── 取樣緒 ────────────────────────────────────────────────────
class MetricsRecorder(threading.Thread):
    """每 interval 秒取一列寫進 CSV；stop() 後用 summary() 拿摘要文字。

    app_stats：主程式提供的 callable，回傳 dict（fps、road_mode、各階段 ms、recording 等，
    key 對應 COLUMNS 裡程式那一段）；hailo_temp 也由它提供（要用到 VDevice）。
    hailo_names：{"resnet": HEF 檔名 stem, ...}，用來把 monitor 表格裡的模型名對到固定欄位。
    """

    def __init__(self, cfg: MetricsSettings, app_stats: Callable[[], dict], hailo_names: dict[str, str]):
        super().__init__(daemon=True, name="metrics")
        self.cfg = cfg
        self.app_stats = app_stats
        self.hailo_names = hailo_names
        self._stop_evt = threading.Event()
        self.rows: list[dict] = []
        self.throttled_start: int | None = None

        cfg.dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = cfg.dir / f"{stamp}.csv"
        self.summary_path = cfg.dir / f"{stamp}_summary.txt"
        self._fh = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=COLUMNS)
        self._writer.writeheader()

        self._monitor = HailoMonitor()
        self._fan = _find_hwmon("pwmfan")
        self._max_khz = int(_read(_CPUFREQ / "cpuinfo_max_freq") or 0)
        self._unmatched: set[str] = set()

    def stop(self) -> None:
        self._stop_evt.set()

    # 主緒呼叫：把 monitor 的模型名對到 resnet/asphalt/cement/yolo
    def _match_model(self, mon_name: str) -> str | None:
        for key, stem in self.hailo_names.items():
            if mon_name == stem or mon_name in stem or stem in mon_name:
                return key
        if mon_name not in self._unmatched:
            self._unmatched.add(mon_name)
            print(f"[metrics] hailortcli monitor 裡的模型 {mon_name} 對不到任何欄位，略過", file=sys.stderr)
        return None

    def run(self) -> None:
        if not self._monitor.start():
            print("[metrics] 找不到 hailortcli，Hailo 使用率欄位會是空的", file=sys.stderr)
        if not read_pmic():
            print("[metrics] vcgencmd pmic_read_adc 讀不到資料，功耗欄位會是空的", file=sys.stderr)
        self.throttled_start = read_throttled()

        t_start = time.perf_counter()
        prev_tis = _time_in_state()
        prev_cpu = _cpu_times()
        next_t = t_start
        while not self._stop_evt.is_set():
            next_t += self.cfg.interval
            row: dict = {"time": datetime.now().strftime("%H:%M:%S"),
                         "elapsed_s": round(time.perf_counter() - t_start, 1)}

            # 功耗
            pm = read_pmic()
            if pm:
                row["board_w"] = round(pm["board_w"], 3)
                for r in PMIC_RAILS:
                    row[f"w_{r.lower()}"] = round(pm[r], 4)
                row["ext5v_v"] = round(pm["ext5v_v"], 3)
                row["vdd_core_v"] = round(pm["vdd_core_v"], 4)

            # 頻率：瞬時值 + 這一段 interval 內各檔停留時間（時間加權平均、最高檔佔比）
            cur = _read(_CPUFREQ / "scaling_cur_freq")
            if cur:
                row["arm_mhz"] = int(cur) // 1000
            tis = _time_in_state()
            delta = {f: tis[f] - prev_tis.get(f, 0) for f in tis}
            total = sum(delta.values())
            if total > 0:
                row["arm_avg_mhz"] = round(sum(f * t for f, t in delta.items()) / total / 1000)
                row["arm_max_share"] = round(delta.get(self._max_khz, 0) / total, 3)
            prev_tis = tis

            # 溫度 / 降頻 / 風扇
            temp = _read(_THERMAL)
            if temp:
                row["soc_temp"] = round(int(temp) / 1000, 1)
            thr = read_throttled()
            if thr is not None:
                row["throttled"] = f"0x{thr:x}"
            if self._fan:
                row["fan_pwm"] = _read(self._fan / "pwm1")
                row["fan_rpm"] = _read(self._fan / "fan1_input")

            # 負載
            cpu = _cpu_times()
            if cpu and prev_cpu and len(cpu) == len(prev_cpu):
                for i, ((b1, t1), (b0, t0)) in enumerate(zip(cpu, prev_cpu)):
                    key = "cpu_pct" if i == 0 else f"cpu{i - 1}_pct"
                    if key in COLUMNS and t1 > t0:
                        row[key] = round(100.0 * (b1 - b0) / (t1 - t0), 1)
            prev_cpu = cpu
            mem = _mem_used_mb()
            if mem is not None:
                row["mem_used_mb"] = round(mem)

            # Hailo
            dev_util, models = self._monitor.snapshot()
            if dev_util is not None:
                row["hailo_util"] = dev_util
            for name, (util, fps) in models.items():
                key = self._match_model(name)
                if key:
                    row[f"{key}_util"] = util
                    row[f"{key}_fps"] = fps

            # 程式端
            try:
                app = self.app_stats()
            except Exception as e:      # 取樣不能弄垮主程式
                app = {}
                print(f"[metrics] 讀取程式狀態失敗: {e}", file=sys.stderr)
            for k, v in app.items():
                if k in COLUMNS and v is not None:
                    row[k] = round(v, 2) if isinstance(v, float) else v

            apply_estimates(row, self.cfg)

            self.rows.append(row)
            self._writer.writerow(row)
            self._fh.flush()

            self._stop_evt.wait(timeout=max(0.0, next_t - time.perf_counter()))

        self._monitor.stop()
        self._fh.close()

    def summary(self) -> str:
        text = summarize(self.rows, self.csv_path, self.throttled_start)
        try:
            self.summary_path.write_text(text + "\n", encoding="utf-8")
        except OSError as e:
            print(f"[metrics] 寫摘要失敗: {e}", file=sys.stderr)
        return text


def apply_estimates(row: dict, c: MetricsSettings) -> None:
    """依 metrics.yaml 的估算參數填 est_* 欄位。取樣時呼叫；重算摘要時也用它，改了參數不用重跑。"""
    def f(key):
        v = row.get(key)
        try:
            return float(v) if v not in (None, "") else None
        except (TypeError, ValueError):
            return None

    util, pwm, board = f("hailo_util"), f("fan_pwm"), f("board_w")
    est_hailo = c.hailo_idle_w + (c.hailo_full_w - c.hailo_idle_w) * (util / 100.0 if util is not None else 0.0)
    est_fan = c.fan_full_w * pwm / 255.0 if pwm else 0.0
    row["est_hailo_w"] = round(est_hailo, 3)
    row["est_camera_w"] = c.camera_w
    row["est_fan_w"] = round(est_fan, 3)
    row["est_total_w"] = round((board + est_hailo + c.camera_w + est_fan) / c.efficiency, 3) if board is not None else ""


# ── 摘要 ──────────────────────────────────────────────────────
def _num(rows: list[dict], key: str) -> list[float]:
    out = []
    for r in rows:
        v = r.get(key)
        if v in (None, ""):
            continue
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            continue
    return out


def _stat(rows: list[dict], key: str, fmt: str = "{:.2f}", unit: str = "") -> str:
    v = _num(rows, key)
    if not v:
        return "（無資料）"
    return (f"平均 {fmt.format(sum(v) / len(v))}{unit}，最高 {fmt.format(max(v))}{unit}，"
            f"最低 {fmt.format(min(v))}{unit}")


def _mean(rows: list[dict], key: str) -> float | None:
    v = _num(rows, key)
    return sum(v) / len(v) if v else None


def summarize(rows: list[dict], csv_path: Path | None = None, throttled_start: int | None = None) -> str:
    L: list[str] = []
    n = len(rows)
    dur = float(rows[-1]["elapsed_s"]) if rows else 0.0
    L.append("═══ 指標摘要 ═══")
    if csv_path:
        L.append(f"檔案：{csv_path}")
    L.append(f"{n} 筆取樣，共 {dur:.0f} 秒（{dur / 60:.1f} 分）")
    if not rows:
        return "\n".join(L)

    # 功耗
    L.append("")
    L.append("[功耗]")
    L.append(f"  Pi 5 板上實測（PMIC 各軌加總）：{_stat(rows, 'board_w', unit=' W')}")
    rails = sorted(((_mean(rows, f'w_{r.lower()}') or 0.0, r) for r in PMIC_RAILS), reverse=True)[:5]
    L.append("    主要電源軌平均：" + "、".join(f"{r} {w:.2f} W" for w, r in rails))
    ext = _num(rows, "ext5v_v")
    if ext:
        L.append(f"    5V 輸入電壓：平均 {sum(ext) / len(ext):.3f} V，最低 {min(ext):.3f} V"
                 + ("（< 4.8 V 表示供電吃緊）" if min(ext) < 4.8 else ""))
    eh, ec, ef = _mean(rows, "est_hailo_w"), _mean(rows, "est_camera_w"), _mean(rows, "est_fan_w")
    hu = _mean(rows, "hailo_util")
    L.append(f"  估算（規格值折算，非實測）：Hailo-8 {eh or 0:.2f} W"
             + (f"（使用率平均 {hu:.0f}%）" if hu is not None else "（沒有使用率資料，以待機值計）")
             + f"、相機 {ec or 0:.2f} W、風扇 {ef or 0:.2f} W")
    L.append(f"  整套估算（含 DC-DC 效率換算）：{_stat(rows, 'est_total_w', unit=' W')}")
    mean_total = _mean(rows, "est_total_w")
    if mean_total and dur > 0:
        L.append(f"    ≈ 每小時 {mean_total:.2f} Wh，這次執行共 {mean_total * dur / 3600:.2f} Wh")

    # 頻率
    L.append("")
    L.append("[SoC 頻率]")
    avg = _num(rows, "arm_avg_mhz")
    share = _num(rows, "arm_max_share")
    inst = _num(rows, "arm_mhz")
    if avg:
        L.append(f"  ARM 時間加權平均 {sum(avg) / len(avg):.0f} MHz，"
                 f"最高檔（{max(inst):.0f} MHz）佔 {100 * sum(share) / len(share):.1f}% 的時間，"
                 f"取樣到的最低瞬時值 {min(inst):.0f} MHz" if inst else "")
    else:
        L.append("  （無資料）")

    # 溫度 / 降頻
    L.append("")
    L.append("[溫度 / 降頻]")
    L.append(f"  SoC：{_stat(rows, 'soc_temp', '{:.1f}', ' °C')}")
    if _num(rows, "hailo_temp"):
        L.append(f"  Hailo-8 晶片：{_stat(rows, 'hailo_temp', '{:.1f}', ' °C')}")
    if _num(rows, "fan_pwm"):
        L.append(f"  風扇 PWM：{_stat(rows, 'fan_pwm', '{:.0f}')}（0~255）；轉速 {_stat(rows, 'fan_rpm', '{:.0f}', ' rpm')}")
    seen_now = 0
    last = None
    for r in rows:
        v = r.get("throttled")
        if v:
            bits = int(str(v), 16)
            seen_now |= bits & 0xF
            last = bits
    if last is None:
        L.append("  get_throttled：（無資料）")
    else:
        new_ever = (last >> 16) & 0xF & ~(((throttled_start or 0) >> 16) & 0xF)
        if seen_now or new_ever:
            what = [n for b, n in THROTTLE_BITS.items() if (seen_now | new_ever) & b]
            L.append(f"  get_throttled：執行期間出現 → {'、'.join(what)}（最後值 0x{last:x}）")
        else:
            L.append(f"  get_throttled：執行期間沒有出現任何低電壓 / 降頻旗標（最後值 0x{last:x}"
                     + (f"，開機以來曾：{describe_throttled(last & 0xF0000)[6:]}" if last & 0xF0000 else "") + "）")

    # 負載
    L.append("")
    L.append("[負載]")
    L.append(f"  CPU 總體：{_stat(rows, 'cpu_pct', '{:.0f}', '%')}；各核平均 "
             + " / ".join(f"{_mean(rows, f'cpu{i}_pct') or 0:.0f}%" for i in range(N_CPU)))
    L.append(f"  記憶體使用：{_stat(rows, 'mem_used_mb', '{:.0f}', ' MB')}")
    if hu is not None:
        L.append(f"  Hailo 裝置使用率：{_stat(rows, 'hailo_util', '{:.0f}', '%')}")
        per = []
        for m in HAILO_MODELS:
            u, f = _mean(rows, f"{m}_util"), _mean(rows, f"{m}_fps")
            if u is not None:
                per.append(f"{m} {u:.0f}% / {f or 0:.1f} fps")
        if per:
            L.append("    各模型（使用率 / fps）：" + "、".join(per))

    # 程式
    L.append("")
    L.append("[程式]")
    L.append(f"  顯示 fps：{_stat(rows, 'fps', '{:.1f}')}")
    modes = {}
    for r in rows:
        m = r.get("road_mode")
        if m:
            modes[m] = modes.get(m, 0) + 1
    if modes:
        L.append("  路面模式時間占比：" + "、".join(f"{m} {100 * c / n:.0f}%" for m, c in sorted(modes.items())))
    def ms(key):
        v = _mean(rows, key)
        return f"{v:.1f} ms" if v is not None else "—"
    L.append(f"  路面分析緒每輪 {ms('road_round_ms')}（ResNet 推論 {ms('resnet_infer_ms')}、"
             f"分級推論 {ms('grade_infer_ms')}、裂縫偵測 {ms('crack_ms')}）")
    L.append(f"  偵測緒每輪 {ms('det_round_ms')}（YOLO 推論 {ms('yolo_infer_ms')}）")
    L.append(f"  主緒每幀：疊圖 {ms('draw_ms')}、寫檔 {ms('write_ms')}、顯示 {ms('show_ms')}")
    rec = _num(rows, "recording")
    if rec:
        L.append(f"  錄影中的時間占比：{100 * sum(rec) / len(rec):.0f}%")
    return "\n".join(L)


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("用法: python -m src.metrics outputs/metrics/<檔名>.csv", file=sys.stderr)
        return 2
    from .settings import SettingsError, load_settings
    path = Path(argv[0])
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # 估算欄位依「現在的」metrics.yaml 重算，所以改了估算參數只要重跑這個指令，不用重新量
    try:
        cfg = load_settings().metrics
    except SettingsError as e:
        print(f"設定錯誤，估算欄位沿用 CSV 裡的值: {e}", file=sys.stderr)
    else:
        for r in rows:
            apply_estimates(r, cfg)
        print(f"（估算參數取自 configs/metrics.yaml：Hailo {cfg.hailo_idle_w}~{cfg.hailo_full_w} W、"
              f"相機 {cfg.camera_w} W、風扇 {cfg.fan_full_w} W、效率 {cfg.efficiency}）")
    print(summarize(rows, path))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
