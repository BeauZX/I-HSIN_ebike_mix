"""讀取 configs/ 底下各專案對應的 YAML，整理成程式好用的 dataclass 並先驗證。

一個專案一個檔：
    camera.yaml     鏡頭
    road_type.yaml  路面種類（ResNet）+ 共用 ROI
    asphalt.yaml    瀝青分級
    cement.yaml     水泥分級 + 裂縫
    detect.yaml     人車偵測 / 追蹤 / 警戒區
    door.yaml       車門開啟 / 關閉偵測
    output.yaml     畫面、錄影與偵測結果 log
    metrics.yaml    系統指標取樣（功耗 / 頻率 / 降頻）與估算參數

設定檔裡的相對路徑一律相對於專案根目錄，在哪個目錄下執行都一樣。
模型載入要好幾秒，所以先把明顯寫錯的值擋下來。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "configs"


class SettingsError(Exception):
    pass


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise SettingsError(f"找不到設定檔: {path}")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise SettingsError(f"{path.name} 的最上層必須是 key: value")
    return data


def _require_file(path: Path, what: str, cfg_name: str) -> Path:
    if not path.exists():
        raise SettingsError(f"找不到{what}: {path}（{cfg_name}）")
    return path


@dataclass
class CameraSettings:
    index: int
    width: int
    height: int
    fps: int
    rotation: int
    exposure: str
    awb: str
    denoise: str
    warmup: float


@dataclass
class RoadTypeSettings:
    model_dir: Path
    hef: Path
    grading: dict[str, str]       # 類別名稱 → "asphalt" / "cement"
    smooth_window: int
    confirm_count: int
    roi_store: Path
    roi_window_max_w: int
    roi_window_max_h: int


@dataclass
class GridSettings:
    rows: int
    cols: int


@dataclass
class AsphaltSettings:
    hef: Path
    class_names: list[str]
    grid: GridSettings
    ema_alpha: float
    switch_margin: float


@dataclass
class CementSettings:
    hef: Path
    class_names: list[str]
    grid: GridSettings
    crack_preset: Path
    fusion: dict
    smoother: str                 # "pid" / "ema"
    pid_params: Path
    ema_alpha: float
    ema_switch_margin: float


@dataclass
class DetectSettings:
    hef: Path
    conf: float
    classes: dict[int, str]
    assoc_dist_px: float
    max_miss: int
    prediction_horizon_sec: float
    alert_cooldown_sec: float
    zone_store: Path
    default_zone: list[float]


@dataclass
class DoorSettings:
    hef: Path
    conf: float
    class_names: list[str]
    open_classes: list[str]


@dataclass
class MotorSettings:
    pin_motor_plus: int
    pin_motor_minus: int
    pin_pulse: int
    pulses_per_rev: int
    pulse_filter_us: int
    pos_tight: int
    pos_mid: int
    pos_loose: int
    stop_margin_pulses: int
    position_tolerance: int
    brake_time_ms: int
    move_timeout_ms: int
    stall_ratio_num: int
    stall_ratio_den: int
    min_stall_floor_us: int
    start_timeout_ms: int
    stall_timeout_ms: int
    pothole_severe_cell_threshold: int
    position_store: Path
    confirm_count: int
    min_switch_interval_sec: float


@dataclass
class OutputSettings:
    width: int              # 顯示與錄影的畫面尺寸（相機擷取尺寸在 CameraSettings）
    height: int
    dir: Path
    segment_seconds: float
    fps: float
    codec: str
    window_title: str
    log_dir: Path
    log_stable_sec: float


@dataclass
class MetricsSettings:
    enabled: bool
    interval: float
    dir: Path
    hailo_idle_w: float
    hailo_k_w: float
    camera_w: float
    fan_full_w: float
    usb_5v_w: float
    misc_5v_w: float
    efficiency: float
    hat_efficiency: float
    battery_cells: int
    battery_capacity_mah: float
    battery_cutoff_v_per_cell: float
    battery_buck_efficiency: float


@dataclass
class Settings:
    camera: CameraSettings
    road_type: RoadTypeSettings
    asphalt: AsphaltSettings
    cement: CementSettings
    detect: DetectSettings
    door: DoorSettings
    motor: MotorSettings
    output: OutputSettings
    metrics: MetricsSettings
    config_dir: Path = field(default=DEFAULT_CONFIG_DIR)


def _grid(raw: dict, cfg_name: str) -> GridSettings:
    g = raw.get("grid", {})
    rows, cols = int(g.get("rows", 3)), int(g.get("cols", 5))
    if rows < 1 or cols < 1:
        raise SettingsError(f"grid.rows / grid.cols 必須至少是 1（{cfg_name}）")
    return GridSettings(rows, cols)


def load_settings(config_dir: Path | str = DEFAULT_CONFIG_DIR) -> Settings:
    config_dir = Path(config_dir)
    if not config_dir.is_dir():
        raise SettingsError(f"找不到設定資料夾: {config_dir}")

    # ── camera ──
    c = _load_yaml(config_dir / "camera.yaml")
    camera = CameraSettings(
        index=int(c.get("index", 0)),
        width=int(c.get("width", 1280)),
        height=int(c.get("height", 720)),
        fps=int(c.get("fps", 25)),
        rotation=int(c.get("rotation", 0)),
        exposure=str(c.get("exposure", "normal")),
        awb=str(c.get("awb", "auto")),
        denoise=str(c.get("denoise", "cdn_off")),
        warmup=float(c.get("warmup", 1.0)),
    )
    if camera.rotation not in (0, 180):
        raise SettingsError(f"camera.rotation 只能是 0 或 180: {camera.rotation}")
    if camera.fps < 1:
        raise SettingsError("camera.fps 必須大於 0")

    # ── road_type ──
    r = _load_yaml(config_dir / "road_type.yaml")
    model_dir = _resolve(r.get("model_dir", "models/resnet"))
    road_type = RoadTypeSettings(
        model_dir=model_dir,
        hef=_require_file(_resolve(r.get("hef", model_dir / "model.hef")), "ResNet HEF", "road_type.yaml"),
        grading={str(k): str(v) for k, v in (r.get("grading") or {}).items()},
        smooth_window=max(1, int(r.get("smooth_window", 5))),
        confirm_count=max(1, int(r.get("confirm_count", 3))),
        roi_store=_resolve(r.get("roi_store", "presets/roi.json")),
        roi_window_max_w=int(r.get("roi_window_max_w", 1600)),
        roi_window_max_h=int(r.get("roi_window_max_h", 900)),
    )
    for what in ("config.json", "classes.txt"):
        _require_file(model_dir / what, f"ResNet 的 {what}", "road_type.yaml")
    bad = {v for v in road_type.grading.values()} - {"asphalt", "cement"}
    if bad:
        raise SettingsError(f"road_type.yaml 的 grading 只能對應到 asphalt 或 cement，不認得: {bad}")

    # ── asphalt ──
    a = _load_yaml(config_dir / "asphalt.yaml")
    asphalt = AsphaltSettings(
        hef=_require_file(_resolve(a.get("hef", "models/asphalt_cls.hef")), "asphalt HEF", "asphalt.yaml"),
        class_names=[str(n) for n in a.get("class_names", [])],
        grid=_grid(a, "asphalt.yaml"),
        ema_alpha=float(a.get("ema_alpha", 0.45)),
        switch_margin=float(a.get("switch_margin", 0.1)),
    )
    if not asphalt.class_names:
        raise SettingsError("asphalt.yaml 缺 class_names（HEF 裡沒有類別名稱）")

    # ── cement ──
    m = _load_yaml(config_dir / "cement.yaml")
    ema = m.get("ema", {})
    cement = CementSettings(
        hef=_require_file(_resolve(m.get("hef", "models/cement_cls.hef")), "cement HEF", "cement.yaml"),
        class_names=[str(n) for n in m.get("class_names", [])],
        grid=_grid(m, "cement.yaml"),
        crack_preset=_require_file(_resolve(m.get("crack_preset", "presets/rough.json")),
                                   "裂縫參數檔", "cement.yaml"),
        fusion=dict(m.get("fusion") or {}),
        smoother=str(m.get("smoother", "pid")).lower(),
        pid_params=_resolve(m.get("pid_params", "presets/pid.json")),
        ema_alpha=float(ema.get("alpha", 0.45)),
        ema_switch_margin=float(ema.get("switch_margin", 0.08)),
    )
    if not cement.class_names:
        raise SettingsError("cement.yaml 缺 class_names（HEF 裡沒有類別名稱）")
    if cement.smoother not in ("pid", "ema"):
        raise SettingsError(f"cement.yaml 的 smoother 只能是 pid 或 ema: {cement.smoother}")

    # ── detect ──
    d = _load_yaml(config_dir / "detect.yaml")
    t = d.get("tracker", {})
    zone = [float(v) for v in d.get("default_zone", [0.27, 0.28, 0.73, 0.90])]
    if len(zone) != 4:
        raise SettingsError("detect.yaml 的 default_zone 要有 4 個值（x1, y1, x2, y2，0~1 比例）")
    detect = DetectSettings(
        hef=_require_file(_resolve(d.get("hef", "models/yolov8n.hef")), "YOLO HEF", "detect.yaml"),
        conf=float(d.get("conf", 0.4)),
        classes={int(k): str(v) for k, v in (d.get("classes") or {}).items()},
        assoc_dist_px=float(t.get("assoc_dist_px", 160)),
        max_miss=int(t.get("max_miss", 4)),
        prediction_horizon_sec=float(d.get("prediction_horizon_sec", 2.0)),
        alert_cooldown_sec=float(d.get("alert_cooldown_sec", 1.5)),
        zone_store=_resolve(d.get("zone_store", "presets/warning_zone.json")),
        default_zone=zone,
    )
    if not detect.classes:
        raise SettingsError("detect.yaml 的 classes 不能是空的")

    # ── door ──
    dr = _load_yaml(config_dir / "door.yaml")
    door = DoorSettings(
        hef=_require_file(_resolve(dr.get("hef", "models/car_door_yolov11m.hef")), "車門 HEF", "door.yaml"),
        conf=float(dr.get("conf", 0.4)),
        class_names=[str(n) for n in dr.get("class_names", [])],
        open_classes=[str(n) for n in dr.get("open_classes", ["open"])],
    )
    if not door.class_names:
        raise SettingsError("door.yaml 缺 class_names（HEF 裡沒有類別名稱）")
    bad = set(door.open_classes) - set(door.class_names)
    if bad:
        raise SettingsError(f"door.yaml 的 open_classes 不在 class_names 裡: {bad}")

    # ── motor ──
    mo = _load_yaml(config_dir / "motor.yaml")
    motor = MotorSettings(
        pin_motor_plus=int(mo.get("pin_motor_plus", 5)),
        pin_motor_minus=int(mo.get("pin_motor_minus", 6)),
        pin_pulse=int(mo.get("pin_pulse", 13)),
        pulses_per_rev=int(mo.get("pulses_per_rev", 692)),
        pulse_filter_us=int(mo.get("pulse_filter_us", 150)),
        pos_tight=int(mo.get("pos_tight", 0)),
        pos_mid=int(mo.get("pos_mid", -280)),
        pos_loose=int(mo.get("pos_loose", -560)),
        stop_margin_pulses=int(mo.get("stop_margin_pulses", 1)),
        position_tolerance=int(mo.get("position_tolerance", 4)),
        brake_time_ms=int(mo.get("brake_time_ms", 100)),
        move_timeout_ms=int(mo.get("move_timeout_ms", 5000)),
        stall_ratio_num=int(mo.get("stall_ratio_num", 17)),
        stall_ratio_den=int(mo.get("stall_ratio_den", 10)),
        min_stall_floor_us=int(mo.get("min_stall_floor_us", 100000)),
        start_timeout_ms=int(mo.get("start_timeout_ms", 500)),
        stall_timeout_ms=int(mo.get("stall_timeout_ms", 400)),
        pothole_severe_cell_threshold=int(mo.get("pothole_severe_cell_threshold", 2)),
        position_store=_resolve(mo.get("position_store", "presets/motor_position.json")),
        confirm_count=max(1, int(mo.get("confirm_count", 5))),
        min_switch_interval_sec=float(mo.get("min_switch_interval_sec", 3.0)),
    )
    if len({motor.pin_motor_plus, motor.pin_motor_minus, motor.pin_pulse}) != 3:
        raise SettingsError("motor.yaml 的 pin_motor_plus / pin_motor_minus / pin_pulse 不能重複")
    if motor.pulses_per_rev <= 0:
        raise SettingsError("motor.yaml 的 pulses_per_rev 必須大於 0")
    if not motor.pos_loose <= motor.pos_mid <= motor.pos_tight:
        raise SettingsError("motor.yaml 的位置必須滿足 pos_loose <= pos_mid <= pos_tight"
                            f"（目前 {motor.pos_loose} / {motor.pos_mid} / {motor.pos_tight}）")
    if motor.stall_ratio_num <= 0 or motor.stall_ratio_den <= 0:
        raise SettingsError("motor.yaml 的 stall_ratio_num / stall_ratio_den 必須大於 0")
    if motor.pothole_severe_cell_threshold < 0:
        raise SettingsError("motor.yaml 的 pothole_severe_cell_threshold 不能是負的")
    if motor.min_switch_interval_sec < 0:
        raise SettingsError("motor.yaml 的 min_switch_interval_sec 不能是負的")

    # ── output ──
    o = _load_yaml(config_dir / "output.yaml")
    output = OutputSettings(
        width=int(o.get("width", camera.width)),
        height=int(o.get("height", camera.height)),
        dir=_resolve(o.get("dir", "outputs")),
        segment_seconds=float(o.get("segment_seconds", 60)),
        fps=float(o.get("fps", camera.fps)),
        codec=str(o.get("codec", "mp4v")),
        window_title=str(o.get("window_title", "Road Integration")),
        log_dir=_resolve(o.get("log_dir", "outputs/logs")),
        log_stable_sec=float(o.get("log_stable_sec", 0.5)),
    )
    if output.width < 1 or output.height < 1:
        raise SettingsError("output.yaml 的 width / height 必須大於 0")
    # 容許 1% 誤差：IMX219 原生 3280×2464 不是精確的 4:3（縮成 1280×960 差 0.16%，看不出變形）
    aspect_err = (output.width / output.height) / (camera.width / camera.height) - 1
    if abs(aspect_err) > 0.01:
        raise SettingsError(f"output.yaml 的畫面 {output.width}×{output.height} 與相機 "
                            f"{camera.width}×{camera.height} 長寬比不同，縮放會變形")
    if output.log_stable_sec < 0:
        raise SettingsError("output.yaml 的 log_stable_sec 不能是負的")
    if output.segment_seconds <= 0:
        raise SettingsError("output.yaml 的 segment_seconds 必須大於 0")
    if not output.window_title.isascii():
        raise SettingsError("output.yaml 的 window_title 只能用 ASCII（OpenCV Qt 的限制）")

    # ── metrics ──
    mt = _load_yaml(config_dir / "metrics.yaml")
    est = mt.get("estimate") or {}
    bat = mt.get("battery") or {}
    metrics = MetricsSettings(
        enabled=bool(mt.get("enabled", False)),
        interval=float(mt.get("interval", 1.0)),
        dir=_resolve(mt.get("dir", "outputs/metrics")),
        hailo_idle_w=float(est.get("hailo_idle_w", 0.5)),
        hailo_k_w=float(est.get("hailo_k_w", 0.76)),
        camera_w=float(est.get("camera_w", 0.0)),
        fan_full_w=float(est.get("fan_full_w", 0.4)),
        usb_5v_w=float(est.get("usb_5v_w", 0.10)),
        misc_5v_w=float(est.get("misc_5v_w", 0.15)),
        efficiency=float(est.get("efficiency", 0.88)),
        hat_efficiency=float(est.get("hat_efficiency", 0.90)),
        battery_cells=int(bat.get("cells", 4)),
        battery_capacity_mah=float(bat.get("capacity_mah", 2600)),
        battery_cutoff_v_per_cell=float(bat.get("cutoff_v_per_cell", 3.70)),
        battery_buck_efficiency=float(bat.get("buck_efficiency", 0.88)),
    )
    if metrics.interval <= 0:
        raise SettingsError("metrics.yaml 的 interval 必須大於 0")
    if not 0 < metrics.efficiency <= 1:
        raise SettingsError("metrics.yaml 的 estimate.efficiency 必須在 0~1 之間")
    if not 0 < metrics.hat_efficiency <= 1:
        raise SettingsError("metrics.yaml 的 estimate.hat_efficiency 必須在 0~1 之間")

    return Settings(camera, road_type, asphalt, cement, detect, door, motor, output, metrics, config_dir)
