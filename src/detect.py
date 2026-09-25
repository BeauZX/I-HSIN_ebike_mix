"""人車偵測 + Kalman 追蹤 + 警戒區 + 警報（複製自 OverlayView 專案 OverayView.py，改動處見各段註解）。

改動：
    * HailoDetector 改走共用的 HailoModel，輸入直接吃 Camera.read() 備好的 640×640 RGB
      （ISP 出的顯示畫面再縮一次），不再自己 cvtColor + resize。
    * 警戒區改存 0~1 比例到 json（放開滑鼠即存），下次自動沿用；原版關掉就沒了。
    * 偵測與追蹤搬到背景緒（DetectorThread），主緒只拿最新結果畫圖，不等 Hailo。
    * 畫面翻轉改由 ISP 處理（camera.yaml rotation），拿掉 cv2.flip。
"""

from __future__ import annotations

import colorsys
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .hailo import HailoModel
from .roi import select_rect
from .settings import DetectSettings

# 視窗標題只能用 ASCII（理由見 roi.py）
ZONE_WINDOW_TITLE = "Select warning zone - drag, ENTER to confirm, C to cancel"


# ── 1. 卡爾曼濾波器 ─────────────────────────────────────────
class SimpleKalman2D:
    """2D 定速卡爾曼濾波器，狀態向量 [x, y, vx, vy]。"""

    def __init__(self, x0: float, y0: float, init_vel: float = 0.0):
        self.X = np.array([x0, y0, init_vel, init_vel], dtype=np.float32)
        self.P = np.diag([50, 50, 100, 100]).astype(np.float32)
        self.Q = np.diag([1, 1, 10, 10]).astype(np.float32)
        self.R = np.diag([25, 25]).astype(np.float32)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        self.F = np.eye(4, dtype=np.float32)
        self._vx = 0.0
        self._vy = 0.0
        self.alpha = 0.5  # 速度平滑係數

    def set_delta_t(self, dt: float):
        dt = max(1e-3, dt)
        self.F = np.eye(4, dtype=np.float32)
        self.F[0, 2] = dt
        self.F[1, 3] = dt
        s = max(1e-3, dt)
        self.Q = np.diag([s, s, 10 * s, 10 * s]).astype(np.float32)

    def predict(self):
        self.X = self.F @ self.X
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, zx: float, zy: float):
        z = np.array([zx, zy], dtype=np.float32)
        y = z - self.H @ self.X
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.X = self.X + K @ y
        I_KH = np.eye(4) - K @ self.H
        self.P = I_KH @ self.P
        new_vx = float(self.X[2])
        new_vy = float(self.X[3])
        self._vx = (1 - self.alpha) * self._vx + self.alpha * new_vx
        self._vy = (1 - self.alpha) * self._vy + self.alpha * new_vy

    @property
    def x(self): return float(self.X[0])
    @property
    def y(self): return float(self.X[1])
    @property
    def vx(self): return self._vx
    @property
    def vy(self): return self._vy


# ── 2. 軌跡物件 ─────────────────────────────────────────────
@dataclass
class Track:
    id: int
    kf: SimpleKalman2D
    last_update_ms: float
    miss: int = 0
    history: list = field(default_factory=list)
    MAX_HISTORY: int = 32

    @classmethod
    def create(cls, track_id: int, cx: float, cy: float, now_ms: float):
        t = cls(id=track_id, kf=SimpleKalman2D(cx, cy), last_update_ms=now_ms)
        t.history.append((cx, cy))
        return t

    def predict(self, now_ms: float):
        dt = max(0.001, (now_ms - self.last_update_ms) / 1000.0)
        self.kf.set_delta_t(dt)
        self.kf.predict()

    def update(self, cx: float, cy: float, now_ms: float):
        dt = max(0.001, (now_ms - self.last_update_ms) / 1000.0)
        self.kf.set_delta_t(dt)
        self.kf.update(cx, cy)
        self.last_update_ms = now_ms
        self.history.append((self.kf.x, self.kf.y))
        if len(self.history) > self.MAX_HISTORY:
            self.history.pop(0)
        self.miss = 0

    def predicted_point(self, future_sec: float) -> tuple:
        return (self.kf.x + self.kf.vx * future_sec, self.kf.y + self.kf.vy * future_sec)


# ── 3. 追蹤器 ───────────────────────────────────────────────
class Tracker:
    def __init__(self, assoc_dist_px: float = 160.0, max_miss: int = 4):
        self.tracks: list[Track] = []
        self.next_id = 1
        self.assoc_dist_px = assoc_dist_px
        self.max_miss = max_miss

    def update(self, detections: list[tuple], now_ms: float) -> list[Track]:
        """detections: list of (cx, cy)。回傳目前所有存活 Track。"""
        for t in self.tracks:
            t.predict(now_ms)

        matched_track_ids = set()
        matched_det_ids = set()
        for di, (cx, cy) in enumerate(detections):     # 貪婪最近鄰配對
            best_dist = self.assoc_dist_px
            best_tid = None
            for t in self.tracks:
                if t.id in matched_track_ids:
                    continue
                dist = np.hypot(t.kf.x - cx, t.kf.y - cy)
                if dist < best_dist:
                    best_dist = dist
                    best_tid = t.id
            if best_tid is not None:
                for t in self.tracks:
                    if t.id == best_tid:
                        t.update(cx, cy, now_ms)
                        break
                matched_track_ids.add(best_tid)
                matched_det_ids.add(di)

        for di, (cx, cy) in enumerate(detections):     # 未匹配偵測 → 新 track
            if di not in matched_det_ids:
                new_track = Track.create(self.next_id, cx, cy, now_ms)
                self.tracks.append(new_track)
                matched_track_ids.add(new_track.id)
                self.next_id += 1

        for t in self.tracks:                          # 未匹配 track → miss++
            if t.id not in matched_track_ids:
                t.miss += 1
        self.tracks = [t for t in self.tracks if t.miss <= self.max_miss]
        return self.tracks


# ── 4. 偵測器（Hailo NMS 輸出解析）────────────────────────────
Detection = tuple[int, int, str, float, int, int, int, int]   # cx, cy, label, score, x1, y1, x2, y2


class HailoDetector:
    """yolov8n.hef：NMS 已編進模型，輸出每類一個 (n, 5) 陣列，
    每列 [y_min, x_min, y_max, x_max, score]，座標為 0~1 正規化。"""

    def __init__(self, model: HailoModel, cfg: DetectSettings):
        if not model.is_nms:
            raise ValueError(f"{model.name} 不是 NMS 輸出的偵測模型")
        self.model = model
        self.conf = cfg.conf
        self.classes = cfg.classes
        self.input_size = (int(model.input_shape[1]), int(model.input_shape[0]))   # (W, H)

    def detect(self, rgb_input: np.ndarray, out_w: int, out_h: int) -> list[Detection]:
        """rgb_input 必須已是模型輸入尺寸的 RGB；框換算到 out_w×out_h（main 串流）座標。"""
        per_class = self.model.infer([rgb_input])[0]
        results: list[Detection] = []
        for class_id, boxes in enumerate(per_class):
            if class_id not in self.classes or len(boxes) == 0:
                continue
            label = self.classes[class_id]
            for ymin, xmin, ymax, xmax, score in boxes:
                if score < self.conf:
                    continue
                x1 = int(np.clip(xmin * out_w, 0, out_w - 1))
                y1 = int(np.clip(ymin * out_h, 0, out_h - 1))
                x2 = int(np.clip(xmax * out_w, 0, out_w - 1))
                y2 = int(np.clip(ymax * out_h, 0, out_h - 1))
                results.append(((x1 + x2) // 2, (y1 + y2) // 2, label, float(score), x1, y1, x2, y2))
        return results


# ── 5. 警戒區（可拖曳、會存檔）────────────────────────────────
Zone = tuple[int, int, int, int]


class WarningZone:
    """警戒區，以 0~1 比例存進 json，下次啟動自動沿用。

    兩種設定方式：啟動時 select() 跳視窗框（沒存過、或加 --ui 時）；
    執行中在主視窗滑鼠左鍵拖曳，放開即存。
    """

    def __init__(self, cfg: DetectSettings, frame_w: int, frame_h: int):
        self.store = Path(cfg.zone_store)
        self.w, self.h = frame_w, frame_h
        self.start: Optional[tuple[int, int]] = None
        self._lock = threading.Lock()
        self.loaded = False                 # 是否成功從 json 讀到上次的警戒區
        frac = list(cfg.default_zone)
        if self.store.exists():
            try:
                loaded = json.loads(self.store.read_text(encoding="utf-8"))
                if isinstance(loaded, list) and len(loaded) == 4:
                    frac = [float(v) for v in loaded]
                    self.loaded = True
            except Exception as exc:
                print(f"[警告] 讀取 {self.store} 失敗，改用預設：{exc}")
        self.region: Zone = self._to_px(frac)

    def select(self, first_frame: np.ndarray, force_ui: bool, max_w: int, max_h: int) -> None:
        """啟動時決定警戒區：加了 --ui 就重新框；沒加就沿用存的；從沒存過也跳視窗框一次。
        取消（按 C）就保留目前的（上次存的或 default_zone），不存檔。"""
        if self.loaded and not force_ui:
            print(f"沿用上次的警戒區 {self.store}（拖曳滑鼠可重設）")
            return
        if not force_ui:
            print("[警戒區] 還沒框過，跳出視窗讓你框一次（之後就不用了）")
        # 黃框、不畫十字線，跟路面 ROI 的藍框＋十字線區分；黃色與主視窗畫警戒區的顏色相同
        rect = select_rect(first_frame, ZONE_WINDOW_TITLE,
                           "請用滑鼠拖出人車警戒區，按 Enter 確認，按 C 取消（沿用目前的）",
                           max_w, max_h, color=(0, 215, 255), crosshair=False)
        if rect is None:
            src = "上次存的警戒區" if self.loaded else "detect.yaml 的 default_zone"
            print(f"未選取，這次使用{src}（沒有存檔）")
            return
        with self._lock:
            self.region = rect
        self._save()
        print(f"警戒區: ({rect[0]}, {rect[1]}) → ({rect[2]}, {rect[3]})，已存到 {self.store}")

    def _to_px(self, frac: list[float]) -> Zone:
        x1 = int(np.clip(frac[0] * self.w, 0, self.w - 1))
        y1 = int(np.clip(frac[1] * self.h, 0, self.h - 1))
        x2 = int(np.clip(frac[2] * self.w, 0, self.w - 1))
        y2 = int(np.clip(frac[3] * self.h, 0, self.h - 1))
        return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

    def get(self) -> Zone:
        with self._lock:
            return self.region

    def _save(self) -> None:
        x1, y1, x2, y2 = self.region
        self.store.parent.mkdir(parents=True, exist_ok=True)
        self.store.write_text(json.dumps([x1 / self.w, y1 / self.h, x2 / self.w, y2 / self.h]),
                              encoding="utf-8")

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.start is not None:
            x1, y1 = self.start
            if abs(x - x1) >= 20 and abs(y - y1) >= 20:
                with self._lock:
                    self.region = (min(x1, x), min(y1, y), max(x1, x), max(y1, y))
                self._save()
                print(f"[INFO] 警戒區已更新並存檔：{self.region}")
            self.start = None


def check_in_zone(px: float, py: float, zone: Zone) -> bool:
    rx1, ry1, rx2, ry2 = zone
    return rx1 <= px <= rx2 and ry1 <= py <= ry2


class AlertController:
    """只在進入區域的瞬間發聲，避免每幀連續響。"""

    def __init__(self, cooldown_sec: float = 1.5):
        self.was_alerting = False
        self.last_sound_time = 0.0
        self.cooldown_sec = cooldown_sec

    def update(self, alerting: bool):
        now = time.time()
        if alerting and not self.was_alerting and now - self.last_sound_time >= self.cooldown_sec:
            print("\a[ALERT] 物體進入警戒範圍！", flush=True)
            self.last_sound_time = now
        self.was_alerting = alerting


# ── 6. 背景偵測緒 ───────────────────────────────────────────
@dataclass
class TrackView:
    """給主緒畫圖用的軌跡快照（Track 物件本身只在偵測緒動）。"""
    id: int
    x: int
    y: int
    px: int
    py: int
    history: list[tuple[float, float]]


@dataclass
class DetectResult:
    detections: list[Detection]
    tracks: list[TrackView]
    alert: bool


class DetectorThread(threading.Thread):
    """持續對「最新的 lores 畫面」做偵測 + 追蹤 + 警戒判斷；主緒隨時取最近一次結果。
    只保留最新一幀、舊的直接丟：偵測若比相機慢，排隊處理只會越來越落後。"""

    def __init__(self, detector: HailoDetector, zone: WarningZone, cfg: DetectSettings,
                 frame_w: int, frame_h: int):
        super().__init__(daemon=True, name="detect")
        self.detector = detector
        self.zone = zone
        self.cfg = cfg
        self.frame_w, self.frame_h = frame_w, frame_h
        self.tracker = Tracker(cfg.assoc_dist_px, cfg.max_miss)
        self.alerts = AlertController(cfg.alert_cooldown_sec)

        self._pending: np.ndarray | None = None
        self._lock = threading.Lock()
        self._new_frame = threading.Event()
        self._stop_evt = threading.Event()      # 不能取名 _stop：會蓋掉 Thread 內部方法
        self._reset_req = threading.Event()
        self._result: DetectResult | None = None
        self._result_lock = threading.Lock()
        self.update_count = 0
        self.last_ms = 0.0

    def submit(self, lores_rgb: np.ndarray) -> None:
        with self._lock:
            self._pending = lores_rgb
        self._new_frame.set()

    def latest(self) -> DetectResult | None:
        with self._result_lock:
            return self._result

    def reset_tracks(self) -> None:
        self._reset_req.set()

    def stop(self) -> None:
        self._stop_evt.set()
        self._new_frame.set()

    def run(self) -> None:
        horizon = self.cfg.prediction_horizon_sec
        while not self._stop_evt.is_set():
            self._new_frame.wait(timeout=1.0)
            if self._stop_evt.is_set():
                break
            with self._lock:
                frame = self._pending
                self._pending = None
            self._new_frame.clear()
            if frame is None:
                continue
            if self._reset_req.is_set():
                self.tracker = Tracker(self.cfg.assoc_dist_px, self.cfg.max_miss)
                self._reset_req.clear()

            t0 = time.perf_counter()
            try:
                dets = self.detector.detect(frame, self.frame_w, self.frame_h)
            except Exception as e:
                print(f"[偵測] 推論失敗: {e}", flush=True)
                time.sleep(0.5)
                continue
            now_ms = time.time() * 1000
            tracks = self.tracker.update([(cx, cy) for (cx, cy, *_) in dets], now_ms)
            zone = self.zone.get()
            alert = any(check_in_zone(t.kf.x, t.kf.y, zone) for t in tracks if t.miss == 0)
            self.alerts.update(alert)

            views = []
            for t in tracks:
                px, py = t.predicted_point(horizon)
                views.append(TrackView(t.id, int(t.kf.x), int(t.kf.y), int(px), int(py), list(t.history)))
            with self._result_lock:
                self._result = DetectResult(dets, views, alert)
            self.update_count += 1
            self.last_ms = (time.perf_counter() - t0) * 1000.0


# ── 7. 覆蓋層繪製 ───────────────────────────────────────────
class OverlayRenderer:
    def draw(self, frame: np.ndarray, result: DetectResult | None, zone: Zone) -> None:
        h, w = frame.shape[:2]
        alert = bool(result and result.alert)

        # 警戒範圍：正常為黃色，觸發警報後為紅色
        rx1, ry1, rx2, ry2 = zone
        zone_color = (0, 0, 255) if alert else (0, 215, 255)
        overlay = frame.copy()
        cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), zone_color, -1)
        cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), zone_color, 3)
        cv2.putText(frame, "WARNING ZONE", (rx1 + 8, max(25, ry1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, zone_color, 2)
        if result is None:
            return

        for (cx, cy, label, score, x1, y1, x2, y2) in result.detections:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {score:.2f}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

        for t in result.tracks:
            color = self._track_color(t.id)
            if len(t.history) >= 2:
                pts = np.array(t.history, dtype=np.int32)
                for i in range(1, len(pts)):
                    cv2.line(frame, tuple(pts[i - 1]), tuple(pts[i]), color, 2)
            cv2.circle(frame, (t.x, t.y), 8, color, -1)
            cv2.putText(frame, f"ID{t.id}", (t.x + 10, t.y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.circle(frame, (t.px, t.py), 14, (255, 0, 255), 2)
            cv2.putText(frame, "pred", (t.px + 8, t.py + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)

        if alert:
            self._draw_alert(frame, w)

    def _draw_alert(self, frame, w):
        msg = "ALERT: OBJECT IN WARNING ZONE"      # OpenCV 內建字型不支援中文
        font, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
        (tw, th), _ = cv2.getTextSize(msg, font, font_scale, thickness)
        x, y, pad = w // 2 - tw // 2, 100, 12
        cv2.rectangle(frame, (x - pad, y - th - pad), (x + tw + pad, y + pad), (255, 255, 255), -1)
        cv2.rectangle(frame, (x - pad, y - th - pad), (x + tw + pad, y + pad), (0, 0, 255), 2)
        cv2.putText(frame, msg, (x, y), font, font_scale, (0, 0, 200), thickness)

    @staticmethod
    def _track_color(track_id: int) -> tuple:
        hue = (track_id * 47) % 360
        r, g, b = colorsys.hsv_to_rgb(hue / 360.0, 0.9, 1.0)
        return (int(b * 255), int(g * 255), int(r * 255))
