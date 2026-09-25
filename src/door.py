"""車門開啟 / 關閉偵測（改寫自 car_door 專案 src/detector.py、postprocess.py、visualize.py）。

改動：
    * 推論改走共用的 HailoModel（同一個 VDevice、scheduler 排程），不再自己開 VDevice。
    * 輸入共用人車偵測的 640×640 RGB（Camera.read() 從顯示畫面拉伸縮放），不做 letterbox；
      因此框座標直接用 0~1 比例換回顯示畫面，與 detect.py 的 HailoDetector 相同。
    * 偵測搬到背景緒（DoorThread），只處理最新一幀、全速跑；主緒只拿最新結果畫圖。
    * 類別名稱與「哪些類別算開啟」改由 configs/door.yaml 指定（原本是 sidecar json + 關鍵字比對）。
    * 狀態不另畫橫幅，併進 main.py 左上角的狀態列；框上的標籤改用描邊文字（OpenCV 5 相容）。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .draw import put_text_outlined
from .hailo import HailoModel
from .settings import DoorSettings

OPEN_COLOR = (0, 0, 255)       # 紅（BGR），沿用原專案
CLOSED_COLOR = (0, 200, 0)     # 綠


@dataclass
class DoorBox:
    label: str
    score: float
    is_open: bool
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass
class DoorResult:
    boxes: list[DoorBox]

    @property
    def state(self) -> str:
        """open / closed / none（畫面上沒有車門）。"""
        if not self.boxes:
            return "none"
        return "open" if any(b.is_open for b in self.boxes) else "closed"


class DoorDetector:
    """car_door_yolov11m.hef：NMS 已編進模型，輸出每類一個 (n, 5) 陣列，
    每列 [y_min, x_min, y_max, x_max, score]，座標為 0~1 正規化。"""

    def __init__(self, model: HailoModel, cfg: DoorSettings):
        if not model.is_nms:
            raise ValueError(f"{model.name} 不是 NMS 輸出的偵測模型")
        self.model = model
        self.conf = cfg.conf
        self.class_names = cfg.class_names
        self.open_classes = set(cfg.open_classes)
        self.input_size = (int(model.input_shape[1]), int(model.input_shape[0]))   # (W, H)

    def detect(self, rgb_input: np.ndarray, out_w: int, out_h: int) -> DoorResult:
        """rgb_input 必須已是模型輸入尺寸的 RGB；框換算到 out_w×out_h（main 串流）座標。"""
        per_class = self.model.infer([rgb_input])[0]
        boxes: list[DoorBox] = []
        for class_id, arr in enumerate(per_class):
            if len(arr) == 0:
                continue
            label = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
            is_open = label in self.open_classes
            for ymin, xmin, ymax, xmax, score in arr:
                if score < self.conf:
                    continue
                boxes.append(DoorBox(
                    label, float(score), is_open,
                    int(np.clip(xmin * out_w, 0, out_w - 1)), int(np.clip(ymin * out_h, 0, out_h - 1)),
                    int(np.clip(xmax * out_w, 0, out_w - 1)), int(np.clip(ymax * out_h, 0, out_h - 1))))
        boxes.sort(key=lambda b: b.score, reverse=True)
        return DoorResult(boxes)


class DoorThread(threading.Thread):
    """持續對「最新的 lores 畫面」做車門偵測；主緒隨時取最近一次結果。
    只保留最新一幀、舊的直接丟（與 detect.py 的 DetectorThread 相同）。"""

    def __init__(self, detector: DoorDetector, frame_w: int, frame_h: int):
        super().__init__(daemon=True, name="door")
        self.detector = detector
        self.frame_w, self.frame_h = frame_w, frame_h

        self._pending: np.ndarray | None = None
        self._lock = threading.Lock()
        self._new_frame = threading.Event()
        self._stop_evt = threading.Event()      # 不能取名 _stop：會蓋掉 Thread 內部方法
        self._result: DoorResult | None = None
        self._result_lock = threading.Lock()
        self.update_count = 0
        self.last_ms = 0.0

    def submit(self, lores_rgb: np.ndarray) -> None:
        with self._lock:
            self._pending = lores_rgb
        self._new_frame.set()

    def latest(self) -> DoorResult | None:
        with self._result_lock:
            return self._result

    def stop(self) -> None:
        self._stop_evt.set()
        self._new_frame.set()

    def run(self) -> None:
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

            t0 = time.perf_counter()
            try:
                result = self.detector.detect(frame, self.frame_w, self.frame_h)
            except Exception as e:
                print(f"[車門] 推論失敗: {e}", flush=True)
                time.sleep(0.5)
                continue
            with self._result_lock:
                self._result = result
            self.update_count += 1
            self.last_ms = (time.perf_counter() - t0) * 1000.0


def draw_doors(frame: np.ndarray, result: DoorResult | None) -> None:
    """車門框：開啟紅框、關閉綠框，框上標類別與分數。"""
    if result is None:
        return
    for b in result.boxes:
        color = OPEN_COLOR if b.is_open else CLOSED_COLOR
        cv2.rectangle(frame, (b.x1, b.y1), (b.x2, b.y2), color, 2)
        # 框太靠上時標籤改畫在框內，免得壓到左上角的狀態列（y≈24）
        ty = b.y1 - 8 if b.y1 >= 50 else max(b.y1, 32) + 20
        put_text_outlined(frame, f"door {b.label} {b.score:.2f}", (b.x1 + 3, ty), 0.55, color, 1)
