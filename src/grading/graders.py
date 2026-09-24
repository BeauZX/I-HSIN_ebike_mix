"""兩個分級管線的整合層，介面相同：

    grader.analyze(roi_bgr) -> result      在分析緒呼叫（含該管線自己的時序平滑）
    grader.draw(roi_bgr, result) -> vis    在主緒呼叫，把結果畫到 ROI 副本上
    grader.reset()                          路面種類切換時清掉平滑狀態，不讓舊路面的記憶拖累新路面
    grader.grade_counts(result) -> dict     各等級格數 {"severe": n, "slight": n, "smooth": n}（給 log 用）
    grader.rows / grader.cols

GridClassifier 取代 asphalt / cement 原本的 HailoClassifier：前處理一字不改
（短邊 resize → 置中裁切 → BGR→RGB，raw uint8），只是推論改走共用的 HailoModel。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from ..hailo import HailoModel
from ..settings import AsphaltSettings, CementSettings
from . import grid as asphalt_grid
from . import yolo_grid
from .fusion import FrameResult, FusionConfig, analyze_frame
from .overlay import draw_overlay
from .pothole.crack import CrackDetector
from .presets import load_crack_config

GRADE_NAMES = ("severe", "slight", "smooth")     # 兩個分級管線統一回報的等級名稱


class GridClassifier:
    """網格分類器：__call__(cells) -> probs[N, C]、names。與原專案 backend 介面相同。"""

    def __init__(self, model: HailoModel, class_names: list[str]):
        self.model = model
        self.names = dict(enumerate(class_names))
        self.imgsz = int(model.input_shape[0])
        n_out = int(np.prod(model.output_shape))
        if n_out != len(class_names):
            raise ValueError(f"{model.name}: HEF 輸出 {n_out} 類，但 class_names 有 {len(class_names)} 個")

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        # 與 ultralytics classify 一模一樣：短邊 resize → 置中裁切 → BGR→RGB，不做 mean/std
        size = self.imgsz
        h, w = img.shape[:2]
        scale = size / min(h, w)
        img = cv2.resize(img, (max(size, round(w * scale)), max(size, round(h * scale))),
                         interpolation=cv2.INTER_LINEAR)
        h, w = img.shape[:2]
        top, left = (h - size) // 2, (w - size) // 2
        img = img[top:top + size, left:left + size]
        return np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def __call__(self, cells: list[np.ndarray]) -> np.ndarray:
        outs = self.model.infer([self._preprocess(c) for c in cells])
        return np.stack(outs).astype(np.float32).reshape(len(cells), -1)


class _TimedCrackDetector:
    """包一層記錄 detect() 耗時（水泥輪裡唯一的純 CPU 大項，供 metrics 取樣）；其餘屬性透傳。"""

    def __init__(self, inner: CrackDetector):
        self._inner = inner
        self.last_ms = 0.0

    def detect(self, *args, **kwargs):
        t0 = time.perf_counter()
        out = self._inner.detect(*args, **kwargs)
        self.last_ms = (time.perf_counter() - t0) * 1000.0
        return out

    def __getattr__(self, name):
        return getattr(self._inner, name)


class AsphaltGrader:
    name = "asphalt"

    def __init__(self, model: HailoModel, cfg: AsphaltSettings):
        self.cfg = cfg
        self.rows, self.cols = cfg.grid.rows, cfg.grid.cols
        self.classifier = GridClassifier(model, cfg.class_names)
        self.reset()

    def reset(self) -> None:
        self.tracker = asphalt_grid.GridTracker(self.rows, self.cols,
                                                self.cfg.ema_alpha, self.cfg.switch_margin)

    def analyze(self, roi_bgr: np.ndarray):
        probs, names = asphalt_grid.classify_grid(self.classifier, roi_bgr, self.rows, self.cols)
        return self.tracker.update(probs, names)

    def draw(self, roi_bgr: np.ndarray, result) -> np.ndarray:
        return asphalt_grid.draw_grid_overlay(roi_bgr.copy(), result, self.rows, self.cols)

    def grade_counts(self, result) -> dict[str, int]:
        # 類別名稱像 dry_asphalt_severe，依字尾歸到 severe / slight / smooth
        counts = dict.fromkeys(GRADE_NAMES, 0)
        for row in result:
            for _, name, _ in row:
                grade = next((g for g in GRADE_NAMES if g in name.lower()), None)
                if grade:
                    counts[grade] += 1
        return counts


class CementGrader:
    name = "cement"

    def __init__(self, model: HailoModel, cfg: CementSettings):
        self.cfg = cfg
        self.rows, self.cols = cfg.grid.rows, cfg.grid.cols
        self.classifier = GridClassifier(model, cfg.class_names)
        self.crack_detector = _TimedCrackDetector(CrackDetector(load_crack_config(cfg.crack_preset)))
        self.fusion = FusionConfig(rows=self.rows, cols=self.cols, **cfg.fusion)
        self.reset()

    def _load_pid(self) -> dict:
        params = {"kp": 0.35, "ki": 0.03, "kd": 0.08, "min_dwell": 6, "switch_margin": 0.08}
        p = Path(self.cfg.pid_params)
        if p.exists():
            try:
                saved = json.loads(p.read_text(encoding="utf-8"))
                params.update({k: saved[k] for k in params if k in saved})
            except Exception as exc:
                print(f"[警告] 讀取 PID 參數 {p} 失敗，改用預設：{exc}")
        return params

    def reset(self) -> None:
        if self.cfg.smoother == "pid":
            pid = self._load_pid()
            self.smoother = yolo_grid.PIDSeveritySmoother(
                kp=pid["kp"], ki=pid["ki"], kd=pid["kd"],
                t_slight=self.fusion.t_slight, t_severe=self.fusion.t_severe,
                switch_margin=pid["switch_margin"], min_dwell=pid["min_dwell"])
        else:
            self.smoother = yolo_grid.TemporalSmoother(
                alpha=self.cfg.ema_alpha, switch_margin=self.cfg.ema_switch_margin,
                t_slight=self.fusion.t_slight, t_severe=self.fusion.t_severe)

    def analyze(self, roi_bgr: np.ndarray) -> tuple[FrameResult, np.ndarray]:
        result = analyze_frame(roi_bgr, self.classifier, self.crack_detector, self.fusion)
        _, level = self.smoother.update(result.sev_score)
        return result, level

    def draw(self, roi_bgr: np.ndarray, result) -> np.ndarray:
        frame_result, level = result
        return draw_overlay(roi_bgr, frame_result, level=level)

    def grade_counts(self, result) -> dict[str, int]:
        _, level = result       # 0 / 1 / 2 = smooth / slight / severe（yolo_grid.SEVERITY_NAMES）
        level_of = {name: lv for lv, name in yolo_grid.SEVERITY_NAMES.items()}
        return {g: int(np.count_nonzero(level == level_of[g])) for g in GRADE_NAMES}
