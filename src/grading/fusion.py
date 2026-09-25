"""融合核心：YOLO-cls 語意分級（守門）× 影像法裂縫偵測（定位＋量化）。

單幀函式 analyze_frame：
  1) YOLO 網格分類 → 每格期望嚴重度 + top1 分級
  2) 影像法裂縫偵測一次（整個 ROI）→ 候選裂縫 Detection
  3) 守門：YOLO 判為 smooth 的格子，只採信 score 夠高的裂縫（壓紋理誤判）
  4) 量化：每格裂縫像素比 crack_ratio
  5) 融合：每格嚴重度 = w_yolo*期望嚴重度 + w_crack*正規化裂縫密度
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .pothole.crack import CrackDetector
from .pothole.detector import Detection

from . import yolo_grid as yg


@dataclass
class FusionConfig:
    rows: int = 4
    cols: int = 10

    # 融合權重：YOLO 語意 vs 影像法裂縫密度
    w_yolo: float = 0.6
    w_crack: float = 0.4
    # 裂縫密度正規化基準：格內裂縫像素比達此值 → 裂縫項視為滿分
    crack_ratio_norm: float = 0.06

    # 嚴重度分級門檻（單幀用；影片改由 TemporalSmoother 帶相同門檻）
    t_slight: float = 0.30
    t_severe: float = 0.60

    # 守門：YOLO 判為 smooth 的格子，裂縫 Detection.score 需 ≥ 此值才採信
    smooth_gate_min_score: float = 0.55


@dataclass
class FrameResult:
    sev_score: np.ndarray      # [rows, cols] 融合連續嚴重度 0..1
    final_level: np.ndarray    # [rows, cols] 分級 0/1/2（單幀直接門檻）
    yolo_level: np.ndarray     # [rows, cols] 純 YOLO top1 分級
    crack_ratio: np.ndarray    # [rows, cols] 格內裂縫像素比
    cracks: list[tuple[np.ndarray, Detection]]  # (ROI 座標輪廓, Detection)
    names: dict
    roi_size: tuple[int, int]  # 分析時的 ROI (寬, 高)；畫到不同大小的 ROI 上時用來縮放裂縫輪廓


def _detector_scale(detector: CrackDetector, h: int, w: int) -> tuple[float, float]:
    """偵測器內部會把影像 resize 到 target_width；算出「偵測座標 → ROI 座標」縮放比。"""
    tw = detector.config.target_width
    if w == tw:
        return 1.0, 1.0
    dh = int(round(h * tw / float(w)))
    return w / float(tw), h / float(dh)


def _centroid(contour: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    m = cv2.moments(contour)
    if m["m00"] > 0:
        return m["m10"] / m["m00"], m["m01"] / m["m00"]
    x, y, bw, bh = bbox
    return x + bw / 2.0, y + bh / 2.0


def analyze_frame(
    roi_bgr: np.ndarray,
    model,
    crack_detector: CrackDetector,
    cfg: FusionConfig,
) -> FrameResult:
    rows, cols = cfg.rows, cfg.cols
    h, w = roi_bgr.shape[:2]
    cell_h = h // rows
    cell_w = w // cols

    # 1) YOLO 網格分類 → 期望嚴重度 + 分級
    probs, names = yg.classify_grid(model, roi_bgr, rows, cols)
    expected, yolo_level, _ = yg.probs_to_severity(probs, names)

    # 2) 影像法裂縫偵測（一次跑整個 ROI）
    dets, _ = crack_detector.detect(roi_bgr)
    sx, sy = _detector_scale(crack_detector, h, w)

    # 3) 守門 + 座標換回 ROI：逐條裂縫，看它落在哪一格的 YOLO 分級
    kept: list[tuple[np.ndarray, Detection]] = []
    for d in dets:
        cx, cy = _centroid(d.contour, d.bbox)
        cx *= sx
        cy *= sy
        r = min(rows - 1, max(0, int(cy // cell_h)))
        c = min(cols - 1, max(0, int(cx // cell_w)))
        if int(yolo_level[r, c]) == yg.SEVERITY_SMOOTH and d.score < cfg.smooth_gate_min_score:
            continue  # 平滑格內的弱裂縫 → 視為紋理誤判，丟棄
        cnt = d.contour.astype(np.float32).copy()
        cnt[:, 0, 0] *= sx
        cnt[:, 0, 1] *= sy
        kept.append((cnt.astype(np.int32), d))

    # 4) 量化：每格裂縫像素比
    crack_mask = np.zeros((h, w), np.uint8)
    for cnt, _ in kept:
        cv2.drawContours(crack_mask, [cnt], -1, 255, thickness=cv2.FILLED)
    crack_ratio = np.zeros((rows, cols), np.float32)
    for r in range(rows):
        for c in range(cols):
            sub = crack_mask[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w]
            if sub.size:
                crack_ratio[r, c] = float(np.count_nonzero(sub)) / float(sub.size)

    # 5) 融合
    crack_term = np.clip(crack_ratio / max(1e-6, cfg.crack_ratio_norm), 0.0, 1.0)
    sev_score = np.clip(cfg.w_yolo * expected + cfg.w_crack * crack_term, 0.0, 1.0)
    final_level = np.where(
        sev_score >= cfg.t_severe, 2,
        np.where(sev_score >= cfg.t_slight, 1, 0),
    ).astype(np.int32)

    return FrameResult(
        sev_score=sev_score,
        final_level=final_level,
        yolo_level=yolo_level,
        crack_ratio=crack_ratio,
        cracks=kept,
        names=names,
        roi_size=(w, h),
    )
