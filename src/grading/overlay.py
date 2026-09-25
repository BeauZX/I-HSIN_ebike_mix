"""融合結果視覺化：彩色嚴重度網格 + 裂縫輪廓 + 每格量化文字。"""

from __future__ import annotations

import cv2
import numpy as np

from ..draw import put_text_outlined
from .fusion import FrameResult

# 嚴重度分級顏色（BGR）
SEVERITY_COLORS = {
    0: (0, 255, 0),    # 綠：平整 smooth
    1: (0, 255, 255),  # 黃：輕微 slight
    2: (0, 0, 255),    # 紅：嚴重 severe
}
SEVERITY_NAMES = {0: "smooth", 1: "slight", 2: "severe"}

# 裂縫類型顏色（與 pothole2 viz 一致）
_CRACK_COLORS = {
    "crack": (0, 255, 255),
    "mesh": (0, 140, 255),
    "pothole": (0, 0, 255),
}


def draw_overlay(
    roi_bgr: np.ndarray,
    result: FrameResult,
    level: np.ndarray | None = None,
    alpha: float = 0.35,
    show_text: bool = True,
) -> np.ndarray:
    """把融合結果畫到 ROI 副本上並回傳。

    level：可傳入時序平滑後的分級覆寫 result.final_level（影片用）。
    roi_bgr 可以和分析時的 ROI 不同大小（例如 1080p 分析、720p 顯示），裂縫輪廓會依比例縮放。
    """
    lvl = result.final_level if level is None else level
    rows, cols = lvl.shape
    h, w = roi_bgr.shape[:2]
    cell_h = h // rows
    cell_w = w // cols

    vis = roi_bgr.copy()
    overlay = vis.copy()
    for r in range(rows):
        for c in range(cols):
            color = SEVERITY_COLORS.get(int(lvl[r, c]), (0, 255, 0))
            x1, y1 = c * cell_w, r * cell_h
            x2, y2 = (c + 1) * cell_w, (r + 1) * cell_h
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)
    cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)

    # 裂縫輪廓畫在網格之上（精準定位）
    src_w, src_h = result.roi_size
    scale = None if (src_w, src_h) == (w, h) else np.array([w / src_w, h / src_h], np.float32)
    for cnt, det in result.cracks:
        if scale is not None:
            cnt = (cnt * scale).astype(np.int32)
        cv2.drawContours(vis, [cnt], -1, _CRACK_COLORS.get(det.label, (0, 255, 0)), 2)

    if show_text:
        for r in range(rows):
            for c in range(cols):
                name = SEVERITY_NAMES.get(int(lvl[r, c]), "?")
                s = float(result.sev_score[r, c])
                label = f"{name} {s:.2f}"
                font_scale = max(0.3, min(cell_w, cell_h) / 600)
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
                )
                tx = c * cell_w + (cell_w - tw) // 2
                ty = r * cell_h + (cell_h + th) // 2
                # OpenCV 5 的字距隨 thickness 變，粗黑細白的描邊會錯開，改用偏移描邊（見 src/draw.py）
                put_text_outlined(vis, label, (tx, ty), font_scale)
    return vis


def summarize(level: np.ndarray, cracks: list) -> str:
    """回傳一行文字摘要：各級格數 + 裂縫條數。"""
    n_smooth = int(np.count_nonzero(level == 0))
    n_slight = int(np.count_nonzero(level == 1))
    n_severe = int(np.count_nonzero(level == 2))
    n_crack = sum(1 for _, d in cracks if d.label == "crack")
    n_mesh = sum(1 for _, d in cracks if d.label == "mesh")
    return (f"格: 平整 {n_smooth} / 輕微 {n_slight} / 嚴重 {n_severe}；"
            f"裂縫 {n_crack}、網狀 {n_mesh}")
