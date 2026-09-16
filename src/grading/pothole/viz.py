"""共用的偵測結果視覺化。"""

from __future__ import annotations

import cv2
import numpy as np

from .detector import Detection

# 依缺陷類型上色（BGR）
LABEL_COLORS: dict[str, tuple[int, int, int]] = {
    "pothole": (0, 0, 255),   # 紅：坑洞
    "crack": (0, 255, 255),   # 黃：條狀裂縫
    "mesh": (0, 140, 255),    # 橘：網狀裂縫（龜裂）
}
_DEFAULT_COLOR = (0, 255, 0)


def annotate(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """把偵測結果畫到影像副本上並回傳。"""
    vis = image.copy()
    for det in detections:
        color = LABEL_COLORS.get(det.label, _DEFAULT_COLOR)
        cv2.drawContours(vis, [det.contour], -1, color, 2)
        # 坑洞與網狀裂縫另加外框，凸顯「整片區域」
        if det.label in ("pothole", "mesh"):
            x, y, bw, bh = det.bbox
            cv2.rectangle(vis, (x, y), (x + bw, y + bh), color, 2)
        x, y = det.bbox[0], det.bbox[1]
        cv2.putText(
            vis, det.label, (x, max(0, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )
    return vis
