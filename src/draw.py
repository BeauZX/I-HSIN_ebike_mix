"""共用的文字繪製。

OpenCV 5.0 的 putText 字距會隨 thickness 改變（實測同一串字 thickness=1 寬 221 px、
thickness≥2 寬 240 px），所以原專案「先畫粗黑字再畫細白字」的描邊寫法在 5.0 上
會變成兩層錯開的字。這裡改成：黑字用同一個 thickness 往上下左右各偏移 1 px 畫四次，
再畫一次前景色，字距完全一致。
"""

import cv2
import numpy as np

_OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1))

# 畫面頂端留給左上角狀態區（main.py 的 ROAD / GRADING / DOOR / SUSPENSION 四行黑底面板）的高度；
# 警戒區標籤、警報橫幅（src/detect.py）要避開這一塊
STATUS_TOP_RESERVED = 235

# 右上角 FPS 黑底面板的高度；警戒區標籤畫在右上角時要從這底下開始
TOP_RIGHT_RESERVED = 70

# 畫面上所有文字統一大小（報告投影用）：左上角面板、FPS、ROI / 警戒區標籤、警報橫幅、人車 / 車門 / 追蹤標籤
UI_FONT_SCALE = 1.3
UI_THICKNESS = 2


def put_text_outlined(img: np.ndarray, text: str, org: tuple[int, int], font_scale: float,
                      color=(255, 255, 255), thickness: int = 1, outline=(0, 0, 0),
                      font=cv2.FONT_HERSHEY_SIMPLEX) -> None:
    x, y = org
    for dx, dy in _OFFSETS:
        cv2.putText(img, text, (x + dx, y + dy), font, font_scale, outline, thickness, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def put_text_panel(img: np.ndarray, lines: list[tuple[str, tuple]], x: int, y: int, font_scale: float,
                   thickness: int = 2, line_gap: int = 15, pad: int = 10, alpha: float = 0.6,
                   font=cv2.FONT_HERSHEY_SIMPLEX) -> int:
    """在 (x, y)（面板左上角）畫半透明黑底面板，裡面一行一段描邊文字；lines 是 [(文字, BGR 顏色)]。
    黑底讓白字在亮背景（天空、白牆、反光路面）上也看得清楚。回傳面板的底 y。"""
    sizes = [cv2.getTextSize(text, font, font_scale, thickness) for text, _ in lines]
    text_h = max(h for (_, h), _ in sizes)
    base = max(b for _, b in sizes)
    line_h = text_h + base + line_gap
    x2 = min(img.shape[1], x + max(w for (w, _), _ in sizes) + 2 * pad)
    y2 = min(img.shape[0], y + len(lines) * line_h - line_gap + 2 * pad)
    region = img[y:y2, x:x2]
    region[:] = (region * (1.0 - alpha)).astype(np.uint8)      # 往黑色混 alpha，只動面板範圍
    for i, (text, color) in enumerate(lines):
        put_text_outlined(img, text, (x + pad, y + pad + text_h + i * line_h), font_scale, color, thickness,
                          font=font)
    return y2


def fit_text_x(img: np.ndarray, text: str, x: int, font_scale: float, thickness: int,
               font=cv2.FONT_HERSHEY_SIMPLEX, margin: int = 5) -> int:
    """標籤靠近畫面右邊時往左推，整串字留在畫面內（字放大後物件靠邊時標籤會被切掉）。"""
    (tw, _), _ = cv2.getTextSize(text, font, font_scale, thickness)
    return max(margin, min(x, img.shape[1] - tw - margin))
