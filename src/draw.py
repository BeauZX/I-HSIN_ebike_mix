"""共用的文字繪製。

OpenCV 5.0 的 putText 字距會隨 thickness 改變（實測同一串字 thickness=1 寬 221 px、
thickness≥2 寬 240 px），所以原專案「先畫粗黑字再畫細白字」的描邊寫法在 5.0 上
會變成兩層錯開的字。這裡改成：黑字用同一個 thickness 往上下左右各偏移 1 px 畫四次，
再畫一次前景色，字距完全一致。
"""

import cv2
import numpy as np

_OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1))

# 畫面頂端留給左上角狀態區（main.py 的 ROAD / GRADING / DOOR / SUSPENSION 四行）的高度；
# 警戒區標籤、警報橫幅（src/detect.py）要避開這一塊
STATUS_TOP_RESERVED = 180


def put_text_outlined(img: np.ndarray, text: str, org: tuple[int, int], font_scale: float,
                      color=(255, 255, 255), thickness: int = 1, outline=(0, 0, 0),
                      font=cv2.FONT_HERSHEY_SIMPLEX) -> None:
    x, y = org
    for dx, dy in _OFFSETS:
        cv2.putText(img, text, (x + dx, y + dy), font, font_scale, outline, thickness, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
