"""網格切分、分類、上色，以及讓結果穩定下來的時序處理（複製自 asphalt 專案 src/grid.py）。"""

import cv2
import numpy as np

from ..draw import put_text_outlined

# 類別索引 → 顏色（BGR）。asphalt 模型的類別順序是字母序：
#   0: dry_asphalt_severe → 紅、1: dry_asphalt_slight → 黃、2: dry_asphalt_smooth → 綠
COLORS = [
    (0, 0, 255),    # 紅 - severe
    (0, 255, 255),  # 黃 - slight
    (0, 255, 0),    # 綠 - smooth
]


def classify_grid(classifier, frame: np.ndarray, rows: int,
                  cols: int) -> tuple[np.ndarray, dict]:
    """
    把 frame 切成 rows×cols 格，整批丟給分類器。回傳 (probs[rows,cols,C], names)。
    classifier 是 backend.py 的任一後端，介面見該檔。
    """
    h, w = frame.shape[:2]
    cell_h = h // rows
    cell_w = w // cols

    cells = []
    for r in range(rows):
        for c in range(cols):
            y1, y2 = r * cell_h, (r + 1) * cell_h
            x1, x2 = c * cell_w, (c + 1) * cell_w
            cells.append(frame[y1:y2, x1:x2])

    probs = classifier(cells)                       # [rows*cols, C]
    return probs.reshape(rows, cols, -1), classifier.names


def draw_grid_overlay(frame: np.ndarray, grid: list[list[tuple]], rows: int, cols: int,
                      alpha: float = 0.35) -> np.ndarray:
    h, w = frame.shape[:2]
    cell_h = h // rows
    cell_w = w // cols
    overlay = frame.copy()

    for r, row in enumerate(grid):
        for c, (class_id, class_name, conf) in enumerate(row):
            y1, y2 = r * cell_h, (r + 1) * cell_h
            x1, x2 = c * cell_w, (c + 1) * cell_w
            color = COLORS[class_id % len(COLORS)]

            # Filled rectangle
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

            # Border
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label
            label = f"{class_name} {conf:.2f}"
            font_scale = max(0.3, min(cell_w, cell_h) / 600)
            thickness = 1
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            tx = x1 + (cell_w - tw) // 2
            ty = y1 + (cell_h + th) // 2
            # OpenCV 5 的字距隨 thickness 變，粗黑細白的描邊會錯開，改用偏移描邊（見 src/draw.py）
            put_text_outlined(frame, label, (tx, ty), font_scale, thickness=thickness)

    # Blend fill
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame


class GridTracker:
    """
    把每一輪的分類機率做時序平滑（EMA）+ 遲滯切換（hysteresis），輸出穩定的網格類別。

    沒有這層處理的話，同一格會因為畫面雜訊在兩個相近的類別之間反覆跳動，
    看起來就是顏色一直閃。EMA 讓機率變化平滑，遲滯則要求新類別要「明顯」贏過
    舊類別才准換色。

    影片模式和即時模式共用這份邏輯，差別只在呼叫頻率：
    影片模式每一幀更新一次，即時模式每次分析（約 1~2 秒）更新一次。
    """

    def __init__(self, rows: int, cols: int, ema_alpha: float, switch_margin: float):
        self.rows = rows
        self.cols = cols
        self.ema_alpha = ema_alpha
        self.switch_margin = switch_margin
        self.ema_probs = None       # 每格類別機率的時序 EMA
        self.prev_class = None      # 每格上一輪確定的類別 (含遲滯切換)

    def update(self, probs: np.ndarray, names: dict) -> list[list[tuple]]:
        """吃一輪的原始機率，回傳 [[(class_id, class_name, conf), ...], ...]。"""
        if self.ema_probs is None:
            self.ema_probs = probs.copy()
            self.prev_class = np.argmax(self.ema_probs, axis=-1)
        else:
            self.ema_probs = self.ema_alpha * probs + (1.0 - self.ema_alpha) * self.ema_probs

        # 用遲滯 (hysteresis) 切換類別：新類別必須贏過舊類別 switch_margin 才更換
        top1 = np.argmax(self.ema_probs, axis=-1)
        for r in range(self.rows):
            for c in range(self.cols):
                old = int(self.prev_class[r, c])
                new = int(top1[r, c])
                if new != old and (self.ema_probs[r, c, new] - self.ema_probs[r, c, old]) < self.switch_margin:
                    top1[r, c] = old
        self.prev_class = top1

        grid = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                cls = int(top1[r, c])
                conf = float(self.ema_probs[r, c, cls])
                row.append((cls, names[cls], conf))
            grid.append(row)
        return grid
