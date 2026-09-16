"""YOLO-cls 網格分類工具。

改寫自 pothole 專案的 main.py（classify_grid / select_roi / EMA 遲滯），
抽成可重用、與執行區段解耦的函式，供融合管線使用。

嚴重度採「與模型類別排序無關」的表示：以名稱關鍵字對應到固定分級
    0 = smooth（平整）
    1 = slight（輕微）
    2 = severe（嚴重）
避免不同模型（cement / asphalt）類別索引順序不一致造成的錯位。
"""

from __future__ import annotations

import cv2
import numpy as np

SEVERITY_SMOOTH = 0
SEVERITY_SLIGHT = 1
SEVERITY_SEVERE = 2
SEVERITY_NAMES = {0: "smooth", 1: "slight", 2: "severe"}
ROI_WINDOW_TITLE = "Select ROI (Enter=confirm / C=cancel)"


def classify_grid(model, frame: np.ndarray, rows: int, cols: int) -> tuple[np.ndarray, dict]:
    """把 frame 切成 rows*cols 網格，逐格分類。

    model 是 combined_road.backend 的分類器（Hailo 或 CPU YOLO），
    介面為 model(cells) -> probs[N, C]、model.names。
    回傳 (probs[rows, cols, C], names)。names 為 {class_index: class_name}。
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

    flat = model(cells)                         # [rows*cols, C]
    probs = np.asarray(flat, dtype=np.float32).reshape(rows, cols, -1)
    return probs, model.names


def severity_class_index(names: dict) -> dict[int, int]:
    """由 model.names 建出 {severity_level: class_index}。

    以名稱關鍵字判斷，避免寫死索引順序。找不到對應者留空。
    """
    mapping: dict[int, int] = {}
    for idx, name in names.items():
        low = str(name).lower()
        if "severe" in low:
            mapping[SEVERITY_SEVERE] = idx
        elif "slight" in low or "moderate" in low:
            mapping[SEVERITY_SLIGHT] = idx
        elif "smooth" in low or "good" in low or "intact" in low:
            mapping[SEVERITY_SMOOTH] = idx
    return mapping


def probs_to_severity(
    probs: np.ndarray, names: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把每格類別機率轉為嚴重度表示。

    回傳：
      expected[rows, cols]  — 期望嚴重度 0..1 = P(slight)*0.5 + P(severe)*1.0
      level[rows, cols]     — top1 類別對應的嚴重分級 (0/1/2)
      conf[rows, cols]      — top1 類別機率
    """
    idx_map = severity_class_index(names)
    i_slight = idx_map.get(SEVERITY_SLIGHT)
    i_severe = idx_map.get(SEVERITY_SEVERE)

    rows, cols = probs.shape[:2]
    expected = np.zeros((rows, cols), np.float32)
    if i_slight is not None:
        expected += 0.5 * probs[:, :, i_slight]
    if i_severe is not None:
        expected += 1.0 * probs[:, :, i_severe]

    class_to_level = {ci: lv for lv, ci in idx_map.items()}
    argmax = np.argmax(probs, axis=-1)
    level = np.zeros((rows, cols), np.int32)
    conf = np.zeros((rows, cols), np.float32)
    for r in range(rows):
        for c in range(cols):
            ci = int(argmax[r, c])
            level[r, c] = class_to_level.get(ci, SEVERITY_SMOOTH)
            conf[r, c] = float(probs[r, c, ci])
    return expected, level, conf


def select_roi(first_frame: np.ndarray) -> tuple[int, int, int, int] | None:
    """手動拖拉框選矩形分析範圍（沿用 pothole 專案作法）。回傳 (x1,y1,x2,y2)。"""
    print("請用滑鼠拖拉選取分析範圍，按 Enter 確認，按 C 取消（全畫面）")
    # 視窗標題必須是 ASCII：OpenCV 5 的 Qt 後端用名稱找視窗，中文會找不到而崩潰
    roi = cv2.selectROI(ROI_WINDOW_TITLE, first_frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(ROI_WINDOW_TITLE)
    x, y, w, h = roi
    if w == 0 or h == 0:
        return None
    return x, y, x + w, y + h


class TemporalSmoother:
    """對每格「連續嚴重度分數」做 EMA 時序平滑，並在分級時加遲滯避免抖動。

    改寫自 pothole 專案 main.py 的 EMA + hysteresis 邏輯，但改為作用在
    融合後的連續分數上（原版作用在類別機率）。
    """

    def __init__(
        self,
        alpha: float = 0.45,
        switch_margin: float = 0.08,
        t_slight: float = 0.30,
        t_severe: float = 0.60,
    ) -> None:
        self.alpha = alpha
        self.switch_margin = switch_margin
        self.t_slight = t_slight
        self.t_severe = t_severe
        self.ema: np.ndarray | None = None
        self.prev_level: np.ndarray | None = None

    def _discretize(self, score: np.ndarray) -> np.ndarray:
        rows, cols = score.shape
        out = np.empty((rows, cols), np.int32)
        prev = self.prev_level
        for r in range(rows):
            for c in range(cols):
                s = float(score[r, c])
                if s >= self.t_severe:
                    lv = 2
                elif s >= self.t_slight:
                    lv = 1
                else:
                    lv = 0
                if prev is not None:
                    p = int(prev[r, c])
                    if lv > p:  # 升級需超過邊界 + margin
                        boundary = self.t_severe if lv == 2 else self.t_slight
                        if s < boundary + self.switch_margin:
                            lv = p
                    elif lv < p:  # 降級需低於邊界 - margin
                        boundary = self.t_severe if p == 2 else self.t_slight
                        if s > boundary - self.switch_margin:
                            lv = p
                out[r, c] = lv
        return out

    def update(self, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """吃當幀每格分數，回傳 (平滑後分數, 遲滯後分級)。"""
        if self.ema is None:
            self.ema = score.astype(np.float32).copy()
        else:
            self.ema = self.alpha * score + (1.0 - self.alpha) * self.ema
        level = self._discretize(self.ema)
        self.prev_level = level
        return self.ema, level


class PIDSeveritySmoother:
    """每格「連續嚴重度分數」的 PID 平滑 + 遲滯 + 逐級緩衝。

    目標：路面分級隨時間平滑過渡、避免抖動，且**不會跨級跳變**——
    嚴重 severe 與 平整 smooth 之間一定要經過 輕微 slight，且每次換級前
    需連續確認數幀，形成「緩衝時間」。介面與 TemporalSmoother 相同
    （update(score) → (平滑分數, 分級)），可直接替換。

    三段式：
      1) PID：以當幀每格原始融合分數為設定點(setpoint)，驅動顯示分數
         平滑逼近。積分項讓短暫尖峰不會立刻改變輸出（緩衝），微分項抑制
         突變。當 ki=kd=0 時退化為原本 EMA（y ← y + kp·(r−y)，等效
         alpha=kp）。
      2) 遲滯：離散化門檻加 switch_margin，避免在邊界來回抖動。
      3) 逐級緩衝：每次最多變動一級，且新目標級需連續維持 min_dwell 幀
         才確認 → severe↔smooth 不會直接跳，必經 slight，且各段各有緩衝。
    """

    def __init__(
        self,
        kp: float = 0.35,
        ki: float = 0.03,
        kd: float = 0.08,
        i_limit: float = 1.0,
        t_slight: float = 0.30,
        t_severe: float = 0.60,
        switch_margin: float = 0.08,
        min_dwell: int = 6,
    ) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.i_limit = i_limit
        self.t_slight = t_slight
        self.t_severe = t_severe
        self.switch_margin = switch_margin
        self.min_dwell = max(1, int(min_dwell))

        self.output: np.ndarray | None = None       # 每格顯示分數（PID 輸出 / PV）
        self.integral: np.ndarray | None = None
        self.prev_error: np.ndarray | None = None
        self.level: np.ndarray | None = None         # 已確認的離散分級
        self.pending_dir: np.ndarray | None = None   # 待確認的變動方向 (-1/0/+1)
        self.pending_count: np.ndarray | None = None

    def _pid(self, target: np.ndarray) -> np.ndarray:
        """對每格跑一步 PID，回傳更新後的顯示分數。"""
        if self.output is None:
            self.output = target.astype(np.float32).copy()
            self.integral = np.zeros_like(self.output)
            self.prev_error = np.zeros_like(self.output)
            return self.output
        error = target - self.output
        self.integral = np.clip(self.integral + error, -self.i_limit, self.i_limit)
        derivative = error - self.prev_error
        delta = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.output = np.clip(self.output + delta, 0.0, 1.0)
        self.prev_error = error
        return self.output

    def _target_level(self, score: np.ndarray) -> np.ndarray:
        """遲滯離散化：回傳每格「想要」達到的分級（尚未套逐級限制）。"""
        rows, cols = score.shape
        out = np.empty((rows, cols), np.int32)
        prev = self.level
        for r in range(rows):
            for c in range(cols):
                s = float(score[r, c])
                if s >= self.t_severe:
                    lv = 2
                elif s >= self.t_slight:
                    lv = 1
                else:
                    lv = 0
                if prev is not None:
                    p = int(prev[r, c])
                    if lv > p:  # 升級需超過邊界 + margin
                        boundary = self.t_severe if lv == 2 else self.t_slight
                        if s < boundary + self.switch_margin:
                            lv = p
                    elif lv < p:  # 降級需低於邊界 - margin
                        boundary = self.t_severe if p == 2 else self.t_slight
                        if s > boundary - self.switch_margin:
                            lv = p
                out[r, c] = lv
        return out

    def _step_limit(self, target: np.ndarray) -> np.ndarray:
        """逐級 + 緩衝：每格朝目標一次最多動一級，且需連續確認 min_dwell 幀。"""
        if self.level is None:
            self.level = target.copy()
            self.pending_dir = np.zeros_like(target)
            self.pending_count = np.zeros_like(target)
            return self.level
        rows, cols = target.shape
        for r in range(rows):
            for c in range(cols):
                cur = int(self.level[r, c])
                tgt = int(target[r, c])
                if tgt == cur:  # 已到位 → 清空待確認
                    self.pending_dir[r, c] = 0
                    self.pending_count[r, c] = 0
                    continue
                step = 1 if tgt > cur else -1
                if int(self.pending_dir[r, c]) == step:
                    self.pending_count[r, c] += 1
                else:  # 方向改變 → 重新計時
                    self.pending_dir[r, c] = step
                    self.pending_count[r, c] = 1
                if int(self.pending_count[r, c]) >= self.min_dwell:
                    self.level[r, c] = cur + step  # 一次只動一級
                    self.pending_dir[r, c] = 0
                    self.pending_count[r, c] = 0  # 下一級需各自再累積緩衝
        return self.level

    def update(self, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """吃當幀每格原始分數，回傳 (PID 平滑後分數, 逐級緩衝後分級)。"""
        y = self._pid(score.astype(np.float32))
        target = self._target_level(y)
        level = self._step_limit(target)
        return y.copy(), level.copy()
