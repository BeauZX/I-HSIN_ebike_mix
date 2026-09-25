"""依路面種類、坑洞代理訊號、人員/車門偵測，自動決定避震器目標位置（沒有實體按鈕）。

沒有正式的加權優先序，採「依序覆蓋」，最後一條覆蓋規則贏：
    1. 路面種類定基準：asphalt/cement（柏油/水泥）-> tight；Forest Road（森林路）-> loose；
       Belgian Block（比利時路）-> mid
    2. 坑洞代理（見下方註記）覆蓋 -> loose
    3. 安全（畫面裡有人、或車門開啟）最後覆蓋 -> tight，永遠贏

⚠️ 兩個已知的訊號落差（見專案 README「需要你確認的假設」，決定要不要調整這裡的邏輯）：
    - 這個系統沒有真正的坑洞偵測（grading/pothole/detector.py 的 PotholeDetector 是死程式碼，
      沒有被接進任何分級流程），這裡用「severe 格數達門檻」當代理訊號，門檻在 motor.yaml 調整。
    - 沒有人員移動速度訊號（TrackView 只有座標歷史，沒有速度欄位），這裡簡化成「畫面裡偵測到
      person 類別就觸發」，不特別判斷是否真的在動。
"""

from __future__ import annotations

from .settings import MotorSettings

_ROAD_LABEL_TARGET = {
    "Forest Road": "loose",
    "Belgian Block": "mid",
}


class TargetDebouncer:
    """decide_target() 每幀都可能因為辨識結果的雜訊而抖動（路面標籤、坑洞格數、人員偵測都可能
    一幀一幀跳），這裡加防抖動：同一個目標要連續 confirm_count 次決策都一樣才算數確認，
    仿 road_type.yaml 的 confirm_count（同一手法，防的是同一類問題）。

    跟切換頻率的硬上限（motor.yaml 的 min_switch_interval_sec）是分開兩層：這裡只負責把「確認
    穩定的目標」算出來，多久送一次交給呼叫端（main.py）決定，因為那還牽涉馬達忙碌中命令被拒絕
    要不要重試，屬於執行面而不是決策面。
    """

    def __init__(self, confirm_count: int):
        self.confirm_count = max(1, confirm_count)
        self._confirmed: str | None = None
        self._pending: str | None = None
        self._pending_count = 0

    def update(self, raw: str | None) -> str | None:
        """餵這一幀的瞬時決策（decide_target() 的輸出），回傳目前確認穩定的目標（還沒確認過就是 None）。"""
        if raw is None:
            return self._confirmed
        if raw == self._confirmed:
            self._pending, self._pending_count = None, 0
        elif raw == self._pending:
            self._pending_count += 1
        else:
            self._pending, self._pending_count = raw, 1
        if self._pending is not None and self._pending_count >= self.confirm_count:
            self._confirmed = self._pending
            self._pending, self._pending_count = None, 0
        return self._confirmed


def decide_target(rr, det_res, dr, motor_cfg: MotorSettings) -> str | None:
    """rr: RoadResult | None, det_res: DetectResult | None, dr: DoorResult | None。
    還沒有路面分析結果時回傳 None（不下任何馬達命令）。"""
    if rr is None:
        return None

    target = {"asphalt": "tight", "cement": "tight"}.get(rr.mode)
    target = _ROAD_LABEL_TARGET.get(rr.label, target)

    if rr.result is not None and rr.grader is not None:
        counts = rr.grader.grade_counts(rr.result)   # {"severe", "slight", "smooth"}，跟 event_log.py 同一套
        if counts.get("severe", 0) >= motor_cfg.pothole_severe_cell_threshold:
            target = "loose"

    if det_res is not None and any(label == "person" for (_, _, label, *_) in det_res.detections):
        target = "tight"

    if dr is not None and dr.state == "open":
        target = "tight"

    return target
