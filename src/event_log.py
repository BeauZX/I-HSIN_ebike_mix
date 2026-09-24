"""偵測結果 log：只在狀態改變時記一筆（JSON Lines，每行一個 JSON）。

記的是摘要，不是每個框：
    road      路面種類（ResNet top-1），road_conf 是該筆寫入當下的信心（信心變動本身不觸發紀錄）
    grading   分級模式 asphalt / cement / none
    grid      網格各等級格數 {"severe", "slight", "smooth"}；不分級的路面為 {}（null 表示還沒有結果）
    objects   各類人車數量（detect.yaml 的 classes，沒看到的記 0）
    door      車門狀態 open / closed / none

防閃動：偵測每秒十幾次，漏抓一兩幀就會讓數量跳一下。每個欄位的新值要穩定維持
log_stable_sec 秒才算改變（output.yaml），紀錄的 time 是新值「開始出現」的時間，不是確認的時間。

切檔：跟錄影同名、同一時間切（outputs/logs/20260923_144950.jsonl ↔ outputs/20260923_144950.mp4）；
沒錄影（--no-save）時自己每 segment_seconds 秒切一檔。每個檔的第一行是 event=segment_start，
記當下的完整狀態，所以每個檔單獨拿出來看也知道起始狀態；之後每行是 event=change，
changed 列出這次改變的欄位，其餘欄位是當下的狀態。offset_sec 是距離該檔（該段影片）開頭的秒數。
"""

from __future__ import annotations

import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from .settings import OutputSettings

FIELDS = ("road", "grading", "grid", "objects", "door")


def snapshot(rr, det_result, door_result, classes: list[str]) -> dict:
    """把三條分析緒的最新結果整理成摘要；還沒有結果的欄位是 None（不會觸發紀錄）。"""
    state = dict.fromkeys(FIELDS)
    state["road_conf"] = None
    if rr is not None:
        state["road"] = rr.label
        state["road_conf"] = round(float(rr.confidence), 3)
        state["grading"] = rr.mode
        # 不分級的路面沒有網格；用空 dict 表示「確定沒有」，與「還沒結果」的 None 區分
        state["grid"] = rr.grader.grade_counts(rr.result) if rr.result is not None else {}
    if det_result is not None:
        seen = Counter(label for (_, _, label, *_) in det_result.detections)
        state["objects"] = {c: seen.get(c, 0) for c in classes}
    if door_result is not None:
        state["door"] = door_result.state
    return state


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t).isoformat(timespec="milliseconds")


class EventLogger:
    def __init__(self, cfg: OutputSettings):
        self.dir = Path(cfg.log_dir)
        self.segment_sec = cfg.segment_seconds
        self.stable_sec = cfg.log_stable_sec
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path: Path | None = None
        self._f = None
        self._seg_start = 0.0
        self._stable: dict = dict.fromkeys(FIELDS)      # 已確認（已寫進 log）的值
        self._cand: dict = {}                           # 欄位 → (候選新值, 開始出現的時間)
        self.records = 0
        print(f"偵測結果 log 輸出到 {self.dir}（狀態改變才記，需穩定 {self.stable_sec:g} 秒）")

    def update(self, state: dict, now: float, segment: tuple[str, float] | None = None) -> None:
        """主緒每幀呼叫。segment = (錄影檔名不含副檔名, 該段開始時間)；沒錄影時傳 None。"""
        if segment is None:
            if self._f is None or now - self._seg_start >= self.segment_sec:
                self._rotate(time.strftime("%Y%m%d_%H%M%S", time.localtime(now)), now, now, state)
        elif self.path is None or self.path.stem != segment[0]:
            self._rotate(segment[0], segment[1], now, state)

        changed, since = [], None
        for k in FIELDS:
            v = state[k]
            if v is None or v == self._stable[k]:
                self._cand.pop(k, None)
                continue
            cand = self._cand.get(k)
            if cand is None or cand[0] != v:
                self._cand[k] = cand = (v, now)
            if now - cand[1] >= self.stable_sec:
                self._stable[k] = v
                del self._cand[k]
                changed.append(k)
                since = cand[1] if since is None else min(since, cand[1])
        if changed:
            self._write("change", since, state, changed)

    def _rotate(self, stem: str, seg_start: float, now: float, state: dict) -> None:
        self.close()
        self.path = self.dir / f"{stem}.jsonl"
        self._f = open(self.path, "a", encoding="utf-8")
        self._seg_start = seg_start
        self._write("segment_start", now, state, [])

    def _write(self, event: str, t: float, state: dict, changed: list[str]) -> None:
        rec = {"time": _iso(t), "offset_sec": round(max(0.0, t - self._seg_start), 2), "event": event}
        if changed:
            rec["changed"] = changed
        rec.update({k: self._stable[k] for k in FIELDS})
        rec["road_conf"] = state.get("road_conf") if self._stable["road"] is not None else None
        self._f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._f.flush()             # 紀錄很稀疏，每筆都寫到磁碟，斷電也只丟最後一筆
        self.records += 1

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None
