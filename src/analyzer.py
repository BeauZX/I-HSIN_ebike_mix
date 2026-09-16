"""路面分析緒：ResNet 判路面種類 → 依種類跑 asphalt 或 cement 分級。

主緒維持相機幀率顯示與錄影，這裡在背景持續分析「最新的一幀」，
主緒隨時取最近一次結果填補。submit() 只保留最新一幀、舊的直接丟：
分析比相機慢時，排隊處理會越積越舊，畫面上標的顏色會離現況越來越遠。

每輪耗時（Pi 5 + Hailo-8，ROI 987×290 實測，2026-09-15）：ResNet ~8 ms；瀝青 15 格 ~33 ms（純推論 28 ms）；
水泥 15 格 ~28 ms + CPU 裂縫偵測 ~42 ms + 融合 ≈ 76 ms。單獨跑一輪：瀝青 ~40 ms、水泥 ~85 ms。
偵測緒同時在跑 YOLO（每幀一次）時，scheduler 在四個 HEF 間輪流切換，一輪拉長到瀝青 ~78 ms、
水泥 ~115 ms，所以實際瀝青路網格每秒約更新 12 次、水泥路約 8 次。

路面種類切換（含連續確認遲滯，見 road_type.py）時會 reset 分級器，
把上一種路面的平滑記憶清掉。
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass

import numpy as np

from .grading.graders import AsphaltGrader, CementGrader
from .road_type import MODE_NONE, RoadTypeClassifier


@dataclass
class RoadResult:
    mode: str                   # asphalt / cement / none
    label: str                  # ResNet top-1 類別名
    confidence: float
    grader: object | None       # 產生 result 的分級器（主緒用它的 draw()）
    result: object | None       # 該分級器的結果；mode 為 none 時是 None


class RoadAnalyzer(threading.Thread):
    def __init__(self, road_type: RoadTypeClassifier,
                 asphalt: AsphaltGrader, cement: CementGrader):
        super().__init__(daemon=True, name="road")
        self.road_type = road_type
        self.graders = {"asphalt": asphalt, "cement": cement}

        self._pending: np.ndarray | None = None
        self._lock = threading.Lock()
        self._new_frame = threading.Event()
        self._stop_evt = threading.Event()      # 不能取名 _stop：會蓋掉 Thread 內部方法

        self._result: RoadResult | None = None
        self._result_lock = threading.Lock()
        self.update_count = 0
        self.last_ms = 0.0
        self.switches = 0

    def submit(self, roi_crop: np.ndarray) -> None:
        """主緒呼叫：丟最新一幀的 ROI 進來（會覆寫掉還沒被處理的舊幀）。"""
        with self._lock:
            self._pending = roi_crop
        self._new_frame.set()

    def latest(self) -> RoadResult | None:
        with self._result_lock:
            return self._result

    def stop(self) -> None:
        self._stop_evt.set()
        self._new_frame.set()

    def run(self) -> None:
        while not self._stop_evt.is_set():
            self._new_frame.wait(timeout=1.0)
            if self._stop_evt.is_set():
                break
            with self._lock:
                frame = self._pending
                self._pending = None
            self._new_frame.clear()
            if frame is None:
                continue

            t0 = time.perf_counter()
            try:
                mode, switched = self.road_type.update(frame)
                grader = self.graders.get(mode)
                if switched:
                    self.switches += 1
                    print(f"[路面] 切換為 {self.road_type.label}"
                          f" → {'不分級' if grader is None else grader.name + ' 分級'}", flush=True)
                    if grader is not None:
                        grader.reset()
                result = grader.analyze(frame) if grader is not None else None
            except Exception as e:
                # 單次推論失敗不該讓整個錄影中斷，記錄後繼續等下一幀
                print(f"[路面] 分析失敗: {e}", file=sys.stderr, flush=True)
                time.sleep(0.5)
                continue

            with self._result_lock:
                self._result = RoadResult(mode, self.road_type.label, self.road_type.confidence,
                                          grader, result)
            self.update_count += 1
            self.last_ms = (time.perf_counter() - t0) * 1000.0
