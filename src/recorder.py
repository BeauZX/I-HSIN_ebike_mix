"""分段錄影（改寫自 OverlayView 的 SegmentRecorder）：每 segment_seconds 秒切一個新檔，
檔名為該段開始時間 YYYYMMDD_HHMMSS.mp4，存到 outputs/。

主迴圈的實際幀率會隨負載浮動，VideoWriter 卻需要固定 fps，所以依牆上時鐘決定
每幀該寫幾次（慢時重複、快時丟棄），播放速度才會與真實時間一致。

close() 是必要的：mp4 的索引（moov）在 release 時才寫，沒關的檔案無法播放；
呼叫端把它包在 try/finally 裡，Ctrl+C 也能留下可播放的片段。
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from .settings import OutputSettings


class SegmentRecorder:
    def __init__(self, cfg: OutputSettings):
        self.out_dir = Path(cfg.dir)
        self.segment_sec = cfg.segment_seconds
        self.fps = cfg.fps
        self.fourcc = cv2.VideoWriter_fourcc(*cfg.codec)
        self.writer: cv2.VideoWriter | None = None
        self.path: Path | None = None
        self.seg_start = 0.0
        self.written = 0
        self.total_written = 0
        self.out_dir.mkdir(parents=True, exist_ok=True)
        print(f"錄影輸出到 {self.out_dir}（每 {self.segment_sec:g} 秒一段，{self.fps:g} fps）")

    def write(self, frame: np.ndarray) -> None:
        now = time.time()
        if self.writer is None or now - self.seg_start >= self.segment_sec:
            self._rotate(frame, now)
        target = int((now - self.seg_start) * self.fps) + 1    # 到目前為止這段應該有幾幀
        while self.written < target:
            self.writer.write(frame)
            self.written += 1
            self.total_written += 1

    def _rotate(self, frame: np.ndarray, now: float) -> None:
        self.close()
        h, w = frame.shape[:2]
        self.path = self.out_dir / (time.strftime("%Y%m%d_%H%M%S", time.localtime(now)) + ".mp4")
        self.writer = cv2.VideoWriter(str(self.path), self.fourcc, self.fps, (w, h))
        if not self.writer.isOpened():
            self.writer = None
            raise RuntimeError(f"無法建立影片檔：{self.path}")
        self.seg_start = now
        self.written = 0
        print(f"[錄影] 開始新片段：{self.path.name}")

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            print(f"[錄影] 已存檔：{self.path.name}")
            self.writer = None
