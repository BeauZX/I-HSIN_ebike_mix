"""CSI 鏡頭讀取（Picamera2）。

一顆鏡頭同時要兩路輸出，由 ISP 硬體各自縮放，不耗 CPU：
    main   1280×720 BGR  → 顯示、錄影、路面 ROI 裁切
    lores  640×640  RGB  → 直接餵 YOLO（與 OverlayView 原本 cv2.resize 到 640×640 的拉伸方式相同，
                           偵測框用 0~1 比例換算回 main 座標）

為什麼不用 cv2.VideoCapture：Pi 5 的 /dev/video0 只吐 raw Bayer，OpenCV 打得開卻讀不到可用畫面。
為什麼不用 asphalt 的 rpicam-vid pipe：pipe 讀得慢會撕裂（要另開讀取緒補救）、只能一路 YUV 輸出、
Python 還要自己轉 BGR。Picamera2 每次拿到的都是完整的一幀。

曝光組合沿用 asphalt 專案在 rpicam-vid 上實測的設定，對應到 libcamera control：
    --exposure normal → AeExposureMode Normal
    --awb auto        → AwbMode Auto
    --denoise cdn_off → NoiseReductionMode Minimal（關彩色降噪、保留空間降噪）
    --rotation 180    → Transform(hflip=1, vflip=1)
"""

from __future__ import annotations

import time

import numpy as np

from .settings import CameraSettings


class Camera:
    def __init__(self, cfg: CameraSettings, lores_size: tuple[int, int]):
        try:
            from libcamera import Transform, controls
            from picamera2 import Picamera2
        except ImportError:
            raise RuntimeError(
                "找不到 Picamera2。它是 apt 的 python3-picamera2，裝在系統 Python；"
                "venv 必須用 `uv venv --system-site-packages` 建立才看得到。") from None

        self.cfg = cfg
        self.lores_size = lores_size
        self.cam = None
        self.started = False
        self.frames = 0

        exposure_modes = {
            "normal": controls.AeExposureModeEnum.Normal,
            "sport": controls.AeExposureModeEnum.Short,
            "short": controls.AeExposureModeEnum.Short,
            "long": controls.AeExposureModeEnum.Long,
        }
        awb_modes = {
            "auto": controls.AwbModeEnum.Auto,
            "incandescent": controls.AwbModeEnum.Incandescent,
            "tungsten": controls.AwbModeEnum.Tungsten,
            "fluorescent": controls.AwbModeEnum.Fluorescent,
            "indoor": controls.AwbModeEnum.Indoor,
            "daylight": controls.AwbModeEnum.Daylight,
            "cloudy": controls.AwbModeEnum.Cloudy,
        }
        nr = controls.draft.NoiseReductionModeEnum
        denoise_modes = {
            "off": nr.Off, "cdn_off": nr.Minimal, "cdn_fast": nr.Fast,
            "cdn_hq": nr.HighQuality, "auto": nr.Fast,
        }
        for value, table, key in ((cfg.exposure, exposure_modes, "exposure"),
                                  (cfg.awb, awb_modes, "awb"),
                                  (cfg.denoise, denoise_modes, "denoise")):
            if value not in table:
                raise ValueError(f"camera.yaml 的 {key} 不認得 {value!r}，可用：{sorted(table)}")

        frame_us = max(1, round(1_000_000 / cfg.fps))
        flip = 1 if cfg.rotation == 180 else 0
        self.cam = Picamera2(cfg.index)
        try:
            config = self.cam.create_video_configuration(
                # Picamera2 的 RGB888 在記憶體裡實際是 [B, G, R]，可直接給 OpenCV；
                # BGR888 反而是 [R, G, B]，剛好是 YOLO 要的順序，lores 就用它省一次轉換。
                main={"size": (cfg.width, cfg.height), "format": "RGB888"},
                lores={"size": lores_size, "format": "BGR888"},
                transform=Transform(hflip=flip, vflip=flip),
                controls={
                    "FrameDurationLimits": (frame_us, frame_us),
                    "AeExposureMode": exposure_modes[cfg.exposure],
                    "AwbMode": awb_modes[cfg.awb],
                    "NoiseReductionMode": denoise_modes[cfg.denoise],
                },
                buffer_count=6,
                queue=True,
            )
            self.cam.configure(config)
            self.cam.start()
            self.started = True
            if cfg.warmup > 0:
                time.sleep(cfg.warmup)       # 等 AEC/AGC 收斂，不然前幾幀過曝或全黑
        except Exception:
            self.close()
            raise

    def read(self) -> tuple[np.ndarray, np.ndarray] | None:
        """回傳同一瞬間的 (main BGR, lores RGB)；相機停了回傳 None。"""
        if self.cam is None:
            return None
        req = self.cam.capture_request()
        try:
            main = req.make_array("main")
            lores = req.make_array("lores")
        finally:
            req.release()       # 一定要還 buffer，不然 6 個用完相機就停了
        self.frames += 1
        return np.ascontiguousarray(main), np.ascontiguousarray(lores)

    def close(self) -> None:
        if self.cam is None:
            return
        if self.started:
            try:
                self.cam.stop()
            except Exception:
                pass
            self.started = False
        try:
            self.cam.close()
        except Exception:
            pass
        self.cam = None
