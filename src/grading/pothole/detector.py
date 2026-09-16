"""傳統影像處理的坑洞偵測管線。

核心假設：在一般行車視角下，坑洞相對於周圍路面會呈現「較暗」的區域
（凹陷處的陰影 + 積水/碎裂），且形狀為塊狀而非細長。我們用一連串
規則式（rule-based）的影像處理步驟把這種區域圈出來：

    灰階 → CLAHE 對比強化 → 模糊去噪 → 暗區門檻化
    → 形態學清理 → 找輪廓 → 依面積/形狀過濾

這是啟發式方法，沒有「學習」，所以對光線、陰影、油漬、人孔蓋等會有
誤判。所有門檻都放在 DetectorConfig，方便針對你的資料反覆調參。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class DetectorConfig:
    """坑洞偵測的可調參數。"""

    # 前處理：先把影像縮放到固定寬度，讓門檻在不同解析度下行為一致。
    target_width: int = 900

    # CLAHE 對比限制與網格大小（強化局部對比，凸顯凹陷陰影）。
    clahe_clip_limit: float = 2.0
    clahe_grid_size: int = 8

    # 高斯模糊核大小（奇數），抑制路面紋理造成的雜訊。
    blur_kernel: int = 7

    # 暗區門檻：判定為「暗」的條件是 亮度 < mean - dark_k * std。
    # dark_k 越大，只抓越暗的區域（越保守、誤判少但可能漏抓）。
    dark_k: float = 0.8

    # 形態學運算核大小（先 open 去小雜點，再 close 填補破洞）。
    morph_kernel: int = 5

    # 輪廓過濾條件 ──────────────────────────────────────────
    # 面積佔整張影像的比例範圍（過濾太小的雜訊與太大的整片陰影）。
    min_area_ratio: float = 0.002
    max_area_ratio: float = 0.35
    # 外接矩形長寬比上限（過濾細長的裂縫，那不是坑洞）。
    max_aspect_ratio: float = 4.0
    # extent = 輪廓面積 / 外接矩形面積，太低代表細長/破碎，非塊狀坑洞。
    min_extent: float = 0.35


@dataclass
class Detection:
    """單一缺陷偵測結果（坑洞或裂縫）。"""

    bbox: tuple[int, int, int, int]  # x, y, w, h（相對於處理後的影像尺寸）
    area: float                       # 輪廓面積（像素）
    contour: np.ndarray               # 多邊形輪廓點
    score: float                      # 啟發式信心分數 0~1
    label: str = "pothole"            # 缺陷類型："pothole" 或 "crack"


class PotholeDetector:
    """以傳統 CV 步驟偵測坑洞。

    用法：
        detector = PotholeDetector()
        detections, vis = detector.detect(image_bgr)
    """

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DetectorConfig()

    def detect(self, image_bgr: np.ndarray) -> tuple[list[Detection], np.ndarray]:
        """偵測坑洞。

        回傳 (detections, annotated_image)，兩者皆基於縮放後的影像尺寸。
        """
        cfg = self.config
        image = self._resize(image_bgr)

        # 1) 灰階
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 2) CLAHE 局部對比強化，讓凹陷陰影更明顯
        clahe = cv2.createCLAHE(
            clipLimit=cfg.clahe_clip_limit,
            tileGridSize=(cfg.clahe_grid_size, cfg.clahe_grid_size),
        )
        enhanced = clahe.apply(gray)

        # 3) 模糊去噪
        k = cfg.blur_kernel | 1  # 確保為奇數
        blurred = cv2.GaussianBlur(enhanced, (k, k), 0)

        # 4) 暗區門檻化：低於 (mean - dark_k*std) 的像素視為候選坑洞
        mean, std = blurred.mean(), blurred.std()
        thresh_value = max(0.0, mean - cfg.dark_k * std)
        _, mask = cv2.threshold(
            blurred, thresh_value, 255, cv2.THRESH_BINARY_INV
        )

        # 5) 形態學清理：open 去除小雜點，close 填補坑洞內部破洞
        mk = cfg.morph_kernel | 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mk, mk))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 6) 找外輪廓並依形狀過濾
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        detections = self._filter_contours(contours, image.shape)

        # 7) 視覺化
        from .viz import annotate
        vis = annotate(image, detections)
        return detections, vis

    def _resize(self, image_bgr: np.ndarray) -> np.ndarray:
        cfg = self.config
        h, w = image_bgr.shape[:2]
        if w == cfg.target_width:
            return image_bgr.copy()
        scale = cfg.target_width / float(w)
        return cv2.resize(
            image_bgr, (cfg.target_width, int(round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )

    def _filter_contours(
        self, contours: list[np.ndarray], shape: tuple[int, ...]
    ) -> list[Detection]:
        cfg = self.config
        h, w = shape[:2]
        image_area = float(h * w)
        results: list[Detection] = []

        for contour in contours:
            area = cv2.contourArea(contour)
            area_ratio = area / image_area
            if not (cfg.min_area_ratio <= area_ratio <= cfg.max_area_ratio):
                continue

            x, y, bw, bh = cv2.boundingRect(contour)
            aspect = max(bw, bh) / max(1.0, min(bw, bh))
            if aspect > cfg.max_aspect_ratio:
                continue  # 細長 → 比較像裂縫，不是坑洞

            extent = area / float(bw * bh)
            if extent < cfg.min_extent:
                continue  # 太破碎/不填滿外框

            # 啟發式分數：塊狀（extent 高）且面積適中者分數高
            score = float(np.clip(extent, 0.0, 1.0))
            results.append(
                Detection(
                    bbox=(x, y, bw, bh),
                    area=area,
                    contour=contour,
                    score=score,
                    label="pothole",
                )
            )

        results.sort(key=lambda d: d.area, reverse=True)
        return results
