"""路面種類辨識（ResNet34 HEF，改寫自 road_classification 專案 hailo_classifier.py）。

前處理與 timm 的評估流程一致：短邊縮到 image_size / crop_pct 後置中裁切、BGR→RGB、
0~255 uint8 直接餵（HEF 編譯時已含 normalization 層，載入時檢查輸入量化參數 scale=1、zp=0）。
輸出是 logits，在 host 端做 softmax。

之上再做兩層穩定化（原專案只有第一層）：
    1. smooth_window：平均最近 N 次的機率，減少閃爍
    2. confirm_count：新的分級模式要連續 N 次勝出才切換 → 瀝青/水泥邊界不會來回跳、
       後面的網格平滑器不會一直被重置
"""

from __future__ import annotations

import json
import math
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from .hailo import HailoModel
from .settings import RoadTypeSettings

_INTERPOLATIONS = {
    "bicubic": cv2.INTER_CUBIC,
    "bilinear": cv2.INTER_LINEAR,
    "nearest": cv2.INTER_NEAREST,
}

MODE_NONE = "none"          # 沒有對應分級模型的路面（Belgian Block / Forest Road）


def load_model_metadata(model_dir: Path) -> tuple[dict, list[str]]:
    """讀取並驗證 config.json 與 classes.txt。"""
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    names = [ln.strip() for ln in (model_dir / "classes.txt").read_text(encoding="utf-8-sig").splitlines()
             if ln.strip()]
    if not names:
        raise ValueError(f"{model_dir / 'classes.txt'} 未包含任何類別名稱")
    config_names = [str(n) for n in config.get("classes", [])]
    if config_names and config_names != names:
        raise ValueError("classes.txt 與 config.json 的類別名稱或順序不一致")
    return config, names


class RoadTypeClassifier:
    def __init__(self, model: HailoModel, cfg: RoadTypeSettings):
        self.model = model
        self.cfg = cfg
        config, self.names = load_model_metadata(cfg.model_dir)

        if len(model.input_shape) != 3 or model.input_shape[0] != model.input_shape[1]:
            raise ValueError(f"ResNet HEF 輸入必須是正方形 HxWx3（目前 {model.input_shape}）")
        self.image_size = int(model.input_shape[0])
        if int(np.prod(model.output_shape)) != len(self.names):
            raise ValueError(f"ResNet HEF 輸出 {int(np.prod(model.output_shape))} 類，"
                             f"但 classes.txt 有 {len(self.names)} 個")
        q = model.input_quant
        if not math.isclose(q.qp_scale, 1.0, rel_tol=1e-6) or q.qp_zp != 0:
            raise ValueError(
                "ResNet HEF 輸入的量化參數不是 scale=1、zero_point=0，表示編譯時未加入 normalization 層"
                f"（目前 scale={q.qp_scale}, zp={q.qp_zp}）")
        configured_size = config.get("image_size")
        if configured_size is not None and int(configured_size) != self.image_size:
            print(f"警告：config.json 的 image_size={configured_size}，但 HEF 輸入為 {self.image_size}；以 HEF 為準")

        pre = config.get("pretrained_cfg") or {}
        self.crop_pct = float(pre.get("crop_pct", 224 / 256))
        self.interpolation = _INTERPOLATIONS.get(str(pre.get("interpolation", "bicubic")).lower(),
                                                 cv2.INTER_CUBIC)

        # 類別 → 分級模式（asphalt / cement / none）
        self.mode_of = {n: cfg.grading.get(n, MODE_NONE) for n in self.names}
        unknown = set(cfg.grading) - set(self.names)
        if unknown:
            print(f"警告：road_type.yaml 的 grading 有 classes.txt 裡沒有的類別：{unknown}")

        self._history: deque[np.ndarray] = deque(maxlen=cfg.smooth_window)
        self.mode = MODE_NONE           # 目前已確認的分級模式
        self._pending_mode = MODE_NONE
        self._pending_count = 0
        self.label = "?"                # 目前平滑後的 top-1 類別名
        self.confidence = 0.0

    def preprocess(self, crop: np.ndarray) -> np.ndarray:
        h, w = crop.shape[:2]
        scale_size = math.floor(self.image_size / self.crop_pct)
        scale = scale_size / min(h, w)
        new_w = max(scale_size, round(w * scale))
        new_h = max(scale_size, round(h * scale))
        resized = cv2.resize(crop, (new_w, new_h), interpolation=self.interpolation)
        top = (new_h - self.image_size) // 2
        left = (new_w - self.image_size) // 2
        square = resized[top:top + self.image_size, left:left + self.image_size]
        return np.ascontiguousarray(cv2.cvtColor(square, cv2.COLOR_BGR2RGB))

    def update(self, roi_bgr: np.ndarray) -> tuple[str, bool]:
        """推論一次並更新狀態。回傳 (目前分級模式, 這次是否切換了模式)。"""
        logits = self.model.infer([self.preprocess(roi_bgr)])[0].astype(np.float64).reshape(-1)
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        self._history.append(probs.astype(np.float32))
        smoothed = np.mean(self._history, axis=0)
        top = int(np.argmax(smoothed))
        self.label = self.names[top]
        self.confidence = float(smoothed[top])

        # 連續確認才切換
        want = self.mode_of[self.label]
        switched = False
        if want == self.mode:
            self._pending_mode, self._pending_count = self.mode, 0
        else:
            if want == self._pending_mode:
                self._pending_count += 1
            else:
                self._pending_mode, self._pending_count = want, 1
            if self._pending_count >= self.cfg.confirm_count:
                self.mode = want
                self._pending_count = 0
                switched = True
        return self.mode, switched
