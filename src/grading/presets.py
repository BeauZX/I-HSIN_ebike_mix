"""載入 pothole2 的 smooth/rough 參數檔，轉成 CrackConfig。

沿用 pothole2 build_detector() 的常數 → 欄位對應，讓使用者在 pothole2
調好的兩套預設能原封不動被融合管線採用。
"""

from __future__ import annotations

import json
from pathlib import Path

from .pothole.crack import CrackConfig

# presets/*.json 的大寫常數名 → CrackConfig 欄位名（對應自 pothole2/main.py）
_KEY_MAP = {
    "CRACK_METHOD": "method",
    "CRACK_BLUR": "blur_method",
    "THRESH_K": "thresh_k",
    "THRESH_K_HIGH": "thresh_k_high",
    "THRESH_K_LOW": "thresh_k_low",
    "ADAPTIVE_C": "adaptive_c",
    "FRANGI_BETA": "frangi_beta",
    "MIN_CRACK_SCORE": "min_score",
    "MIN_CONTRAST": "min_contrast",
    "MIN_CRACK_LENGTH": "min_length",
    "MAX_CRACK_WIDTH": "max_width",
    "DETECT_MESH": "detect_mesh",
    "MESH_MIN_AREA": "mesh_min_area",
    "MESH_MIN_FILL": "mesh_min_fill",
    "MESH_MAX_FILL": "mesh_max_fill",
    "MIN_SHARPNESS": "min_sharpness",
    "EXCLUDE_MARKINGS": "exclude_markings",
    "MARKING_DILATE": "marking_dilate",
    "POST_MORPHOLOGY": "post_morphology",
    "SUPPRESS_TEXTURE": "suppress_texture",
    "LINE_LENGTH": "line_length",
    "CLAHE_CLIP": "clahe_clip_limit",
    "REMOVE_DARK_BLOBS": "remove_thick",
}


def load_crack_config(
    preset_path: str | Path,
    honor_raw: bool = False,
    target_width: int | None = None,
) -> CrackConfig:
    """讀 presets/<name>.json → CrackConfig。

    honor_raw=False（預設）：即使 JSON 內 RAW_DETECT_ALL=true 也「保留形狀過濾」，
        讓 rough 預設輸出的是「像裂縫」的候選（融合時才好量化與守門）。
    honor_raw=True：完全比照 pothole2 的 RAW 模式（關掉所有過濾，等於二值圖直接標）。
    target_width：覆寫 CrackConfig.target_width（None = 沿用檔內/預設 900）。
    """
    data = json.loads(Path(preset_path).read_text(encoding="utf-8"))
    cfg = CrackConfig()
    for key, field in _KEY_MAP.items():
        if key in data:
            setattr(cfg, field, data[key])
    if "FRANGI_SCALES" in data:
        cfg.frangi_scales = tuple(data["FRANGI_SCALES"])

    if honor_raw and data.get("RAW_DETECT_ALL", False):
        cfg.min_area = 0
        cfg.min_length = 0
        cfg.min_elongation = 0.0
        cfg.max_width = float("inf")
        cfg.min_contrast = 0.0
        cfg.min_sharpness = 0.0
        cfg.min_score = 0.0
        cfg.exclude_markings = False

    if target_width is not None:
        cfg.target_width = target_width
    return cfg
