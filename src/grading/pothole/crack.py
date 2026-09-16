"""水泥路面裂縫偵測（傳統影像處理）。

核心假設：裂縫是路面上「細而暗」的線狀物。與坑洞相反，裂縫細長、
不填滿外接矩形。

二值化方式可由 CrackConfig.method 切換：

  "adaptive"（預設）— 灰階 → CLAHE → 模糊 → adaptiveThreshold（逐區塊
                      比周圍暗就標白）。直觀、對光線不均穩定。
  "blackhat"        — 灰階 → CLAHE → black-hat 形態學（凸顯比結構元素
                      細的暗線）→ 相對門檻。對整體亮度變化不敏感。

兩者得到二值遮罩後共用相同後處理：
    清除小連通元件 → 形態學 close 接線 → 找輪廓 → 依長度/細長度/寬度過濾。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .detector import Detection


@dataclass
class CrackConfig:
    """裂縫偵測的可調參數。"""

    target_width: int = 900

    clahe_clip_limit: float = 2.0
    clahe_grid_size: int = 8

    # 去噪方法：
    #   "bilateral" → 邊緣保留：抹平水泥掃紋顆粒、保留裂縫銳利邊（建議）
    #   "median"    → 中值：對顆粒/胡椒鹽雜訊很有效，亦保線
    #   "gaussian"  → 高斯：最快，但會連裂縫邊一起糊掉
    blur_method: str = "bilateral"
    blur_kernel: int = 5            # 核大小（奇數）：越大去噪越強，太大會糊掉細裂縫
    bilateral_sigma: int = 50       # bilateral 的色彩/空間 sigma：越大越平滑

    # 二值化方法："adaptive"（自適應二值化）或 "blackhat"（形態學）。
    method: str = "adaptive"

    # ── adaptive 專用 ──
    # adaptiveThreshold 的鄰域大小（奇數）：要大於裂縫寬度、涵蓋周圍路面。
    adaptive_block: int = 31
    # adaptiveThreshold 的常數 C：越大越保守（誤判少、易漏抓細裂縫）。
    adaptive_c: int = 7

    # ── blackhat 專用 ──
    # black-hat 結構元素大小（奇數）：需大於裂縫寬度才能突顯裂縫。
    blackhat_kernel: int = 15
    # black-hat / frangi 後的相對門檻：mean + thresh_k*std。
    thresh_k: float = 1.6

    # ── blackhat_hyst 專用（雙門檻 hysteresis）──
    # 不用單一門檻，而是高低兩條線：高門檻挑「確定是裂縫」的亮種子，低門檻當
    # 延伸；只保留「含種子」的連通塊，與種子不相連的孤立弱紋理一律丟掉。
    # 都是相對門檻：mean + k*std。
    #   thresh_k_high 調高 → 種子更嚴（雜點更少、易漏淡裂縫）。
    #   thresh_k_low  調低 → 延伸更寬鬆（裂縫淡段接得起來，但太低會撿回紋理）。
    thresh_k_high: float = 2.5
    thresh_k_low: float = 1.0

    # ── frangi 專用（Hessian 脊狀濾波）──
    # 偵測「連續線狀/脊狀結構」，對掃紋顆粒不敏感。挑「大尺度」可壓掉細掃紋
    # （1~2px）只留較寬的裂縫（3~6px）。scales 為高斯 σ（像素），約等於裂縫半寬。
    #   想多抓細裂縫 → 加入較小 σ（如 2.0）；掃紋太多 → 只留大 σ（4、5、6）。
    frangi_scales: tuple = (3.0, 4.0, 5.0)
    frangi_beta: float = 0.5        # 線狀 vs 塊狀的辨別度（越小越只收「細長線」）

    # 二值化（black-hat / adaptive）之後的遮罩後處理總開關。
    # False = 只做到二值化就直接找輪廓，跳過「線狀濾波／去小區塊／close 接線」，
    #         用來檢視純 black-hat 的原始能力（會比較雜、線也較斷）。
    post_morphology: bool = True

    # 移除面積小於此值的連通元件（清掉零散紋理點，但不會咬斷細裂縫）。
    min_blob_area: int = 25
    # close：把斷掉的裂縫線段接起來。
    close_kernel: int = 7

    # ── 磨掉填實的深色髒污（保留細裂縫/龜裂網路）────────────────────────
    # 原理：用「比裂縫寬的圓形」做開運算——圓塞不進細裂縫（裂縫被清掉），
    # 但塞得進胖髒污（髒污被保留）；把這團髒污從遮罩減掉，就只留細的線狀物。
    # 因為看的是「局部線寬」，龜裂的網狀細線會留下，只有填實胖暗塊被磨掉。
    # 直徑（像素）：要「大於最寬的真裂縫」才不會咬到裂縫。0 = 不啟用。
    remove_thick: int = 0

    # ── 方向性線狀濾波（壓掉掃紋/拉毛紋等「短平行細紋」）──────────────
    # 原理：裂縫是「一條連續的長線」，掃紋是「一段段的短紋」。對二值遮罩沿
    # 多個角度各做一次「線狀結構元素的開運算」，只保留「在某個方向上夠長」
    # 的前景——短紋路放不下這根長線會被清掉，長裂縫則留下。
    # 注意：line_length 設太長會連較短的真裂縫一起濾掉；彎太兇的裂縫也可能斷。
    suppress_texture: bool = False
    line_length: int = 31          # 線狀結構元素長度（像素）：要短於最短的真裂縫
    line_angles: int = 12          # 取樣角度數（0~180 度均分）：越多越保留彎曲裂縫

    # 過濾條件（要全中才算裂縫，藉此排除髒污/紋理）──────────────────
    # 細長度與寬度皆以「最小旋轉矩形」計算，斜向裂縫也準確。
    min_area: int = 80            # 最小面積（像素），濾掉雜點與小髒污
    min_length: int = 80          # 最短長度（旋轉矩形長邊）：裂縫要夠長
    min_elongation: float = 3.2   # 長寬比下限：髒污圓胖會被擋掉
    max_width: float = 18.0       # 寬度上限（旋轉矩形短邊）：裂縫全程都細，
                                  # 中間鼓成一坨的髒污會被擋掉

    # ── 網狀裂縫（龜裂/鱷魚紋 alligator）──────────────────────────────
    # 上面的長度/細長度/寬度是為「單一直線」設計的，整片網狀裂縫接近方形、又被
    # RETR_EXTERNAL 當成一大塊胖區，會被那些條件砍掉。改用「填充率」另外辨識：
    # 把外輪廓填實當作區域，數其中真正屬於裂縫的像素占比——
    #   實心暗塊（髒污/坑洞）填充率≈1；網狀裂縫大多是完好水泥島、只有細線是暗的，
    #   填充率低（約 0.06~0.5）。符合「大面積 + 填充率落在網狀區間」者判為網狀。
    detect_mesh: bool = False
    mesh_min_area: int = 4000     # 網狀區最小填實面積（像素）：只收「明顯」的大片龜裂
    mesh_min_fill: float = 0.06   # 填充率下限：太低多半是稀疏雜訊/掃紋而非真網路
    mesh_max_fill: float = 0.5    # 填充率上限：超過視為實心暗塊（髒污）而非網狀裂縫

    # 邊緣銳利度（深度替身）：裂縫是又窄又陡的暗谷（Laplacian 大），
    # 平的污漬是緩和色塊（Laplacian 小）。低於門檻就丟棄。0 = 不過濾。
    min_sharpness: float = 0.0

    # 最低暗度：區域平均暗度反應低於此值就丟棄。
    # 裂縫比周圍暗（反應為正），路面標線比周圍亮（反應≈0）→ 標線被擋掉。
    # 調高 → 連較淡的裂縫也會被濾掉；調低 → 可能放行標線。
    min_contrast: float = 4.0

    # 排除路面標線（白/黃漆）：先找出亮色標線，在其周圍畫禁區，
    # 落在禁區內的偵測一律丟棄（標線邊緣/縫隙常被誤判成裂縫）。
    exclude_markings: bool = True
    marking_bright_k: float = 1.0   # 亮度門檻 = mean + k*std（相對）
    marking_min_bright: int = 190   # 絕對亮度下限：要夠白才算標線。水泥路本身
                                    # 偏亮，靠這個避免把亮水泥誤當標線而劃禁區
    marking_dilate: int = 11        # 標線周圍禁區寬度（像素），越大排除範圍越廣
    marking_overlap: float = 0.25   # 與禁區重疊比例超過此值就視為標線而丟棄

    # 信心門檻：score 低於此值就丟棄（0 = 不過濾）。
    # score 綜合「細長 + 細 + 對比強」三項，越接近 1 越像真裂縫。
    min_score: float = 0.0
    # 對比正規化基準：暗度反應達到此值即視為對比滿分（越大越嚴）。
    contrast_norm: float = 45.0


class CrackDetector:
    """以 black-hat + 二值化偵測水泥路面裂縫。"""

    def __init__(self, config: CrackConfig | None = None) -> None:
        self.config = config or CrackConfig()

    def _preprocess_to_mask(self, image_bgr: np.ndarray):
        """共用前處理：resize→CLAHE→去噪→二值化→遮罩後處理。

        回傳 (resize 後 BGR, 最終二值遮罩, response, sharpness, marking)。
        detect 與 compute_mask 共用，確保「最終遮罩」只有單一來源。
        """
        cfg = self.config
        image = self._resize(image_bgr)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # CLAHE 強化局部對比
        clahe = cv2.createCLAHE(
            clipLimit=cfg.clahe_clip_limit,
            tileGridSize=(cfg.clahe_grid_size, cfg.clahe_grid_size),
        )
        enhanced = clahe.apply(gray)

        # 去噪：抑制水泥紋理顆粒（依方法保留裂縫邊緣程度不同）
        bn = cfg.blur_kernel | 1
        if cfg.blur_method == "bilateral":
            blurred = cv2.bilateralFilter(
                enhanced, bn, cfg.bilateral_sigma, cfg.bilateral_sigma
            )
        elif cfg.blur_method == "median":
            blurred = cv2.medianBlur(enhanced, bn)
        else:
            blurred = cv2.GaussianBlur(enhanced, (bn, bn), 0)

        # 依方法產生二值遮罩 mask 與「反應圖」response（供信心分數用）
        if cfg.method == "blackhat":
            mask, response = self._mask_blackhat(blurred)
        elif cfg.method == "blackhat_hyst":
            mask, response = self._mask_blackhat_hyst(blurred)
        elif cfg.method == "frangi":
            mask, response = self._mask_frangi(blurred)
        else:
            mask, response = self._mask_adaptive(blurred)

        # 亮色標線禁區（白/黃漆周圍），落在裡面的偵測之後會被丟棄
        marking = self._marking_mask(gray) if cfg.exclude_markings else None

        # 邊緣銳利度圖（深度替身）：暗谷越窄越陡，Laplacian 絕對值越大
        sharpness = np.abs(cv2.Laplacian(blurred, cv2.CV_32F, ksize=3))

        # 二值化之後的遮罩後處理（總開關 post_morphology 可整段跳過）
        if cfg.post_morphology:
            # 方向性線狀濾波：只留「夠長的線」，壓掉掃紋/拉毛紋等短平行細紋
            if cfg.suppress_texture:
                mask = self._suppress_texture(mask)
            # 移除面積太小的連通元件（清掉零散紋理點，不咬斷細裂縫）
            mask = self._remove_small_blobs(mask, cfg.min_blob_area)
            # close：接合斷掉的裂縫線段
            ck = cfg.close_kernel | 1
            close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ck, ck))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k, iterations=1)

        # 磨掉填實的深色髒污（保留細裂縫/龜裂）——獨立於 post_morphology，RAW 模式也作用
        if cfg.remove_thick > 0:
            mask = self._remove_thick_blobs(mask, cfg.remove_thick)

        return image, mask, response, sharpness, marking

    def detect(self, image_bgr: np.ndarray) -> tuple[list[Detection], np.ndarray]:
        image, mask, response, sharpness, marking = self._preprocess_to_mask(
            image_bgr
        )
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        detections = self._filter_contours(
            contours, response, marking, sharpness, mask
        )

        from .viz import annotate
        vis = annotate(image, detections)
        return detections, vis

    def compute_mask(
        self, image_bgr: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """回傳 (resize 後 BGR, 最終二值遮罩)。供遮罩影片輸出等用途。"""
        image, mask, _, _, _ = self._preprocess_to_mask(image_bgr)
        return image, mask

    def compute_response(
        self, image_bgr: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """回傳 (resize 後 BGR, 反應圖 response)。

        response 是「二值化前」的連續高頻反應（blackhat / frangi 脊狀 / 暗度），
        值越大代表高頻、越像裂縫——拿來套 colormap 就是熱力圖。沒有經過任何門檻。
        """
        image, _, response, _, _ = self._preprocess_to_mask(image_bgr)
        return image, response

    def debug_stages(self, image_bgr: np.ndarray) -> "dict[str, np.ndarray]":
        """回傳前處理各階段的中間影像（debug 用，與 detect 共用同一套邏輯）。

        key 為步驟名稱、value 為可直接顯示的影像（灰階圖會是單通道）。
        順序即 pipeline 順序，方便排成拼圖逐步檢視該調哪個參數。
        """
        cfg = self.config
        stages: dict[str, np.ndarray] = {}

        image = self._resize(image_bgr)
        stages["1.resize (BGR)"] = image

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        stages["2.grayscale"] = gray

        clahe = cv2.createCLAHE(
            clipLimit=cfg.clahe_clip_limit,
            tileGridSize=(cfg.clahe_grid_size, cfg.clahe_grid_size),
        )
        enhanced = clahe.apply(gray)
        stages[f"3.CLAHE clip={cfg.clahe_clip_limit}"] = enhanced

        bn = cfg.blur_kernel | 1
        if cfg.blur_method == "bilateral":
            blurred = cv2.bilateralFilter(
                enhanced, bn, cfg.bilateral_sigma, cfg.bilateral_sigma
            )
        elif cfg.blur_method == "median":
            blurred = cv2.medianBlur(enhanced, bn)
        else:
            blurred = cv2.GaussianBlur(enhanced, (bn, bn), 0)
        stages[f"4.denoise ({cfg.blur_method} k={bn})"] = blurred

        if cfg.method == "blackhat":
            mask, response = self._mask_blackhat(blurred)
            stages["5.blackhat response"] = self._normalize(response)
        elif cfg.method == "blackhat_hyst":
            mask, response = self._mask_blackhat_hyst(blurred)
            stages["5.blackhat response"] = self._normalize(response)
        elif cfg.method == "frangi":
            mask, response = self._mask_frangi(blurred)
            stages[f"5.frangi ridge {cfg.frangi_scales}"] = response
        else:
            mask, response = self._mask_adaptive(blurred)
            stages["5.darkness response"] = self._normalize(response)
        stages[f"6.binarize ({cfg.method})"] = mask.copy()

        # 邊緣銳利度圖（深度替身）
        sharpness = np.abs(cv2.Laplacian(blurred, cv2.CV_32F, ksize=3))
        stages["7.sharpness (Laplacian)"] = self._normalize(sharpness)

        # 標線禁區（若啟用）
        if cfg.exclude_markings:
            stages["8.marking zone"] = self._marking_mask(gray)

        # 二值化之後的遮罩後處理（post_morphology=False 時整段跳過、直接抓輪廓）
        if cfg.post_morphology:
            if cfg.suppress_texture:
                mask = self._suppress_texture(mask)
                stages[f"6b.line filter (len={cfg.line_length})"] = mask.copy()

            mask = self._remove_small_blobs(mask, cfg.min_blob_area)
            stages[f"9.remove blobs <{cfg.min_blob_area}px"] = mask.copy()

            ck = cfg.close_kernel | 1
            close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ck, ck))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k, iterations=1)
            stages[f"10.close (join, k={ck})"] = mask.copy()

        if cfg.remove_thick > 0:
            mask = self._remove_thick_blobs(mask, cfg.remove_thick)
            stages[f"10b.de-stain (open d={cfg.remove_thick | 1})"] = mask.copy()

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        marking = self._marking_mask(gray) if cfg.exclude_markings else None
        detections = self._filter_contours(
            contours, response, marking, sharpness, mask
        )

        # 所有候選輪廓（過濾前，灰）vs 通過過濾的裂縫（過濾後，黃）
        all_contours = image.copy()
        cv2.drawContours(all_contours, contours, -1, (160, 160, 160), 1)
        stages[f"11.all contours ({len(contours)})"] = all_contours

        from .viz import annotate
        stages[f"12.final cracks ({len(detections)})"] = annotate(image, detections)
        return stages

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        """把任意範圍的浮點/反應圖正規化成 0~255 的 uint8 灰階圖（供顯示）。"""
        out = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX)
        return out.astype(np.uint8)

    def _mask_adaptive(
        self, blurred: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """自適應二值化：逐區塊比周圍暗就標白。回傳 (遮罩, 暗度反應圖)。"""
        cfg = self.config
        block = cfg.adaptive_block | 1
        mask = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block, cfg.adaptive_c,
        )
        # 暗度反應 = 局部平均 − 像素值（越亮代表越比周圍暗，即越像裂縫）
        local_mean = cv2.boxFilter(
            blurred, ddepth=-1, ksize=(block, block), normalize=True
        )
        response = cv2.subtract(local_mean, blurred)
        return mask, response

    def _mask_blackhat(
        self, blurred: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """black-hat 形態學 + 相對門檻。回傳 (遮罩, black-hat 反應圖)。"""
        cfg = self.config
        bk = cfg.blackhat_kernel | 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bk, bk))
        blackhat = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, kernel)
        thresh_value = blackhat.mean() + cfg.thresh_k * blackhat.std()
        _, mask = cv2.threshold(blackhat, thresh_value, 255, cv2.THRESH_BINARY)
        return mask, blackhat

    def _mask_blackhat_hyst(
        self, blurred: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """black-hat + 雙門檻 hysteresis。回傳 (遮罩, black-hat 反應圖)。

        高門檻挑「確定是裂縫」的亮種子，低門檻當延伸；只保留含種子的連通塊，
        孤立的弱紋理（沒有任何亮種子）整塊丟掉。
        """
        cfg = self.config
        bk = cfg.blackhat_kernel | 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bk, bk))
        blackhat = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, kernel)
        mu, sd = blackhat.mean(), blackhat.std()
        high_t = mu + cfg.thresh_k_high * sd
        low_t = mu + cfg.thresh_k_low * sd
        strong = blackhat >= high_t
        weak = (blackhat >= low_t).astype(np.uint8)
        num, labels = cv2.connectedComponents(weak)
        keep = np.zeros(num, dtype=bool)
        keep[labels[strong]] = True   # 只保留「含亮種子」的連通塊
        keep[0] = False               # 背景
        mask = np.where(keep[labels], 255, 0).astype(np.uint8)
        return mask, blackhat

    def _frangi_response(self, blurred: np.ndarray) -> np.ndarray:
        """多尺度 Hessian 脊狀濾波（Frangi）：回傳 0~255 的「裂縫像線程度」反應圖。

        對暗裂縫（亮背景上的暗谷）而言，跨線方向的二階導為正，故最大特徵值
        mu2>0；只在此情況計分。挑大尺度 σ 可壓掉細掃紋、突顯較寬的連續裂縫。
        """
        cfg = self.config
        g = blurred.astype(np.float32)
        beta = cfg.frangi_beta
        out = np.zeros_like(g)
        for s in cfg.frangi_scales:
            sm = cv2.GaussianBlur(g, (0, 0), s)
            # 尺度正規化的 Hessian 二階導
            dxx = cv2.Sobel(sm, cv2.CV_32F, 2, 0, ksize=3) * (s * s)
            dyy = cv2.Sobel(sm, cv2.CV_32F, 0, 2, ksize=3) * (s * s)
            dxy = cv2.Sobel(sm, cv2.CV_32F, 1, 1, ksize=3) * (s * s)
            tmp = np.sqrt((dxx - dyy) ** 2 + 4.0 * dxy * dxy)
            l1 = 0.5 * (dxx + dyy + tmp)
            l2 = 0.5 * (dxx + dyy - tmp)
            # 依絕對值排序：|mu1| <= |mu2|
            swap = np.abs(l1) > np.abs(l2)
            mu1 = np.where(swap, l2, l1)
            mu2 = np.where(swap, l1, l2)
            rb = mu1 / (mu2 + 1e-9)          # 線狀度（線狀≈0、塊狀≈1）
            sness = np.sqrt(mu1 * mu1 + mu2 * mu2)  # 結構強度
            c = 0.5 * float(sness.max())
            v = np.exp(-(rb * rb) / (2 * beta * beta)) * (
                1.0 - np.exp(-(sness * sness) / (2 * c * c + 1e-9))
            )
            v[mu2 < 0] = 0.0                 # 只收暗裂縫（亮背景上的暗谷）
            out = np.maximum(out, v)
        return cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def _mask_frangi(
        self, blurred: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Frangi 脊狀濾波 + 相對門檻。回傳 (遮罩, 脊狀反應圖)。"""
        cfg = self.config
        ridge = self._frangi_response(blurred)
        thresh_value = ridge.mean() + cfg.thresh_k * ridge.std()
        _, mask = cv2.threshold(ridge, thresh_value, 255, cv2.THRESH_BINARY)
        return mask, ridge

    @staticmethod
    def _line_kernel(length: int, angle_deg: float) -> np.ndarray:
        """產生一條過中心、指定角度的線狀結構元素（length×length 內畫線）。"""
        size = max(3, length | 1)
        k = np.zeros((size, size), np.uint8)
        c = size // 2
        rad = np.deg2rad(angle_deg)
        dx, dy = np.cos(rad), np.sin(rad)
        p0 = (int(round(c - dx * c)), int(round(c - dy * c)))
        p1 = (int(round(c + dx * c)), int(round(c + dy * c)))
        cv2.line(k, p0, p1, 1, 1)
        return k

    def _suppress_texture(self, mask: np.ndarray) -> np.ndarray:
        """方向性線狀濾波（含形態學重建）：丟掉「沒有長直線段」的短紋路。

        步驟：
          1) 對每個角度做線狀開運算並聯集 → seed：只剩「夠長的直線段」像素。
          2) 以 seed 當種子做形態學重建——保留「與 seed 相連」的整個連通元件。
        如此一來彎曲的裂縫只要有一段夠直就整條留下，而完全找不到長直線段的
        掃紋/拉毛紋短紋路會被整塊清掉，避免把裂縫切碎誤刪。
        """
        cfg = self.config
        seed = np.zeros_like(mask)
        for i in range(max(1, cfg.line_angles)):
            angle = 180.0 * i / max(1, cfg.line_angles)
            kernel = self._line_kernel(cfg.line_length, angle)
            opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            seed = cv2.bitwise_or(seed, opened)

        # 形態學重建：保留「含有 seed 像素」的連通元件（整條保留，含彎曲處）
        num, labels, _, _ = cv2.connectedComponentsWithStats(mask, 8)
        keep = np.unique(labels[seed > 0])
        keep = keep[keep != 0]  # 去掉背景標籤
        return np.where(np.isin(labels, keep), 255, 0).astype(np.uint8)

    @staticmethod
    def _remove_thick_blobs(mask: np.ndarray, diameter: int) -> np.ndarray:
        """磨掉「比裂縫寬的填實暗塊」（深色髒污），保留細裂縫與龜裂網路。

        用直徑 diameter 的圓做開運算：圓塞得進的（夠寬的胖塊）會留在 opened，
        塞不進的細線則消失。把 opened 從原遮罩減掉 → 只留細的線狀前景。
        diameter 要大於最寬的真裂縫，否則會把較寬的裂縫也一起磨掉。
        """
        d = max(3, diameter | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))
        thick = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return cv2.subtract(mask, thick)

    @staticmethod
    def _remove_small_blobs(mask: np.ndarray, min_area: int) -> np.ndarray:
        """移除面積小於 min_area 的連通元件，保留細線寬度不變。"""
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        out = np.zeros_like(mask)
        for i in range(1, num):  # 0 是背景
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                out[labels == i] = 255
        return out

    @staticmethod
    def _overlaps_marking(
        contour: np.ndarray, marking: np.ndarray, max_overlap: float
    ) -> bool:
        """輪廓與標線禁區的重疊比例是否超過 max_overlap。"""
        comp = np.zeros(marking.shape, np.uint8)
        cv2.drawContours(comp, [contour], -1, 255, thickness=cv2.FILLED)
        n = int(cv2.countNonZero(comp))
        if n == 0:
            return False
        inter = int(cv2.countNonZero(cv2.bitwise_and(comp, marking)))
        return inter / n > max_overlap

    def _marking_mask(self, gray: np.ndarray) -> np.ndarray:
        """找出亮色路面標線（白/黃漆）並向外擴張成禁區。

        門檻取「相對(mean+k*std)」與「絕對亮度下限」兩者較大者——確保只有
        夠白的漆才算標線；避免在偏亮的水泥路上把整片亮水泥誤判成標線。
        """
        cfg = self.config
        thr = max(
            float(cfg.marking_min_bright),
            gray.mean() + cfg.marking_bright_k * gray.std(),
        )
        _, bright = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
        dk = cfg.marking_dilate | 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dk, dk))
        return cv2.dilate(bright, kernel, iterations=1)

    @staticmethod
    def _mean_response(contour: np.ndarray, response: np.ndarray) -> float:
        """計算輪廓內部的平均暗度反應（衡量暗線對比強度）。"""
        comp_mask = np.zeros(response.shape, np.uint8)
        cv2.drawContours(comp_mask, [contour], -1, 255, thickness=cv2.FILLED)
        return float(cv2.mean(response, mask=comp_mask)[0])

    @staticmethod
    def _fill_ratio(contour: np.ndarray, mask: np.ndarray) -> float:
        """外輪廓填實後，其中真正屬於遮罩（裂縫）的像素占比。

        用來分辨「網狀裂縫」與「實心暗塊」：實心髒污/坑洞占比≈1；網狀裂縫因
        中間都是完好水泥島、只有細線是前景，占比低（約 0.06~0.5）。
        """
        comp = np.zeros(mask.shape, np.uint8)
        cv2.drawContours(comp, [contour], -1, 255, thickness=cv2.FILLED)
        region = int(cv2.countNonZero(comp))
        if region == 0:
            return 1.0
        crack = int(cv2.countNonZero(cv2.bitwise_and(comp, mask)))
        return crack / region

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
        self, contours: list[np.ndarray], response: np.ndarray,
        marking: np.ndarray | None = None,
        sharpness: np.ndarray | None = None,
        mask: np.ndarray | None = None,
    ) -> list[Detection]:
        cfg = self.config
        results: list[Detection] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < cfg.min_area:
                continue

            # 長度用最小旋轉矩形長邊；寬度改用「面積÷長度＝平均寬度」。
            # 因為斜向/彎曲的裂縫，旋轉矩形短邊會鬆鬆框住而虛胖，
            # 平均寬度才反映裂縫真正的纖細程度。
            long_side = max(cv2.minAreaRect(contour)[1])
            long_side = max(1.0, long_side)
            mean_width = area / long_side
            elongation = long_side / max(1.0, mean_width)

            # 形狀關：符合「條狀」或「網狀」任一種即通過，否則視為髒污/紋理捨棄。
            #   條狀：夠長 + 夠細長 + 夠細（單一線條）
            #   網狀：夠大 + 填充率落在網狀區間（大片龜裂，非實心暗塊）
            is_line = (
                long_side >= cfg.min_length
                and elongation >= cfg.min_elongation
                and mean_width <= cfg.max_width
            )
            is_mesh = False
            if cfg.detect_mesh and mask is not None and area >= cfg.mesh_min_area:
                fill = self._fill_ratio(contour, mask)
                is_mesh = cfg.mesh_min_fill <= fill <= cfg.mesh_max_fill
            if not (is_line or is_mesh):
                continue
            label = "crack" if is_line else "mesh"

            # 落在標線禁區內 → 多半是車道標線的邊緣/縫隙，捨棄
            if marking is not None and self._overlaps_marking(
                contour, marking, cfg.marking_overlap
            ):
                continue

            # 綜合信心分數（0~1）：三項線索取平均
            #   細長：越長條越像裂縫
            #   細  ：越細越像裂縫（胖的偏髒污）
            #   對比：暗度反應越強 → 像銳利暗線，而非模糊髒污
            contrast = self._mean_response(contour, response)
            # 不夠暗 → 多半是路面標線（比周圍亮）或淡髒污，捨棄
            if contrast < cfg.min_contrast:
                continue

            # 邊緣不夠銳利（暗谷不夠陡）→ 多半是平的污漬而非有深度的裂縫，捨棄
            if sharpness is not None and cfg.min_sharpness > 0:
                sharp = self._mean_response(contour, sharpness)
                if sharp < cfg.min_sharpness:
                    continue

            contrast_term = float(np.clip(contrast / cfg.contrast_norm, 0.0, 1.0))
            if is_line:
                elong_term = float(np.clip(elongation / 8.0, 0.0, 1.0))
                thin_term = float(
                    np.clip(1.0 - mean_width / cfg.max_width, 0.0, 1.0)
                )
                score = (elong_term + thin_term + contrast_term) / 3.0
                if score < cfg.min_score:
                    continue
            else:
                # 網狀：細長/細的形狀項對整片網路沒意義，信心以對比為準
                score = contrast_term

            x, y, bw, bh = cv2.boundingRect(contour)
            results.append(
                Detection(
                    bbox=(x, y, bw, bh),
                    area=area,
                    contour=contour,
                    score=score,
                    label=label,
                )
            )

        results.sort(key=lambda d: d.area, reverse=True)
        return results
