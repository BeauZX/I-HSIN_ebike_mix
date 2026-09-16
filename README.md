# 路面辨識整合系統

把桌面上四個專案整合成**一個程序、一顆鏡頭、一顆 Hailo-8**：

| 來源專案 | 在這裡的角色 | 模型 |
|---|---|---|
| `road_classification` | 先判路面種類，決定用哪個分級模型 | `resnet_v1_34.hef`（Asphalt / Belgian Block / Concrete / Forest） |
| `asphalt` | 瀝青路：3×5 網格劣化分級（紅/黃/綠） | `asphalt_cls.hef` |
| `cement` | 水泥路：3×5 網格分級 × 影像法裂縫定位 融合 | `cement_cls.hef` + CPU 裂縫偵測 |
| `OverlayView` | 人車偵測 + Kalman 追蹤 + 警戒區警報 | `yolov8n.hef` |

流程：ResNet 判成 **Asphalt Road** → 跑 asphalt 分級；**Concrete road** → 跑 cement 分級；
Belgian Block / Forest Road 本來就不平整 → 只顯示種類、不畫網格。
人車偵測與警戒區同時跑在整張畫面上。全部疊在一個 1280×720 畫面，即時顯示並每分鐘存一段。

硬體：Raspberry Pi 5 + Hailo-8 AI HAT + IMX219 Stereo Camera（兩顆朝同一方向、倒裝；目前只用 cam0）。

---

## 執行

```bash
cd /home/beau/Desktop/System_Integration

uv run main.py                 # 沿用上次框的路面 ROI 與警戒區，開視窗、錄影
uv run main.py --ui            # 重新框選路面 ROI 與警戒區（鏡頭位置動過、或改了 rotation 時）
uv run main.py --no-save       # 只顯示不錄影
```

視窗操作：

| 操作 | 作用 |
|---|---|
| `q` 或 Ctrl+C | 結束（會正常收檔） |
| 滑鼠左鍵拖曳 | 執行中重設警戒區，放開即存到 `presets/warning_zone.json`，下次自動沿用 |
| `r` | 重設人車追蹤 |

第一次執行（或加 `--ui`）會依序跳出兩個框選視窗：先拖出路面範圍（藍框＋十字線）、再拖出人車警戒區（黃框），各按 Enter 確認
（按 C 取消：路面改用全畫面、警戒區沿用目前的，都不存檔）。
結果以 0~1 比例分別存在 `presets/roi.json`（三個路面模型共用）與 `presets/warning_zone.json`，之後不用再框。

---

## 設定檔：一個專案一個檔

都在 [`configs/`](configs/)，改完存檔重新執行即生效。「這次怎麼跑」用命令列旗標，「跑起來用什麼參數」改 yaml。

| 檔案 | 對應專案 | 主要內容 |
|---|---|---|
| [`camera.yaml`](configs/camera.yaml) | — | 鏡頭編號、解析度、fps、`rotation`、曝光組合（`exposure` / `awb` / `denoise`，沿用 asphalt 實測值） |
| [`road_type.yaml`](configs/road_type.yaml) | road_classification | ResNet 模型、`grading`（類別 → 分級模式）、`smooth_window`、`confirm_count`（切換遲滯）、共用 ROI 檔 |
| [`asphalt.yaml`](configs/asphalt.yaml) | asphalt | 模型、類別名、網格、EMA 平滑與遲滯 |
| [`cement.yaml`](configs/cement.yaml) | cement | 模型、類別名、網格、裂縫參數檔、融合權重、PID / EMA 平滑 |
| [`detect.yaml`](configs/detect.yaml) | OverlayView | YOLO 模型、信心門檻、保留類別、追蹤器、警戒區檔與預設 |
| [`output.yaml`](configs/output.yaml) | — | 輸出資料夾、分段秒數、錄影 fps、視窗標題 |

### 路面種類切換的遲滯

`road_type.yaml` 的 `confirm_count`（預設 3）：新的路面種類要連續 N 次分析都勝出才切換分級模型。
沒有它，在瀝青/水泥邊界或 ResNet 機率接近時會每輪來回切換，網格顏色閃爍、
兩個分級器的時序平滑一直被重置。代價是真正換路面時晚 N 次分析（約 0.1～0.5 秒）才反應。設 1 = 不遲滯。

---

## 輸出

`outputs/20260915_160140.mp4`（該段開始時間命名，每 60 秒一段，可在 `output.yaml` 改）。
錄的就是螢幕上看到的合併畫面。依牆上時鐘補幀/丟幀，處理變慢時播放速度仍與真實時間一致。

---

## 程式結構

```
main.py                 進入點：讀設定、載模型、開鏡頭、主迴圈（疊圖 / 顯示 / 錄影）
configs/                各專案的設定檔
models/                 四個 .hef；resnet/ 另含 config.json、classes.txt
presets/                rough.json / smooth.json / pid.json（複製自 cement）
                        roi.json / warning_zone.json（程式自動產生）
src/
  settings.py           讀 configs/*.yaml 並驗證
  hailo.py              共用 VDevice（scheduler）+ HailoModel 包裝（run_async 多張並排）
  camera.py             Picamera2：main 720p BGR + lores 640×640 RGB（給 YOLO），ISP 翻轉與曝光控制
  roi.py                路面 ROI 框選與存讀（自 asphalt）
  road_type.py          ResNet 前處理（自 road_classification）+ 機率平滑 + 連續確認切換
  analyzer.py           路面分析緒：ResNet → asphalt / cement 分級
  detect.py             YOLO NMS 解析、Kalman、追蹤、警戒區（可存檔）、警報、偵測緒、繪製（自 OverlayView）
  recorder.py           分段錄影（自 OverlayView，時間戳命名）
  draw.py               OpenCV 5 相容的描邊文字
  grading/
    graders.py          AsphaltGrader / CementGrader 統一介面 + GridClassifier
    grid.py             網格切分、上色、EMA 平滑（自 asphalt）
    yolo_grid.py, fusion.py, overlay.py, presets.py, pothole/   （自 cement）
```

執行緒：主緒讀相機、疊圖、顯示、錄影，維持相機幀率、從不等 Hailo；
路面分析緒與偵測緒各自只處理「最新的一幀」，來不及就丟，主緒拿最近一次結果填補。

---

## 實測（Pi 5 8GB + Hailo-8，1280×720 @ 25 fps）

| 項目 | 數值 |
|---|---|
| 顯示 / 錄影幀率 | 25 fps |
| ResNet 單次 | ~4 ms |
| asphalt 15 格（Hailo） | ~28 ms（含前處理 ~40 ms/輪） |
| cement 15 格 + CPU 裂縫偵測 | ~95 ms/輪 |
| yolov8n | ~7 ms（含追蹤 ~10 ms/輪） |

四個 HEF 放在同一個 VDevice，由 HailoRT scheduler 輪流排程。

---

## 環境

```bash
sudo apt install hailo-all python3-picamera2     # hailo_platform 與 picamera2 都是系統套件
uv venv --system-site-packages
uv sync
```

**`--system-site-packages` 不能省**：`hailo_platform` 與 `picamera2` 不在 PyPI，venv 要加這個旗標才看得到。
`pyproject.toml` 已設 `python-preference = "only-system"`，uv 一定用系統的 `/usr/bin/python3.13`。
之後若刪掉 `.venv` 重建，記得先 `uv venv --system-site-packages` 再 `uv sync`。

只跑 Hailo，不裝 torch / ultralytics（原專案的 `.pt` CPU 備援不納入）。

---

## 與原專案的差異（整合時必要的改動）

- **Hailo 後端**：asphalt / cement 原本的 `InferVStreams` + `activate()` 會獨佔裝置、與 scheduler 不相容，
  改寫成 `InferModel` API（前處理一字不改）。用 `run_async` 一次排 15 格，實測 28 ms，比原本的 34 ms 快。
- **鏡頭**：asphalt / cement 的 `rpicam-vid` pipe 改為 Picamera2；曝光組合以 libcamera control 對應保留。
  OverlayView 的 `cv2.flip(-1)` 改由 ISP `rotation` 處理。
- **網格**：cement 由 4×10 統一為 3×5。
- **警戒區**：OverlayView 原本只在記憶體裡，改為存 json 自動沿用。
- **描邊文字**：OpenCV 5.0 的 `putText` 字距隨 thickness 改變，原專案「粗黑字 + 細白字」的描邊會錯開成兩層；
  這裡改用同 thickness 偏移描邊（`src/draw.py`）。**原四個專案在 OpenCV 5 上也有同樣現象**，未動。
- 一台 Hailo-8 同時只能被一個程序開啟；若原專案有程式在跑，這裡會出現 `HAILO_OUT_OF_PHYSICAL_DEVICES`。

原四個專案與 `imx_video/` 未做任何修改。
