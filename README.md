# 路面辨識整合系統

把桌面上五個專案整合成**一個程序、一顆鏡頭、一顆 Hailo-8**：

| 來源專案 | 在這裡的角色 | 模型 |
|---|---|---|
| `road_classification` | 先判路面種類，決定用哪個分級模型 | `resnet_v1_34.hef`（Asphalt / Belgian Block / Concrete / Forest） |
| `asphalt` | 瀝青路：3×5 網格劣化分級（紅/黃/綠） | `asphalt_cls.hef` |
| `cement` | 水泥路：3×5 網格分級 × 影像法裂縫定位 融合 | `cement_cls.hef` + CPU 裂縫偵測 |
| `OverlayView` | 人車偵測 + Kalman 追蹤 + 警戒區警報 | `yolov8n.hef` |
| `car_door` | 車門開啟 / 關閉偵測 | `car_door_yolov11m.hef`（closed / open） |

流程：ResNet 判成 **Asphalt Road** → 跑 asphalt 分級；**Concrete road** → 跑 cement 分級；
Belgian Block / Forest Road 本來就不平整 → 只顯示種類、不畫網格。
人車偵測與警戒區、車門偵測同時跑在整張畫面上（車門開啟紅框、關閉綠框，狀態顯示在左上角狀態列，不發警報）。全部疊在一個 1280×960 畫面（相機以原生 3280×2464 全幅擷取，最高約 21 fps；路面 ROI 從原圖裁切分析，顯示與錄影再縮小），即時顯示並每分鐘存一段。

另外依上述辨識結果自動控制避震器鎖緊/放鬆（沒有實體按鈕，`src/motor.py` + `src/motor_policy.py`，
邏輯移植自 Arduino `0709_3btn_edge.ino`，細節見「避震器馬達控制」一節）。

硬體：Raspberry Pi 5 + Hailo-8 AI HAT + IMX219 Stereo Camera（兩顆朝同一方向、倒裝；目前只用 cam0）
+ 避震器調整馬達（M+/M- 兩條控制線、1 條編碼器脈衝回饋，接 Pi 5 的 GPIO）。

---

## 執行

```bash
cd /home/beau/Desktop/System_Integration

uv run main.py                 # 沿用上次框的路面 ROI 與警戒區，開視窗、錄影
uv run main.py --ui            # 重新框選路面 ROI 與警戒區（鏡頭位置動過、或改了 rotation 時）
uv run main.py --no-save       # 只顯示不錄影
uv run main.py --no-display    # 只錄影不顯示即時畫面（省顯示開銷；框選視窗照常；Ctrl+C 結束）
uv run main.py --metrics       # 同時取樣功耗 / 頻率 / 降頻 / 各階段耗時（見下方「量測功耗與降頻」）
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
| [`camera.yaml`](configs/camera.yaml) | — | 鏡頭編號、擷取解析度（路面分析用；顯示/錄影尺寸在 `output.yaml`）、fps、`rotation`、曝光組合（`exposure` / `awb` / `denoise`，沿用 asphalt 實測值） |
| [`road_type.yaml`](configs/road_type.yaml) | road_classification | ResNet 模型、`grading`（類別 → 分級模式）、`smooth_window`、`confirm_count`（切換遲滯）、共用 ROI 檔 |
| [`asphalt.yaml`](configs/asphalt.yaml) | asphalt | 模型、類別名、網格、EMA 平滑與遲滯 |
| [`cement.yaml`](configs/cement.yaml) | cement | 模型、類別名、網格、裂縫參數檔、融合權重、PID / EMA 平滑 |
| [`detect.yaml`](configs/detect.yaml) | OverlayView | YOLO 模型、信心門檻、保留類別、追蹤器、警戒區檔與預設 |
| [`door.yaml`](configs/door.yaml) | car_door | 車門模型、分數門檻、類別名稱順序、哪些類別算開啟 |
| [`motor.yaml`](configs/motor.yaml) | — | 避震器馬達：GPIO 腳位、三個目標位置、堵轉偵測門檻、坑洞代理門檻、防抖動（confirm_count / 最短切換間隔） |
| [`output.yaml`](configs/output.yaml) | — | 顯示與錄影畫面尺寸、輸出資料夾、分段秒數、錄影 fps、視窗標題、偵測結果 log 資料夾與防閃動秒數 |
| [`metrics.yaml`](configs/metrics.yaml) | — | 指標取樣開關、間隔、輸出資料夾、Hailo / 相機 / 風扇的估算參數與 DC-DC 效率 |

### 路面種類切換的遲滯

`road_type.yaml` 的 `confirm_count`（預設 3）：新的路面種類要連續 N 次分析都勝出才切換分級模型。
沒有它，在瀝青/水泥邊界或 ResNet 機率接近時會每輪來回切換，網格顏色閃爍、
兩個分級器的時序平滑一直被重置。代價是真正換路面時晚 N 次分析（約 0.1～0.5 秒）才反應。設 1 = 不遲滯。

---

## 避震器馬達控制

沒有實體按鈕，`main.py` 主迴圈每幀依當下的路面/坑洞/人員/車門辨識結果算出目標位置（`src/motor_policy.py`
的 `decide_target()`），目標改變時才呼叫 `MotorController.move_to()`（`src/motor.py`）。
沒有加權優先序，依序覆蓋、最後一條規則贏：

1. **路面種類定基準**：`Asphalt Road` / `Concrete road` → 鎖緊（`tight`）；`Forest Road` → 全鬆（`loose`）；
   `Belgian Block` → 中間（`mid`）
2. **坑洞覆蓋** → 全鬆：目前畫面 3×5 網格 `severe` 格數達 `motor.yaml` 的 `pothole_severe_cell_threshold`
3. **安全覆蓋（永遠贏）** → 鎖緊：畫面裡偵測到 `person`，或車門狀態為 `open`

馬達控制邏輯（雙模式堵轉偵測 + 動態歸零）移植自 Arduino `0709_3btn_edge.ino`，用 `lgpio`
驅動（Pi 5 的 RP1 晶片，`RPi.GPIO` 不支援）。跟 Arduino 版不同：全緊、全鬆兩端都是實測確認的機構死點，
目標是兩端時不看計數、一律轉到堵轉才停並把位置校正回該端（計數有約 10% 漂移，靠計數停會鎖不緊）；
中間位置照計數停。行程實測約 197 個脈衝、3 秒；校正時 log 會印出這一趟的計數漂移
（例如 `位置自動校正：-8 → 0（漂移 -8）`），細節見 `MOTOR_INTEGRATION.md`。

**位置持久化**：每次移動完成後把目前 pulse 位置存進 `motor.yaml` 的 `position_store`
（預設 `presets/motor_position.json`，atomic write），開機時讀回來，不像 Arduino 版每次開機都假設全緊
——樹莓派有真正的檔案系統，不需要 ESP32 那種 NVS。讀不到/壞掉/超出範圍時才退回 `pos_tight`。

**防抖動（避免頻繁切換磨損齒輪）**：`decide_target()` 是每幀重算的瞬時決策，路面標籤、坑洞格數、
人員偵測在臨界點附近都可能一幀一幀跳。兩層保護（都在 `motor.yaml`）：

- `confirm_count`（預設 5）：同一個目標要連續這麼多次決策都一樣才算確認（`src/motor_policy.py` 的
  `TargetDebouncer`，仿 `road_type.yaml` 的 `confirm_count`，同一手法防同一類問題）
- `min_switch_interval_sec`（預設 3 秒）：就算目標確認變了，離上次真的送出切換命令沒過這麼久也先不送
  （`main.py` 主迴圈裡實作，仿 `detect.yaml` 的 `alert_cooldown_sec`），這是切換頻率的硬上限

**已知的訊號落差**（跟原本 Arduino 版按鈕觸發不同，這裡是自動決策，兩個規則用的是代理訊號，不是真正對應的偵測）：

- **坑洞**：這個系統沒有接真正的坑洞偵測（`src/grading/pothole/detector.py` 的 `PotholeDetector`
  是死程式碼，沒被接進 `CementGrader`/`AsphaltGrader` 的流程），目前用「severe 格數」代理，門檻在
  `motor.yaml` 調整
- **人員移動**：`TrackView`（`src/detect.py`）沒有速度欄位，只有座標歷史，目前簡化成「畫面裡有 person
  類別就觸發」，不判斷是否真的在動

**時間精度**：ESP32 是專用微控制器，中斷延遲微秒等級且穩定；這裡跑 Linux + Python，`lgpio` 的回呼
是一批一批送進來的，所以脈衝間隔（濾波、堵轉基準）一律用回呼帶的核心時間戳記 tick，不用回呼被呼叫的時間
（後者實測會把約三分之一的真脈衝當雜訊丟掉）。堵轉判斷的「多久沒脈衝」仍受系統排程與 Hailo 負載影響，抖動比
ESP32 大很多。`motor.yaml` 裡跟 ESP32 版本抄過來的堵轉偵測門檻（`stall_ratio_*`、
`min_stall_floor_us`、`start_timeout_ms`、`stall_timeout_ms`）幾乎確定需要在 Pi 上重新實測調整。

---

## 輸出

`outputs/20260915_160140.mp4`（該段開始時間命名，每 60 秒一段，可在 `output.yaml` 改）。
錄的就是螢幕上看到的合併畫面。依牆上時鐘補幀/丟幀，處理變慢時播放速度仍與真實時間一致。

### 偵測結果 log

`outputs/logs/20260915_160140.jsonl`：跟同名的 mp4 對應，同時切檔（`--no-save` 時照記，自己每 60 秒切一檔）。
JSON Lines，每行一筆，**只在狀態改變時記**：

| 欄位 | 內容 |
|---|---|
| `time` / `offset_sec` | 改變開始的時間 / 距該段影片開頭幾秒（可直接拿去影片裡找） |
| `event` | `segment_start`（每檔第一行，記當下完整狀態）或 `change` |
| `changed` | 這次改變的欄位 |
| `road` / `road_conf` | 路面種類 / 寫入當下的信心（信心變動不觸發紀錄） |
| `grading` | `asphalt` / `cement` / `none` |
| `grid` | 網格各等級格數 `{"severe", "slight", "smooth"}`；不分級的路面為 `{}` |
| `objects` | 各類人車數量（`detect.yaml` 的 classes） |
| `door` | 車門 `open` / `closed` / `none` |
| `motor` | 避震器目標位置 `tight` / `mid` / `loose`（記目標名稱，不是忙碌狀態，見「避震器馬達控制」） |

防閃動：新值要穩定維持 `output.yaml` 的 `log_stable_sec`（預設 0.5 秒）才算改變，
偵測漏抓一兩幀不會被記；`time` 記的是新值開始出現的時間。

```bash
# 例：列出這段影片裡車門開啟的時間點
grep '"door": "open"' outputs/logs/20260923_161122.jsonl | grep '"changed": \[[^]]*door'
```

---

## 量測功耗與降頻

`uv run main.py --metrics`（或把 `metrics.yaml` 的 `enabled` 設 true）會在同一個程序裡每秒取樣一次，
寫到 `outputs/metrics/<開始時間>.csv`，結束時印摘要並存 `<開始時間>_summary.txt`。
重新產摘要：`uv run python -m src.metrics outputs/metrics/<檔名>.csv`（估算欄位會依當時的 `metrics.yaml` 重算，改參數不用重新量）。

哪些是實測、哪些是估算（沒有外接功率計時的極限）：

| | 來源 | 說明 |
|---|---|---|
| **實測** Pi 5 板上功耗 | `vcgencmd pmic_read_adc` | 12 條電源軌各自的電流 × 電壓，加總為 `board_w` |
| **實測** ARM 頻率 | cpufreq `time_in_state` | 每秒內各檔停留時間 → 時間加權平均、2.4 GHz 佔比 |
| **實測** 降頻 / 低電壓 | `vcgencmd get_throttled` + SoC 溫度 + 風扇 PWM | 摘要會說執行期間有沒有出現旗標 |
| **實測** Hailo 使用率 | `hailortcli monitor` | 程式自動設 `HAILO_MONITOR=1`，裝置與四個模型各自的使用率 / fps |
| **實測** Hailo 晶片溫度 | HailoRT `get_chip_temperature()` | 只能在持有裝置的程序內讀 |
| **估算** Hailo / 相機 / 風扇瓦數 | `metrics.yaml` 的 `estimate` | 這塊 AI HAT 沒有電流感測器（`hailortcli measure-power` 不支援），Hailo 依使用率在待機 / 滿載之間線性內插，風扇依 PWM 折算 |
| **估算** 整套 | `(board_w + Hailo + 相機 + 風扇) / efficiency` | `est_total_w`；報告時請註明估算部分與參數 |

另外每列也記 CPU 各核使用率、記憶體、顯示 fps、路面模式、各階段耗時
（ResNet / 分級 / 裂縫偵測 / YOLO / 車門推論、主緒疊圖 / 寫檔 / 顯示），方便把功耗曲線對到程式狀態。
要看穩態，建議至少跑 10～15 分鐘讓溫度穩定。

---

## 程式結構

```
main.py                 進入點：讀設定、載模型、開鏡頭、主迴圈（疊圖 / 顯示 / 錄影）
configs/                各專案的設定檔
models/                 五個 .hef；resnet/ 另含 config.json、classes.txt
presets/                rough.json / smooth.json / pid.json（複製自 cement）
                        roi.json / warning_zone.json / motor_position.json（程式自動產生）
tests/
  simulate_motor.py         不需要真實硬體的馬達模擬測試（假 lgpio + 背景執行緒模擬轉動），
                            改動 src/motor.py 或 src/motor_policy.py 後先跑這個當回歸測試
  manual_jog.py             寸動工具：一次轉一小段，把避震器轉回全緊、重設位置存檔、實測 pos_loose
  manual_gpio_test.py       階段1：空轉測試，鍵盤手動控制正反轉 + 看脈衝數，不掛避震器
  manual_motor_control.py   階段2：裝上避震器後，鍵盤手動觸發 move_to()，測真正的堵轉偵測/
                            動態歸零/位置持久化，還不接 CV 自動決策
                            （階段3就是直接 uv run main.py，接上自動辨識，不用另外寫腳本）
src/
  settings.py           讀 configs/*.yaml 並驗證
  hailo.py              共用 VDevice（scheduler）+ HailoModel 包裝（run_async 多張並排）
  camera.py             Picamera2：main 3280×2464 BGR（只複製路面 ROI）+ lores 1280×960 BGR（顯示/錄影，再縮成 640×640 RGB 給 YOLO），ISP 翻轉與曝光控制
  roi.py                路面 ROI 框選與存讀（自 asphalt）
  road_type.py          ResNet 前處理（自 road_classification）+ 機率平滑 + 連續確認切換
  analyzer.py           路面分析緒：ResNet → asphalt / cement 分級
  detect.py             YOLO NMS 解析、Kalman、追蹤、警戒區（可存檔）、警報、偵測緒、繪製（自 OverlayView）
  door.py               車門 YOLOv11m NMS 解析、車門緒、繪製（自 car_door）
  motor.py              避震器馬達控制：lgpio 驅動 + 堵轉偵測 + 動態歸零（改寫自 Arduino 0709_3btn_edge.ino）
  motor_policy.py        依路面/坑洞/人員/車門結果決定馬達目標位置（decide_target()）
  recorder.py           分段錄影（自 OverlayView，時間戳命名）
  event_log.py          偵測結果 log（狀態改變才記、防閃動、跟錄影同名切檔）
  draw.py               OpenCV 5 相容的描邊文字
  grading/
    graders.py          AsphaltGrader / CementGrader 統一介面 + GridClassifier
    grid.py             網格切分、上色、EMA 平滑（自 asphalt）
    yolo_grid.py, fusion.py, overlay.py, presets.py, pothole/   （自 cement）
```

執行緒：主緒讀相機、疊圖、顯示、錄影，維持相機幀率、從不等 Hailo；
路面分析緒、偵測緒與車門緒各自只處理「最新的一幀」，來不及就丟，主緒拿最近一次結果填補。

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

### 加入車門偵測後（2026-09-23 實測）

`car_door_yolov11m.hef` 是 multi-context（4 段），單獨跑 38.6 ms/幀。與其他模型共用 Hailo 時，
round-robin scheduler 會平分晶片時間，其他分析緒的更新頻率明顯下降（各緒不限速、同時全速跑的量測）：

| | 不加車門 | 車門全速（目前設定） | 車門限 10 次/秒 | 車門限 5 次/秒 |
|---|---|---|---|---|
| 人車偵測 yolov8n | 32.6 次/秒 | 14.1 | 19.4 | 25.9 |
| 路面分析（ResNet + asphalt 15 格） | 10.9 次/秒 | 4.7 | 6.5 | 8.7 |
| 車門偵測 | — | 14.1 | 10 | 5 |

實機（asphalt 模式、錄影、`--no-display`）：顯示 / 錄影 24.9 fps 不受影響，人車與車門各約 13 次/秒，
路面分析約 4.6 次/秒（每輪 ~207 ms），Hailo 使用率 ~93%。

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

**馬達控制需要 `lgpio`**，走 apt 的系統套件 `python3-lgpio`（Raspberry Pi OS 預設已裝；沒有的話
`sudo apt install python3-lgpio`），跟 `hailo_platform`/`picamera2` 一樣靠 `--system-site-packages` 讀到。
不列進 `pyproject.toml`：PyPI 上的 lgpio 只有原始碼，沒有 swig 會編譯失敗，連帶 `uv run` 同步也會失敗。

第一次接上馬達前，建議先跑 `uv run tests/simulate_motor.py`（不需要真實硬體）確認邏輯本身沒問題，
再上機測 `lgpio` 的實際腳位/中斷是否正常——這部分模擬測試沒辦法取代，細節見 `MOTOR_INTEGRATION.md`。

---

## 與原專案的差異（整合時必要的改動）

- **Hailo 後端**：asphalt / cement 原本的 `InferVStreams` + `activate()` 會獨佔裝置、與 scheduler 不相容，
  改寫成 `InferModel` API（前處理一字不改）。用 `run_async` 一次排 15 格，實測 28 ms，比原本的 34 ms 快。
- **鏡頭**：asphalt / cement 的 `rpicam-vid` pipe 改為 Picamera2；曝光組合以 libcamera control 對應保留。
  OverlayView 的 `cv2.flip(-1)` 改由 ISP `rotation` 處理。
- **網格**：cement 由 4×10 統一為 3×5。
- **警戒區**：OverlayView 原本只在記憶體裡，改為存 json 自動沿用。
- **車門偵測**：car_door 原本自己開 VDevice、從 BGR 畫面 letterbox 到 640×640；這裡改走共用 VDevice，輸入直接共用人車偵測的 640×640（4:3 全幅畫面拉伸成正方形，與原專案的 letterbox 比例不同）。類別順序 `[closed, open]` 沿用原專案的假設、**尚未驗證**，見 `door.yaml`。
- **描邊文字**：OpenCV 5.0 的 `putText` 字距隨 thickness 改變，原專案「粗黑字 + 細白字」的描邊會錯開成兩層；
  這裡改用同 thickness 偏移描邊（`src/draw.py`）。**原四個專案在 OpenCV 5 上也有同樣現象**，未動。
- 一台 Hailo-8 同時只能被一個程序開啟；若原專案有程式在跑，這裡會出現 `HAILO_OUT_OF_PHYSICAL_DEVICES`。

原五個專案與 `imx_video/` 未做任何修改。
