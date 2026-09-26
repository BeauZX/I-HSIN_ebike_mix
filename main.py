#!/usr/bin/env python3
"""路面辨識整合系統 — 進入點。

整合五個專案，一顆 CSI 鏡頭、一個程序、一顆 Hailo-8：
    road_classification  ResNet34 判路面種類 → 決定用哪個分級模型
    asphalt              瀝青路：3×5 網格劣化分級
    cement               水泥路：3×5 網格分級 × 影像法裂縫定位 融合
    OverlayView          人車偵測 + Kalman 追蹤 + 警戒區警報
    car_door             車門開啟 / 關閉偵測

另外依上述辨識結果自動控制避震器鎖緊/放鬆（沒有實體按鈕，見 src/motor.py、src/motor_policy.py）。

執行：
    uv run main.py                 沿用上次框的路面 ROI 與警戒區，開視窗、錄影
    uv run main.py --ui            重新框選路面 ROI 與警戒區（鏡頭位置動過時）
    uv run main.py --no-save       只顯示不錄影
    uv run main.py --no-display    只錄影不顯示即時畫面（框選視窗照常；Ctrl+C 結束）
    uv run main.py --metrics       同時取樣功耗 / 頻率 / 降頻 / 各階段耗時（見 src/metrics.py，configs/metrics.yaml）

偵測結果（路面種類、網格等級、人車數量、車門）只在狀態改變時記到 outputs/logs/*.jsonl（見 src/event_log.py）。

視窗操作：q 結束、滑鼠左鍵拖曳 = 設定警戒區（放開即存檔）、r 重設追蹤。
參數在 configs/ 底下，一個專案一個檔（camera / road_type / asphalt / cement / detect / door / output）。

執行緒：
    主緒        讀相機 → 疊圖 → 顯示 → 錄影（維持相機幀率，從不等 Hailo）
    路面分析緒  ResNet → asphalt 或 cement 分級（見 src/analyzer.py）
    偵測緒      YOLO → 追蹤 → 警戒判斷（見 src/detect.py）
    車門緒      YOLOv11m 車門開 / 關（見 src/door.py）
    馬達緒      依主緒算出的目標位置驅動避震器馬達、讀編碼器脈衝（見 src/motor.py）
"""

import argparse
import os
import signal
import sys
import time

import src  # noqa: F401  先 import：它會設 QT_QPA_PLATFORM，必須早於 cv2
import cv2

from src.draw import UI_FONT_SCALE, UI_THICKNESS, put_text_panel
from src.settings import SettingsError, load_settings


# 左上角狀態區：一項一行（報告投影用，比其他標籤大；分析緒的次數與耗時不上畫面，看 metrics / log）
_HUD_X, _HUD_Y = 4, 4          # 黑底面板左上角；面板的底要在 src/draw.py 的 STATUS_TOP_RESERVED 之內
_WHITE, _YELLOW = (255, 255, 255), (0, 215, 255)
_GRADING_TEXT = {"asphalt": "asphalt grid", "cement": "cement grid+crack", "none": "none"}
# 畫面上的顯示名稱（程式內部、log、設定檔仍用原本的代號）
_SUSPENSION_TEXT = {"tight": "Lock", "mid": "Half-Lock", "loose": "Unlock"}
_DOOR_TEXT = {"open": "OPEN", "closed": "CLOSED", "none": "NOT DETECTED"}

# 沒有到達目標的停止原因：畫面上照實標出來，不顯示成已經到位（mid 沒有死點，stall 代表中途卡住）
_MOTOR_NOT_REACHED = ("timeout", "stall_start", "aborted")


def _suspension_line(ms) -> tuple[str, tuple]:
    """避震器狀態只用真實數值：停著顯示目前檔位，切換中（黃字）顯示
    「從哪一檔 -> 到哪一檔」加即時 pulse（跟終端機 log 對得上）。"""
    if ms is None:
        return "SUSPENSION: ...", _WHITE
    current = _SUSPENSION_TEXT.get(ms.target, "--")
    if ms.busy and ms.moving_to:
        return f"SUSPENSION: {current} -> {_SUSPENSION_TEXT[ms.moving_to]}  ({ms.position_pulses})", _YELLOW
    text = f"SUSPENSION: {current}"
    reason = ms.last_stop_reason
    if reason in _MOTOR_NOT_REACHED or (reason == "stall" and ms.target == "mid"):
        text += f"  ({reason})"
    return text, _WHITE


def _draw_status(frame, fps_shown, rr, dr, ms) -> None:
    """左上角黑底面板一項一行：路面種類與信心、分級模式、車門、避震器；FPS 黑底放右上角。"""
    lines = [
        ("ROAD: ..." if rr is None else f"ROAD: {rr.label} ({rr.confidence:.0%})", _WHITE),
        ("GRADING: ..." if rr is None else f"GRADING: {_GRADING_TEXT[rr.mode]}", _WHITE),
        ("DOOR: ..." if dr is None else f"DOOR: {_DOOR_TEXT[dr.state]}", _WHITE),
        _suspension_line(ms),
    ]
    put_text_panel(frame, lines, _HUD_X, _HUD_Y, UI_FONT_SCALE, UI_THICKNESS)

    fps_text = f"{fps_shown:.1f} FPS"
    (tw, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, UI_FONT_SCALE, UI_THICKNESS)
    put_text_panel(frame, [(fps_text, _WHITE)], frame.shape[1] - tw - 24, _HUD_Y, UI_FONT_SCALE, UI_THICKNESS)


def main() -> None:
    parser = argparse.ArgumentParser(description="路面辨識整合系統（路面種類 → 劣化分級 + 人車警戒 + 車門偵測）",
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog="參數請改 configs/ 底下的各個 yaml")
    parser.add_argument("--ui", action="store_true",
                        help="重新框選路面 ROI 與警戒區（不沿用 presets/roi.json、warning_zone.json）")
    parser.add_argument("--no-save", action="store_true", help="只顯示不錄影")
    parser.add_argument("--no-display", action="store_true",
                        help="只錄影不開即時畫面視窗（省顯示開銷；框選 ROI / 警戒區的視窗照常，"
                             "執行中沒有 q / r 鍵與滑鼠改警戒區，用 Ctrl+C 或 SIGTERM 結束）")
    parser.add_argument("--metrics", action="store_true",
                        help="取樣功耗 / 頻率 / 降頻 / 各階段耗時到 CSV（metrics.yaml 的 enabled 為 false 時臨時打開）")
    parser.add_argument("--configs", default=None, help="設定資料夾（預設 configs/）")
    args = parser.parse_args()
    if args.no_save and args.no_display:
        parser.error("--no-save 與 --no-display 不能同時用（不顯示也不錄影就沒有輸出了）")
    show = not args.no_display

    try:
        cfg = load_settings(args.configs) if args.configs else load_settings()
    except SettingsError as e:
        print(f"設定錯誤: {e}", file=sys.stderr)
        sys.exit(1)
    metrics_on = cfg.metrics.enabled or args.metrics
    if metrics_on:
        # HailoRT 的 scheduler 只在建 VDevice 時看這個變數；有它 `hailortcli monitor` 才拿得到使用率
        os.environ.setdefault("HAILO_MONITOR", "1")

    # 模型與硬體相關的模組放在設定檢查通過之後才 import，設定寫錯時能馬上得到回饋
    from src.analyzer import RoadAnalyzer
    from src.camera import Camera
    from src.detect import DetectorThread, HailoDetector, OverlayRenderer, WarningZone
    from src.door import DoorDetector, DoorThread, draw_doors
    from src.event_log import EventLogger, snapshot
    from src.grading.graders import AsphaltGrader, CementGrader
    from src.hailo import HailoDevice
    from src.metrics import MetricsRecorder
    from src.motor import MotorController
    from src.motor_policy import TargetDebouncer, decide_target
    from src.recorder import SegmentRecorder
    from src.road_type import RoadTypeClassifier
    from src.roi import camera_key, resolve_roi

    # systemd 用 SIGTERM 結束；轉成 KeyboardInterrupt 走同一條收尾路徑（Hailo 資源要依序釋放）
    def _on_sigterm(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _on_sigterm)

    dev = HailoDevice()
    cam = None
    road = det = door = motor = None
    recorder = None
    logger = None
    metrics = None
    loop_ms = {"draw_ms": 0.0, "write_ms": 0.0, "show_ms": 0.0}    # 主緒每幀各段耗時，供 metrics 取樣
    frame_idx = 0
    interrupted = False
    t_start = time.perf_counter()
    try:
        # ── 模型 ──
        resnet = dev.load(cfg.road_type.hef)
        asphalt_model = dev.load(cfg.asphalt.hef, pool=cfg.asphalt.grid.rows * cfg.asphalt.grid.cols)
        cement_model = dev.load(cfg.cement.hef, pool=cfg.cement.grid.rows * cfg.cement.grid.cols)
        yolo_model = dev.load(cfg.detect.hef)
        door_model = dev.load(cfg.door.hef)
        road_type = RoadTypeClassifier(resnet, cfg.road_type)
        asphalt = AsphaltGrader(asphalt_model, cfg.asphalt)
        cement = CementGrader(cement_model, cfg.cement)
        detector = HailoDetector(yolo_model, cfg.detect)
        door_detector = DoorDetector(door_model, cfg.door)
        # 車門模型共用人車偵測的 YOLO 輸入，兩個模型的輸入尺寸必須一樣
        if door_detector.input_size != detector.input_size:
            raise RuntimeError(f"車門模型輸入 {door_detector.input_size} 與人車偵測 {detector.input_size} "
                               "不同，無法共用同一份輸入")

        # ── 鏡頭 ──
        W, H = cfg.output.width, cfg.output.height
        cam = Camera(cfg.camera, view_size=(W, H), yolo_size=detector.input_size)
        c = cfg.camera
        print(f"相機 cam{c.index} 已啟動：{c.width}×{c.height} @ {c.fps}fps，顯示/錄影 {W}×{H}，"
              f"YOLO {detector.input_size}"
              f"（曝光 {c.exposure} / 白平衡 {c.awb} / 降噪 {c.denoise} / 旋轉 {c.rotation}°）")
        first = cam.read_full()
        if first is None:
            raise RuntimeError(f"讀不到 cam{c.index} 的畫面，請確認鏡頭有接好（rpicam-hello --list-cameras）")

        # ── 路面 ROI（三個路面模型共用）與警戒區 ──
        # 路面 ROI 在擷取原圖上框、裁切（分析用原解析度）；疊圖、人車、車門、警戒區都在縮小後的 W×H 座標
        win_w, win_h = cfg.road_type.roi_window_max_w, cfg.road_type.roi_window_max_h
        cx1, cy1, cx2, cy2 = resolve_roi(first, cfg.road_type.roi_store, camera_key(c.index),
                                         args.ui, win_w, win_h)
        roi_is_full = (cx1, cy1, cx2, cy2) == (0, 0, c.width, c.height)
        sx, sy = W / c.width, H / c.height
        rx1, ry1, rx2, ry2 = round(cx1 * sx), round(cy1 * sy), round(cx2 * sx), round(cy2 * sy)
        cam.set_roi((cx1, cy1, cx2, cy2))
        # 框警戒區時把剛框好的路面 ROI 畫在畫面上當參考（畫在副本，first 之後不再用）
        zone = WarningZone(cfg.detect, W, H)
        preview = cv2.resize(first, (W, H), interpolation=cv2.INTER_AREA)
        if not roi_is_full:
            cv2.rectangle(preview, (rx1, ry1), (rx2, ry2), (255, 255, 255), 1)
        zone.select(preview, args.ui, win_w, win_h)

        window = cfg.output.window_title
        if show:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window, W, H)      # 原生尺寸顯示，拉大會經插值讓小字糊掉
            cv2.setMouseCallback(window, zone.mouse_callback)
        renderer = OverlayRenderer()

        # ── 背景緒 ──
        road = RoadAnalyzer(road_type, asphalt, cement)
        det = DetectorThread(detector, zone, cfg.detect, W, H)
        door = DoorThread(door_detector, W, H)
        motor = MotorController(cfg.motor)
        road.start()
        det.start()
        door.start()
        motor.start()
        motor_debouncer = TargetDebouncer(cfg.motor.confirm_count)
        last_motor_target = None      # 上次成功送出的目標，只在確認變了才下命令
        last_motor_switch_time = 0.0  # 上次成功送出切換命令的時間，搭配 min_switch_interval_sec 限頻
        motor_busy_warned = None      # 已經印過「忙碌中被略過」的目標

        if not args.no_save:
            recorder = SegmentRecorder(cfg.output)
        logger = EventLogger(cfg.output)
        det_classes = list(cfg.detect.classes.values())

        fps_shown = 0.0
        t_start = last_t = time.perf_counter()
        last_n = 0

        if metrics_on:
            def _app_stats() -> dict:
                rr = road.latest()
                mode = rr.mode if rr else None
                grade_model = {"asphalt": asphalt_model, "cement": cement_model}.get(mode)
                return {
                    "fps": fps_shown if last_n > 0 else None,     # 第一秒還沒算出 fps，不要記 0

                    "road_mode": mode,
                    "road_label": rr.label if rr else None,
                    "road_round_ms": road.last_ms,
                    "resnet_infer_ms": resnet.last_ms,
                    "grade_infer_ms": grade_model.last_ms if grade_model else None,
                    "crack_ms": cement.crack_detector.last_ms if mode == "cement" else None,
                    "det_round_ms": det.last_ms,
                    "yolo_infer_ms": yolo_model.last_ms,
                    "door_round_ms": door.last_ms,
                    "door_infer_ms": door_model.last_ms,
                    "door_state": dr.state if (dr := door.latest()) else None,
                    **loop_ms,
                    "recording": 1 if recorder else 0,
                    "hailo_temp": dev.chip_temperature(),
                }
            metrics = MetricsRecorder(cfg.metrics, _app_stats,
                                      {"resnet": resnet.name, "asphalt": asphalt_model.name,
                                       "cement": cement_model.name, "yolo": yolo_model.name,
                                       "door": door_model.name})
            metrics.start()
            print(f"指標取樣中（每 {cfg.metrics.interval:g} 秒）→ {metrics.csv_path}")

        if show:
            print("開始。按 q 或 Ctrl+C 結束；滑鼠拖曳設定警戒區；r 重設追蹤。")
        else:
            print("開始（不顯示畫面）。按 Ctrl+C 結束。")

        while True:
            got = cam.read()
            if got is None:
                print("\n相機串流結束（可能是 CSI 排線接觸不良）", file=sys.stderr)
                break
            roi_crop, frame, yolo_in = got

            # 三條分析緒都只留最新一幀。roi_crop（擷取原圖裁切）與 frame（顯示畫面）是分開的新陣列，
            # 在 frame 上畫圖不影響分析緒；yolo_in 兩條偵測緒都只讀不寫，可共用同一份
            road.submit(roi_crop)
            det.submit(yolo_in)
            door.submit(yolo_in)

            # 路面：網格疊回 ROI，白框標出分析範圍
            t_draw = time.perf_counter()
            rr = road.latest()
            if rr is not None and rr.result is not None:
                frame[ry1:ry2, rx1:rx2] = rr.grader.draw(frame[ry1:ry2, rx1:rx2], rr.result)
            if not roi_is_full:
                cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 255, 255), 1)
            if rr is not None:
                cv2.putText(frame, rr.label, (rx1 + 6, max(ry1 - 8, 50)),
                            cv2.FONT_HERSHEY_SIMPLEX, UI_FONT_SCALE, (255, 255, 255), UI_THICKNESS, cv2.LINE_AA)

            # 人車：偵測框、軌跡、預測點、警戒區、警報
            det_res = det.latest()
            renderer.draw(frame, det_res, zone.get())

            # 車門：開啟紅框、關閉綠框（狀態併進左上角狀態列）
            dr = door.latest()
            draw_doors(frame, dr)

            # 馬達：依路面/坑洞/人員/車門決定目標位置。兩層防抖動避免頻繁切換磨損齒輪：
            # TargetDebouncer 過濾掉辨識結果本身的抖動（見 src/motor_policy.py），
            # min_switch_interval_sec 是切換頻率的硬上限，就算目標真的確認變了也要間隔夠久才送
            raw_target = decide_target(rr, det_res, dr, cfg.motor)
            motor_target = motor_debouncer.update(raw_target)
            if motor_target is not None and motor_target != last_motor_target:
                now_t = time.monotonic()
                if now_t - last_motor_switch_time < cfg.motor.min_switch_interval_sec:
                    pass   # 還在最短切換間隔內，先不送，下一幀再檢查
                elif motor.move_to(motor_target):
                    last_motor_target = motor_target
                    last_motor_switch_time = now_t
                elif motor_busy_warned != motor_target:
                    # 忙碌時每幀都會重試，同一個目標只印一次，避免 20 fps 洗版
                    print(f"[馬達] 目標改為 {motor_target}，但馬達忙碌中，命令被略過（完成後重試）", flush=True)
                    motor_busy_warned = motor_target
            ms = motor.latest()

            now = time.perf_counter()
            if now - last_t >= 1.0:
                fps_shown = (frame_idx - last_n) / (now - last_t)
                last_t, last_n = now, frame_idx
            _draw_status(frame, fps_shown, rr, dr, ms)
            t_write = time.perf_counter()

            if recorder:
                recorder.write(frame)
            # 放在 recorder.write 之後：錄影剛切新檔時，log 用的是新檔名
            logger.update(snapshot(rr, det_res, dr, det_classes, ms), time.time(),
                          (recorder.path.stem, recorder.seg_start) if recorder else None)
            t_show = time.perf_counter()
            key = -1
            if show:
                cv2.imshow(window, frame)
                key = cv2.waitKey(1) & 0xFF
            if metrics:
                t_end = time.perf_counter()
                loop_ms["draw_ms"] = (t_write - t_draw) * 1000.0
                loop_ms["write_ms"] = (t_show - t_write) * 1000.0
                loop_ms["show_ms"] = (t_end - t_show) * 1000.0
            if key == ord("q"):
                break
            if key == ord("r"):
                det.reset_tracks()
                print("[INFO] 追蹤器已重設")
            frame_idx += 1

    except KeyboardInterrupt:
        interrupted = True
        print("\n偵測到中斷，正在收尾存檔...")
    finally:
        # 順序很重要：先停取樣緒（它會讀 Hailo 溫度）與三條分析緒（它們還在用 Hailo），
        # 再收 writer（moov 在 release 時才寫），關相機，最後才釋放 Hailo
        for t in (metrics, road, det, door, motor):
            if t:
                t.stop()
        for t in (metrics, road, det, door, motor):
            if t:
                t.join(timeout=5)
        if motor:
            motor.close()
        if recorder:
            recorder.close()
        if logger:
            logger.close()
        if cam:
            cam.close()
        dev.close()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    elapsed = time.perf_counter() - t_start
    avg = frame_idx / elapsed if elapsed > 0 else 0
    print(f"{'已中斷。' if interrupted else ''}共 {frame_idx} 幀 / {elapsed:.0f} 秒（平均 {avg:.1f} fps）"
          + (f"，路面分析 {road.update_count} 次、切換 {road.switches} 次" if road else "")
          + (f"，偵測 {det.update_count} 次" if det else "")
          + (f"，車門偵測 {door.update_count} 次" if door else "")
          + (f"，馬達最終 pulse = {ms.position_pulses}" if motor and (ms := motor.latest()) else ""))
    if recorder:
        print(f"錄影輸出到: {cfg.output.dir}")
    if logger:
        print(f"偵測結果 log：{logger.records} 筆 → {cfg.output.log_dir}")
    if metrics:
        print()
        print(metrics.summary())
        print(f"（CSV：{metrics.csv_path}；摘要：{metrics.summary_path}）")


if __name__ == "__main__":
    main()
