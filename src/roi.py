"""框選要分析的路面範圍，並把結果存起來重複使用（改寫自 asphalt 專案 src/roi.py）。

與原版的差別：roi.json 的路徑由呼叫端傳入（來自 configs/road_type.yaml），
而不是寫死在模組裡；三個路面模型（ResNet / asphalt / cement）共用同一組 ROI。
"""

import json
from pathlib import Path

import cv2
import numpy as np

# 視窗標題只能用 ASCII：OpenCV 5.0 的 Qt 後端用視窗名稱查找視窗，
# 含中文時查不到、回傳空指標，setMouseCallback 就會炸掉。
ROI_WINDOW_TITLE = "Select road ROI - drag, ENTER to confirm, C to cancel"

Roi = tuple[int, int, int, int]


def camera_key(camera_id: int) -> str:
    """相機的 roi.json 鍵值。每顆鏡頭各存一組，cam0 和 cam1 可以框不同範圍。"""
    return f"camera{camera_id}"


def _load_store(store: Path) -> dict:
    if not store.exists():
        return {}
    try:
        with open(store, encoding="utf-8") as f:
            return json.load(f) or {}
    except (json.JSONDecodeError, OSError) as e:
        print(f"[ROI] {store.name} 讀取失敗（{e}），當作沒有存過")
        return {}


def save_roi(store: Path, key: str, roi: Roi, width: int, height: int) -> None:
    """存成 0~1 的比例而不是像素座標，之後改解析度不必重框。"""
    data = _load_store(store)
    x1, y1, x2, y2 = roi
    data[key] = [x1 / width, y1 / height, x2 / width, y2 / height]
    store.parent.mkdir(parents=True, exist_ok=True)
    with open(store, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[ROI] 已存到 {store}（下次不用再框，除非鏡頭位置有動過）")


def load_roi(store: Path, key: str, width: int, height: int) -> Roi | None:
    """讀回指定來源的範圍並換算成目前解析度的像素座標；沒存過就回傳 None。"""
    data = _load_store(store)
    if key not in data:
        return None
    try:
        fx1, fy1, fx2, fy2 = data[key]
    except (ValueError, TypeError):
        print(f"[ROI] {key} 的設定格式不對，當作沒有存過")
        return None
    x1 = max(0, min(width, int(round(fx1 * width))))
    y1 = max(0, min(height, int(round(fy1 * height))))
    x2 = max(0, min(width, int(round(fx2 * width))))
    y2 = max(0, min(height, int(round(fy2 * height))))
    if x2 - x1 < 1 or y2 - y1 < 1:
        print(f"[ROI] {key} 的範圍太小，當作沒有存過")
        return None
    return x1, y1, x2, y2


class _DragBox:
    """select_rect 用的滑鼠狀態：左鍵按下記起點、拖曳時更新終點、放開定案。"""

    def __init__(self):
        self.p0 = self.p1 = None
        self.dragging = False

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.p0 = self.p1 = (x, y)
            self.dragging = True
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.p1 = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.p1 = (x, y)
            self.dragging = False

    def box(self) -> tuple[int, int, int, int] | None:
        if self.p0 is None or self.p1 is None:
            return None
        (ax, ay), (bx, by) = self.p0, self.p1
        return min(ax, bx), min(ay, by), max(ax, bx), max(ay, by)


def select_rect(first_frame: np.ndarray, title: str, prompt: str, max_w: int, max_h: int,
                color: tuple[int, int, int] = (255, 0, 0), crosshair: bool = True) -> Roi | None:
    """
    開視窗讓使用者用滑鼠拖出一個矩形，回傳原始解析度下的 (x1, y1, x2, y2)。
    Enter / 空白鍵確認，按 C 取消或沒框到東西時回傳 None。路面 ROI 與警戒區共用這個流程，
    靠 color / crosshair 區分（不用 cv2.selectROI 是因為它的框色寫死藍色、改不了）。
    畫面比螢幕大時先等比例縮小顯示，框完再換算回原始解析度。
    """
    h, w = first_frame.shape[:2]
    scale = min(1.0, max_w / w, max_h / h)
    if scale < 1.0:
        view = cv2.resize(first_frame, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_AREA)
    else:
        view = first_frame

    print(prompt)
    drag = _DragBox()
    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(title, drag.on_mouse)
    cancelled = False
    while True:
        canvas = view.copy()
        b = drag.box()
        if b is not None:
            cv2.rectangle(canvas, (b[0], b[1]), (b[2], b[3]), color, 2)
            if crosshair:
                cx, cy = (b[0] + b[2]) // 2, (b[1] + b[3]) // 2
                cv2.line(canvas, (cx, b[1]), (cx, b[3]), color, 1)
                cv2.line(canvas, (b[0], cy), (b[2], cy), color, 1)
        cv2.imshow(title, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10, 32) and not drag.dragging:     # Enter / 空白鍵
            break
        if key in (ord("c"), ord("C"), 27):                # C / Esc
            cancelled = True
            break
    cv2.destroyWindow(title)
    cv2.waitKey(1)          # 讓 Qt 有機會真正把視窗關掉，否則視窗會殘留在畫面上

    b = drag.box()
    if cancelled or b is None:
        return None
    x, y, bw, bh = b[0], b[1], b[2] - b[0], b[3] - b[1]
    if bw == 0 or bh == 0:
        return None
    x1 = max(0, min(w, int(round(x / scale))))
    y1 = max(0, min(h, int(round(y / scale))))
    x2 = max(0, min(w, int(round((x + bw) / scale))))
    y2 = max(0, min(h, int(round((y + bh) / scale))))
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return x1, y1, x2, y2


def select_roi(first_frame: np.ndarray, max_w: int, max_h: int) -> Roi | None:
    """框路面分析範圍；取消時回傳 None（呼叫端會改用全畫面）。"""
    return select_rect(first_frame, ROI_WINDOW_TITLE,
                       "請用滑鼠拖拉選取路面分析範圍，按 Enter 確認，按 C 取消（全畫面）",
                       max_w, max_h)


def resolve_roi(first_frame: np.ndarray, store: Path, key: str, force_ui: bool,
                max_w: int, max_h: int) -> Roi:
    """
    決定這次要分析的範圍。加了 --ui 就重新框；沒加就沿用 roi.json 存的；
    這個來源從來沒框過也會跳出視窗框一次（直接分析全畫面會把天空、路邊餵給分類器）。
    """
    h, w = first_frame.shape[:2]
    saved = None if force_ui else load_roi(store, key, w, h)
    if saved is not None:
        print(f"沿用上次框選的路面範圍: ({saved[0]}, {saved[1]}) → ({saved[2]}, {saved[3]})　[{key}]")
        return saved

    if not force_ui:
        print(f"[ROI] {key} 還沒框過，跳出視窗讓你框一次（之後就不用了）")
    roi = select_roi(first_frame, max_w, max_h)
    if roi is None:
        print("未選取範圍，這次使用全畫面（沒有存檔）")
        return (0, 0, w, h)
    print(f"路面分析範圍: ({roi[0]}, {roi[1]}) → ({roi[2]}, {roi[3]})")
    save_roi(store, key, roi, w, h)
    return roi
