"""路面辨識整合系統 — 內部模組。

這裡放的是「必須在 cv2 被載入之前」執行的環境設定：
任何模組要用 cv2 都得先經過這個 package，所以放這裡最保險。
"""

import os

# OpenCV 內建的 Qt 只附了 X11 版的視窗元件（libqxcb.so），沒有 Wayland 版。
# 樹莓派桌面預設跑 Wayland，直接開窗會失敗；指定走 X11（XWayland）就能正常顯示。
# 用 setdefault 讓使用者仍能從外部覆寫（例如 QT_QPA_PLATFORM=offscreen）。
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
