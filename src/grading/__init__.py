"""路面劣化分級：asphalt（網格分類）與 cement（網格分類 × 裂縫偵測融合）。

複製自兩個原專案的模組：
    grid.py                 asphalt：classify_grid / draw_grid_overlay / GridTracker
    yolo_grid.py, fusion.py, overlay.py, presets.py, pothole/   cement
graders.py 是整合層：把兩者包成相同介面給分析緒用。
"""
