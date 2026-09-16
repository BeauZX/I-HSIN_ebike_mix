"""Hailo-8 共用裝置與模型包裝。

為什麼要共用一個 VDevice：Hailo-8 預設一次只能被一個程序開啟（第二個會拿到
HAILO_OUT_OF_PHYSICAL_DEVICES），所以四個模型全部放進同一個程序、同一個 VDevice，
交給 HailoRT 的 scheduler 輪流排程。這也是 asphalt / cement 原本的舊式
InferVStreams + activate() 寫法必須改掉的原因：activate() 會獨佔裝置，跟 scheduler 不相容。

多張影像用 run_async 一起排進去、再一起等，比逐張同步快一倍以上
（實測 asphalt 15 格：同步逐張 64 ms、async 27 ms；原專案舊 API 批次是 34 ms）。
set_batch_size() 在這幾個 HEF 上會卡死，所以不用。

HailoRT 物件之間有 DMA 映射的相依關係，交給直譯器結束時任意順序解構會
segfault / bus error，因此 close() 依 bindings → buffer → configured → infer_model → vdevice
的順序手動釋放；請用 Ctrl+C 或 SIGTERM 結束，不要 kill -9。
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

INFER_TIMEOUT_MS = 10_000


def _import_hailo():
    try:
        from hailo_platform import FormatType, HailoSchedulingAlgorithm, VDevice
    except ImportError as e:
        raise ImportError(
            "找不到 hailo_platform。它是 apt 的 python3-hailort 帶的，裝在系統 Python；"
            "venv 必須用 `uv venv --system-site-packages` 建立才看得到（見 README）。"
        ) from e
    return FormatType, HailoSchedulingAlgorithm, VDevice


class HailoModel:
    """單一 HEF 的推論介面：infer(images) -> outputs。

    pool 決定預先建立幾組 bindings / buffer；一次 infer 最多同時排 pool 張，
    超過就分批。buffer 整段執行都重用，避免每幀新建、被回收後在 shutdown 時
    釋放到已失效的記憶體。
    """

    def __init__(self, vdevice, hef_path: Path, pool: int = 1):
        FormatType, _, _ = _import_hailo()
        self.hef_path = Path(hef_path)
        self.name = self.hef_path.stem
        self.infer_model = None
        self.configured = None
        self._bindings: list = []
        self._inputs: list[np.ndarray] = []
        self._lock = threading.Lock()

        self.infer_model = vdevice.create_infer_model(str(self.hef_path))
        ins, outs = self.infer_model.inputs, self.infer_model.outputs
        if len(ins) != 1 or len(outs) != 1:
            raise ValueError(f"{self.name}: 只支援單輸入單輸出的 HEF（目前 {len(ins)} 入 {len(outs)} 出）")
        self.infer_model.input().set_format_type(FormatType.UINT8)
        self.infer_model.output().set_format_type(FormatType.FLOAT32)
        self.input_shape = tuple(ins[0].shape)          # (H, W, C)
        self.output_shape = tuple(outs[0].shape)
        self.input_quant = ins[0].quant_infos[0]        # 供呼叫端檢查前處理假設
        self.is_nms = "nms" in outs[0].name.lower()     # YOLO 這種 NMS 輸出是「每類一個 (n,5) 陣列」

        try:
            self.configured = self.infer_model.configure()
            self.pool = max(1, int(pool))
            for _ in range(self.pool):
                b = self.configured.create_bindings()
                inp = np.zeros(self.input_shape, np.uint8)
                b.input().set_buffer(inp)
                if not self.is_nms:
                    b.output().set_buffer(np.zeros(self.output_shape, np.float32))
                else:
                    # NMS 輸出長度不固定，交給 HailoRT 自己配置，讀時用 get_buffer()
                    b.output().set_buffer(np.zeros(self.output_shape, np.float32))
                self._bindings.append(b)
                self._inputs.append(inp)
        except Exception:
            self.close()
            raise

    def infer(self, images: list[np.ndarray], timeout_ms: int = INFER_TIMEOUT_MS) -> list:
        """images 每張都要是 input_shape 的 uint8（HWC）。回傳每張的輸出（float32 陣列副本）。"""
        if self.configured is None:
            raise RuntimeError(f"{self.name}: 模型已關閉")
        results: list = []
        with self._lock:
            for start in range(0, len(images), self.pool):
                chunk = images[start:start + self.pool]
                for i, img in enumerate(chunk):
                    np.copyto(self._inputs[i], img)
                jobs = [self.configured.run_async([self._bindings[i]]) for i in range(len(chunk))]
                for j in jobs:
                    j.wait(timeout_ms)
                for i in range(len(chunk)):
                    out = self._bindings[i].output().get_buffer()
                    if isinstance(out, np.ndarray):
                        results.append(out.copy())
                    else:                       # NMS：list of per-class arrays
                        results.append([np.array(a, dtype=np.float32, copy=True) for a in out])
        return results

    def close(self) -> None:
        # 不能用區域變數暫存這些物件：多一份參照就會讓解構延後，順序又亂掉
        self._bindings = []
        self._inputs = []
        if self.configured is not None:
            try:
                self.configured.shutdown()
            except Exception:
                pass
            self.configured = None
        self.infer_model = None


class HailoDevice:
    """整個程序唯一的 VDevice；所有模型從這裡 load()，結束時 close() 依相反順序釋放。"""

    def __init__(self):
        _, HailoSchedulingAlgorithm, VDevice = _import_hailo()
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        self.vdevice = VDevice(params)
        self._models: list[HailoModel] = []
        try:
            ids = ", ".join(self.vdevice.get_physical_devices_ids())
        except Exception:
            ids = "?"
        print(f"Hailo 裝置：{ids}（scheduler: round-robin）")

    def load(self, hef_path: Path, pool: int = 1) -> HailoModel:
        m = HailoModel(self.vdevice, hef_path, pool)
        self._models.append(m)
        print(f"  已載入 {m.name}：輸入 {m.input_shape} → 輸出 {m.output_shape}"
              f"{'（NMS）' if m.is_nms else ''}，pool {m.pool}")
        return m

    def close(self) -> None:
        for m in reversed(self._models):
            m.close()
        self._models.clear()
        if self.vdevice is not None:
            try:
                self.vdevice.release()
            except Exception:
                pass
            self.vdevice = None
