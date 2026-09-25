#!/usr/bin/env python3
"""階段 2：裝上避震器後，用真正的 MotorController 手動切換測試（還不接自動辨識）。

跟正式跑 main.py 的差別：這裡用鍵盤指令手動觸發 move_to()，取代 src/motor_policy.py 的
decide_target() 自動決策——main.py 的其他部分（Hailo、攝影機、CV 分析）完全沒有牽扯進來，
純粹測馬達本身：位置追蹤、堵轉偵測、兩端動態歸零、位置持久化，這些邏輯在真實負載
（已經裝上避震器）下到底準不準。

執行：uv run tests/manual_motor_control.py

確認這支腳本都正常之後，才進到最後一步：接上 main.py 讓 CV 辨識結果自動觸發（不用再改這支腳本，
main.py 本來就已經整合好了，直接 uv run main.py 即可）。
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.motor import MotorController        # noqa: E402
from src.settings import SettingsError, load_settings  # noqa: E402

try:
    cfg = load_settings()
except SettingsError as e:
    print(f"設定錯誤: {e}", file=sys.stderr)
    sys.exit(1)

print(f"馬達設定：全緊={cfg.motor.pos_tight}　中間={cfg.motor.pos_mid}　全鬆={cfg.motor.pos_loose}"
      f"　（pulse 數，來自 configs/motor.yaml）")

motor = MotorController(cfg.motor)
motor.start()


def show_status() -> None:
    st = motor.latest()
    print(f"  目前狀態：pulse={st.position_pulses}　busy={st.busy}　"
          f"上次目標={st.target}　上次停止原因={st.last_stop_reason}　跟目標差={st.diff_from_target}")


print(__doc__)
print("指令：1=移到全緊　2=移到中間　3=移到全鬆　p=印狀態　q=結束")
print("（移動中會持續印 log；同一時間只接受一個命令，忙碌中按其他指令會顯示被略過）")

TARGET_NAMES = {"1": "tight", "2": "mid", "3": "loose"}

try:
    while True:
        cmd = input("> ").strip().lower()
        if cmd in TARGET_NAMES:
            target = TARGET_NAMES[cmd]
            if motor.move_to(target):
                print(f"已送出命令：移到 {target}，等待完成...")
                while motor.latest().busy:
                    time.sleep(0.1)
                show_status()
            else:
                print("馬達忙碌中，命令被略過，請稍後再試（或按 p 看目前狀態）")
        elif cmd == "p":
            show_status()
        elif cmd == "q":
            break
        else:
            print("看不懂，指令是 1 / 2 / 3 / p / q")
except KeyboardInterrupt:
    print("\n中斷，收尾中...")
finally:
    motor.stop()
    motor.join(timeout=5)
    motor.close()
    print("已釋放 GPIO。")

print("\n檢查重點：")
print("  1. 三個位置分別移動過去，機構實際鎖緊/放鬆的狀態跟你預期的一致嗎？")
print("  2. 移到全緊 / 全鬆時，log 有沒有印出「撞到…死點…位置自動校正」且停止原因是 stall？")
print("     兩端一律轉到頂住死點才停；看印出的漂移值，穩定在同一範圍就正常")
print("  3. 反覆在三個位置間切換幾次，最後的 pulse 數值會不會偏移很多（漂移）？")
print("  4. 關掉這支腳本、重新執行，開機讀回的位置是不是跟上次結束時一致？（測位置持久化）")
print("  5. 如果堵轉偵測太敏感（正常移動就提早停）或太遲鈍（真的卡住撐很久才停），")
print("     回頭調整 configs/motor.yaml 的 stall_ratio_num/den、min_stall_floor_us、stall_timeout_ms")
sys.exit(0)
