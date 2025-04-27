#!/usr/bin/env python3
import shutil
from pathlib import Path
import subprocess

# 生データ置き場と、処理済み置き場
RAW_DIR = Path(__file__).parent / "raw_data"
PROC_DIR = Path(__file__).parent / "processed_data"
RAW_DIR.mkdir(exist_ok=True)
PROC_DIR.mkdir(exist_ok=True)

# VSCode ターミナルから: python3 batch_process.py
# もしくは chmod +x batch_process.py && ./batch_process.py

for csv_file in sorted(RAW_DIR.glob("*.csv")):
    print(f"\n=== Processing {csv_file.name} ===")
    # GUI モードでファイルを指定して起動。ユーザーがウィンドウを閉じるまで待機します。
    # ※ run_gait_analysis.py は引数があるとそのファイルをロードして GUI を起動する前提です。
    subprocess.run(
        ["python3", "run_gait_analysis.py", str(csv_file)],
        check=True
    )
    # GUI を閉じたら、処理済みフォルダへ移動
    dest = PROC_DIR / csv_file.name
    shutil.move(str(csv_file), str(dest))
    print(f"→ Moved to {dest}")

print("\n=== All files processed ===")