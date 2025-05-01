# gait_analytics

体幹加速度・下腿角速度センサーデータから歩行周期（IC/FO）を同定し，歩行パラメータや Harmonic Ratio（HR／iHR）を算出する GUI 解析ツールです。

---

## 主な機能

- **IMU 前処理＆同期**  
  左右・体幹センサーの加速度 Y 軸ピークからデータを同期  
- **歩行周期同定**  
  下腿 (Gyro Z) と体幹 (Acc Z/Y+Gyro Y) による IC/FO 検出  
- **定常歩行区間抽出**  
  トライアル分割・前後トリミングによる定常サイクル抽出  
- **歩行パラメータ計算**  
  時間パラメータ，運動学パラメータ，PCI，Symmetry Index など  
- **Harmonic Ratio (HR/iHR)**  
  実際の歩行周期ごとの周波数解析で HR と iHR を算出  
- **GUI モード**  
  スライダーで閾値調整 → 「同期実行 & 全解析」「体幹IC更新」「結果フィードバック」

---

## 動作環境・依存ライブラリ

- Python 3.7 以降  
- 必要パッケージ:
  ```bash
  pip install numpy scipy pandas matplotlib japanize-matplotlib
  ```

---

## インストール

```bash
git clone https://github.com/HideyukiWanaka/gait_analytics.git
cd gait_analytics
# 任意で仮想環境を作成・有効化
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install numpy scipy pandas matplotlib japanize-matplotlib
```

---

## 使い方

### 1. 単一ファイル解析 (GUI モード)

解析したい CSV ファイルがあるフォルダで：
```bash
python run_gait_analysis.py                # 最新ファイルを自動で検索して GUI 起動
python run_gait_analysis.py path/to/data.csv  # 特定ファイルを直接 GUI で開く
```
1. Height/Prominence/Distance スライダーで同期ピーク検出のしきい値を設定  
2. 「1. 同期実行 & 全解析」ボタンを押して全ステップを実行  
3. 必要に応じて「2. 体幹IC更新＆プロット」ボタンで閾値再調整  
4. 「3. 結果フィードバック」で選択セグメントの詳細表示  

### 2. バッチ処理ワークフロー

複数の生データ CSV をまとめて処理したい場合：  
1. プロジェクト直下に `raw_data/` フォルダを作成し，未処理の CSV を格納  
2. VSCode などで以下を実行:
   ```bash
   python batch_process.py
   ```
3. `batch_process.py` が：
   - `raw_data/` 内の CSV を1件ずつ GUI 起動で自動実行  
   - 各ファイルでウィンドウを閉じると，解析結果を `analysis_results_<ID>/` に保存  
   - 終了後，該当ファイルを `processed_data/` に移動  
4. 扱うファイル数が多いときも，手動で1つずつ確認しながら安全にバッチ実行が可能  

---

## フォルダ構成例

```
gait_analytics/
├── run_gait_analysis.py       # メインスクリプト
├── feedback_ui.py             # フィードバックウィンドウ定義
├── preprocessing.py           # IMU 前処理・同期
├── gait_cycles.py             # IC/FO 同定ロジック
├── temporal_parameters.py     # 時間パラメータ計算
├── kinematic_parameters.py    # 運動学パラメータ計算
├── pci.py                     # PCI 計算
├── symmetry_indices.py        # Symmetry Index 計算
├── harmonic_ratio.py          # HR / iHR 計算
├── file_utils.py              # ファイル入出力ユーティリティ
├── batch_process.py           # バッチ実行スクリプト
├── raw_data/                  # 未処理生データ
├── processed_data/            # 解析後 CSV 移動先
└── README.md                  # この README
```

---

## 設定・カスタマイズ

- スクリプト冒頭の定数 (`DATA_FOLDER`, `OUTPUT_FOLDER`, 閾値など) で基本動作を調整可能  
- GUI スライダーで同期／IC検出の閾値を動的に変更  

---