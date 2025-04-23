# run_gait_analysis.py

# --- 各機能ファイルをインポート ---
import matplotlib.font_manager as fm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from preprocessing import preprocess_and_sync_imu_data
from gait_cycles import identify_gait_cycles, identify_ics_from_trunk_accel
from pci import calculate_pci
from file_utils import find_latest_csv_file, save_results
from temporal_parameters import calculate_temporal_params
from kinematic_parameters import calculate_kinematic_params
from symmetry_indices import calculate_symmetry_index

# --- 必要な標準ライブラリ・外部ライブラリをインポート ---
import pandas as pd
import numpy as np
from pathlib import Path

# --- GUIとプロットのためのインポート ---
import tkinter as tk
from tkinter import Frame, Label, Button, BOTH, W, LEFT, messagebox, HORIZONTAL, Scale, Toplevel, DISABLED, NORMAL
import matplotlib
matplotlib.use('TkAgg')

# --- 日本語フォント設定 (japanize-matplotlib推奨) ---
try:
    import japanize_matplotlib
    print("japanize_matplotlib をインポートしました。")
except ImportError:
    print("警告: japanize-matplotlib が見つかりません。'pip install japanize-matplotlib' でインストールしてください。")
    try:
        # 環境に合わせてフォント名を調整
        jp_font_name = 'Hiragino Sans'  # macOS の例
        plt.rcParams['font.family'] = jp_font_name
        print(f"フォールバック: 日本語フォントとして '{jp_font_name}' を試みます。")
    except Exception as e_font:
        print(f"警告: フォールバックの日本語フォント設定エラー: {e_font}")

# --- 解析実行のための設定 ---
DATA_FOLDER = Path("./")  # データ検索フォルダ (デフォルト: スクリプトと同じ場所)
OUTPUT_FOLDER = Path("./analysis_results")  # 出力先フォルダ
OUTPUT_SUFFIX_SYNC = '_synchronized_gyro_z.csv'  # 同期済み角速度データの接尾辞
OUTPUT_SUFFIX_GAIT_TRIMMED = '_gait_events_steady.csv'   # 前後除外後の歩行周期データの接尾辞
OUTPUT_ALL_PARAMS_FILE = '_all_gait_parameters.csv'  # 全パラメータを保存するファイル名

# 前処理パラメータ
ROWS_TO_SKIP = 11                 # スキップするヘッダー前の行数
SAMPLING_INTERVAL_MS = 5          # サンプリング周期 (ms)
SYNC_SIGNAL_SUFFIX = '_Acc_Y'     # 同期に使う信号の軸
ALIGN_TARGET_SUFFIX = '_Gyro_Z'   # 同期を適用する信号
RIGHT_PREFIX = 'R'                # 右センサーのプレフィックス
LEFT_PREFIX = 'L'                 # 左センサーのプレフィックス
TRUNK_PREFIX = 'T'                # 体幹センサーのプレフィックス (列名設定用)

# find_peaks のデフォルトパラメータ (GUIの初期値)
DEFAULT_PEAK_HEIGHT = 10.0        # find_peaks デフォルト Height
DEFAULT_PEAK_PROMINENCE = 0.3     # find_peaks デフォルト Prominence
DEFAULT_PEAK_DISTANCE = 50        # find_peaks デフォルト Distance
NUM_SAMPLES_TO_PLOT = 1000        # 初期プロットで表示するサンプル数

# 自動トライアル分割用パラメータ
MAX_IC_INTERVAL_SEC = 2.0  # IC間の最大許容時間（これを超えたら別トライアル）
MIN_ICS_PER_TRIAL = 11    # トライアルとみなす最小IC数 (これ未満は除外)

# 定常歩行抽出用パラメータ
NUM_ICS_REMOVE_START = 3  # 除外する先頭IC数
NUM_ICS_REMOVE_END = 5   # 除外する末尾IC数

# 体幹検出用パラメータ
TRUNK_IC_FILTER_CUTOFF = 20.0  # 体幹加速度フィルター周波数 (Hz)
TRUNK_IC_PEAK_HEIGHT = 0.1     # 体幹AP加速度 正ピークの最小高さ (単位注意!)
TRUNK_IC_PEAK_PROMINENCE = 0.1  # 体幹AP加速度 正ピークの最小突出度
TRUNK_IC_PEAK_DISTANCE_MS = 200  # 体幹AP加速度 正ピーク間の最小時間 (ms)
TRUNK_LR_THRESHOLD = 0.0         # 体幹ML加速度 左右判定の閾値
MIN_STEP_TIME_SEC = 0.3        # IC間最小時間 (秒) - ノイズ除去用
TRUNK_LR_GYRO_THRESHOLD = 5.0  # 左右判定のヨー角速度閾値 (deg/s?)

# --- 列名定義関数 ---


def get_expected_column_names(right_prefix, left_prefix, trunk_prefix):
    # 33列分のリストを返す
    return [
        'Time', f'{right_prefix}_Acc_X', f'{right_prefix}_Acc_Y', f'{right_prefix}_Acc_Z', f'{right_prefix}_Ch4_V',
        f'{right_prefix}_Gyro_X', f'{right_prefix}_Gyro_Y', f'{right_prefix}_Gyro_Z', f'{right_prefix}_Ch8_V',
        f'{left_prefix}_Acc_X', f'{left_prefix}_Acc_Y', f'{left_prefix}_Acc_Z', 'Blank_1',
        f'{left_prefix}_Gyro_X', f'{left_prefix}_Gyro_Y', f'{left_prefix}_Gyro_Z', 'Blank_2',
        f'{trunk_prefix}_Acc_X', f'{trunk_prefix}_Acc_Y', f'{trunk_prefix}_Acc_Z', 'Blank_3',
        f'{trunk_prefix}_Gyro_X', f'{trunk_prefix}_Gyro_Y', f'{trunk_prefix}_Gyro_Z', 'Blank_4',
        'Blank_5', 'Blank_6', 'Blank_7', 'Blank_8', 'Blank_9', 'Blank_10', 'Blank_11', 'Blank_12'
    ]

# --- 歩行トライアル分割関数 ---


def segment_walking_trials(events_df, max_interval_sec, min_ics_per_trial):
    """IC間の時間差に基づき歩行トライアルを分割し、短すぎるトライアルを除外"""
    print(
        f"--- 歩行トライアルの自動分割開始 (最大IC間隔: {max_interval_sec}s, 最小IC数: {min_ics_per_trial}) ---")
    if events_df is None or events_df.empty:
        return pd.DataFrame()
    df_sorted = events_df.sort_values(by="IC_Time").copy()
    df_sorted['Time_Diff'] = df_sorted['IC_Time'].diff()
    df_sorted['Trial_ID_Raw'] = (
        df_sorted['Time_Diff'] > max_interval_sec).cumsum() + 1
    df_sorted['Trial_IC_Count'] = df_sorted.groupby(
        'Trial_ID_Raw')['IC_Index'].transform('size')
    df_segmented = df_sorted[df_sorted['Trial_IC_Count']
                             >= min_ics_per_trial].copy()
    if df_segmented.empty:
        print("警告: 有効な歩行トライアル検出不可")
        return pd.DataFrame()
    df_segmented['Trial_ID'] = df_segmented.groupby(
        'Trial_ID_Raw').ngroup() + 1
    valid_trials = df_segmented['Trial_ID'].unique()
    print(
        f"  -> {len(valid_trials)} 個の有効な歩行トライアル (ID: {valid_trials.tolist()}) を検出")
    return df_segmented.drop(columns=['Time_Diff', 'Trial_ID_Raw', 'Trial_IC_Count'])

# --- トライアルの最初と最後を除外する関数 ---


def trim_trial_ends(df_segmented, n_start=3, n_end=5):
    """Trial_ID と Leg でグループ化し、各グループの先頭n_start個と末尾n_end個のICを除外"""
    print(f"--- トライアルの前後除外開始 (先頭: {n_start}歩, 末尾: {n_end}歩) ---")
    if df_segmented is None or df_segmented.empty:
        return pd.DataFrame()
    min_required_ics = n_start + n_end + 1
    trimmed_groups = []
    grouped = df_segmented.sort_values(by='IC_Index').groupby(
        ['Trial_ID', 'Leg'], sort=False)
    total_removed_count = 0
    original_count = len(df_segmented)
    for name, group in grouped:
        if len(group) >= min_required_ics:
            trimmed_group = group.iloc[n_start:-n_end]
            trimmed_groups.append(trimmed_group)
            total_removed_count += len(group) - len(trimmed_group)
        else:
            total_removed_count += len(group)

    if not trimmed_groups:
        print("警告: 前後除外の結果、有効歩行周期なし")
        return pd.DataFrame()

    df_trimmed = pd.concat(trimmed_groups).reset_index(drop=True)
    print(
        f"  前後除外処理完了。 {original_count} -> {len(df_trimmed)} イベント ({total_removed_count} イベント除外)")
    return df_trimmed


# --- Tkinter GUI アプリケーションクラス ---
class GaitAnalysisApp:
    def __init__(self, master):
        self.master = master
        master.title("歩行データ同期・周期同定・パラメータ算出ツール")
        master.geometry("900x700")
        self.input_file_path = None
        self.sampling_rate = 1000.0 / SAMPLING_INTERVAL_MS

        # --- GUI要素の作成 ---
        top_frame = Frame(master, pady=5)
        top_frame.pack(side=tk.TOP, fill=tk.X)
        self.file_label = Label(top_frame, text="処理対象ファイル: 検索中...")
        self.file_label.pack(side=LEFT, padx=10)
        self.status_label = Label(top_frame, text="初期化中...")
        self.status_label.pack(side=LEFT, padx=10)

        # Matplotlibグラフ描画エリア
        self.fig = Figure(figsize=(8, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=BOTH, expand=True)
        toolbar_frame = Frame(master)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

        # パラメータ入力エリア
        param_frame = Frame(master, pady=10)
        param_frame.pack(side=tk.TOP, fill=tk.X)
        self.height_var = tk.DoubleVar(value=DEFAULT_PEAK_HEIGHT)
        self.prominence_var = tk.DoubleVar(value=DEFAULT_PEAK_PROMINENCE)
        self.distance_var = tk.IntVar(value=DEFAULT_PEAK_DISTANCE)

        # スライダーと値表示ラベル (Height, Prominence, Distance)
        row_idx = 0
        Label(param_frame, text="Height:").grid(
            row=row_idx, column=0, padx=5, pady=2, sticky=W)
        self.height_scale = Scale(param_frame, from_=0, to=50, resolution=0.1,
                                  orient=HORIZONTAL, variable=self.height_var, length=150)
        self.height_scale.grid(row=row_idx, column=1, padx=5, pady=2)
        self.height_label_val = Label(
            param_frame, text=f"{self.height_var.get():.1f}", width=5)
        self.height_label_val.grid(row=row_idx, column=2, padx=5, pady=2)
        self.height_scale.config(
            command=lambda v: self.height_label_val.config(text=f"{float(v):.1f}"))

        Label(param_frame, text="Prominence:").grid(
            row=row_idx, column=3, padx=5, pady=2, sticky=W)
        self.prominence_scale = Scale(param_frame, from_=0, to=10, resolution=0.1,
                                      orient=HORIZONTAL, variable=self.prominence_var, length=150)
        self.prominence_scale.grid(row=row_idx, column=4, padx=5, pady=2)
        self.prominence_label_val = Label(
            param_frame, text=f"{self.prominence_var.get():.1f}", width=5)
        self.prominence_label_val.grid(row=row_idx, column=5, padx=5, pady=2)
        self.prominence_scale.config(
            command=lambda v: self.prominence_label_val.config(text=f"{float(v):.1f}"))

        row_idx += 1
        Label(param_frame, text="Distance:").grid(
            row=row_idx, column=0, padx=5, pady=2, sticky=W)
        self.distance_scale = Scale(param_frame, from_=1, to=200, resolution=1,
                                    orient=HORIZONTAL, variable=self.distance_var, length=150)
        self.distance_scale.grid(row=row_idx, column=1, padx=5, pady=2)
        self.distance_label_val = Label(
            param_frame, text=f"{self.distance_var.get():d}", width=5)
        self.distance_label_val.grid(row=row_idx, column=2, padx=5, pady=2)
        self.distance_scale.config(
            command=lambda v: self.distance_label_val.config(text=f"{int(float(v)):d}"))

        # 実行ボタン (最初は無効)
        self.run_button = Button(param_frame, text="同期実行 ＆ 歩行周期同定・パラメータ算出",
                                 command=self.run_analysis_pipeline, width=30, state=DISABLED)
        self.run_button.grid(row=row_idx-1, column=6, rowspan=2,
                             padx=20, pady=5, sticky=tk.W + tk.E + tk.S)

        # 初期データロード（GUI準備後に実行）
        self.status_label.config(text="最新ファイル検索中...")
        self.master.update_idletasks()
        self.master.after(100, self.load_and_plot_initial_data)  # 100ms後に実行

    def load_and_plot_initial_data(self):
        """最新CSVをロードし、初期プロットを表示、ボタンを有効化"""
        print("\n[準備] 最新のCSVデータを検索中...")
        self.input_file_path = find_latest_csv_file(DATA_FOLDER)
        if self.input_file_path is None:
            messagebox.showerror(
                "エラー", f"CSVファイルが見つかりません ({DATA_FOLDER.resolve()})")
            self.master.quit()
            return
        self.file_label.config(text=f"処理対象: {self.input_file_path.name}")

        self.status_label.config(text="データ読込・プロット中...")
        self.master.update_idletasks()

        try:
            temp_df = None
            try:
                temp_df = pd.read_csv(
                    self.input_file_path, skiprows=ROWS_TO_SKIP, encoding='cp932')
            except UnicodeDecodeError:
                temp_df = pd.read_csv(
                    self.input_file_path, skiprows=ROWS_TO_SKIP, encoding='shift_jis')

            expected_cols = get_expected_column_names(
                RIGHT_PREFIX, LEFT_PREFIX, TRUNK_PREFIX)
            if len(temp_df.columns) == len(expected_cols):
                temp_df.columns = expected_cols
                blank_cols = [
                    col for col in temp_df.columns if 'Blank_' in col]
                temp_df = temp_df.drop(columns=blank_cols)
            else:
                raise ValueError(
                    f"初期読み込みで列数が不一致 ({len(temp_df.columns)} vs {len(expected_cols)})")

            sync_l_full = temp_df[f'{LEFT_PREFIX}{SYNC_SIGNAL_SUFFIX}'].fillna(
                0).values
            sync_r_full = temp_df[f'{RIGHT_PREFIX}{SYNC_SIGNAL_SUFFIX}'].fillna(
                0).values
            sync_t_full = temp_df[f'{TRUNK_PREFIX}{SYNC_SIGNAL_SUFFIX}'].fillna(
                0).values
            plot_len = min(len(sync_l_full), len(
                sync_r_full), NUM_SAMPLES_TO_PLOT)
            if plot_len <= 0:
                raise ValueError("プロットするデータがありません。")

            sync_l_short_plot = sync_l_full[:plot_len]
            sync_r_short_plot = sync_r_full[:plot_len]
            sync_t_short_plot = sync_t_full[:plot_len]
            time_plot = np.arange(plot_len) * (SAMPLING_INTERVAL_MS / 1000.0)

            self.ax.clear()
            self.ax.plot(time_plot, sync_l_short_plot,
                         label=f'左 ({LEFT_PREFIX}{SYNC_SIGNAL_SUFFIX})', alpha=0.8)
            self.ax.plot(time_plot, sync_r_short_plot,
                         label=f'右 ({RIGHT_PREFIX}{SYNC_SIGNAL_SUFFIX})', linestyle='--', alpha=0.8)
            self.ax.plot(time_plot, sync_t_short_plot,
                         label=f'体幹 ({TRUNK_PREFIX}{SYNC_SIGNAL_SUFFIX})', linestyle=':', alpha=0.7)
            self.ax.set_title(f'同期用信号 先頭 {plot_len} サンプル (パラメータ調整用)')
            self.ax.set_xlabel('時間 (s)')
            self.ax.set_ylabel('信号値 (単位?)')
            self.ax.legend()
            self.ax.grid(True)
            self.canvas.draw()

            self.run_button.config(state=NORMAL)
            self.status_label.config(text="準備完了. パラメータ調整後、ボタン実行.")
            print(f"  プロット完了。パラメータを調整してボタンを押してください。")

        except Exception as e:
            self.status_label.config(text="エラー発生！")
            messagebox.showerror(
                "エラー", f"初期データのプロット中にエラー:\n{e}", parent=self.master)
            self.run_button.config(state=DISABLED)  # エラー時はボタン無効のまま

    def run_analysis_pipeline(self):
        """解析パイプライン全体を実行するメソッド"""
        self.run_button.config(state=DISABLED)
        self.status_label.config(text="解析処理実行中...")
        self.master.update_idletasks()

        # 変数初期化
        sync_gyro_df = None
        gait_events_df_segmented = pd.DataFrame()
        gait_events_df_steady = pd.DataFrame()
        filtered_signals = {}
        time_vector = None

        try:
            print("\n========================================")
            print("=== 解析パイプライン実行 ===")
            user_peak_height = self.height_var.get()
            user_peak_prominence = self.prominence_var.get()
            user_peak_distance = self.distance_var.get()
            print(
                f"  使用 find_peaks パラメータ: h={user_peak_height:.2f}, p={user_peak_prominence:.2f}, d={user_peak_distance}")

            # === ステップ 1: 前処理 ===
            print("\n[ステップ1] 角速度データの前処理...")
            preprocess_result = preprocess_and_sync_imu_data(  # または preprocess_angular_velocity
                data_file=self.input_file_path, rows_to_skip=ROWS_TO_SKIP, sampling_interval_ms=SAMPLING_INTERVAL_MS,
                right_prefix=RIGHT_PREFIX, left_prefix=LEFT_PREFIX, trunk_prefix=TRUNK_PREFIX,
                # run_gait_analysis の設定を使用 (_Acc_Y になっているはず)
                sync_axis_suffix=SYNC_SIGNAL_SUFFIX,
                align_gyro_suffix=ALIGN_TARGET_SUFFIX,
                peak_height=user_peak_height, peak_prominence=user_peak_prominence, peak_distance=user_peak_distance
            )

            # 戻り値のチェック
            if preprocess_result is None:
                messagebox.showerror(
                    "エラー", "前処理中に致命的なエラーが発生しました。", parent=self.master)
                return  # finallyブロックへ
            if not isinstance(preprocess_result, tuple) or len(preprocess_result) != 3:
                messagebox.showerror(
                    "エラー", "前処理からの戻り値が予期せぬ形式です。", parent=self.master)
                return  # finallyブロックへ

            sync_data_df, lags_info, self.sampling_rate = preprocess_result  # ★ アンパック ★

            # sync_data_df のチェック
            if sync_data_df is None or not isinstance(sync_data_df, pd.DataFrame) or sync_data_df.empty:
                messagebox.showerror(
                    "エラー", "前処理は完了しましたが、有効な同期データフレームを取得できませんでした。", parent=self.master)
                return  # finallyブロックへ

            # ラグ情報を表示
            lag_L_R = lags_info.get('L_vs_R', 'N/A')
            lag_T_R = lags_info.get('T_vs_R', 'N/A')
            result_message = f"前処理完了。\nLag (R基準, {SYNC_SIGNAL_SUFFIX}ピーク): L={lag_L_R}, T={lag_T_R} [samples]"
            print(f"\n[成功] {result_message}")

            # 保存 (DataFrameには加速度も含まれる)
            base_filename = self.input_file_path.stem
            output_sync_file = OUTPUT_FOLDER / \
                (base_filename + '_synchronized_imu_data.csv')  # ★ ファイル名変更推奨 ★
            print(f"\n[ステップ1.1] 同期済み全IMUデータ保存中...")
            save_results(output_sync_file, sync_data_df,
                         "同期済みIMUデータ")  # ★ 説明変更 ★

            # === ステップ 2: 歩行周期同定 (全体データに対して) ===
            gait_events_df_all = None  # 初期化

            if sync_data_df is not None:
                print("\n[ステップ2] 歩行周期の同定 (全体) を実行中...")
                current_swing_threshold = 100  # ★必要なら調整★

                # identify_gait_cycles の呼び出し (引数はデフォルトを使用する例)
                gait_events_result = identify_gait_cycles(
                    sync_gyro_df=sync_data_df,
                    sampling_rate_hz=self.sampling_rate,
                    swing_threshold=current_swing_threshold
                    # 必要なら他のパラメータ(prominence, distanceなど)もここで指定
                )
                if gait_events_result is not None and isinstance(gait_events_result, dict):
                    gait_events_df_all = gait_events_result.get("events_df")
                    filtered_signals = gait_events_result.get(
                        "filtered_signals", {})
                    time_vector = gait_events_result.get("time_vector")

            # === ステップ 2bis: 歩行周期同定 (体幹 Acc Z/X ベース IC+LR) ===
            ic_events_df_trunk = None  # 結果格納用
            filtered_ap_signal = None  # ★ AP信号用変数
            filtered_lr_gyro_signal = None  # ★ ML信号用変数

            if sync_data_df is not None:
                print("\n[ステップ2bis] IC同定(左右判定含む, 体幹Acc) を実行中...")
                # ★ 体幹ベースの関数呼び出し ★
                ic_detection_result_trunk = identify_ics_from_trunk_accel(
                    sync_data_df=sync_data_df,
                    sampling_rate_hz=self.sampling_rate,
                    ap_axis_col='T_Acc_Z_aligned',
                    lr_gyro_axis_col='T_Gyro_Y_aligned',  # ★ Yaw軸を指定 ★
                    filter_cutoff_acc=TRUNK_IC_FILTER_CUTOFF,
                    ic_peak_height=TRUNK_IC_PEAK_HEIGHT,
                    ic_peak_prominence=TRUNK_IC_PEAK_PROMINENCE,
                    ic_peak_distance_ms=TRUNK_IC_PEAK_DISTANCE_MS,
                    min_step_time_sec=MIN_STEP_TIME_SEC,  # ★ 追加 ★
                    lr_gyro_threshold=TRUNK_LR_GYRO_THRESHOLD  # ★ 追加 ★
                )
                # ★ 結果の受け取り（キー名が変わる可能性に注意）★
                if ic_detection_result_trunk is not None and isinstance(ic_detection_result_trunk, dict):
                    ic_events_df_trunk = ic_detection_result_trunk.get(
                        "ic_events_df")
                    filtered_ap_signal = ic_detection_result_trunk.get(
                        "filtered_ap_signal")
                    filtered_lr_gyro_signal = ic_detection_result_trunk.get(
                        "filtered_lr_gyro_signal")  # ★ ML -> Yaw ★
                    if time_vector is None:
                        time_vector = ic_detection_result_trunk.get("time_vector")

                if ic_events_df_trunk is None or ic_events_df_trunk.empty:
                    print("  体幹AccベースのICイベントを検出できませんでした。")
                else:
                    print(
                        f"  {len(ic_events_df_trunk)} 個の体幹AccベースICイベント(左右判定試行済)を検出。")
                    base_filename = self.input_file_path.stem
                    output_gait_file_trunk_ic = OUTPUT_FOLDER / \
                        (base_filename + "_trunk_ic_events.csv")
                    print(f"\n[ステップ2bis.1] 体幹ICイベントデータを保存中...")
                    save_results(output_gait_file_trunk_ic,
                                 ic_events_df_trunk, "ICイベントデータ(体幹AccZ/LR)")
                    print("\n--- 検出された体幹ICイベント (最初の5件) ---")
                    print(ic_events_df_trunk[[
                          'Leg', 'Cycle', 'IC_Time']].head().to_string())
                    print("---")
                    # ★ 体幹ICプロット関数呼び出し ★
                    # plot_gait_events は下腿用なので、別名のプロット関数を呼び出す
                    if filtered_ap_signal is not None and filtered_lr_gyro_signal is not None and time_vector is not None:
                        self.plot_trunk_ics(  # ★ 呼び出す関数名を plot_trunk_ics とする ★
                            ic_events_df=ic_events_df_trunk,  # ★ 正しい変数 ★
                            filtered_ap_signal=filtered_ap_signal,  # AP信号
                            filtered_lr_gyro_signal=filtered_lr_gyro_signal,  # ML信号
                            time_vector=time_vector              # 時間ベクトル
                        )
                    else:
                        print("  警告: 体幹ICプロットに必要な信号データまたは時間ベクトルが不足しています。")
                # === ステップ 2.1: 歩行トライアル分割 ===
                if gait_events_df_all is not None and not gait_events_df_all.empty:
                    print("\n[ステップ2.1] 歩行トライアルの自動分割を実行中...")
                    gait_events_df_segmented = segment_walking_trials(
                        events_df=gait_events_df_all,
                        max_interval_sec=MAX_IC_INTERVAL_SEC,
                        min_ics_per_trial=MIN_ICS_PER_TRIAL
                    )

                # === ステップ 2.2: トライアルの前後除外 ===
                if not gait_events_df_segmented.empty:
                    print("\n[ステップ2.2] 定常歩行部分の抽出（前後除外）を実行中...")
                    gait_events_df_steady = trim_trial_ends(
                        df_segmented=gait_events_df_segmented,
                        n_start=NUM_ICS_REMOVE_START,
                        n_end=NUM_ICS_REMOVE_END
                    )

                # === ステップ 3 & 4: パラメータ計算 & 保存/表示 (有効な定常データがある場合のみ) ===
                if not gait_events_df_steady.empty:
                    # ステップ 2.3: 定常歩行周期データ保存
                    output_gait_file = OUTPUT_FOLDER / \
                        (base_filename + OUTPUT_SUFFIX_GAIT_TRIMMED)
                    print(f"\n[ステップ2.3] 定常歩行周期データを保存中...")
                    save_results(output_gait_file,
                                 gait_events_df_steady, "定常歩行周期データ")
                    print("\n--- 抽出された定常歩行周期 (最初の5件) ---")
                    print(gait_events_df_steady[[
                        'Leg', 'Trial_ID', 'Cycle', 'IC_Time', 'FO_Time']].head().to_string())
                    print("---")

                    # ステップ 2.4: IC/FOイベントのプロット
                    self.plot_gait_events(
                        gait_events_df_steady, filtered_signals, time_vector)

                    # ステップ 3: 各種パラメータ計算
                    print("\n[ステップ3] 各種歩行パラメータを計算中...")
                    results_all = {}  # 結果をまとめる辞書

                    # 3a. 時間パラメータ
                    temporal_params = calculate_temporal_params(
                        gait_events_df_steady)
                    if temporal_params:
                        results_all.update(temporal_params)

                    # 3b. 運動学パラメータ
                    if filtered_signals and time_vector is not None:
                        kinematic_params = calculate_kinematic_params(
                            gait_events_df_steady, filtered_signals, time_vector, self.sampling_rate)
                        if kinematic_params:
                            results_all.update(kinematic_params)
                else:
                    # ★★★ このメッセージが表示されている ★★★
                    print("\n[情報] 有効な定常歩行周期データがないため、パラメータ計算・イベントプロットはスキップします。")

                # 3c. PCI パラメータ
                pci_params = calculate_pci(gait_events_df_steady)
                if pci_params:
                    results_all.update(pci_params)

                # 3d. 対称性指数
                print("  対称性指数 (Symmetry Index) を計算中...")
                symmetry_results = {}
                param_pairs = {  # SIを計算したいパラメータのキーペア
                    'Stride_Time_%': ('Mean_Stride_Time_L_s', 'Mean_Stride_Time_R_s'),
                    'Swing_Time_%': ('Mean_Swing_Time_L_s', 'Mean_Swing_Time_R_s'),
                    'Stance_Time_%': ('Mean_Stance_Time_L_s', 'Mean_Stance_Time_R_s'),
                    'Max_Swing_Vel_%': ('Mean_Max_Swing_Vel_L', 'Mean_Max_Swing_Vel_R'),
                    'Peak_Stance_Vel_%': ('Mean_Peak_Stance_Vel_L', 'Mean_Peak_Stance_Vel_R')
                    # 必要なら他のペアも追加
                }
                for si_name, (key_L, key_R) in param_pairs.items():
                    val_L, val_R = results_all.get(
                        key_L), results_all.get(key_R)
                    # 値が両方存在する場合のみ計算
                    if pd.notna(val_L) and pd.notna(val_R):
                        symmetry_results[f'SI_{si_name}'] = calculate_symmetry_index(
                            val_L, val_R)
                    else:
                        symmetry_results[f'SI_{si_name}'] = np.nan
                results_all.update(symmetry_results)
                print("  対称性指数 計算完了。")

                # ステップ 3.1: 全パラメータ結果の表示と保存
                print(f"\n--- 全計算結果 ---")
                if results_all:
                    # コンソール表示
                    for key, value in sorted(results_all.items()):  # キーでソートして表示
                        if isinstance(value, (float, np.floating)):
                            print(f"  {key}: {value:.3f}")
                        else:
                            print(f"  {key}: {value}")
                    print("--------------------")
                    # CSV保存
                    try:
                        all_params_df = pd.DataFrame(
                            [results_all])  # 辞書を1行のDataFrameに
                        output_all_params_file = OUTPUT_FOLDER / \
                            (base_filename + OUTPUT_ALL_PARAMS_FILE)  # 新しいファイル名
                        print(f"\n[ステップ3.1] 全パラメータを保存中...")
                        save_results(output_all_params_file,
                                     all_params_df, "全歩行パラメータ")
                        # Pop-up message update
                        pci_val = results_all.get('PCI')
                        pci_str = f"{pci_val:.2f}" if pd.notna(
                            pci_val) else 'N/A'
                        cad_val = results_all.get('Cadence_steps_per_min')
                        cad_str = f"{cad_val:.1f}" if pd.notna(
                            cad_val) else 'N/A'
                        si_stride = results_all.get('SI_Stride_Time_%')
                        si_str = f"{si_stride:.1f}" if pd.notna(
                            si_stride) else 'N/A'
                        messagebox.showinfo(
                            "パラメータ計算完了", f"計算完了。\nPCI: {pci_str}, Cadence: {cad_str}\nStride SI: {si_str}%", parent=self.master)
                    except Exception as e_save:
                        print(f"エラー: 全パラメータ保存エラー: {e_save}")
                else:
                    print("  警告: 計算されたパラメータがありません。")
            else:  # gait_events_df_steady is empty
                print("\n[情報] 有効な定常歩行周期データがないため、パラメータ計算・イベントプロットはスキップします。")

            # === ステップ 4: 同期済み角速度の最終プロット ===
            self.plot_final_synchronized_data(sync_gyro_df)

            print("\n--- 解析パイプライン 完了 ---")
            self.status_label.config(text="解析完了")

        except Exception as e:
            print(f"\n[エラー] 解析パイプライン実行中にエラー: {e}")
            messagebox.showerror(
                "実行時エラー", f"解析中にエラー:\n{e}", parent=self.master)
            self.status_label.config(text="エラー発生")
        finally:
            # 処理終了後 (成功・失敗問わず) ボタンを再度有効化
            self.run_button.config(state=NORMAL)
            self.master.update_idletasks()

    # --- IC/FO イベントプロット用メソッド ---

    def plot_gait_events(self, gait_events_df, filtered_signals, time_vector):
        print("\n[ステップ2.4] IC/FO イベント (定常歩行区間) をグラフにプロットします...")
        if gait_events_df is None or gait_events_df.empty:
            print("  プロットするイベントデータがありません。")
            return
        try:
            fig_events, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
            fig_events.suptitle(
                f'検出された歩行イベント (定常区間) - {self.input_file_path.name}')
            plot_successful = False
            # 時間ベクトルがない場合は再生成を試みる
            if time_vector is None or len(time_vector) == 0:
                sig_len = len(filtered_signals.get('L', []))
                if sig_len == 0:
                    sig_len = len(filtered_signals.get('R', []))
                if sig_len > 0:
                    time_vector = np.arange(sig_len) * (1.0/self.sampling_rate)
                else:
                    raise ValueError("プロット用時間ベクトル不明")

            for i, leg in enumerate(['L', 'R']):
                ax = axes[i]
                filt_signal = filtered_signals.get(leg)
                # スライスされていないので、時間ベクトルと信号長は一致するはず
                if filt_signal is None or len(filt_signal) != len(time_vector):
                    print(f"  警告: {leg}脚のプロット用信号データ不整合。")
                    ax.set_title(f'{leg} 脚 - データ不整合')
                    continue

                ax.plot(time_vector, filt_signal,
                        label=f'{leg} Gyro Z (Filtered)', alpha=0.7)
                leg_events = gait_events_df[gait_events_df['Leg'] == leg]
                ic_times = leg_events['IC_Time'].dropna().values
                fo_times = leg_events['FO_Time'].dropna().values
                # 時間に対応するインデックスを推定 (マーカー位置のため)
                ic_indices_plot = np.searchsorted(
                    time_vector, ic_times, side='left')
                fo_indices_plot = np.searchsorted(
                    time_vector, fo_times, side='left')
                # 配列境界チェック
                ic_indices_plot = ic_indices_plot[ic_indices_plot < len(
                    filt_signal)]
                fo_indices_plot = fo_indices_plot[fo_indices_plot < len(
                    filt_signal)]

                if len(ic_indices_plot) > 0:
                    ax.plot(time_vector[ic_indices_plot], filt_signal[ic_indices_plot],
                            'ro', markersize=6, label='IC (踵接地)', linestyle='None')
                if len(fo_indices_plot) > 0:
                    ax.plot(time_vector[fo_indices_plot], filt_signal[fo_indices_plot], 'gx',
                            markersize=8, markeredgewidth=2, label='FO (爪先離地)', linestyle='None')

                ax.set_title(f'{leg} 脚')
                ax.set_ylabel('角速度 (単位?)')
                ax.legend(loc='upper right')
                ax.grid(True)
                plot_successful = True

            if plot_successful:
                axes[1].set_xlabel('時間 (s)')
                plt.tight_layout(rect=[0, 0.03, 1, 0.96])
                plt.show(block=False)  # 非同期表示
            else:
                plt.close(fig_events)  # プロットできなかったFigureは閉じる
                print("  イベントグラフはプロットされませんでした。")

        except Exception as e:
            print(f"  エラー: IC/FOイベントのプロット中にエラー: {e}")

# --- 体幹ICイベントプロット用メソッド ---
    def plot_trunk_ics(self, ic_events_df, filtered_ap_signal, filtered_lr_gyro_signal, time_vector):
        print("\n[ステップ2.4(暫定)] IC イベント (体幹Accベース) をグラフにプロットします...")
        # 入力データの基本的なチェック
        if ic_events_df is None or ic_events_df.empty:
            print("プロットするICイベントなし")
            return
        if filtered_ap_signal is None or filtered_lr_gyro_signal is None or time_vector is None:
            print("プロット用信号/時間ベクトルなし")
            return  # ★ Yaw をチェック ★
        if len(filtered_ap_signal) != len(time_vector) or len(filtered_lr_gyro_signal) != len(time_vector):
            print("警告:信号/時間ベクトル長不整合")
            return  # ★ Yaw をチェック ★

        if filtered_ap_signal is None or filtered_lr_gyro_signal is None or time_vector is None:
            print("プロット用信号/時間ベクトルなし")
            return
        if len(filtered_ap_signal) != len(time_vector) or len(filtered_lr_gyro_signal,) != len(time_vector):
            print("警告: プロット用信号/時間ベクトル長不整合")
            return

        try:
            fig_events, axes = plt.subplots(
                2, 1, figsize=(14, 8), sharex=True)
            fig_events.suptitle(
                f'検出されたICイベント (体幹Acc) - {self.input_file_path.name}')
            plot_successful = False

            signal_ap = filtered_ap_signal  # AP信号(Z軸仮定)
            signal_ml = filtered_lr_gyro_signal  # ML信号(X軸仮定)

            # --- 上段: AP加速度とICマーカー ---
            ax_ap = axes[0]
            ax_ap.plot(time_vector, signal_ap,
                       label=f'Trunk AP Acc (Z, Filtered)', alpha=0.7, color='k')
            for leg, color, marker in [('L', 'blue', 'o'), ('R', 'red', 'o')]:  # 左右色分け
                leg_events = ic_events_df[ic_events_df['Leg'] == leg]
                ic_indices = leg_events['IC_Index'].dropna().astype(
                    int).values
                valid_ic = ic_indices[(ic_indices >= 0) & (
                    ic_indices < len(signal_ap))]
                if len(valid_ic) > 0:
                    ax_ap.plot(time_vector[valid_ic], signal_ap[valid_ic], marker=marker,
                               color=color, markersize=6, label=f'IC ({leg})', linestyle='None')
            # Unknownもプロット
            unknown_events = ic_events_df[ic_events_df['Leg'] == 'Unknown']
            if not unknown_events.empty:
                unk_ic_indices = unknown_events['IC_Index'].dropna().astype(
                    int).values
                valid_unk_ic = unk_ic_indices[(unk_ic_indices >= 0) & (
                    unk_ic_indices < len(signal_ap))]
                if len(valid_unk_ic) > 0:
                    ax_ap.plot(time_vector[valid_unk_ic], signal_ap[valid_unk_ic], marker='x',
                               color='gray', markersize=6, label=f'IC (Unknown)', linestyle='None')
            ax_ap.set_title(f'体幹 AP(Z軸) 加速度 と検出されたIC')
            ax_ap.set_ylabel('加速度 (単位?)')
            ax_ap.legend(loc='upper right')
            ax_ap.grid(True)
            plot_successful = True

            # --- 下段: ML加速度とICマーカー ---
            ax_ml = axes[1]
            ax_ml.plot(time_vector, signal_ml,
                       label=f'Trunk ML Acc (X, Filtered)', alpha=0.7, color='k')
            ax_ml.axhline(0, color='gray', linestyle='--', linewidth=0.5)
            for leg, color, marker in [('L', 'blue', 'o'), ('R', 'red', 'o')]:  # 左右色分け
                leg_events = ic_events_df[ic_events_df['Leg'] == leg]
                ic_indices = leg_events['IC_Index'].dropna().astype(
                    int).values
                valid_ic = ic_indices[(ic_indices >= 0) & (
                    ic_indices < len(signal_ml))]
                if len(valid_ic) > 0:
                    ax_ml.plot(time_vector[valid_ic], signal_ml[valid_ic], marker=marker,
                               color=color, markersize=6, label=f'IC ({leg})', linestyle='None')
            if not unknown_events.empty:  # Unknownもプロット
                unk_ic_indices = unknown_events['IC_Index'].dropna().astype(
                    int).values
                valid_unk_ic = unk_ic_indices[(unk_ic_indices >= 0) & (
                    unk_ic_indices < len(signal_ml))]
                if len(valid_unk_ic) > 0:
                    ax_ml.plot(time_vector[valid_unk_ic], signal_ml[valid_unk_ic], marker='x',
                               color='gray', markersize=6, label=f'IC (Unknown)', linestyle='None')
            ax_ml.set_title(f'体幹 ML(X軸) 加速度 と検出されたIC（左右判定に使用）')
            ax_ml.set_ylabel('加速度 (単位?)')
            ax_ml.legend(loc='upper right')
            ax_ml.grid(True)

            if plot_successful:
                ax_ml.set_xlabel('時間 (s)')
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.show(block=False)
            else:
                plt.close(fig_events)
                print("ICプロット不可")
        except Exception as e:
            print(f"  エラー: ICイベントプロット中にエラー: {e}")
            if 'fig_events' in locals() and plt.fignum_exists(fig_events.number):
                plt.close(fig_events)

    # --- 同期済み角速度の最終プロット用メソッド ---

    def plot_final_synchronized_data(self, sync_gyro_df):
        if sync_gyro_df is not None:
            print(f"\n[ステップ4] 同期済み{ALIGN_TARGET_SUFFIX}のグラフ (先頭部分) を表示します...")
            try:
                if not sync_gyro_df.empty:
                    num_samples_final_plot = 1000
                    sync_df_short = sync_gyro_df.head(num_samples_final_plot)
                    actual_plot_len = len(sync_df_short)
                    plt.figure(figsize=(12, 6))  # 新しいウィンドウ
                    left_col_name = f'L{ALIGN_TARGET_SUFFIX}_aligned'
                    right_col_name = f'R{ALIGN_TARGET_SUFFIX}_aligned'
                    time_col_name = 'time_aligned_sec'

                    if left_col_name in sync_df_short.columns and \
                       right_col_name in sync_df_short.columns and \
                       time_col_name in sync_df_short.columns:
                        plt.plot(
                            sync_df_short[time_col_name], sync_df_short[left_col_name], label=f'同期後 左 {ALIGN_TARGET_SUFFIX}')
                        plt.plot(sync_df_short[time_col_name], sync_df_short[right_col_name],
                                 label=f'同期後 右 {ALIGN_TARGET_SUFFIX}', linestyle='--', alpha=0.8)
                        plt.title(
                            f'同期済み角速度 - 先頭 {actual_plot_len} サンプル ({self.input_file_path.name})')
                        plt.xlabel('時間 (s)')
                        plt.ylabel('角速度 (単位?)')
                        plt.legend()
                        plt.grid(True)
                        plt.show(block=False)
                    else:
                        print(f"  警告: 最終プロットに必要な列が見つかりません。")
                else:
                    print("  警告: 同期済みデータが空です。")
            except Exception as e:
                print(f"  エラー: 最終グラフ表示エラー: {e}")


# --- メイン実行ブロック ---
if __name__ == "__main__":
    # Tkinter ルートウィンドウの作成と非表示
    root = tk.Tk()
    root.withdraw()

    # Toplevelをメインウィンドウとしてアプリケーションインスタンスを作成
    app_window = Toplevel(root)
    app = GaitAnalysisApp(app_window)
    # ウィンドウが閉じられたときにTkinterのメインループを終了させる
    app_window.protocol("WM_DELETE_WINDOW", root.destroy)

    try:
        root.mainloop()  # イベントループ開始
    except KeyboardInterrupt:
        print("\nCtrl+C により中断されました。")
    finally:
        # Mainloop終了後
        print("\n========================================")
        print("=== アプリケーション終了 ===")
        print("========================================")
