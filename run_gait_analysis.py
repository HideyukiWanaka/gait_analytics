# run_gait_analysis.py

# --- 各機能ファイルをインポート ---
from scipy.signal import find_peaks
from scipy.signal import butter, filtfilt
from harmonic_ratio import calculate_harmonic_ratio
from harmonic_ratio import calculate_integrated_harmonic_ratio
from symmetry_indices import calculate_symmetry_index
from kinematic_parameters import calculate_kinematic_params
from temporal_parameters import calculate_temporal_params
from file_utils import find_latest_csv_file, save_results
from pci import calculate_pci
from gait_cycles import identify_gait_cycles, identify_ics_from_trunk_accel
from preprocessing import preprocess_and_sync_imu_data  # 関数名注意
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.font_manager as fm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import Frame, Label, Button, BOTH, W, LEFT, messagebox, HORIZONTAL, Scale, Toplevel, DISABLED, NORMAL
import matplotlib
matplotlib.use('TkAgg')  # Tkinter連携用バックエンドを明示的に指定

# --- Stumble detection utility ---


def detect_stumble(time_vector, signal_lr, neg_thresh=-30, pos_thresh=70, window_sec=0.3):
    """
    Detect first “stumble” event: a negative trough below neg_thresh
    followed within window_sec by a positive peak above pos_thresh.
    Returns the time of the trough if found, else None.
    """
    # find negative troughs
    troughs, _ = find_peaks(-signal_lr, height=-neg_thresh)
    if len(troughs) == 0:
        return None
    dt = time_vector[1] - time_vector[0] if len(time_vector) > 1 else 0.01
    window_samples = int(window_sec / dt)
    # find positive peaks
    peaks, _ = find_peaks(signal_lr, height=pos_thresh)
    for t_idx in troughs:
        if np.any((peaks > t_idx) & (peaks <= t_idx + window_samples)):
            return time_vector[t_idx]
    return None

# データ処理・数値計算関連

# 自作モジュール


# --- 日本語フォント設定 (japanize-matplotlib推奨) ---
try:
    import japanize_matplotlib
    print("japanize_matplotlib をインポートしました。")
except ImportError:
    print("警告: japanize-matplotlib が見つかりません。'pip install japanize-matplotlib' でインストールしてください。")
    try:
        # 環境に合わせてフォント名を調整
        jp_font_name = 'Hiragino Sans'  # macOS の例
        # jp_font_name = 'Yu Gothic' # Windows の例
        # jp_font_name = 'IPAexGothic' # Linux (要インストール) の例
        plt.rcParams['font.family'] = jp_font_name
        print(f"フォールバック: 日本語フォントとして '{jp_font_name}' を試みます。")
    except Exception as e_font:
        print(f"警告: フォールバックの日本語フォント設定エラー: {e_font}")

# --- 解析実行のための設定 ---
DATA_FOLDER = Path("./")  # データ検索フォルダ (デフォルト: スクリプトと同じ場所)
OUTPUT_FOLDER = Path("./analysis_results")  # 出力先フォルダ
OUTPUT_SUFFIX_SYNC = '_synchronized_imu_data.csv'  # 同期済みIMUデータ
OUTPUT_SUFFIX_GAIT_TRIMMED = '_gait_events_shank_steady.csv'  # 下腿ベースの定常歩行イベント
OUTPUT_SUFFIX_GAIT_TRUNK_IC = '_trunk_ic_events.csv'     # 体幹ベースICイベント
OUTPUT_ALL_PARAMS_FILE = '_all_gait_parameters.csv'    # 下腿ベースの全パラメータ

# 前処理パラメータ
ROWS_TO_SKIP = 11                 # スキップするヘッダー前の行数
SAMPLING_INTERVAL_MS = 5          # サンプリング周期 (ms)
SYNC_SIGNAL_SUFFIX = '_Acc_Y'     # 同期に使う信号の軸
ALIGN_TARGET_SUFFIX = '_Gyro_Z'   # 同期を適用する角速度信号(下腿)
RIGHT_PREFIX = 'R'                # 右センサーのプレフィックス
LEFT_PREFIX = 'L'                 # 左センサーのプレフィックス
TRUNK_PREFIX = 'T'                # 体幹センサーのプレフィックス (列名設定用)

# find_peaks のデフォルトパラメータ (GUIの初期値 - 同期用)
DEFAULT_PEAK_HEIGHT = 15.0        # find_peaks デフォルト Height
DEFAULT_PEAK_PROMINENCE = 0.3     # find_peaks デフォルト Prominence
DEFAULT_PEAK_DISTANCE = 50        # find_peaks デフォルト Distance
NUM_SAMPLES_TO_PLOT = 1000        # 初期プロットで表示するサンプル数

# 体幹IC検出用パラメータ (GUIの初期値 - 要調整)
TRUNK_IC_FILTER_CUTOFF = 20.0
TRUNK_IC_PEAK_HEIGHT = 0.1
TRUNK_IC_PEAK_PROMINENCE = 0.1
TRUNK_IC_PEAK_DISTANCE_MS = 200
MIN_STEP_TIME_SEC = 0.3
TRUNK_LR_GYRO_THRESHOLD = 12.5  # Gyro Y 左右判定閾値

# 下腿IC/FO検出用パラメータ (identify_gait_cyclesのデフォルト値を使用)
SHANK_SWING_THRESHOLD = 100  # 下腿GyroZ用

MAX_IC_INTERVAL_SEC = 0.8  # IC間隔上限を0.8秒に設定（大きめのギャップでセグメント分割）
MIN_ICS_PER_TRIAL = 15


# 下腿ベース定常歩行抽出用パラメータ (先頭3歩・末尾5歩)
NUM_ICS_REMOVE_START = 3
NUM_ICS_REMOVE_END = 5

# 体幹ベース定常歩行抽出用パラメータ (先頭5歩・末尾8歩)
TRUNK_NUM_ICS_REMOVE_START = 5
TRUNK_NUM_ICS_REMOVE_END = 8


# --- 列名定義関数 ---
def get_expected_column_names(right_prefix, left_prefix, trunk_prefix):
    """期待される33列の列名リストを生成する"""
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
    # IC_Time 列が存在するか確認
    if 'IC_Time' not in events_df.columns:
        print("エラー: events_df に IC_Time 列がありません。")
        return pd.DataFrame()
    # NaNを除外してからソート
    df_sorted = events_df.dropna(
        subset=['IC_Time']).sort_values(by="IC_Time").copy()
    if df_sorted.empty:
        return pd.DataFrame()

    df_sorted['Time_Diff'] = df_sorted['IC_Time'].diff()
    # 最初の行の Time_Diff は NaN なので 0 などで埋めるか、cumsum の挙動に任せる
    # cumsum は NaN を 0 として扱う傾向があるが、明示的に fillna(0) の方が安全かも
    df_sorted['Trial_ID_Raw'] = (
        df_sorted['Time_Diff'].fillna(0) > max_interval_sec).cumsum() + 1

    # Trial ID ごとの IC数を計算
    if 'IC_Index' not in df_sorted.columns:  # IC_Index がなければ size() を使う
        df_sorted['Trial_IC_Count'] = df_sorted.groupby(
            'Trial_ID_Raw')['Trial_ID_Raw'].transform('size')
    else:
        df_sorted['Trial_IC_Count'] = df_sorted.groupby(
            'Trial_ID_Raw')['IC_Index'].transform('size')

    # IC数が閾値以上のトライアルのみを抽出
    df_segmented = df_sorted[df_sorted['Trial_IC_Count']
                             >= min_ics_per_trial].copy()
    if df_segmented.empty:
        print("警告: 有効な歩行トライアル検出不可")
        return pd.DataFrame()

    # Trial ID を1から振り直す
    df_segmented['Trial_ID'] = df_segmented.groupby(
        'Trial_ID_Raw').ngroup() + 1
    valid_trials = df_segmented['Trial_ID'].unique()
    print(
        f"  -> {len(valid_trials)} 個の有効な歩行トライアル (ID: {valid_trials.tolist()}) を検出")
    # 不要な中間列を削除
    return df_segmented.drop(columns=['Time_Diff', 'Trial_ID_Raw', 'Trial_IC_Count'])


# --- トライアルの最初と最後を除外する関数 ---
def trim_trial_ends(df_segmented, n_start=3, n_end=5):
    """
    Trial_ID ごとに IC_Time で時系列ソートした上で、
    * 先頭から n_start 個
    * 末尾から n_end 個
    をまとめて除外する。

    この操作は左右の脚を合わせた連続シーケンス単位で行われます。
    """
    print(f"--- トライアル前後除外開始 (先頭: {n_start}歩, 末尾: {n_end}歩) ---")
    if df_segmented is None or df_segmented.empty:
        return pd.DataFrame()
    if 'Trial_ID' not in df_segmented.columns or 'IC_Time' not in df_segmented.columns:
        print("警告: 前後除外に必要な列が不足しています。")
        return df_segmented

    trimmed_trials = []
    total_removed = 0
    original_count = len(df_segmented)

    # Trial_ID ごとに一連の IC を扱う
    for trial_id, grp in df_segmented.groupby('Trial_ID', sort=False):
        grp_sorted = grp.sort_values('IC_Time').copy()
        count = len(grp_sorted)
        # 十分な数がないトライアルはそのままスキップ
        if count <= (n_start + n_end):
            total_removed += count
            continue
        # 先頭 n_start と末尾 n_end をまとめて除外
        grp_trimmed = grp_sorted.iloc[n_start: count - n_end]
        trimmed_trials.append(grp_trimmed)
        total_removed += count - len(grp_trimmed)

    if not trimmed_trials:
        print("警告: 前後除外の結果、有効なトライアルがありません。")
        return pd.DataFrame()

    df_out = pd.concat(trimmed_trials).reset_index(drop=True)
    print(
        f"  前後除外処理完了。 {original_count} -> {len(df_out)} イベント削除数: {total_removed}")
    return df_out


# --- Tkinter GUI アプリケーションクラス ---
class GaitAnalysisApp:
    def __init__(self, master):
        self.master = master
        master.title("歩行データ同期・周期同定・パラメータ算出ツール")
        master.geometry("900x800")  # 高さを確保
        self.input_file_path = None
        self.sampling_rate = None
        self.sync_data_df = None             # 同期済み全IMUデータ
        self.trunk_ic_results = None         # 体幹IC検出結果(辞書)
        self.gait_events_shank_steady = None  # 下腿ベース最終イベント
        self.time_vector = None              # 時間ベクトル

        # --- GUI要素の作成 ---
        # トップフレーム (ファイル名、ステータス)
        top_frame = Frame(master, pady=5)
        top_frame.pack(side=tk.TOP, fill=tk.X)
        self.file_label = Label(top_frame, text="処理対象ファイル: 検索中...")
        self.file_label.pack(side=LEFT, padx=10)
        self.status_label = Label(top_frame, text="初期化中...")
        self.status_label.pack(side=LEFT, padx=10)

        # Matplotlibグラフ描画エリア
        self.fig = Figure(figsize=(8, 3.5), dpi=100)  # 高さを少し縮小
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=BOTH, expand=True)
        toolbar_frame = Frame(master)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

        # --- パラメータ入力エリア ---
        param_area_frame = Frame(master, pady=5)
        param_area_frame.pack(side=tk.TOP, fill=tk.X)

        # --- 同期用パラメータ ---
        param_sync_frame = Frame(
            param_area_frame, borderwidth=1, relief=tk.GROOVE)
        param_sync_frame.pack(side=LEFT, padx=5, pady=2,
                              fill=tk.Y, anchor=tk.N)  # anchor追加
        Label(param_sync_frame, text="同期パラメータ (AccY)").pack(pady=2)
        self.height_var = tk.DoubleVar(value=DEFAULT_PEAK_HEIGHT)
        self.prominence_var = tk.DoubleVar(value=DEFAULT_PEAK_PROMINENCE)
        self.distance_var = tk.IntVar(value=DEFAULT_PEAK_DISTANCE)
        # (Height スライダー)
        frame_h = Frame(param_sync_frame)
        frame_h.pack(fill=tk.X)
        Label(frame_h, text="Height:", width=10, anchor=W).pack(side=LEFT)
        scale_h = Scale(frame_h, from_=0, to=50, resolution=0.1, orient=HORIZONTAL, variable=self.height_var,
                        length=120, command=lambda v: self.height_label_val.config(text=f"{float(v):.1f}"))
        scale_h.pack(side=LEFT)
        self.height_label_val = Label(
            frame_h, text=f"{self.height_var.get():.1f}", width=4)
        self.height_label_val.pack(side=LEFT, padx=2)
        # (Prominence スライダー)
        frame_p = Frame(param_sync_frame)
        frame_p.pack(fill=tk.X)
        Label(frame_p, text="Prominence:", width=10, anchor=W).pack(side=LEFT)
        scale_p = Scale(frame_p, from_=0, to=10, resolution=0.1, orient=HORIZONTAL, variable=self.prominence_var,
                        length=120, command=lambda v: self.prominence_label_val.config(text=f"{float(v):.1f}"))
        scale_p.pack(side=LEFT)
        self.prominence_label_val = Label(
            frame_p, text=f"{self.prominence_var.get():.1f}", width=4)
        self.prominence_label_val.pack(side=LEFT, padx=2)
        # (Distance スライダー)
        frame_d = Frame(param_sync_frame)
        frame_d.pack(fill=tk.X)
        Label(frame_d, text="Distance:", width=10, anchor=W).pack(side=LEFT)
        scale_d = Scale(frame_d, from_=1, to=200, resolution=1, orient=HORIZONTAL, variable=self.distance_var,
                        length=120, command=lambda v: self.distance_label_val.config(text=f"{int(float(v)):d}"))
        scale_d.pack(side=LEFT)
        self.distance_label_val = Label(
            frame_d, text=f"{self.distance_var.get():d}", width=4)
        self.distance_label_val.pack(side=LEFT, padx=2)

        # --- 体幹IC検出用パラメータ ---
        param_trunk_ic_frame = Frame(
            param_area_frame, borderwidth=1, relief=tk.GROOVE)
        param_trunk_ic_frame.pack(
            side=LEFT, padx=5, pady=2, fill=tk.Y, anchor=tk.N)  # anchor追加
        Label(param_trunk_ic_frame, text="体幹IC検出パラメータ").pack(pady=2)
        # 変数定義
        self.trunk_ic_height_var = tk.DoubleVar(value=TRUNK_IC_PEAK_HEIGHT)
        self.trunk_ic_prominence_var = tk.DoubleVar(
            value=TRUNK_IC_PEAK_PROMINENCE)
        self.trunk_lr_thresh_var = tk.DoubleVar(
            value=TRUNK_LR_GYRO_THRESHOLD)  # Gyro用
        self.min_step_time_var = tk.DoubleVar(value=MIN_STEP_TIME_SEC)
        # (IC Height スライダー)
        frame_th = Frame(param_trunk_ic_frame)
        frame_th.pack(fill=tk.X)
        Label(frame_th, text="IC Height:", width=12, anchor=W).pack(side=LEFT)
        scale_th = Scale(frame_th, from_=0, to=5, resolution=0.05, orient=HORIZONTAL, variable=self.trunk_ic_height_var,
                         length=100, command=lambda v: self.trunk_h_label.config(text=f"{float(v):.2f}"))
        scale_th.pack(side=LEFT)
        self.trunk_h_label = Label(
            frame_th, text=f"{self.trunk_ic_height_var.get():.2f}", width=5)
        self.trunk_h_label.pack(side=LEFT, padx=2)
        # (IC Prominence スライダー)
        frame_tp = Frame(param_trunk_ic_frame)
        frame_tp.pack(fill=tk.X)
        Label(frame_tp, text="IC Prominence:",
              width=12, anchor=W).pack(side=LEFT)
        scale_tp = Scale(frame_tp, from_=0, to=5, resolution=0.05, orient=HORIZONTAL, variable=self.trunk_ic_prominence_var,
                         length=100, command=lambda v: self.trunk_p_label.config(text=f"{float(v):.2f}"))
        scale_tp.pack(side=LEFT)
        self.trunk_p_label = Label(
            frame_tp, text=f"{self.trunk_ic_prominence_var.get():.2f}", width=5)
        self.trunk_p_label.pack(side=LEFT, padx=2)
        # (LR Gyro Thresh スライダー)
        frame_lrt = Frame(param_trunk_ic_frame)
        frame_lrt.pack(fill=tk.X)
        Label(frame_lrt, text="LR Gyro Thresh:",
              width=12, anchor=W).pack(side=LEFT)
        scale_lrt = Scale(frame_lrt, from_=0, to=50, resolution=0.01, orient=HORIZONTAL, variable=self.trunk_lr_thresh_var,
                          length=100, command=lambda v: self.trunk_lrt_label.config(text=f"{float(v):.1f}"))
        scale_lrt.pack(side=LEFT)
        self.trunk_lrt_label = Label(
            frame_lrt, text=f"{self.trunk_lr_thresh_var.get():.1f}", width=5)
        self.trunk_lrt_label.pack(side=LEFT, padx=2)
        # (Min Step Time スライダー)
        frame_mst = Frame(param_trunk_ic_frame)
        frame_mst.pack(fill=tk.X)
        Label(frame_mst, text="Min Step Time:",
              width=12, anchor=W).pack(side=LEFT)
        scale_mst = Scale(frame_mst, from_=0.1, to=1.0, resolution=0.01, orient=HORIZONTAL, variable=self.min_step_time_var,
                          length=100, command=lambda v: self.trunk_mst_label.config(text=f"{float(v):.2f}"))
        scale_mst.pack(side=LEFT)
        self.trunk_mst_label = Label(
            frame_mst, text=f"{self.min_step_time_var.get():.2f}", width=5)
        self.trunk_mst_label.pack(side=LEFT, padx=2)

        # --- 実行ボタンエリア ---
        button_frame = Frame(param_area_frame)
        button_frame.pack(side=LEFT, padx=20, fill=tk.Y, anchor=tk.N)
        self.run_button = Button(button_frame, text="1. 同期実行 & 全解析 (下腿ベース)",
                                 command=self.run_analysis_pipeline, width=25, height=2, state=DISABLED)
        self.run_button.pack(pady=5)
        self.update_trunk_ic_button = Button(
            button_frame, text="2. 体幹IC 更新&プロット", command=self.update_trunk_ic_results, width=25, state=DISABLED)
        self.update_trunk_ic_button.pack(pady=5)

        # 初期データロード
        self.status_label.config(text="最新ファイル検索中...")
        self.master.update_idletasks()
        self.master.after(100, self.load_and_plot_initial_data)

    def load_and_plot_initial_data(self):
        """最新CSVをロードし、初期プロットを表示、ボタンを有効化"""
        print("\n[準備] 最新のCSVデータを検索中...")
        self.input_file_path = find_latest_csv_file(DATA_FOLDER)
        if self.input_file_path is None:
            messagebox.showerror(
                "エラー", f"CSVファイルが見つかりません ({DATA_FOLDER.resolve()})", parent=self.master)
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
            plot_len = min(len(sync_l_full), len(sync_r_full),
                           len(sync_t_full), NUM_SAMPLES_TO_PLOT)
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
            self.run_button.config(state=DISABLED)

    def run_analysis_pipeline(self):
        """解析パイプライン全体を実行するメソッド"""
        self.run_button.config(state=DISABLED)
        self.update_trunk_ic_button.config(state=DISABLED)
        self.status_label.config(text="解析処理実行中...")
        self.master.update_idletasks()

        # インスタンス変数をリセット/初期化
        self.sync_data_df = None
        self.trunk_ic_results = None
        self.gait_events_shank_steady = None
        self.time_vector = None
        # ローカル変数も初期化
        sync_data_df = None
        gait_events_df_shank_all = None
        filtered_signals_gyro = {}
        gait_events_df_shank_segmented = pd.DataFrame()
        gait_events_df_shank_steady = pd.DataFrame()
        ic_events_df_trunk = pd.DataFrame()

        try:
            print("\n========================================")
            print("=== 解析パイプライン実行 ===")
            user_peak_height = self.height_var.get()
            user_peak_prominence = self.prominence_var.get()
            user_peak_distance = self.distance_var.get()
            print(
                f"  使用 find_peaks パラメータ (同期用 AccY): h={user_peak_height:.2f}, p={user_peak_prominence:.2f}, d={user_peak_distance}")

            # === ステップ 1: 前処理 ===
            print("\n[ステップ1] 全IMUデータの前処理・同期...")
            preprocess_result = preprocess_and_sync_imu_data(
                data_file=self.input_file_path, rows_to_skip=ROWS_TO_SKIP, sampling_interval_ms=SAMPLING_INTERVAL_MS,
                right_prefix=RIGHT_PREFIX, left_prefix=LEFT_PREFIX, trunk_prefix=TRUNK_PREFIX,
                sync_axis_suffix=SYNC_SIGNAL_SUFFIX, align_gyro_suffix=ALIGN_TARGET_SUFFIX,
                peak_height=user_peak_height, peak_prominence=user_peak_prominence, peak_distance=user_peak_distance
            )
            if preprocess_result is None or not isinstance(preprocess_result, tuple) or len(preprocess_result) != 3:
                messagebox.showerror("エラー", "前処理失敗(戻り値不正)", parent=self.master)
                return  # finallyへ
            sync_data_df, lags_info, self.sampling_rate = preprocess_result
            if sync_data_df is None or sync_data_df.empty:
                messagebox.showerror("エラー", "前処理失敗(データ空)", parent=self.master)
                return  # finallyへ
            self.sync_data_df = sync_data_df  # インスタンス変数に保持
            # self.update_trunk_ic_button.config(state=NORMAL) # 更新ボタン有効化は最後に行う

            lag_L_R = lags_info.get('L_vs_R', 'N/A')
            lag_T_R = lags_info.get('T_vs_R', 'N/A')
            result_message = f"前処理完了。\nLag (R基準, {SYNC_SIGNAL_SUFFIX}ピーク): L={lag_L_R}, T={lag_T_R} [samples]"
            print(f"\n[成功] {result_message}")
            base_filename = self.input_file_path.stem
            output_sync_file = OUTPUT_FOLDER / \
                (base_filename + OUTPUT_SUFFIX_SYNC)
            print(f"\n[ステップ1.1] 同期済み全IMUデータ保存中...")
            save_results(output_sync_file, self.sync_data_df, "同期済みIMUデータ")

            # === ステップ 2: 歩行周期同定 (下腿 Gyro Z ベース) ===
            gait_events_df_shank_all = None
            if self.sync_data_df is not None:
                print("\n[ステップ2] 歩行周期の同定 (下腿 Gyro Z) を実行中...")
                current_swing_threshold = SHANK_SWING_THRESHOLD  # 設定値を使用
                gait_events_result_shank = identify_gait_cycles(
                    sync_gyro_df=self.sync_data_df,
                    sampling_rate_hz=self.sampling_rate,
                    swing_threshold=current_swing_threshold
                )
                if gait_events_result_shank is not None and isinstance(gait_events_result_shank, dict):
                    gait_events_df_shank_all = gait_events_result_shank.get(
                        "events_df")
                    filtered_signals_gyro = gait_events_result_shank.get(
                        "filtered_signals", {})
                    self.time_vector = gait_events_result_shank.get(
                        "time_vector")
                if gait_events_df_shank_all is None or gait_events_df_shank_all.empty:
                    print("  下腿ベース周期検出不可")
                else:
                    print(f"  {len(gait_events_df_shank_all)} 個の下腿ベース候補検出")

            # === ステップ 2bis: 歩行周期同定 (体幹 Acc Z/Y ベース IC+LR) ===
            if self.sync_data_df is not None and self.time_vector is not None:
                print("\n[ステップ2bis] IC同定(左右判定/補間含む, 体幹Acc/Gyro) を実行中...")
                trunk_ic_h = self.trunk_ic_height_var.get()
                trunk_ic_p = self.trunk_ic_prominence_var.get()
                trunk_lr_th = self.trunk_lr_thresh_var.get()
                min_step = self.min_step_time_var.get()
                ic_detection_result_trunk = identify_ics_from_trunk_accel(
                    sync_data_df=self.sync_data_df, sampling_rate_hz=self.sampling_rate,
                    ic_peak_height=trunk_ic_h, ic_peak_prominence=trunk_ic_p, lr_gyro_threshold=trunk_lr_th, min_step_time_sec=min_step
                )
                self.trunk_ic_results = ic_detection_result_trunk  # インスタンス変数に保持
                if self.trunk_ic_results is not None and isinstance(self.trunk_ic_results, dict):
                    ic_events_df_trunk = self.trunk_ic_results.get(
                        "ic_events_df")
                    if ic_events_df_trunk is None or ic_events_df_trunk.empty:
                        print("  体幹IC検出失敗")
                    else:
                        print(f"  {len(ic_events_df_trunk)} 個の体幹IC検出")
                        # --- ステップ2bis.2: 体幹ICを基に歩行区間を分割 (IC_Time間隔 ≥ 2秒) ---
                        print("\n[ステップ2bis.2] 体幹ICを基に歩行区間分割 (IC_Time間隔 ≥2秒)")
                        trunk_segmented = segment_walking_trials(
                            events_df=ic_events_df_trunk,
                            max_interval_sec=0.7,      # 2秒以上空いたら別セグメント
                            min_ics_per_trial=1        # 最低1ICを残す
                        )
                        if trunk_segmented is None or trunk_segmented.empty:
                            print("  体幹ICに基づく歩行区間分割結果なし")
                        else:
                            # 各 Trial_ID ごとの IC 個数を数えて、16 以下のグループを削除
                            before_cnt = len(trunk_segmented)
                            trunk_filtered = trunk_segmented.groupby('Trial_ID') \
                                                            .filter(lambda g: len(g) > 16)
                            removed = before_cnt - len(trunk_filtered)
                            print(
                                f"  小セグメント(≤16 IC)を {removed} イベント分除去 → 残り {len(trunk_filtered)} イベント")

                            # 以降の処理に使う DataFrame を差し替え
                            ic_events_df_trunk = trunk_filtered
                            n_seg = ic_events_df_trunk['Trial_ID'].nunique()
                            print(f"  → {n_seg} 個の体幹歩行セグメントが残っています")
                            # --- 各セグメントの先頭5歩・末尾8歩を除外して定常歩行区間を抽出 ---
                            ic_before = len(ic_events_df_trunk)
                            ic_events_df_trunk = trim_trial_ends(
                                df_segmented=ic_events_df_trunk,
                                n_start=TRUNK_NUM_ICS_REMOVE_START,
                                n_end=TRUNK_NUM_ICS_REMOVE_END
                            )
                            removed_ic = ic_before - len(ic_events_df_trunk)
                            print(
                                f"  定常歩行区間抽出: {removed_ic} 件除去 → 残り {len(ic_events_df_trunk)} 件")

                        output_gait_file_trunk_ic = OUTPUT_FOLDER / \
                            (base_filename + OUTPUT_SUFFIX_GAIT_TRUNK_IC)
                        print(f"\n[ステップ2bis.1] 体幹ICイベントデータ保存中...")
                        save_results(output_gait_file_trunk_ic,
                                     ic_events_df_trunk, "ICイベントデータ(体幹)")
                        print("\n--- 検出体幹IC(最初の5件) ---")
                        print(
                            ic_events_df_trunk[['Leg', 'Cycle', 'IC_Time']].head().to_string())
                        print("---")
                        # プロット実行
                        filtered_ap = self.trunk_ic_results.get(
                            "filtered_ap_signal")
                        filtered_lr = self.trunk_ic_results.get(
                            "filtered_lr_gyro_signal")
                        # preserve originals for shank later
                        orig_time_vector = self.time_vector
                        orig_filtered_ap = filtered_ap
                        orig_filtered_lr = filtered_lr

                        # use locals for trunk truncation
                        tv_trunk = orig_time_vector
                        fa_trunk = orig_filtered_ap
                        fl_trunk = orig_filtered_lr

                        # --- トランクVT加速度のバンドパスフィルタリング (0.1–20Hz) ---
                        vt_raw = self.sync_data_df[f'{TRUNK_PREFIX}_Acc_Y_aligned'].values
                        b, a = butter(4, [0.1, 20], btype='band', fs=self.sampling_rate)
                        filtered_vt = filtfilt(b, a, vt_raw)
                        # --- トランクML加速度のバンドパスフィルタリング (0.1–20Hz) ---
                        ml_raw = self.sync_data_df[f'{TRUNK_PREFIX}_Acc_X_aligned'].values
                        filtered_ml = filtfilt(b, a, ml_raw)

                        # --- ステップ2bis.3: Trunk Harmonic Ratio (AP/VT/ML) の計算 ---
                        print("\n[ステップ2bis.3] Trunk Harmonic Ratio (AP/VT/ML) の計算中...")
                        hr_list_ap, hr_list_vt, hr_list_ml = [], [], []
                        for trial_id, grp in ic_events_df_trunk.groupby('Trial_ID', sort=False):
                            times = grp.sort_values('IC_Time')['IC_Time'].values
                            for i in range(len(times) - 1):
                                t0, t1 = times[i], times[i+1]
                                idx0 = np.searchsorted(self.time_vector, t0, side='left')
                                idx1 = np.searchsorted(self.time_vector, t1, side='left')
                                seg_ap = fa_trunk[idx0:idx1]
                                seg_vt = filtered_vt[idx0:idx1]
                                seg_ml = filtered_ml[idx0:idx1]
                                hr_ap = calculate_harmonic_ratio(seg_ap, self.sampling_rate, axis='AP')
                                hr_vt = calculate_harmonic_ratio(seg_vt, self.sampling_rate, axis='VT')
                                hr_ml = calculate_harmonic_ratio(seg_ml, self.sampling_rate, axis='ML')
                                hr_list_ap.append(hr_ap)
                                hr_list_vt.append(hr_vt)
                                hr_list_ml.append(hr_ml)
                        mean_hr_ap = float(np.nanmean(hr_list_ap)) if hr_list_ap else np.nan
                        mean_hr_vt = float(np.nanmean(hr_list_vt)) if hr_list_vt else np.nan
                        mean_hr_ml = float(np.nanmean(hr_list_ml)) if hr_list_ml else np.nan

                        # Calculate integrated HR (iHR) per cycle
                        ihr_list_ap, ihr_list_vt, ihr_list_ml = [], [], []
                        for trial_id, grp in ic_events_df_trunk.groupby('Trial_ID', sort=False):
                            times = grp.sort_values('IC_Time')['IC_Time'].values
                            for i in range(len(times) - 1):
                                t0, t1 = times[i], times[i+1]
                                idx0 = np.searchsorted(self.time_vector, t0, side='left')
                                idx1 = np.searchsorted(self.time_vector, t1, side='left')
                                seg_ap = fa_trunk[idx0:idx1]
                                seg_vt = filtered_vt[idx0:idx1]
                                seg_ml = filtered_ml[idx0:idx1]
                                ihr_ap = calculate_integrated_harmonic_ratio(seg_ap, self.sampling_rate, axis='AP')
                                ihr_vt = calculate_integrated_harmonic_ratio(seg_vt, self.sampling_rate, axis='VT')
                                ihr_ml = calculate_integrated_harmonic_ratio(seg_ml, self.sampling_rate, axis='ML')
                                ihr_list_ap.append(ihr_ap)
                                ihr_list_vt.append(ihr_vt)
                                ihr_list_ml.append(ihr_ml)
                        mean_ihr_ap = float(np.nanmean(ihr_list_ap)) if ihr_list_ap else np.nan
                        mean_ihr_vt = float(np.nanmean(ihr_list_vt)) if ihr_list_vt else np.nan
                        mean_ihr_ml = float(np.nanmean(ihr_list_ml)) if ihr_list_ml else np.nan
                        print(f"  → iHR_AP: {mean_ihr_ap:.2f}%, iHR_VT: {mean_ihr_vt:.2f}%, iHR_ML: {mean_ihr_ml:.2f}%")

                        print(f"  → HR_AP: {mean_hr_ap:.3f}, HR_VT: {mean_hr_vt:.3f}, HR_ML: {mean_hr_ml:.3f}")
                        print(f"[STEP 2bis.3] Final Harmonic Ratios - AP: {mean_hr_ap:.3f}, VT: {mean_hr_vt:.3f}, ML: {mean_hr_ml:.3f}")
                        # 結果辞書に追加
                        results_all = {}
                        results_all['HR_AP'] = mean_hr_ap
                        results_all['HR_VT'] = mean_hr_vt
                        results_all['HR_ML'] = mean_hr_ml
                        results_all['iHR_AP'] = mean_ihr_ap
                        results_all['iHR_VT'] = mean_ihr_vt
                        results_all['iHR_ML'] = mean_ihr_ml

                        if fa_trunk is not None and fl_trunk is not None and tv_trunk is not None:
                            self.plot_trunk_ics(
                                ic_events_df_trunk,
                                fa_trunk, fl_trunk, tv_trunk
                            )
                        else:
                            print("警告: 体幹ICプロット用データ不足")

            # === ステップ 2.1-3.1: 下腿ベースの解析 ===
            if gait_events_df_shank_all is not None and not gait_events_df_shank_all.empty:
                print("\n[ステップ2.1] 歩行トライアルの自動分割 (下腿ベース) を実行中...")
                gait_events_df_shank_segmented = segment_walking_trials(
                    gait_events_df_shank_all, MAX_IC_INTERVAL_SEC, MIN_ICS_PER_TRIAL)
                # --- per-trial stumble truncation for shank events ---
                cleaned_segments = []
                for trial_id, trial_grp in gait_events_df_shank_segmented.groupby('Trial_ID'):
                    # define time window for this trial
                    start_time = trial_grp['IC_Time'].min()
                    end_time = trial_grp['IC_Time'].max()
                    # convert to sample indices on the global time_vector
                    idx_start = np.searchsorted(
                        self.time_vector, start_time, side='left')
                    idx_end = np.searchsorted(
                        self.time_vector, end_time,   side='right')
                    # slice signals for this trial
                    tv_trial = self.time_vector[idx_start:idx_end]
                    sig_L = filtered_signals_gyro.get('L', np.array([]))[
                        idx_start:idx_end]
                    sig_R = filtered_signals_gyro.get('R', np.array([]))[
                        idx_start:idx_end]
                    # detect stumble per leg within trial window
                    t_L = detect_stumble(
                        tv_trial, sig_L, neg_thresh=-30, pos_thresh=70, window_sec=0.3)
                    t_R = detect_stumble(
                        tv_trial, sig_R, neg_thresh=-30, pos_thresh=70, window_sec=0.3)
                    # pick earliest stumble time if any
                    t_cand = [t for t in (t_L, t_R) if t is not None]
                    if t_cand:
                        t_stumble = min(t_cand)
                        # only apply if at least 3 IC events precede the stumble
                        pre_ic = trial_grp[trial_grp['IC_Time'] < t_stumble]
                        if len(pre_ic) >= 3:
                            # filter out events after stumble
                            trial_grp = trial_grp[pre_ic.index]
                            trial_grp = trial_grp[trial_grp['IC_Time']
                                                  < t_stumble]
                            if 'FO_Time' in trial_grp.columns:
                                trial_grp = trial_grp[trial_grp['FO_Time']
                                                      < t_stumble]
                        else:
                            # do not truncate this trial (stumble too early, likely noise)
                            pass
                    cleaned_segments.append(trial_grp)
                gait_events_df_shank_segmented = pd.concat(
                    cleaned_segments).reset_index(drop=True)
                print(
                    f"  [Shank] Per-trial stumble truncation applied, remaining events: {len(gait_events_df_shank_segmented)}")
                if not gait_events_df_shank_segmented.empty:
                    print("\n[ステップ2.2] 定常歩行部分の抽出（前後除外, 下腿ベース）を実行中...")
                    gait_events_df_shank_steady = trim_trial_ends(
                        gait_events_df_shank_segmented, NUM_ICS_REMOVE_START, NUM_ICS_REMOVE_END)
                    self.gait_events_shank_steady = gait_events_df_shank_steady  # 結果保持
                    if not self.gait_events_shank_steady.empty:
                        # 保存
                        output_gait_file = OUTPUT_FOLDER / \
                            (base_filename + OUTPUT_SUFFIX_GAIT_TRIMMED)
                        print(f"\n[ステップ2.3] 定常歩行周期データ(下腿ベース)保存中...")
                        save_results(
                            output_gait_file, self.gait_events_shank_steady, "定常歩行周期(下腿GyroZ)")
                        print("\n--- 定常歩行周期(下腿 最初の5件) ---")
                        print(self.gait_events_shank_steady[[
                              'Leg', 'Trial_ID', 'Cycle', 'IC_Time', 'FO_Time']].head().to_string())
                        print("---")
                        # プロット (下腿IC/FO)
                        self.plot_gait_events(
                            self.gait_events_shank_steady, filtered_signals_gyro, self.time_vector)

                        # パラメータ計算
                        print("\n[ステップ3] 各種歩行パラメータ計算中 (下腿ベース)...")
                        results_all = {}
                        temporal_params = calculate_temporal_params(
                            self.gait_events_shank_steady)
                        results_all.update(temporal_params or {})
                        if filtered_signals_gyro and self.time_vector is not None:
                            kinematic_params = calculate_kinematic_params(
                                self.gait_events_shank_steady, filtered_signals_gyro, self.time_vector, self.sampling_rate)
                            results_all.update(kinematic_params or {})
                        else:
                            print("警告: Kinematic params計算信号なし")
                        pci_params = calculate_pci(
                            self.gait_events_shank_steady)
                        results_all.update(pci_params or {})
                        print("対称性指数 計算中...")
                        symmetry_results = {}
                        param_pairs = {'Stride_Time_%': ('Mean_Stride_Time_L_s', 'Mean_Stride_Time_R_s'), 'Swing_Time_%': ('Mean_Swing_Time_L_s', 'Mean_Swing_Time_R_s'), 'Stance_Time_%': (
                            'Mean_Stance_Time_L_s', 'Mean_Stance_Time_R_s'), 'Max_Swing_Vel_%': ('Mean_Max_Swing_Vel_L', 'Mean_Max_Swing_Vel_R'), 'Peak_Stance_Vel_%': ('Mean_Peak_Stance_Vel_L', 'Mean_Peak_Stance_Vel_R')}
                        for si_name, (key_L, key_R) in param_pairs.items():
                            val_L, val_R = results_all.get(
                                key_L), results_all.get(key_R)
                            symmetry_results[f'SI_{si_name}'] = calculate_symmetry_index(
                                val_L, val_R) if pd.notna(val_L) and pd.notna(val_R) else np.nan
                        results_all.update(symmetry_results)
                        print("対称性指数 計算完了。")

                        # 結果表示・保存
                        print(f"\n--- 全計算結果 (下腿ベース) ---")
                        if results_all:
                            for key, value in sorted(results_all.items()):
                                print(f"  {key}: {value:.3f}" if isinstance(
                                    value, (float, np.floating)) else f"  {key}: {value}")
                            print("--------------------")
                            # Trunk Harmonic Ratios (from earlier computation)
                            print(f"  HR_AP: {mean_hr_ap:.3f}, HR_VT: {mean_hr_vt:.3f}, HR_ML: {mean_hr_ml:.3f}")
                            # Integrated Harmonic Ratios (iHR)
                            print(f"  iHR_AP: {mean_ihr_ap:.2f}%, iHR_VT: {mean_ihr_vt:.2f}%, iHR_ML: {mean_ihr_ml:.2f}%")
                            # Add iHRs to results_all for CSV
                            results_all['iHR_AP'] = mean_ihr_ap
                            results_all['iHR_VT'] = mean_ihr_vt
                            results_all['iHR_ML'] = mean_ihr_ml
                            output_all_params_file = OUTPUT_FOLDER / \
                                (base_filename + OUTPUT_ALL_PARAMS_FILE)
                            print(f"\n[ステップ3.1] 全パラメータ(下腿ベース)保存中...")
                            save_results(output_all_params_file, pd.DataFrame(
                                [results_all]), "全歩行パラメータ(下腿ベース)")
                            # Pop-up message update (主要なものをいくつか表示)
                            pci_val = results_all.get('PCI')  # 辞書からPCI値を取得
                            # 値が存在しNaNでなければ小数点以下2桁、なければ'N/A'
                            pci_str = f"{pci_val:.2f}" if pd.notna(
                                pci_val) else 'N/A'

                            cad_val = results_all.get(
                                'Cadence_steps_per_min')  # ケイデンス値を取得
                            cad_str = f"{cad_val:.1f}" if pd.notna(
                                cad_val) else 'N/A'

                            si_stride = results_all.get(
                                'SI_Stride_Time_%')  # Stride Time SIを取得
                            si_str = f"{si_stride:.1f}" if pd.notna(
                                si_stride) else 'N/A'
                            messagebox.showinfo(
                                "パラメータ計算完了", f"計算完了(下腿)。\nPCI: {pci_str}, Cadence: {cad_str}\nStride SI: {si_str}%", parent=self.master)
                        else:
                            print("警告: 計算パラメータなし。")
                    else:
                        print("下腿ベース 定常歩行なし")
                else:
                    print("下腿ベース 有効トライアルなし")
            else:
                print("下腿ベース 周期候補なし")

            # === ステップ 4: 同期済み角速度の最終プロット ===
            self.plot_final_synchronized_data(self.sync_data_df)

            print("\n--- 解析パイプライン 完了 ---")
            self.status_label.config(text="解析完了")

        except Exception as e:
            print(f"\n[エラー] 解析パイプライン実行中にエラー: {e}")
            messagebox.showerror(
                "実行時エラー", f"解析中にエラー:\n{e}", parent=self.master)
            self.status_label.config(text="エラー発生")
        finally:
            self.run_button.config(state=NORMAL)
            if self.sync_data_df is not None and not self.sync_data_df.empty:
                self.update_trunk_ic_button.config(state=NORMAL)
            self.master.update_idletasks()

    # ★★★ 新しいメソッド: 体幹IC検出の更新とプロット ★★★

    def update_trunk_ic_results(self):
        print("\n--- [体幹IC 更新&プロット] ボタン実行 ---")
        self.update_trunk_ic_button.config(state=DISABLED)
        self.status_label.config(text="体幹IC更新中...")
        self.master.update_idletasks()
        if self.sync_data_df is None or self.sync_data_df.empty:
            messagebox.showerror(
                "エラー", "先に「同期実行 & 全解析」実行要", parent=self.master)
            self.update_trunk_ic_button.config(state=NORMAL)
            self.status_label.config(text="準備未完了")
            return
        if self.sampling_rate is None:
            messagebox.showerror("エラー", "サンプリングレート不明", parent=self.master)
            self.update_trunk_ic_button.config(state=NORMAL)
            return
        # time_vector は self.time_vector を使う。 load_and_plot で設定されているはず。なければエラー。
        if self.time_vector is None:
            if 'time_aligned_sec' in self.sync_data_df.columns:
                self.time_vector = self.sync_data_df['time_aligned_sec'].values
            else:
                messagebox.showerror("エラー", "時間ベクトル不明")
                self.update_trunk_ic_button.config(state=NORMAL)
                return

        try:
            trunk_ic_h = self.trunk_ic_height_var.get()
            trunk_ic_p = self.trunk_ic_prominence_var.get()
            trunk_lr_th = self.trunk_lr_thresh_var.get()
            min_step = self.min_step_time_var.get()
            print(
                f"  使用する体幹ICパラメータ: h={trunk_ic_h:.2f}, p={trunk_ic_p:.2f}, lr_th={trunk_lr_th:.1f}, min_step={min_step:.2f}")

            # 体幹IC検出関数を再実行し、結果をインスタンス変数に上書き
            self.trunk_ic_results = identify_ics_from_trunk_accel(
                sync_data_df=self.sync_data_df, sampling_rate_hz=self.sampling_rate,
                ap_axis_col='T_Acc_Z_aligned', lr_gyro_axis_col='T_Gyro_Y_aligned',
                filter_cutoff_acc=TRUNK_IC_FILTER_CUTOFF,
                ic_peak_height=trunk_ic_h, ic_peak_prominence=trunk_ic_p, ic_peak_distance_ms=TRUNK_IC_PEAK_DISTANCE_MS,
                min_step_time_sec=min_step, lr_gyro_threshold=trunk_lr_th)

            # 結果のチェックとプロット
            if self.trunk_ic_results is not None and isinstance(self.trunk_ic_results, dict):
                ic_events_df_trunk = self.trunk_ic_results.get("ic_events_df")
                filtered_ap = self.trunk_ic_results.get("filtered_ap_signal")
                filtered_lr = self.trunk_ic_results.get(
                    "filtered_lr_gyro_signal")
                # time_vector は self.time_vector を使用

                # --- stumble detection and cut everything after ---
                stumble_time = detect_stumble(self.time_vector, filtered_lr,
                                              neg_thresh=-30, pos_thresh=70, window_sec=0.3)
                if stumble_time is not None:
                    cut_idx = np.searchsorted(
                        self.time_vector, stumble_time, side='left')
                    # truncate time and signals
                    self.time_vector = self.time_vector[:cut_idx]
                    filtered_ap = filtered_ap[:cut_idx]
                    filtered_lr = filtered_lr[:cut_idx]
                    # remove IC events after stumble
                    ic_events_df_trunk = ic_events_df_trunk[ic_events_df_trunk['IC_Time'] < stumble_time]
                    print(
                        f"[Stumble] Detected at {stumble_time:.2f}s → remaining IC: {len(ic_events_df_trunk)}")

                if ic_events_df_trunk is None or ic_events_df_trunk.empty:
                    print("体幹IC検出失敗(更新)")
                    messagebox.showwarning(
                        "IC同定(更新)", "指定パラメータでIC検出失敗", parent=self.master)
                else:
                    print(f"{len(ic_events_df_trunk)} 個のIC検出(更新)。プロット表示...")
                    if filtered_ap is not None and filtered_lr is not None and self.time_vector is not None:
                        # ★ プロット関数呼び出し ★
                        self.plot_trunk_ics(
                            ic_events_df_trunk, filtered_ap, filtered_lr, self.time_vector)
                        # ★ 更新時にもCSV保存するなら追加 ★
                        # base_filename = self.input_file_path.stem
                        # output_trunk_ic = OUTPUT_FOLDER / (base_filename + OUTPUT_SUFFIX_GAIT_TRUNK_IC)
                        # save_results(output_trunk_ic, ic_events_df_trunk, "体幹ICイベント(更新版)")
                    else:
                        print("警告: 更新後の体幹ICプロット用データ不足")
            else:
                print("エラー: 体幹IC検出関数から予期せぬ結果")
            self.status_label.config(text="体幹IC更新完了")

        except Exception as e:
            print(f"\n[エラー] {e}")
            messagebox.showerror(
                "実行時エラー", f"体幹IC更新中にエラー:\n{e}", parent=self.master)
            self.status_label.config(text="体幹IC更新エラー")
        finally:
            self.update_trunk_ic_button.config(state=NORMAL)
            self.master.update_idletasks()

    # --- IC/FO イベントプロット用メソッド (下腿GyroZ用) ---
    def plot_gait_events(self, gait_events_df, filtered_signals, time_vector):
        print("\n[ステップ2.4] IC/FO イベント (下腿GyroZ定常区間) をプロットします...")
        if gait_events_df is None:
            print("プロットするイベントデータなし")
            return
        if gait_events_df.empty:
            # イベントが空でも信号波形はプロットする
            print("警告: イベントデータが空です。マーカーは表示されません。")
        try:
            fig_events, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
            fig_events.suptitle(
                f'検出歩行イベント(下腿GyroZ定常) - {self.input_file_path.name}')
            plot_successful = False
            if time_vector is None or len(time_vector) == 0:
                # ... (時間ベクトル再生成) ...
                sig_len = len(filtered_signals.get('L', []))
            for i, leg in enumerate(['L', 'R']):
                ax = axes[i]
                filt_signal = filtered_signals.get(leg)
                if filt_signal is None or len(filt_signal) != len(time_vector):
                    print(f"警告:{leg}脚プロットデータ不整合")
                    ax.set_title(f'{leg}脚-データ不整合')
                    continue
                ax.plot(time_vector, filt_signal,
                        label=f'{leg} Gyro Z (Filtered)', alpha=0.7)
                leg_events = gait_events_df[gait_events_df['Leg'] == leg]
                ic_times = leg_events['IC_Time'].dropna().values
                fo_times = leg_events['FO_Time'].dropna().values
                ic_indices_plot = np.searchsorted(
                    time_vector, ic_times, side='left')
                fo_indices_plot = np.searchsorted(
                    time_vector, fo_times, side='left')
                ic_indices_plot = ic_indices_plot[ic_indices_plot < len(
                    filt_signal)]
                fo_indices_plot = fo_indices_plot[fo_indices_plot < len(
                    filt_signal)]
                valid_ic_indices = ic_indices_plot[ic_indices_plot < len(
                    filt_signal)]
                valid_fo_indices = fo_indices_plot[fo_indices_plot < len(
                    filt_signal)]
                if len(valid_ic_indices) > 0:
                    ax.plot(time_vector[valid_ic_indices],
                            filt_signal[valid_ic_indices], 'ro', ms=6, label='IC', ls='None')
                if len(valid_fo_indices) > 0:
                    ax.plot(time_vector[valid_fo_indices], filt_signal[valid_fo_indices],
                            'gx', ms=8, mew=2, label='FO', ls='None')
                ax.set_title(f'{leg} 脚')
                ax.set_ylabel('角速度(単位?)')
                ax.legend(loc='upper right')
                ax.grid(True)
                plot_successful = True
            if plot_successful:
                axes[1].set_xlabel('時間(s)')
                plt.tight_layout(rect=[0, 0.03, 1, 0.96])
                plt.show(block=False)
            else:
                plt.close(fig_events)
                print("下腿イベントグラフプロット不可")
        except Exception as e:
            print(f"エラー:下腿IC/FOプロットエラー:{e}")

    # --- 体幹ICイベントプロット用メソッド ---

    def plot_trunk_ics(self, ic_events_df, filtered_ap_signal, filtered_lr_gyro_signal, time_vector):
        print("[DEBUG] plot_trunk_ics 入力確認 →",
              "IC events rows:", len(ic_events_df),
              "signal_ap len:", len(filtered_ap_signal),
              "time_vector len:", len(time_vector))

        print("\n[ステップ2.4(暫定)] IC イベント (体幹Acc/Gyroベース) をグラフにプロットします...")
        if ic_events_df is None or ic_events_df.empty:
            print("プロットICなし")
            return
        if filtered_ap_signal is None or filtered_lr_gyro_signal is None or time_vector is None:
            print("プロット用信号/時間ベクトルなし")
            return
        if len(filtered_ap_signal) != len(time_vector) or len(filtered_lr_gyro_signal) != len(time_vector):
            print("警告:信号/時間ベクトル長不整合")
            return
        try:
            fig_events, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
            fig_events.suptitle(f'検出ICイベント(体幹) - {self.input_file_path.name}')
            plot_successful = False
            signal_ap = filtered_ap_signal
            signal_lr = filtered_lr_gyro_signal
            # 上段: AP Acc + IC Markers
            ax_ap = axes[0]
            ax_ap.plot(time_vector, signal_ap,
                       label='Trunk AP Acc (Filtered)', alpha=0.7, color='k')
            # plot IC markers at detected IC_Time
            for leg, color, marker in [('L', 'blue', 'o'), ('R', 'red', 'o')]:
                leg_events = ic_events_df[ic_events_df['Leg'] == leg]
                ic_times = leg_events['IC_Time'].dropna().values
                ic_idx = np.searchsorted(time_vector, ic_times, side='left')
                valid = ic_idx[(ic_idx >= 0) & (ic_idx < len(signal_ap))]
                if len(valid) > 0:
                    ax_ap.plot(time_vector[valid], signal_ap[valid],
                               linestyle='None', marker=marker, color=color, ms=6, label=f'IC({leg})')
            # unknown leg
            unk_times = ic_events_df[ic_events_df['Leg']
                                     == 'Unknown']['IC_Time'].dropna().values
            unk_idx = np.searchsorted(time_vector, unk_times, side='left')
            valid_unk = unk_idx[(unk_idx >= 0) & (unk_idx < len(signal_ap))]
            if len(valid_unk) > 0:
                ax_ap.plot(time_vector[valid_unk], signal_ap[valid_unk],
                           linestyle='None', marker='x', color='gray', ms=6, label='IC(Unknown)')
            ax_ap.set_title('体幹 AP(Z) 加速度 – 検出IC位置')
            ax_ap.set_ylabel('加速度 (units)')
            ax_ap.legend(loc='upper right')
            ax_ap.grid(True)
            plot_successful = True
            # 下段: LR Gyro + IC Markers
            ax_lr = axes[1]
            ax_lr.plot(time_vector, signal_lr,
                       label=f'Trunk Yaw Gyro(Y,Filtered)', alpha=0.7, color='k')
            ax_lr.axhline(0, color='gray', ls='--', lw=0.5)
            for leg, color, marker in [('L', 'blue', 'o'), ('R', 'red', 'o')]:
                leg_events = ic_events_df[ic_events_df['Leg'] == leg]
                ic_times = leg_events['IC_Time'].dropna().values
                ic_idx = np.searchsorted(time_vector, ic_times, side='left')
                valid = ic_idx[(ic_idx >= 0) & (ic_idx < len(signal_lr))]
                if len(valid) > 0:
                    ax_lr.plot(time_vector[valid], signal_lr[valid],
                               linestyle='None', marker=marker, color=color, ms=6, label=f'IC({leg})')
            # unknown leg
            unk_times = ic_events_df[ic_events_df['Leg']
                                     == 'Unknown']['IC_Time'].dropna().values
            unk_idx = np.searchsorted(time_vector, unk_times, side='left')
            valid_unk = unk_idx[(unk_idx >= 0) & (unk_idx < len(signal_lr))]
            if len(valid_unk) > 0:
                ax_lr.plot(time_vector[valid_unk], signal_lr[valid_unk],
                           linestyle='None', marker='x', color='gray', ms=6, label='IC(Unknown)')
            ax_lr.set_title('体幹 Yaw(Y軸) Gyro と検出IC (LR判定用)')
            ax_lr.set_ylabel('角速度(単位?)')
            ax_lr.legend(loc='upper right')
            ax_lr.grid(True)
            if plot_successful:
                ax_lr.set_xlabel('時間(s)')
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.show(block=False)
            else:
                plt.close(fig_events)
                print("ICプロット不可")
        except Exception as e:
            print(f"エラー:ICプロット中:{e}")  # ... close fig ...

    # --- 同期済み角速度の最終プロット用メソッド ---

    def plot_final_synchronized_data(self, sync_data_df):
        if sync_data_df is not None:
            print(f"\n[ステップ4] 同期済み{ALIGN_TARGET_SUFFIX}グラフ (先頭) 表示...")
            try:
                if not sync_data_df.empty:
                    num_samples = 1000
                    df_short = sync_data_df.head(num_samples)
                    plot_len = len(df_short)
                    plt.figure(figsize=(12, 6))
                    l_col = f'L{ALIGN_TARGET_SUFFIX}_aligned'
                    r_col = f'R{ALIGN_TARGET_SUFFIX}_aligned'
                    t_col = 'time_aligned_sec'
                    if l_col in df_short.columns and r_col in df_short.columns and t_col in df_short.columns:
                        plt.plot(df_short[t_col], df_short[l_col],
                                 label=f'L {ALIGN_TARGET_SUFFIX}')
                        plt.plot(df_short[t_col], df_short[r_col],
                                 label=f'R {ALIGN_TARGET_SUFFIX}', ls='--', alpha=0.8)
                        plt.title(
                            f'同期済み角速度 - 先頭 {plot_len} Samples ({self.input_file_path.name})')
                        plt.xlabel('Time(s)')
                        plt.ylabel('AngVel(dps?)')
                        plt.legend()
                        plt.grid(True)
                        plt.show(block=False)
                    else:
                        print(f"警告:最終プロット列なし")
                else:
                    print("警告:同期データ空")
            except Exception as e:
                print(f"エラー:最終グラフ表示エラー:{e}")


# --- メイン実行ブロック ---
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    app_window = Toplevel(root)
    app = GaitAnalysisApp(app_window)
    app_window.protocol("WM_DELETE_WINDOW", root.destroy)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\nCtrl+C により中断")
    finally:
        print("\n=== アプリケーション終了 ===")
