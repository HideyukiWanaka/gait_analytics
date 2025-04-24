# run_gait_analysis.py

# --- 各機能ファイルをインポート ---
# GUIおよびMatplotlib関連
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
TRUNK_LR_GYRO_THRESHOLD = 5.0  # Gyro Y 左右判定閾値

# 下腿IC/FO検出用パラメータ (identify_gait_cyclesのデフォルト値を使用)
SHANK_SWING_THRESHOLD = 100  # 下腿GyroZ用

# 自動トライアル分割用パラメータ
MAX_IC_INTERVAL_SEC = 2.0
MIN_ICS_PER_TRIAL = 11

# 定常歩行抽出用パラメータ
NUM_ICS_REMOVE_START = 3
NUM_ICS_REMOVE_END = 5


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
    """Trial_ID と Leg でグループ化し、各グループの先頭n_start個と末尾n_end個のICを除外"""
    print(f"--- トライアルの前後除外開始 (先頭: {n_start}歩, 末尾: {n_end}歩) ---")
    if df_segmented is None or df_segmented.empty:
        return pd.DataFrame()
    # Leg 列が存在するか確認 (体幹ICのみの場合は存在しない可能性 -> スキップ)
    if 'Leg' not in df_segmented.columns or 'Trial_ID' not in df_segmented.columns:
        print("警告: 前後除外に必要な Leg または Trial_ID 列がありません。スキップします。")
        return df_segmented  # そのまま返す

    min_required_ics = n_start + n_end + 1
    trimmed_groups = []
    # Trial_ID と Leg でグループ化し、IC_Indexでソートしておく
    grouped = df_segmented.sort_values(by='IC_Index').groupby(
        ['Trial_ID', 'Leg'], sort=False)
    total_removed_count = 0
    original_count = len(df_segmented)

    for name, group in grouped:
        if len(group) >= min_required_ics:
            # iloc を使って先頭 n_start 個と末尾 n_end 個を除外
            trimmed_group = group.iloc[n_start:-n_end]
            trimmed_groups.append(trimmed_group)
            total_removed_count += len(group) - len(trimmed_group)
        else:
            # 条件を満たさないグループは完全に除外される
            total_removed_count += len(group)

    if not trimmed_groups:
        print("警告: 前後除外の結果、有効な歩行周期が残りませんでした。")
        return pd.DataFrame()

    # 有効なグループを結合
    df_trimmed = pd.concat(trimmed_groups).reset_index(drop=True)
    print(
        f"  前後除外処理完了。 {original_count} -> {len(df_trimmed)} イベント ({total_removed_count} イベント除外)")
    return df_trimmed


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
                        if filtered_ap is not None and filtered_lr is not None and self.time_vector is not None:
                            self.plot_trunk_ics(
                                ic_events_df_trunk, filtered_ap, filtered_lr, self.time_vector)
                        else:
                            print("警告: 体幹ICプロット用データ不足")

            

            # === ステップ 2.1-3.1: 下腿ベースの解析 ===
            if gait_events_df_shank_all is not None and not gait_events_df_shank_all.empty:
                print("\n[ステップ2.1] 歩行トライアルの自動分割 (下腿ベース) を実行中...")
                gait_events_df_shank_segmented = segment_walking_trials(
                    gait_events_df_shank_all, MAX_IC_INTERVAL_SEC, MIN_ICS_PER_TRIAL)
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
                    "filtered_lr_gyro_signal")  # キー名修正
                # time_vector は self.time_vector を使用

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
        if gait_events_df is None or gait_events_df.empty:
            print("プロットするイベントデータなし")
            return
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
                       label=f'Trunk AP Acc(Z,Filt,Offset)', alpha=0.7, color='k')
            for leg, color, marker in [('L', 'blue', 'o'), ('R', 'red', 'o')]:
                leg_events = ic_events_df[ic_events_df['Leg'] == leg]
                if not leg_events.empty:
                    ic_indices = leg_events['IC_Index'].dropna().astype(
                        int).values
                    valid_ic = ic_indices[(ic_indices >= 0) & (
                        ic_indices < len(signal_ap))]
                if len(valid_ic) > 0:
                    ax_ap.plot(time_vector[valid_ic], signal_ap[valid_ic],
                               marker=marker, color=color, ms=6, label=f'IC({leg})', ls='None')
            unknown_events = ic_events_df[ic_events_df['Leg'] == 'Unknown']
            if not unknown_events.empty:
                unk_ic_indices = unknown_events['IC_Index'].dropna().astype(
                    int).values
                valid_unk_ic = unk_ic_indices[(unk_ic_indices >= 0) & (
                    unk_ic_indices < len(signal_ap))]
                if len(valid_unk_ic) > 0:
                    ax_ap.plot(time_vector[valid_unk_ic], signal_ap[valid_unk_ic],
                               marker='x', color='gray', ms=6, label=f'IC(Unknown)', ls='None')
            ax_ap.set_title(f'体幹 AP(Z軸) Acc と検出IC')
            ax_ap.set_ylabel('加速度(単位?)')
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
                if not leg_events.empty:
                    ic_indices = leg_events['IC_Index'].dropna().astype(
                        int).values
                    valid_ic = ic_indices[(ic_indices >= 0) & (
                        ic_indices < len(signal_lr))]
                if len(valid_ic) > 0:
                    ax_lr.plot(time_vector[valid_ic], signal_lr[valid_ic],
                               marker=marker, color=color, ms=6, label=f'IC({leg})', ls='None')
            if not unknown_events.empty:
                unk_ic_indices = unknown_events['IC_Index'].dropna().astype(
                    int).values
                valid_unk_ic = unk_ic_indices[(unk_ic_indices >= 0) & (
                    unk_ic_indices < len(signal_lr))]
                if len(valid_unk_ic) > 0:
                    ax_lr.plot(time_vector[valid_unk_ic], signal_lr[valid_unk_ic],
                           marker='x', color='gray', ms=6, label=f'IC(Unknown)', ls='None')
            ax_lr.set_title(f'体幹 Yaw(Y軸) Gyro と検出IC (LR判定用)')
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
