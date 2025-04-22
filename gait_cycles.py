# gait_cy# gait_cycles.py
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


def lowpass_filter(data, cutoff, fs, order=4):
    """バターワースローパスフィルターを適用する"""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # Wn パラメータが 0 < Wn < 1 の範囲にあるか確認
    if not (0 < normal_cutoff < 1):
        print(
            f"警告: 正規化カットオフ周波数 ({normal_cutoff:.4f}) が (0, 1) の範囲外です。フィルタリングをスキップします。")
        return data  # エラーを防ぐために元のデータを返す
    try:
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y
    except Exception as e:
        print(f"エラー: ローパスフィルター適用中にエラー: {e}")
        return data  # エラー時は元のデータを返す


def identify_gait_cycles(sync_gyro_df, sampling_rate_hz,
                         swing_threshold=100, swing_duration_threshold_ms=40,  # swing_threshold調整済想定
                         filter_cutoff=15.0,
                         min_peak_prominence=10, min_peak_distance_ms=50,  # IC/FO谷検出用
                         swing_peak_height=50, swing_peak_prominence=50,  # Swing山検出用
                         min_fo_wait_time_ms=50):
    """同期済みの角速度データ全体から歩行周期イベント（IC, FO）を同定する。"""
    print(f"--- [Function@gait_cycles] 2. 歩行周期の同定開始 (全体) ---")
    # 入力DataFrameのチェック
    required_gyro_cols = [f'{p}_Gyro_Z_aligned' for p in ['L', 'R']]
    if sync_gyro_df is None or sync_gyro_df.empty or 'time_aligned_sec' not in sync_gyro_df.columns or \
       not all(col in sync_gyro_df.columns for col in required_gyro_cols):
        print("  エラー: 入力DataFrame不備 (Time or Gyro Z)")
        # Noneではなく空の辞書などを返した方がrun_gait_analysis側で扱いやすい可能性がある
        return {"events_df": pd.DataFrame(), "filtered_signals": {}, "time_vector": None}

    results = []  # イベント格納用リスト
    dt = 1.0 / sampling_rate_hz
    # パラメータをサンプル数に変換
    swing_duration_samples = int(
        swing_duration_threshold_ms / 1000 * sampling_rate_hz)
    min_peak_distance_samples = int(
        min_peak_distance_ms / 1000 * sampling_rate_hz)
    min_fo_wait_samples = int(min_fo_wait_time_ms / 1000 * sampling_rate_hz)

    filtered_signals = {}  # フィルタリング済み信号格納用辞書
    time_sec = sync_gyro_df['time_aligned_sec'].values  # 時間ベクトル取得
    # df形状表示
    print(
        f"  Input DataFrame shape: ({len(sync_gyro_df)}, {len(sync_gyro_df.columns)})")
    print(
        f"  Time vector shape: {time_sec.shape}, first 5 values: {time_sec[:5]}")

    for leg_prefix in ['L', 'R']:
        col_name = f'{leg_prefix}_Gyro_Z_aligned'
        signal_raw = sync_gyro_df[col_name].values
        print(
            f"\n  {leg_prefix}脚: Raw signal shape: {signal_raw.shape}, first 5 values: {signal_raw[:5]}")

        # 1. フィルタリング
        signal_filt = lowpass_filter(
            signal_raw, cutoff=filter_cutoff, fs=sampling_rate_hz)
        if signal_filt is None:
            print(f"  エラー: {leg_prefix}脚のフィルタリングで None が返されました。")
            filtered_signals[leg_prefix] = None
            continue  # 次の脚へ
        print(
            f"  {leg_prefix}脚: Filtered signal shape: {signal_filt.shape}, first 5 values: {signal_filt[:5]}")
        filtered_signals[leg_prefix] = signal_filt  # 辞書に格納

        # 2. Swing区間の検出
        above_threshold = signal_filt > swing_threshold
        swing_starts, swing_ends = [], []
        in_swing, count, potential_start = False, 0, -1
        for i in range(len(signal_filt)):
            if above_threshold[i]:
                if not in_swing:
                    potential_start = i
                    count = 1
                    in_swing = True
                else:
                    count += 1
            else:
                if in_swing:
                    if count >= swing_duration_samples:
                        swing_starts.append(potential_start)
                        swing_ends.append(i - 1)
                    in_swing = False
                    count = 0
        # データ末尾がSwingの場合の処理
        if in_swing and count >= swing_duration_samples:
            swing_starts.append(potential_start)
            swing_ends.append(len(signal_filt) - 1)

        if not swing_starts:
            print(f"  {leg_prefix}脚: Swing区間検出不可 (閾値={swing_threshold})")
            continue  # この脚の周期同定はスキップ

        print(f"  {leg_prefix}脚: {len(swing_starts)} 個のSwing区間検出 (全体)")

        # 3. Swing Peak 検出
        swing_peaks, _ = find_peaks(signal_filt, height=swing_peak_height,
                                    prominence=swing_peak_prominence, distance=min_peak_distance_samples)
        # print(f"  {leg_prefix}脚: {len(swing_peaks)} 個のSwingピーク検出") # ログ削減

        # 4. IC候補 (谷) 検出
        ic_candidates, _ = find_peaks(
            -signal_filt, prominence=min_peak_prominence, distance=min_peak_distance_samples)
        # print(f"  {leg_prefix}脚: {len(ic_candidates)} 個のIC候補検出")

        # 5. FO候補 (谷) 検出
        fo_candidates, _ = find_peaks(
            -signal_filt, prominence=min_peak_prominence, distance=min_peak_distance_samples)
        fo_candidate_values = {idx: signal_filt[idx] for idx in fo_candidates}
        # print(f"  {leg_prefix}脚: {len(fo_candidates)} 個のFO候補検出")

        if len(ic_candidates) == 0 or len(fo_candidates) == 0 or len(swing_peaks) == 0:
            print(f"  {leg_prefix}脚: ピーク/谷検出不足のため周期同定不可")
            continue

        # 6. イベントの関連付け
        num_cycles = 0
        for j in range(len(swing_starts)):  # 各Swingについてループ
            current_swing_start = swing_starts[j]
            current_swing_end = swing_ends[j] if j < len(
                swing_ends) else len(signal_filt) - 1

            # --- IC 検出 ---
            possible_ics = ic_candidates[ic_candidates > current_swing_end]
            if len(possible_ics) == 0:
                # print(f"  DEBUG: Swing {j+1} 後にIC候補なし。スキップ。")
                continue  # 次のSwingへ
            ic_idx = possible_ics[0]

            # --- FO 検出のための準備 ---
            relevant_swing_peaks = swing_peaks[(swing_peaks >= current_swing_start) & (
                swing_peaks <= current_swing_end)]
            if len(relevant_swing_peaks) == 0:
                fo_wait_samples_dynamic = min_fo_wait_samples
            else:
                last_swing_peak_in_interval = relevant_swing_peaks[-1]
                fo_wait_samples_dynamic = max(
                    ic_idx - last_swing_peak_in_interval, min_fo_wait_samples)

            next_swing_start = len(signal_filt) if j + \
                1 >= len(swing_starts) else swing_starts[j+1]
            fo_search_start_idx = ic_idx + fo_wait_samples_dynamic
            fo_search_end_idx = next_swing_start
            possible_fos_in_window_idx = [
                idx for idx in fo_candidates if fo_search_start_idx <= idx < fo_search_end_idx]

            if len(possible_fos_in_window_idx) == 0:
                # print(f"  DEBUG: IC {ic_idx} 後にFO候補なし。スキップ。")
                continue  # 次のSwingへ
            fo_idx = min(possible_fos_in_window_idx,
                         key=lambda idx: fo_candidate_values.get(idx, 0))

            # --- イベント記録 ---
            # time_sec の範囲チェックを追加
            ic_time_val = time_sec[ic_idx] if 0 <= ic_idx < len(
                time_sec) else np.nan
            fo_time_val = time_sec[fo_idx] if 0 <= fo_idx < len(
                time_sec) else np.nan
            cycle_data = {
                "Leg": leg_prefix,
                "Cycle": num_cycles + 1,
                "IC_Index": ic_idx,
                "FO_Index": fo_idx,
                "IC_Time": ic_time_val,
                "FO_Time": fo_time_val
            }
            results.append(cycle_data)
            num_cycles += 1
        # --- for j ループ終了 ---

        print(f"  {leg_prefix}脚: {num_cycles} 歩行周期候補 同定")
    # --- for leg_prefix ループ終了 ---

    # --- 結果の準備 ---
    events_df = pd.DataFrame(results)  # resultsが空でも空のDataFrameが作られる
    if not events_df.empty:
        events_df = events_df.sort_values(
            by=['Leg', 'IC_Index']).reset_index(drop=True)

    # デバッグ用最終チェック
    print(f"\n  [最終チェック] events_df is empty: {events_df.empty}")
    print(f"  [最終チェック] filtered_signals keys: {filtered_signals.keys()}")
    for k, v in filtered_signals.items():
        print(
            f"    - filtered_signals['{k}'] shape: {v.shape if v is not None else 'None'}")
    print(
        f"  [最終チェック] Time vector shape: {time_sec.shape if time_sec is not None else 'None'}")

    print(f"--- [Function@gait_cycles] 2. 歩行周期の同定終了 ---")

    # 常に辞書を返す（events_dfが空の場合も含む）
    return {
        "events_df": events_df,
        "filtered_signals": filtered_signals,
        "time_vector": time_sec
    }


def identify_ics_from_trunk_accel(sync_data_df, sampling_rate_hz,
                                  ap_axis_col='T_Acc_Z_aligned',  # 前後(AP)=Z軸
                                  # 左右(ML)=X軸 と仮定
                                  ml_axis_col='T_Acc_X_aligned',
                                  filter_cutoff_acc=20.0,
                                  ic_peak_height=0.1, ic_peak_prominence=0.1, ic_peak_distance_ms=200,
                                  lr_threshold=0.0):
    """
    同期済みの体幹前方(AP)加速度データから IC (正ピーク@ゼロクロス前) を検出し、
    ML加速度から左右を判定する。(FO検出は未実装)
    """
    print(
        f"--- [Function@gait_cycles] IC同定開始 (体幹 Acc AP={ap_axis_col}, ML={ml_axis_col}, cutoff={filter_cutoff_acc}Hz) ---")

    # --- 初期化 ---
    ic_events_df = pd.DataFrame()  # 結果DF (空で初期化)
    filtered_ap_signal = None     # フィルター済みAP信号
    filtered_ml_signal = None     # フィルター済みML信号
    time_sec = None               # 時間ベクトル

    # --- 入力チェック ---
    required_cols = {ap_axis_col, ml_axis_col, 'time_aligned_sec'}
    if sync_data_df is None or sync_data_df.empty or not required_cols.issubset(sync_data_df.columns):
        print(f"  エラー: IC同定に必要なデータ ({required_cols}) が不足。")
        # ★★★ エラー時も None を含む辞書を返す ★★★
        return {"ic_events_df": ic_events_df, "filtered_ap_signal": None, "filtered_ml_signal": None, "time_vector": None}

    try:
        time_sec = sync_data_df['time_aligned_sec'].values
        signal_ap_raw = sync_data_df[ap_axis_col].values
        signal_ml_raw = sync_data_df[ml_axis_col].values
        dt = 1.0 / sampling_rate_hz
        ic_peak_distance_samples = int(
            ic_peak_distance_ms / 1000 * sampling_rate_hz)

        # 1. フィルタリング
        print("  DEBUG: Filtering AP/ML signals...")
        signal_ap_filt = lowpass_filter(
            signal_ap_raw, cutoff=filter_cutoff_acc, fs=sampling_rate_hz)
        signal_ml_filt = lowpass_filter(
            signal_ml_raw, cutoff=filter_cutoff_acc, fs=sampling_rate_hz)
        if signal_ap_filt is None or signal_ml_filt is None:
            raise ValueError("信号フィルタリング失敗。")
        print(
            f"  DEBUG: Filtering successful. AP shape: {signal_ap_filt.shape}, ML shape: {signal_ml_filt.shape}")

        # 2. DCオフセット除去
        print("  DEBUG: Removing DC offset...")
        mean_ap = np.mean(signal_ap_filt)
        signal_ap_proc = signal_ap_filt - mean_ap
        mean_ml = np.mean(signal_ml_filt)
        signal_ml_proc = signal_ml_filt - mean_ml
        # ★★★ プロット用に返す変数を確定 ★★★
        filtered_ap_signal = signal_ap_proc  # 処理後の信号を返す
        filtered_ml_signal = signal_ml_proc  # 処理後の信号を返す

        # 3. IC検出ロジック
        print("  DEBUG: Detecting peaks and zero crossings...")
        positive_peak_indices, _ = find_peaks(
            signal_ap_proc, height=ic_peak_height, prominence=ic_peak_prominence, distance=ic_peak_distance_samples)
        zero_crossings_pos_neg = np.where(
            (signal_ap_proc[:-1] > 0) & (signal_ap_proc[1:] <= 0))[0]
        print(
            f"  DEBUG: Found {len(positive_peak_indices)} positive peaks, {len(zero_crossings_pos_neg)} pos->neg zero crossings.")

        valid_ic_indices = []
        if len(positive_peak_indices) > 0 and len(zero_crossings_pos_neg) > 0:
            print("  DEBUG: Associating peaks and zero crossings...")
            for peak_idx in positive_peak_indices:
                next_zc_indices = zero_crossings_pos_neg[zero_crossings_pos_neg > peak_idx]
                if len(next_zc_indices) > 0:
                    first_next_zc_idx = next_zc_indices[0]
                    intervening_peaks = positive_peak_indices[(positive_peak_indices > peak_idx) & (
                        positive_peak_indices < first_next_zc_idx)]
                    if len(intervening_peaks) == 0:
                        valid_ic_indices.append(peak_idx)
        print(f"  DEBUG: Found {len(valid_ic_indices)} valid IC indices.")

        # 4. 結果のDataFrame作成
        if len(valid_ic_indices) > 0:
            results = []
            for i, ic_idx in enumerate(sorted(valid_ic_indices)):
                ic_time_val = time_sec[ic_idx] if 0 <= ic_idx < len(
                    time_sec) else np.nan
                leg_label = "Unknown"
                if 0 <= ic_idx < len(signal_ml_proc):
                    ml_value_at_ic = signal_ml_proc[ic_idx]
                    if ml_value_at_ic > lr_threshold:
                        leg_label = 'R'
                    elif ml_value_at_ic < -lr_threshold:
                        leg_label = 'L'
                results.append({"Leg": leg_label, "Cycle": i + 1, "IC_Index": ic_idx,
                               "FO_Index": pd.NA, "IC_Time": ic_time_val, "FO_Time": pd.NA})
            ic_events_df = pd.DataFrame(results)  # ★ ic_events_df がここで更新される
        else:
            print("  警告: 条件を満たすICが見つかりませんでした。")
            # ic_events_df は空のまま

    except Exception as e:
        print(f"エラー: IC同定(体幹)処理中に予期せぬエラー: {e}")
        # エラーが発生しても、できるだけフィルター済み信号と時間は返すようにする

    # --- 関数の最後で必ず辞書を返す ---
    print(f"--- [Function@gait_cycles] IC同定(体幹)終了 ---")
    # ★★★ 最終的な戻り値の内容を DEBUG プリント ★★★
    print(f"  DEBUG: Returning dictionary:")
    print(f"    ic_events_df is empty: {ic_events_df.empty}")
    print(
        f"    filtered_ap_signal shape: {filtered_ap_signal.shape if filtered_ap_signal is not None else 'None'}")
    print(
        f"    filtered_ml_signal shape: {filtered_ml_signal.shape if filtered_ml_signal is not None else 'None'}")
    print(
        f"    time_vector shape: {time_sec.shape if time_sec is not None else 'None'}")
    # ★★★ ここまで ★★★
    return {
        "ic_events_df": ic_events_df,
        "filtered_ap_signal": filtered_ap_signal,
        "filtered_ml_signal": filtered_ml_signal,
        "time_vector": time_sec
    }
