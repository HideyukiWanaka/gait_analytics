# kinematic_parameters.py
import pandas as pd
import numpy as np
# scipy.signal.find_peaks はここでは直接使わないが、
# 谷検出などで将来的に使う可能性も考慮し import してもよい
# from scipy.signal import find_peaks


def calculate_kinematic_params(gait_events_df, filtered_signals, time_vector, sampling_rate_hz):
    """
    定常歩行区間のイベントデータとフィルター済み角速度信号から
    運動学的パラメータ（Max Swing Vel, Peak Stance Vel）を計算する。

    Args:
        gait_events_df (pd.DataFrame): trim_trial_ends 処理後の DataFrame.
                                      Trial_ID, Leg, IC_Index, FO_Index が必要。
        filtered_signals (dict): {'L': array, 'R': array} フィルター済み角速度信号。
        time_vector (np.ndarray): filtered_signalsに対応する時間ベクトル。
        sampling_rate_hz (float): サンプリング周波数。

    Returns:
        dict: 計算された運動学パラメータを格納した辞書。
    """
    print("--- [Function@kinematic] 運動学パラメータ計算開始 ---")
    results = {}  # 結果格納用辞書
    # 入力データのチェック
    required_cols = {'Trial_ID', 'Leg', 'IC_Index', 'FO_Index'}
    required_legs = {'L', 'R'}
    if gait_events_df is None or gait_events_df.empty or \
       not filtered_signals or time_vector is None or \
       not required_cols.issubset(gait_events_df.columns) or \
       not required_legs.issubset(gait_events_df['Leg'].unique()):
        print("  エラー: 運動学パラメータ計算に必要なデータが不足しています。")
        return results

    # 次のICインデックスを計算（Swing区間特定のため）
    df = gait_events_df.sort_values(by=['Trial_ID', 'Leg', 'IC_Time']).copy()
    df['Next_IC_Index'] = df.groupby(['Trial_ID', 'Leg'])['IC_Index'].shift(-1)

    # 左右の脚のデータを格納するリスト
    all_max_swing_vel_L, all_max_swing_vel_R = [], []
    all_peak_stance_vel_L, all_peak_stance_vel_R = [], []

    # 左右の脚でループ
    for leg in ['L', 'R']:
        leg_df = df[df['Leg'] == leg].copy()
        signal_filt = filtered_signals.get(leg)

        # フィルター済み信号のチェック
        if signal_filt is None or len(signal_filt) != len(time_vector):
            print(f"  警告: {leg}脚のフィルター信号データ不整合。スキップします。")
            continue

        max_swing_vels_leg = []  # この脚の最大スイング速度を格納
        peak_stance_vels_leg = []  # この脚のピーク立脚速度を格納
        max_signal_idx = len(signal_filt) - 1  # 信号の最大インデックス

        # 各サイクル（IC-FO-NextIC）について処理
        for index, row in leg_df.iterrows():
            ic_idx = row['IC_Index']
            fo_idx = row['FO_Index']
            next_ic_idx = row['Next_IC_Index']  # 最後は NaN になる

            # --- Peak Stance Velocity (Trough) ---
            # ICからFOまでの区間で最小値（最も深い谷）を探す
            if pd.notna(ic_idx) and pd.notna(fo_idx):
                # floatの場合があるのでintに変換
                ic_idx_int = int(ic_idx)
                fo_idx_int = int(fo_idx)
                # 有効なインデックス範囲か確認
                start_s = max(0, ic_idx_int)
                end_s = min(fo_idx_int + 1, max_signal_idx + 1)  # FOインデックスを含む

                if start_s < end_s:  # 区間が存在するか
                    stance_signal = signal_filt[start_s: end_s]
                    if len(stance_signal) > 0:  # 区間にデータがあるか
                        peak_stance_vels_leg.append(np.min(stance_signal))

            # --- Max Swing Velocity ---
            # FOから次のICまでの区間で最大値（最も高い山）を探す
            if pd.notna(fo_idx) and pd.notna(next_ic_idx):
                fo_idx_int = int(fo_idx)
                next_ic_idx_int = int(next_ic_idx)
                # 有効なインデックス範囲か確認
                start_sw = max(0, fo_idx_int)
                end_sw = min(next_ic_idx_int + 1,
                             max_signal_idx + 1)  # 次のICインデックスを含む

                if start_sw < end_sw:  # 区間が存在するか
                    swing_signal = signal_filt[start_sw: end_sw]
                    if len(swing_signal) > 0:  # 区間にデータがあるか
                        max_swing_vels_leg.append(np.max(swing_signal))

        # 全サイクルの値を左右それぞれのリストに格納
        if leg == 'L':
            all_max_swing_vel_L = max_swing_vels_leg
            all_peak_stance_vel_L = peak_stance_vels_leg
        else:  # leg == 'R'
            all_max_swing_vel_R = max_swing_vels_leg
            all_peak_stance_vel_R = peak_stance_vels_leg

    # --- 左右別の統計量計算 ---
    for leg, max_vels, stance_vels in [('L', all_max_swing_vel_L, all_peak_stance_vel_L),
                                       ('R', all_max_swing_vel_R, all_peak_stance_vel_R)]:
        # --- Max Swing Velocity 統計量 ---
        count_sw = len(max_vels)
        results[f'Num_Cycles_SwingVel_{leg}'] = count_sw  # 計算に使用したサイクル数
        if count_sw >= 2:  # 平均・SD・CV計算には最低2つ必要
            arr = np.array(max_vels)
            mean_val = np.mean(arr)
            sd_val = np.std(arr, ddof=0)  # 母標準偏差
            cv_val = (sd_val / mean_val) * \
                100.0 if not np.isclose(mean_val, 0) else np.nan
            results[f'Mean_Max_Swing_Vel_{leg}'] = mean_val
            results[f'SD_Max_Swing_Vel_{leg}'] = sd_val
            results[f'CV_Max_Swing_Vel_{leg}_percent'] = cv_val
        else:
            # データ不足の場合はNaN
            results[f'Mean_Max_Swing_Vel_{leg}'] = np.nan
            results[f'SD_Max_Swing_Vel_{leg}'] = np.nan
            results[f'CV_Max_Swing_Vel_{leg}_percent'] = np.nan

        # --- Peak Stance Velocity 統計量 ---
        count_st = len(stance_vels)
        results[f'Num_Cycles_StanceVel_{leg}'] = count_st  # 計算に使用したサイクル数
        if count_st >= 2:
            arr = np.array(stance_vels)
            mean_val = np.mean(arr)
            sd_val = np.std(arr, ddof=0)
            # CVの分母が0に近い場合NaNにする
            cv_val = (sd_val / mean_val) * \
                100.0 if not np.isclose(mean_val, 0) else np.nan
            results[f'Mean_Peak_Stance_Vel_{leg}'] = mean_val
            results[f'SD_Peak_Stance_Vel_{leg}'] = sd_val
            results[f'CV_Peak_Stance_Vel_{leg}_percent'] = cv_val
        else:
            # データ不足の場合はNaN
            results[f'Mean_Peak_Stance_Vel_{leg}'] = np.nan
            results[f'SD_Peak_Stance_Vel_{leg}'] = np.nan
            results[f'CV_Peak_Stance_Vel_{leg}_percent'] = np.nan

    print(f"--- [Function@kinematic] 運動学パラメータ計算終了 ---")
    return results
