# temporal_parameters.py
import pandas as pd
import numpy as np


def calculate_temporal_params(gait_events_df):
    """
    定常歩行区間のイベントデータから時間関連パラメータを計算する。
    (Cadence, Stride/Stance/Swing Time Mean/SD/CV/Percent)

    Args:
        gait_events_df (pd.DataFrame): trim_trial_ends 処理後の DataFrame.
                                      Trial_ID, Leg, IC_Time, FO_Time が必要。

    Returns:
        dict: 計算された時間パラメータを格納した辞書。計算不可の場合は一部NaN。
    """
    print("--- [Function@temporal] 時間パラメータ計算開始 ---")
    results = {}  # 結果格納用辞書
    # 必要な列が存在するかチェック
    required_cols = {'Trial_ID', 'Leg', 'IC_Time', 'FO_Time'}
    if gait_events_df is None or gait_events_df.empty or not required_cols.issubset(gait_events_df.columns):
        print("  エラー: 時間パラメータ計算に必要なデータ (列名含む) が不足しています。")
        return results  # 空辞書を返す

    # 以降の計算のためにソート
    df = gait_events_df.sort_values(by=['Trial_ID', 'Leg', 'IC_Time']).copy()

    # --- ストライド時間、立脚時間、遊脚時間の計算 ---
    # groupby で Trial と Leg ごとに次の IC_Time を計算
    df['Next_IC_Time'] = df.groupby(['Trial_ID', 'Leg'])['IC_Time'].shift(-1)
    # 各時間パラメータを計算
    df['Stride_Time'] = df['Next_IC_Time'] - df['IC_Time']
    df['Stance_Time'] = df['FO_Time'] - df['IC_Time']
    # Swing_Time は Stride - Stance で計算 (Next_IC_Time - FO_Time でも可)
    # Stride_Time が NaN の場合 Swing_Time も NaN になるように注意
    df['Swing_Time'] = df['Stride_Time'] - df['Stance_Time']

    # NaN (各トライアル/脚の最後のストライド/スイング) や非正の値を除外した有効なデータ
    df_valid = df.dropna(subset=['Stride_Time', 'Stance_Time', 'Swing_Time'])
    df_valid = df_valid[
        (df_valid['Stride_Time'] > 0) &
        (df_valid['Stance_Time'] > 0) &
        (df_valid['Swing_Time'] > 0)
    ]

    if df_valid.empty:
        print("  警告: 有効な時間パラメータ (Stride, Stance, Swing) が計算できませんでした。")
        # 空でもケイデンスは計算試行せず、空辞書を返す
        return results

    # --- 左右別の統計量計算 ---
    for leg in ['L', 'R']:
        leg_df = df_valid[df_valid['Leg'] == leg]
        count = len(leg_df)
        results[f'Num_Strides_{leg}'] = count  # 計算に使用したストライド数

        if count >= 2:  # 標準偏差・変動係数の計算には最低2サンプル必要
            # --- Stride Time ---
            st = leg_df['Stride_Time']
            mean_st = st.mean()
            sd_st = st.std(ddof=0)  # 母標準偏差
            cv_st = (sd_st / mean_st) * \
                100.0 if not np.isclose(mean_st, 0) else np.nan
            results[f'Mean_Stride_Time_{leg}_s'] = mean_st
            results[f'SD_Stride_Time_{leg}_s'] = sd_st
            results[f'CV_Stride_Time_{leg}_percent'] = cv_st

            # --- Stance Time ---
            stat = leg_df['Stance_Time']
            mean_stat = stat.mean()
            sd_stat = stat.std(ddof=0)
            cv_stat = (sd_stat / mean_stat) * \
                100.0 if not np.isclose(mean_stat, 0) else np.nan
            st_percent = (mean_stat / mean_st) * \
                100.0 if not np.isclose(mean_st, 0) else np.nan
            results[f'Mean_Stance_Time_{leg}_s'] = mean_stat
            results[f'SD_Stance_Time_{leg}_s'] = sd_stat
            results[f'CV_Stance_Time_{leg}_percent'] = cv_stat
            # Stride時間に対する割合
            results[f'Stance_Time_Percent_{leg}'] = st_percent

            # --- Swing Time ---
            swt = leg_df['Swing_Time']
            mean_swt = swt.mean()
            sd_swt = swt.std(ddof=0)
            cv_swt = (sd_swt / mean_swt) * \
                100.0 if not np.isclose(mean_swt, 0) else np.nan
            swt_percent = (mean_swt / mean_st) * \
                100.0 if not np.isclose(mean_st, 0) else np.nan
            results[f'Mean_Swing_Time_{leg}_s'] = mean_swt
            results[f'SD_Swing_Time_{leg}_s'] = sd_swt
            results[f'CV_Swing_Time_{leg}_percent'] = cv_swt
            # Stride時間に対する割合
            results[f'Swing_Time_Percent_{leg}'] = swt_percent
        else:
            # データが少ない場合は NaN を入れる
            print(f"  警告: {leg}脚の時間パラメータ統計量計算にはデータが不足 ({count}個)。")
            results[f'Mean_Stride_Time_{leg}_s'] = np.nan
            results[f'SD_Stride_Time_{leg}_s'] = np.nan
            results[f'CV_Stride_Time_{leg}_percent'] = np.nan
            results[f'Mean_Stance_Time_{leg}_s'] = np.nan
            results[f'SD_Stance_Time_{leg}_s'] = np.nan
            results[f'CV_Stance_Time_{leg}_percent'] = np.nan
            results[f'Stance_Time_Percent_{leg}'] = np.nan
            results[f'Mean_Swing_Time_{leg}_s'] = np.nan
            results[f'SD_Swing_Time_{leg}_s'] = np.nan
            results[f'CV_Swing_Time_{leg}_percent'] = np.nan
            results[f'Swing_Time_Percent_{leg}'] = np.nan

    # --- ケイデンス (Cadence) ---
    # ケイデンス(steps/min) = 120 / 平均ストライド時間(秒)
    # 左右の平均を使うか、全体の平均を使うか？ここでは全体の平均を使用。
    mean_stride_overall = df_valid['Stride_Time'].mean()
    if pd.notna(mean_stride_overall) and mean_stride_overall > 0:
        results['Cadence_steps_per_min'] = 120.0 / mean_stride_overall
    else:
        results['Cadence_steps_per_min'] = np.nan

    print(f"--- [Function@temporal] 時間パラメータ計算終了 ---")
    return results
