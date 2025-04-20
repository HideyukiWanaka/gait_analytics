# pci.py
import pandas as pd
import numpy as np


def calculate_pci(gait_events_df):
    """
    歩行周期イベントデータからPCI (Phase Coordination Index) を計算する。
    入力 DataFrame は Trial_ID, Leg, IC_Time, FO_Time を含む必要がある。
    Stride Time CVの計算はここでは行わない。

    Args:
        gait_events_df (pd.DataFrame): trim_trial_ends で処理済みの歩行イベントデータ。

    Returns:
        dict or None: 計算されたPCI値や関連指標を含む辞書。計算不可の場合は None。
    """
    print("--- [Function@pci] 3. PCI の計算開始 ---")
    # 入力データのチェック
    required_cols = {'Trial_ID', 'Leg', 'IC_Time', 'FO_Time'}
    required_legs = {'L', 'R'}
    if gait_events_df is None or gait_events_df.empty or \
       not required_cols.issubset(gait_events_df.columns) or \
       not required_legs.issubset(gait_events_df['Leg'].unique()):
        print("  エラー: PCI計算に必要なデータ/列/脚が不足しています。")
        return None

    all_phases = []  # 全ての有効な位相を格納するリスト

    # Trialごとに処理
    for trial_id, trial_df in gait_events_df.groupby('Trial_ID'):
        # print(f"\n  トライアル {trial_id} の処理中...") # 必要ならコメント解除
        left_df = trial_df[trial_df['Leg'] == 'L'].sort_values(
            by='IC_Time').reset_index()
        right_df = trial_df[trial_df['Leg'] == 'R'].sort_values(
            by='IC_Time').reset_index()

        # ストライド計算には最低2つのICが必要
        if len(left_df) < 2 or len(right_df) < 2:
            # print(f"  警告: トライアル {trial_id} は左右どちらかのICが2未満のためスキップします。")
            continue

        # スイング時間計算 (基準脚決定用)
        left_df['Next_IC_Time'] = left_df['IC_Time'].shift(-1)
        left_df['Swing_Time'] = left_df['Next_IC_Time'] - left_df['FO_Time']
        right_df['Next_IC_Time'] = right_df['IC_Time'].shift(-1)
        right_df['Swing_Time'] = right_df['Next_IC_Time'] - right_df['FO_Time']

        mean_swing_L = left_df['Swing_Time'].mean()
        mean_swing_R = right_df['Swing_Time'].mean()

        # 平均スイング時間が計算できない場合はスキップ
        if pd.isna(mean_swing_L) or pd.isna(mean_swing_R):
            # print(f"  警告: トライアル {trial_id} で平均スイング時間が計算できませんでした。")
            continue

        # 基準脚を決定
        if mean_swing_L >= mean_swing_R:
            ref_leg_df = left_df
            con_leg_df = right_df
            # ref_leg_char = 'L' # デバッグ用
        else:
            ref_leg_df = right_df
            con_leg_df = left_df
            # ref_leg_char = 'R' # デバッグ用
        # print(f"    基準脚: {ref_leg_char}") # デバッグ用

        # 位相を計算
        # 基準脚の各ストライドについてループ (最後から2番目のICまで)
        for i in range(len(ref_leg_df) - 1):
            t_l_i = ref_leg_df.loc[i, 'IC_Time']
            t_l_i_plus_1 = ref_leg_df.loc[i + 1, 'IC_Time']  # = Next_IC_Time
            stride_time = t_l_i_plus_1 - t_l_i

            # 無効なストライド時間はスキップ
            if pd.isna(stride_time) or stride_time <= 0:
                continue

            # この基準脚ストライド内にある対側脚のICを探す
            # t_l_i <= t_s_i < t_l_i_plus_1 となる最初の t_s_i
            con_ic_times_in_stride = con_leg_df.loc[
                (con_leg_df['IC_Time'] >= t_l_i) & (
                    con_leg_df['IC_Time'] < t_l_i_plus_1),
                'IC_Time'
            ]

            if not con_ic_times_in_stride.empty:
                t_s_i = con_ic_times_in_stride.iloc[0]  # 最初のものを採用
                phase = ((t_s_i - t_l_i) / stride_time) * 360.0
                all_phases.append(phase)
            # else:
                # 対側ICが見つからない場合は位相を計算できない
                # print(f"    警告: 基準脚ストライド {i+1} ({t_l_i:.2f}s - {t_l_i_plus_1:.2f}s) 内に対側脚ICが見つかりません。")

    # --- 全トライアルの処理完了後 ---

    # PCI 計算に必要な変数を初期化
    phi_ABS, phi_CV, P_phi_ABS, PCI = np.nan, np.nan, np.nan, np.nan
    mean_phi, std_phi = np.nan, np.nan
    num_strides_for_pci = 0  # PCI計算に使用したストライド数を初期化

    # 位相データが2つ以上ある場合のみPCI関連指標を計算
    if len(all_phases) >= 2:
        phi_array = np.array(all_phases)
        num_strides_for_pci = len(phi_array)  # 正しいストライド数を代入

        mean_phi = np.mean(phi_array)
        std_phi = np.std(phi_array, ddof=0)  # 母標準偏差を使用
        phi_ABS = np.mean(np.abs(phi_array - 180.0))  # 180度からの平均絶対偏差

        # 位相変動係数 (phi_CV)
        if not np.isclose(mean_phi, 0):  # ゼロ除算回避
            phi_CV = (std_phi / mean_phi) * 100.0
        else:
            print("  警告: 平均位相が0に近いため、phi_CV は計算できません。")
            phi_CV = np.nan  # 計算不可の場合はNaN

        # 正規化された位相誤差 (P_phi_ABS)
        P_phi_ABS = 100.0 * (phi_ABS / 180.0)

        # PCI
        if not pd.isna(phi_CV):  # phi_CV が計算できた場合のみPCIを計算
            PCI = phi_CV + P_phi_ABS
        else:
            PCI = np.nan  # phi_CVがNaNならPCIもNaN

    else:
        # 位相データが不足している場合のエラーメッセージ
        print(f"  エラー: 位相データが少なすぎるため ({len(all_phases)} 個)、PCIを計算できません。")
        # 各変数は初期値 NaN または 0 のまま

    # --- Stride Time CV 計算ロジックはここから削除済み ---

    print(f"--- [Function@pci] 3. PCI の計算終了 ---")

    # 戻り値の辞書
    return {
        "PCI": PCI,
        "phi_ABS_deg": phi_ABS,
        "phi_CV_percent": phi_CV,
        "P_phi_ABS": P_phi_ABS,
        "mean_phase_deg": mean_phi,
        "std_phase_deg": std_phi,
        "num_strides_used_pci": num_strides_for_pci
        # Stride Time CV 関連は含まない
    }
