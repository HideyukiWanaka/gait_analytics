# preprocessing.py
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from pathlib import Path

# 列名リストを返すヘルパー関数
def get_expected_column_names(right_prefix, left_prefix, trunk_prefix):
    """期待される33列の列名リストを生成する"""
    return [
        'Time',
        f'{right_prefix}_Acc_X', f'{right_prefix}_Acc_Y', f'{right_prefix}_Acc_Z', f'{right_prefix}_Ch4_V',
        f'{right_prefix}_Gyro_X', f'{right_prefix}_Gyro_Y', f'{right_prefix}_Gyro_Z', f'{right_prefix}_Ch8_V',
        f'{left_prefix}_Acc_X', f'{left_prefix}_Acc_Y', f'{left_prefix}_Acc_Z', 'Blank_1',
        f'{left_prefix}_Gyro_X', f'{left_prefix}_Gyro_Y', f'{left_prefix}_Gyro_Z', 'Blank_2',
        f'{trunk_prefix}_Acc_X', f'{trunk_prefix}_Acc_Y', f'{trunk_prefix}_Acc_Z', 'Blank_3',
        f'{trunk_prefix}_Gyro_X', f'{trunk_prefix}_Gyro_Y', f'{trunk_prefix}_Gyro_Z', 'Blank_4',
        'Blank_5', 'Blank_6', 'Blank_7', 'Blank_8', 'Blank_9', 'Blank_10', 'Blank_11', 'Blank_12'
    ]

# 関数名を preprocess_and_sync_imu_data に変更 (推奨)
def preprocess_and_sync_imu_data(data_file, rows_to_skip=11, sampling_interval_ms=5,
                                 right_prefix='R', left_prefix='L', trunk_prefix='T',
                                 sync_axis_suffix='_Acc_Y',
                                 align_gyro_suffix='_Gyro_Z',
                                 peak_height=None, peak_prominence=None, peak_distance=None):
    """
    CSVを読み込み、左右脚と体幹のIMUデータを同期させ、
    指定された角速度(align_gyro_suffix)と全ての加速度データを含むDataFrameを返す。
    同期は左右体幹の sync_axis_suffix (例: Acc_Z) のピーク検出で行う。

    Args:
        sync_axis_suffix (str): 同期に使用する加速度軸のサフィックス (デフォルト: '_Acc_Z')
        他 Args: (省略)

    Returns:
        tuple: (sync_data_df, lags_dict, sampling_rate_hz)
               エラー時は (None, {}, sampling_rate_hz)
    """
    print(f"--- [Function@preprocessing] 1. 全IMUデータの前処理・同期開始 (同期軸: {sync_axis_suffix}) ---")
    sampling_rate_hz = 1000.0 / sampling_interval_ms
    error_return = (None, {}, sampling_rate_hz) # エラー時の戻り値

    # --- データの読み込みと列名設定 ---
    try:
        data_file_path = Path(data_file) if not isinstance(data_file, Path) else data_file
        try:
            df = pd.read_csv(data_file_path, skiprows=rows_to_skip, encoding='cp932')
        except UnicodeDecodeError:
            print(f"  encoding='cp932' で失敗。encoding='shift_jis' を試します...")
            df = pd.read_csv(data_file_path, skiprows=rows_to_skip, encoding='shift_jis')

        expected_column_names = get_expected_column_names(right_prefix, left_prefix, trunk_prefix)
        if len(df.columns) == len(expected_column_names):
            df.columns = expected_column_names
            blank_cols = [col for col in df.columns if 'Blank_' in col]
            df = df.drop(columns=blank_cols)
            print(f"  ファイル '{data_file_path.name}' 読込・列名修正完了。")
        else:
            print(f"エラー: 列数不一致 ({len(df.columns)} vs {len(expected_column_names)})")
            return error_return
    except FileNotFoundError:
        print(f"エラー: データファイルが見つかりません: {data_file_path}")
        return error_return
    except Exception as e:
        print(f"エラー: データ読み込み/列名設定中に予期せぬエラー: {e}")
        return error_return

    # --- 同期用信号抽出とピーク検出 ---
    peak_indices = {'L': -1, 'R': -1, 'T': -1}
    num_samples_for_sync = 1000
    actual_sync_length = -1

    try:
        min_len = -1
        signals_full = {}
        # 左右体幹の同期用信号(指定軸)を抽出
        for prefix in [left_prefix, right_prefix, trunk_prefix]:
            col_name = f'{prefix}{sync_axis_suffix}'
            if col_name not in df.columns:
                raise KeyError(f"同期用信号列 '{col_name}' が見つかりません。")
            signals_full[prefix] = df[col_name].fillna(0).values
            current_len = len(signals_full[prefix])
            if min_len < 0 or current_len < min_len:
                min_len = current_len

        actual_sync_length = min(min_len, num_samples_for_sync)
        if actual_sync_length <= 0:
            raise ValueError("同期に使用できるデータ長が0以下です。")

        # 各信号の先頭部分でピーク検出
        print(f"  情報: 同期ピーク検出 (軸: {sync_axis_suffix}, 先頭 {actual_sync_length} samples, params: h={peak_height}, p={peak_prominence}, d={peak_distance})")
        peak_found_all = True # 全てのピークが見つかったかフラグ
        for prefix, signal_full in signals_full.items():
            signal_short = signal_full[:actual_sync_length]
            peaks, _ = find_peaks(np.abs(signal_short), height=peak_height, prominence=peak_prominence, distance=peak_distance)
            if len(peaks) > 0:
                peak_indices[prefix] = peaks[0] # 最初のピークを採用
                print(f"    {prefix} ピーク検出 index: {peak_indices[prefix]} (値: {signal_short[peaks[0]]:.2f})")
            else:
                print(f"    警告: {prefix} で同期ピーク({sync_axis_suffix})検出不可。")
                peak_found_all = False
                # エラーメッセージをより具体的に
                error_msg = f"{prefix} の同期ピーク({sync_axis_suffix})を検出できませんでした。\n" \
                            f"データ先頭部分を確認するか、find_peaksパラメータ(高さ/突出度/距離)を調整してください。"
                raise ValueError(error_msg) # ピークが見つからない場合は明確なエラー

    except KeyError as e:
        print(f"エラー: {e}")
        return error_return
    except ValueError as e:
        print(f"エラー: {e}")
        return error_return
    except Exception as e:
        print(f"エラー: 同期用信号処理・ピーク検出中に予期せぬエラー: {e}")
        return error_return

    # --- ラグ計算 (Right基準) ---
    lag_L_vs_R = peak_indices[left_prefix] - peak_indices[right_prefix]
    lag_T_vs_R = peak_indices[trunk_prefix] - peak_indices[right_prefix]
    lags_dict = {'L_vs_R': lag_L_vs_R, 'T_vs_R': lag_T_vs_R}
    print(f"  計算されたラグ (R基準, {sync_axis_suffix}ピーク): L={lag_L_vs_R}, T={lag_T_vs_R} [samples]")

    # --- 全信号のアライメントとDataFrame作成 ---
    sync_data_df = None
    try:
        # アライメントのための開始インデックスと長さを計算
        start_offset = max(0, -lag_L_vs_R, -lag_T_vs_R)
        start_indices = {
            right_prefix: start_offset,
            left_prefix: start_offset + lag_L_vs_R,
            trunk_prefix: start_offset + lag_T_vs_R
        }

        # 各信号の元の長さを取得しておく
        original_lengths = {}
        for prefix in [left_prefix, right_prefix, trunk_prefix]:
            # Acc_X列の存在を仮定 (なければエラーになる)
            col_for_len_check = f'{prefix}_Acc_X'
            if col_for_len_check not in df.columns:
                 raise KeyError(f"アライメント長計算のための列 '{col_for_len_check}' が見つかりません。")
            original_lengths[prefix] = len(df[col_for_len_check])

        # アライメント後の長さを計算
        aligned_length = float('inf')
        for prefix, start_idx in start_indices.items():
            if start_idx < 0 or start_idx >= original_lengths[prefix]:
                 raise ValueError(f"{prefix} の計算された開始インデックス({start_idx})が不正です (元データ長: {original_lengths[prefix]})。")
            len_adj = original_lengths[prefix] - start_idx
            aligned_length = min(aligned_length, len_adj)

        if aligned_length <= 0 or aligned_length == float('inf'):
            raise ValueError(f"アライメント後のデータ長 ({aligned_length}) が不正です。")
        aligned_length = int(aligned_length) # 整数化
        print(f"  アライメント後のデータ長: {aligned_length} samples")

        # アライメント済みデータを格納する辞書
        aligned_data = {}
        # 時間ベクトル
        time_aligned = np.arange(aligned_length) * (sampling_interval_ms / 1000.0)
        aligned_data['time_aligned_sec'] = time_aligned

        # アライメント対象の信号リスト (必要に応じて追加・変更)
        signals_to_process = [
            (left_prefix, align_gyro_suffix), (right_prefix, align_gyro_suffix), # Gyro Z
            (left_prefix, '_Acc_X'), (left_prefix, '_Acc_Y'), (left_prefix, '_Acc_Z'), # Left Acc
            (right_prefix, '_Acc_X'), (right_prefix, '_Acc_Y'), (right_prefix, '_Acc_Z'), # Right Acc
            (trunk_prefix, '_Acc_X'), (trunk_prefix, '_Acc_Y'), (trunk_prefix, '_Acc_Z'),
            (trunk_prefix, '_Gyro_X'), (trunk_prefix, '_Gyro_Y'), (trunk_prefix, '_Gyro_Z')# Trunk Gyro
            # 必要なら Trunk Gyro も追加: (trunk_prefix, '_Gyro_X'), ...
        ]
        print(f"  アライメント対象信号: {[f'{p}{s}' for p, s in signals_to_process]}")

        # 各信号をスライスして辞書に追加
        for prefix, suffix in signals_to_process:
            col_name = f'{prefix}{suffix}'
            output_col_name = f'{col_name}_aligned'
            original_signal = df[col_name].fillna(0).values
            start_idx = start_indices[prefix]
            end_idx = start_idx + aligned_length

            # スライス範囲の最終チェック
            if start_idx < 0 or end_idx > len(original_signal):
                 raise ValueError(f"{output_col_name} のスライス範囲 ({start_idx}:{end_idx}) が元データ長 ({len(original_signal)}) を超えています。")

            aligned_signal = original_signal[start_idx : end_idx]

            # 左 Gyro Z のみ符号反転
            if prefix == left_prefix and suffix == align_gyro_suffix:
                aligned_signal = -aligned_signal
                print(f"  情報: 左 {align_gyro_suffix} の符号を反転しました。")

            aligned_data[output_col_name] = aligned_signal

        # DataFrame作成
        sync_data_df = pd.DataFrame(aligned_data)
        # 列順序調整 (任意)
        cols_order = ['time_aligned_sec'] + sorted([col for col in sync_data_df.columns if col != 'time_aligned_sec'])
        sync_data_df = sync_data_df[cols_order]

    except KeyError as e:
        print(f"エラー: 必要なカラム '{e}' が DF に存在しません (列名設定を確認)。")
        return error_return
    except ValueError as e:
        print(f"エラー: アライメント処理中の値エラー: {e}")
        return error_return
    except Exception as e:
        print(f"エラー: アライメント/DataFrame作成中に予期せぬエラー: {e}")
        return error_return

    print("--- [Function@preprocessing] 1. 全IMUデータの前処理・同期終了 ---")
    # 戻り値を変更 (DataFrame, ラグ辞書, サンプリングレート)
    return sync_data_df, lags_dict, sampling_rate_hz