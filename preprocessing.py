# preprocessing.py
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from pathlib import Path

# 列名リストを返すヘルパー関数 (run_gait_analysis.pyと共通化推奨)


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


def preprocess_angular_velocity(data_file, rows_to_skip=11, sampling_interval_ms=5,
                                right_prefix='R', left_prefix='L', trunk_prefix='T',
                                sync_col_suffix='_Acc_Y', align_col_suffix='_Gyro_Z',
                                peak_height=None, peak_prominence=None, peak_distance=None):
    """
    CSV読込、同期、同期済み角速度(Gyro Z) DataFrame等を返す。(加速度は返さない版)

    Args:
        data_file (str or Path): 入力CSVファイルパス
        rows_to_skip (int): スキップする先頭行数
        sampling_interval_ms (float): サンプリング間隔 (ミリ秒)
        right_prefix (str): 右センサーの列名プレフィックス
        left_prefix (str): 左センサーの列名プレフィックス
        trunk_prefix (str): 体幹センサーの列名プレフィックス
        sync_col_suffix (str): 同期基準にする信号の列名サフィックス (例: '_Acc_Y')
        align_col_suffix (str): 同期を適用する信号の列名サフィックス (例: '_Gyro_Z')
        peak_height (float, optional): find_peaksのheightパラメータ. Defaults to None.
        peak_prominence (float, optional): find_peaksのprominenceパラメータ. Defaults to None.
        peak_distance (int, optional): find_peaksのdistanceパラメータ. Defaults to None.

    Returns:
        tuple: (sync_gyro_df, lag_samples, sampling_rate_hz, peak_idx_l, peak_idx_r, sync_l_short, sync_r_short)
               エラー時は (None, 0, sampling_rate_hz, -1, -1, None, None)
    """
    print(
        f"--- [Function@preprocessing] 1. 角速度データの前処理開始 (find_peaks: h={peak_height}, p={peak_prominence}, d={peak_distance}) ---")
    sampling_rate_hz = 1000.0 / sampling_interval_ms
    # エラー発生時に返すデフォルト値のタプル
    error_return = (None, 0, sampling_rate_hz, -1, -1, None, None)

    # --- データの読み込みと列名設定 ---
    try:
        data_file_path = Path(data_file) if not isinstance(
            data_file, Path) else data_file
        try:
            # encoding='cp932' を試す
            df = pd.read_csv(
                data_file_path, skiprows=rows_to_skip, encoding='cp932')
        except UnicodeDecodeError:
            # cp932がダメならshift_jisを試す
            print(f"  encoding='cp932' で失敗。encoding='shift_jis' を試します...")
            df = pd.read_csv(
                data_file_path, skiprows=rows_to_skip, encoding='shift_jis')

        # 列名の確認と設定
        expected_column_names = get_expected_column_names(
            right_prefix, left_prefix, trunk_prefix)
        if len(df.columns) == len(expected_column_names):
            df.columns = expected_column_names
            # Blank列を削除
            blank_cols = [col for col in df.columns if 'Blank_' in col]
            df = df.drop(columns=blank_cols)
            print(f"  ファイル '{data_file_path.name}' 読込・列名修正完了。")
        else:
            # 列数が一致しない場合はエラー
            print(
                f"エラー: 読み込んだ列数 ({len(df.columns)}) が期待される列数 ({len(expected_column_names)}) と一致しません。")
            return error_return
    except FileNotFoundError:
        print(f"エラー: データファイルが見つかりません: {data_file_path}")
        return error_return
    except Exception as e:
        # その他の読み込み・設定エラー
        print(f"エラー: データ読み込み/列名設定中に予期せぬエラー: {e}")
        return error_return

    # --- 同期用信号抽出とラグ計算 ---
    sync_left_short, sync_right_short = None, None
    actual_sync_length = 0
    lag_samples = 0
    peak_index_left = -1
    peak_index_right = -1
    try:
        # 同期用信号の全体を抽出
        sync_left_full = df[f'{left_prefix}{sync_col_suffix}'].fillna(0).values
        sync_right_full = df[f'{right_prefix}{sync_col_suffix}'].fillna(
            0).values

        # 先頭 N サンプルを切り出し
        num_samples_for_sync = 1000
        actual_sync_length = min(len(sync_left_full), len(
            sync_right_full), num_samples_for_sync)

        if actual_sync_length <= 0:
            raise ValueError("同期に使用できるデータがありません。")

        sync_left_short = sync_left_full[:actual_sync_length]
        sync_right_short = sync_right_full[:actual_sync_length]

        # find_peaks を絶対値に適用してピークインデックスを取得
        # height, prominence, distance は None の場合、find_peaks内で無視される
        left_peaks, _ = find_peaks(np.abs(
            sync_left_short), height=peak_height, prominence=peak_prominence, distance=peak_distance)
        right_peaks, _ = find_peaks(np.abs(
            sync_right_short), height=peak_height, prominence=peak_prominence, distance=peak_distance)

        if len(left_peaks) > 0 and len(right_peaks) > 0:
            # 最初のピークを採用
            peak_index_left = left_peaks[0]
            peak_index_right = right_peaks[0]
            # ラグ計算
            lag_samples = peak_index_left - peak_index_right
            lag_ms = lag_samples * sampling_interval_ms  # 参考情報
            print(f"  計算されたラグ: {lag_samples} サンプル ({lag_ms:.2f} ms)")
        else:
            # ピークが見つからない場合
            print("  警告: find_peaks で左右両方の同期ピークを検出できませんでした。ラグは0とします。")
            lag_samples = 0
            # peak_index は -1 のまま
    except KeyError as e:
        print(f"エラー: 同期用信号のカラム名 '{e}' が見つかりません。")
        return error_return
    except ValueError as e:
        print(f"エラー: 同期ラグ計算中の値エラー: {e}")
        return error_return
    except Exception as e:
        # その他の同期ラグ計算エラー
        print(f"エラー: 同期ラグ計算中に予期せぬエラー: {e}")
        return error_return

    # --- ターゲット信号(GyroZ)抽出とラグ適用 ---
    sync_gyro_df = None  # 初期化
    try:
        # ターゲット信号全体を抽出
        align_left_full = df[f'{left_prefix}{align_col_suffix}'].fillna(
            0).values
        align_right_full = df[f'{right_prefix}{align_col_suffix}'].fillna(
            0).values

        # 左角速度の符号を反転
        align_left = -align_left_full
        align_right = align_right_full

        # ラグ適用ロジック
        if lag_samples > 0:  # 左が遅れている (右基準で左を後ろにずらす -> 左の開始位置をずらす)
            aligned_left_gyro = align_left[lag_samples:]
            aligned_right_gyro = align_right[:len(
                aligned_left_gyro)]  # 長さを左に合わせる
        elif lag_samples < 0:  # 右が遅れている (左基準で右を後ろにずらす -> 右の開始位置をずらす)
            aligned_right_gyro = align_right[abs(lag_samples):]
            aligned_left_gyro = align_left[:len(
                aligned_right_gyro)]  # 長さを右に合わせる
        else:  # lag_samples == 0 (ラグなし)
            min_len = min(len(align_left), len(align_right))
            aligned_left_gyro = align_left[:min_len]
            aligned_right_gyro = align_right[:min_len]

        aligned_length = len(aligned_left_gyro)
        if aligned_length <= 0:
            raise ValueError("アライメント後のデータ長が0以下です。ラグが大きすぎる可能性があります。")

        # 時間ベクトルを作成
        time_aligned = np.arange(aligned_length) * \
            (sampling_interval_ms / 1000.0)

        # 結果をDataFrameにまとめる
        sync_gyro_df = pd.DataFrame({
            'time_aligned_sec': time_aligned,
            f'L{align_col_suffix}_aligned': aligned_left_gyro,
            f'R{align_col_suffix}_aligned': aligned_right_gyro
        })
    except KeyError as e:
        print(f"エラー: ターゲット信号のカラム名 '{e}' が見つかりません。")
        return error_return
    except ValueError as e:
        print(f"エラー: ラグ適用またはDataFrame作成中の値エラー: {e}")
        return error_return
    except Exception as e:
        print(f"エラー: ラグ適用/DataFrame作成中に予期せぬエラー: {e}")
        return error_return

    print("--- [Function@preprocessing] 1. 角速度データの前処理終了 ---")
    # 戻り値 (タプル形式)
    return sync_gyro_df, lag_samples, sampling_rate_hz, peak_index_left, peak_index_right, sync_left_short, sync_right_short
