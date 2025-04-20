# file_utils.py
import pandas as pd
from pathlib import Path
import os  # find_latest_csv_file で最終更新日時の取得に使用 (Path.stat().st_mtime でも可)


def find_latest_csv_file(folder_path: Path):
    """
    指定されたフォルダ内で最終更新日時が最新のCSVファイルを探す。

    Args:
        folder_path (Path): 検索対象フォルダのPathオブジェクト。

    Returns:
        Path or None: 最新のCSVファイルのPathオブジェクト。見つからない場合はNone。
    """
    folder_path = Path(folder_path)  # Pathオブジェクトであることを確認
    latest_file = None

    if not folder_path.is_dir():
        print(f"[エラー@file_utils] 指定されたデータフォルダが見つかりません: {folder_path}")
        return None

    # フォルダ内の .csv ファイルをリストアップ
    try:
        found_csv_files = list(folder_path.glob('*.csv'))
    except Exception as e:
        print(f"[エラー@file_utils] フォルダ内のファイル検索中にエラー: {e}")
        return None

    if not found_csv_files:
        print(f"[エラー@file_utils] データフォルダ内にCSVファイルが見つかりません: {folder_path}")
        return None

    # 最終更新日時で比較して最新のファイルを見つける
    try:
        # os.path.getmtime を使う場合
        # latest_file = max(found_csv_files, key=os.path.getmtime)

        # pathlib.Path.stat().st_mtime を使う場合 (よりモダン)
        latest_file = max(found_csv_files, key=lambda p: p.stat().st_mtime)
    except Exception as e:
        print(f"[エラー@file_utils] 最新ファイルの検索中にエラーが発生しました: {e}")
        return None

    # print(f"最新のCSVファイルが見つかりました: '{latest_file.name}'") # 呼び出し元で表示するのでここではコメントアウト
    return latest_file


def save_results(output_filename, data_to_save, description):
    """
    指定されたDataFrameを指定されたファイル名でCSVとして保存する。

    Args:
        output_filename (str or Path): 出力ファイル名 (フルパス推奨)。
        data_to_save (pd.DataFrame): 保存するデータ。
        description (str): 保存するデータの説明（ログ表示用）。
    """
    # output_filename を Path オブジェクトに変換
    output_path = Path(output_filename) if not isinstance(
        output_filename, Path) else output_filename

    # 保存するデータが有効かチェック
    if data_to_save is not None and isinstance(data_to_save, pd.DataFrame):
        if not data_to_save.empty:
            try:
                # 出力先フォルダが存在しない場合は作成する (親フォルダを含む)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                # CSVファイルに保存 (BOM付きUTF-8, インデックスなし)
                data_to_save.to_csv(
                    output_path, index=False, encoding='utf-8-sig')
                print(f"  結果 ({description}) を '{output_path.name}' に保存しました。")
            except Exception as e:
                # 保存中にエラーが発生した場合
                print(
                    f"  [エラー@file_utils] ファイル '{output_path.name}' ({description}) の保存中にエラーが発生しました: {e}")
        else:
            # 保存するデータフレームが空の場合
            print(f"  情報: 保存するデータ ({description}) が空のため、ファイル出力はスキップされました。")
    else:
        # 保存するデータが無効 (None や DataFrame 以外) の場合
        print(f"  情報: 保存するデータ ({description}) が無効なため、ファイル出力はスキップされました。")
