# symmetry_indices.py
import numpy as np
import pandas as pd  # isnullチェック用 (np.isnanでも可)


def calculate_symmetry_index(value_L, value_R):
    """
    左右の値から対称性指数 (Symmetry Index, SI) を計算する。
    SI = |L - R| / (0.5 * (L + R)) * 100%
    値がNaNの場合や分母が0に近い場合はNaNを返す。
    注意: この計算式は値が0を跨がないパラメータに適している。
          0に近い値や正負が混在するパラメータ (例: 位相、一部の角速度) の
          対称性評価には別の指標(例: Symmetry Ratio) が適切な場合もある。

    Args:
        value_L (float): 左脚のパラメータ値 (例: 平均ストライド時間)
        value_R (float): 右脚のパラメータ値

    Returns:
        float: 計算された対称性指数 (%)。計算不能なら np.nan。
    """
    # 入力値が NaN かどうかチェック
    if pd.isna(value_L) or pd.isna(value_R):
        return np.nan

    # 左右の値が両方ともほぼゼロの場合、SIは定義しにくいので NaN を返す
    if np.isclose(value_L, 0) and np.isclose(value_R, 0):
        # print("  警告: SI計算で左右の値が両方ゼロに近いため NaN を返します。")
        return np.nan

    # 分母 (左右の平均値) を計算
    denominator = 0.5 * (value_L + value_R)

    # 分母がゼロに近い場合、ゼロ除算を避けるために NaN を返す
    # np.isclose で浮動小数点数の比較を行う
    if np.isclose(denominator, 0):
        # print(f"  警告: SI計算で分母がゼロに近いため NaN を返します (L={value_L}, R={value_R})。")
        return np.nan

    # 対称性指数を計算
    # LとRの差の絶対値を、左右の平均値の絶対値で割り、100を掛ける
    # 分母も絶対値を取るのが一般的 (特に負の値を含むパラメータの場合)
    si = np.abs(value_L - value_R) / np.abs(denominator) * 100.0

    return si

# --- 必要なら他の対称性指標（例：Ratio）もここに追加 ---
# def calculate_symmetry_ratio(value_L, value_R, method='min/max'):
#     """左右の値の比率を計算"""
#     if pd.isna(value_L) or pd.isna(value_R): return np.nan
#     # ゼロ除算や無意味な結果を避ける
#     if np.isclose(value_L, 0) and np.isclose(value_R, 0): return np.nan
#     if np.isclose(value_L, 0) or np.isclose(value_R, 0): return 0 # 片方が0なら比率は0? あるいはNaN?
#
#     if method == 'min/max':
#         # 常に 0 <= ratio <= 1 となるように、絶対値の小さい方を大きい方で割る
#         return min(abs(value_L), abs(value_R)) / max(abs(value_L), abs(value_R))
#     elif method == 'L/R':
#         return value_L / value_R
#     else:
#         raise ValueError("未知のRatio計算メソッド")
