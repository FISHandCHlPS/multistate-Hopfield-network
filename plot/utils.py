import numpy as np
import polars as pl
from jaxtyping import ArrayLike


def array2df(arr: ArrayLike, column_names: list[str] | None = None) -> pl.DataFrame:
    """配列を縦長DataFrameに整形する。

    全軸を列化し、値を value 列に格納します。
    `column_names` は全軸名（長さは `arr.ndim`）。
    未指定時は `idx_0, idx_1, ...` を自動付与します。

    Args:
        arr: 任意の配列。
        column_names: 全軸の列名。未指定時は自動付与。

    Returns:
        pd.DataFrame: インデックス列と `value` を持つデータフレーム。

    """
    arr = np.asarray(arr)

    if arr.ndim == 0:
        msg = "arr.ndim は 1 以上である必要があります"
        raise ValueError(msg)
    n_axes = arr.ndim
    if len(column_names) != n_axes:
        msg = "len(column_names) は arr.ndim と等しくなければなりません"
        raise ValueError(msg)

    # 列名リストを生成。未指定時は自動付与。
    idx_names = (
        list(column_names)
        if column_names is not None
        else [f"idx_{i}" for i in range(n_axes)]
    )

    # 各軸の座標を生成して縦に展開
    grids = np.indices(arr.shape)  # 形状: (ndim, *arr.shape)
    flat_coords = [g.reshape(-1) for g in grids]
    flat_values = arr.reshape(-1)

    data = {name: flat_coords[i] for i, name in enumerate(idx_names)}
    data["value"] = flat_values

    column_order = [*idx_names, "value"]
    return pl.DataFrame(data, columns=column_order)
