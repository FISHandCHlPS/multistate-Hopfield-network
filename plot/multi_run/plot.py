"""
マルチラン用の関数（類似度の分散を可視化）
"""

from typing import Optional

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px

from plot.similarity import calc_cos


def plot_multi_run(
    history: np.ndarray,
    weight: np.ndarray,
    path: str = "output",
    bar_filename: Optional[str] = "similarity_variance_bar.html",
    time_filename: Optional[str] = "similarity_variance_time.html",
) -> None:
    """
    粒子ごとの画像と記憶ベクトル（weight）とのコサイン類似度を計算し、
    粒子方向（並列方向）の分散を可視化する。

    - 棒グラフ: 最終時刻 t=-1 における各記憶ごとの類似度分散
    - 折れ線: 時間方向の類似度分散推移（記憶ごと）

    Args:
        history (np.ndarray): 履歴配列 (T, N, D)
        weight (np.ndarray): 記憶ベクトル (D, M)
        path (str): 出力ディレクトリ
        bar_filename (Optional[str]): 最終時刻の分散を描く棒グラフのHTML出力ファイル名
        time_filename (Optional[str]): 分散の時間推移を描く折れ線グラフのHTML出力ファイル名

    Returns:
        None
    """
    if history.ndim != 3:
        raise ValueError(
            f"history.ndim={history.ndim} ですが、(time, parallel, dimension) の3次元配列が必要です。"
        )
    if weight.ndim != 2:
        raise ValueError(
            f"weight.ndim={weight.ndim} ですが、(dimension, memories) の2次元配列が必要です。"
        )

    T, N, D = history.shape
    Dw, M = weight.shape
    if D != Dw:
        raise ValueError(f"history の次元 D={D} と weight の次元 Dw={Dw} が一致しません。")

    # 類似度 (T, N, M)
    sim_matrix: np.ndarray = calc_cos(history, weight.T)

    # 粒子方向の分散（母分散; ddof=0）: (T, M)
    var_over_particles: np.ndarray = sim_matrix.var(axis=1)

    # 1) 最終時刻 t=-1 の分散を記憶ごとに棒グラフ
    final_var: np.ndarray = var_over_particles[-1]  # (M,)
    fig_bar = px.bar(
        x=list(range(M)),
        y=final_var,
        labels={"x": "memory", "y": "variance (over particles)"},
        title="最終時刻における類似度分散（記憶ごと）",
    )
    if bar_filename is not None:
        assert Path(path).is_dir(), f"出力ディレクトリが存在しません: {path}"
        fig_bar.write_html(f"{path}/{bar_filename}")
    fig_bar.show()

    # 2) 分散の時間推移（記憶ごと）を長い形式のDataFrameに整形して折れ線表示
    df = pd.DataFrame(
        var_over_particles,
        columns=[f"mem_{i}" for i in range(M)],
    )
    df["t"] = np.arange(T)
    df_long = df.melt(id_vars=["t"], var_name="memory", value_name="variance")

    fig_time = px.line(
        df_long,
        x="t",
        y="variance",
        color="memory",
        title="類似度分散の時間推移（記憶ごと）",
        markers=False,
    )
    fig_time.update_xaxes(title_text="t")
    fig_time.update_yaxes(title_text="variance (over particles)")
    if time_filename is not None:
        assert Path(path).is_dir(), f"出力ディレクトリが存在しません: {path}"
        fig_time.write_html(f"{path}/{time_filename}")
    fig_time.show()


if __name__ == "__main__":
    # 例: 直近ランの result.npz を指定して確認
    res = np.load("output/multi_pattern_mhn/multirun/lr=0.1,gamma=0.1,c=1.0,beta=30.0/seed=0/result.npz")
    history = res["history"]  # (T, N, D)
    weight = res["weight"]    # (D, M)
    plot_multi_run(history, weight, path="output")



