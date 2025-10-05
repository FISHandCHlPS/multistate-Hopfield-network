"""類似度でソートし、上位の時系列推移をプロットする。"""
from pathlib import Path
from typing import Literal

import numpy as np
import plotly.express as px
from jaxtyping import ArrayLike, Float

from plot.evaluation import calc_cos
from plot.utils import array2df


def sort_history_by_similarity(
    history_images: np.ndarray,
    memory_images: np.ndarray,
    top_k: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """履歴画像群と比較画像群との類似度を計算し、スコアの高い順に履歴画像を並び替える。

    Args:
        history_images (np.ndarray): 履歴画像群 (T, N, D)
        memory_images (np.ndarray): 比較画像群 (M, D)
        top_k (int, optional): 上位k件だけ返す（省略時は全件）

    Returns:
        topk_scores (np.ndarray): 並び替えたスコア (T, N, k)
        topk_indices (np.ndarray): 並び替えたインデックス (N, k)

    """
    # コサイン類似度（全履歴・全粒子・全比較画像）
    sim_matrix = calc_cos(history_images, memory_images.T)

    # 履歴内で最大となる類似度（N, M）
    memory_sim = np.max(sim_matrix, axis=0)  # (N, M)

    # 上位k個の比較画像（インデックスとスコア）を取得
    topk_mem_indices = np.argsort(memory_sim, axis=1)[:, -top_k::-1]  # (N, k)
    topk_mem_scores = np.take_along_axis(sim_matrix, topk_mem_indices[None], axis=2)  # (T, N, k)

    return topk_mem_scores, topk_mem_indices


def plot_similarity_trajectory(history_images: np.ndarray, memory_images: np.ndarray) -> None:
    """上位2つの類似度をx, y軸として、各粒子ごとに時系列の軌跡をプロットする。

    Args:
        history_images (np.ndarray): (T, N, D)
        memory_images (np.ndarray): (M, D)

    """
    topk_scores, _ = sort_history_by_similarity(history_images, memory_images, top_k=2)

    # データをロング形式のDataFrameに整形
    df = array2df(topk_scores)

    fig = px.line(
        df,
        x="top1",
        y="top2",
        animation_frame="t",
        color="particle",
        markers=True,
        line_group="particle",
        hover_data=["t"],
        title="粒子ごとの類似度軌跡（top1 vs top2）",
    )
    fig.update_layout(
        xaxis_title="top1類似度",
        yaxis_title="top2類似度",
        width=900,
        height=600,
    )
    fig.show()
    fig.write_html("./output/similarity_trajectory.html")


def plot_cos_sim(
    history_images: np.ndarray, memory_images: np.ndarray,
    path: str = "output", filename: str = "cosine_similarity.html",
) -> None:
    """各記憶に対するコサイン類似度の時間変化を可視化

    Args:
        history_images (np.ndarray): 形状 (T, N, D)。時刻 T、粒子 N、次元 D の履歴。
        memory_images (np.ndarray): 形状 (D, M)。記憶 M、本数はサブプロットの数になる。
        path (str): 出力先のディレクトリ。
        filename (str): 出力ファイル名。

    """
    sim_matrix = calc_cos(history_images, memory_images.T)  # 類似度 (T, N, M)

    df = array2df(sim_matrix, column_names=["t", "particle", "memory"])

    # サブプロットで各記憶ごとに可視化（粒子ごとに色分け）
    fig = px.line(
        df,
        x="t",
        y="value",
        color="particle",
        facet_col="memory",
        line_group="particle",
        markers=False,
        title="Cosine Similarity over Time per Memory (faceted)",
    )

    fig.update_yaxes(title_text="cosine similarity", range=[-1.0, 1.0])
    fig.update_xaxes(title_text="t")
    fig.update_layout(
        legend_title_text="particle",
        margin={"l": 40, "r": 40, "t": 60, "b": 40},
    )

    # 出力
    assert Path(path).is_dir(), f"出力ディレクトリが存在しません: {path}"
    fig.write_html(str(Path(path) / filename))
    fig.show()


def plot_cos_sim_per_param(
    multirun_data: list[dict[Literal["history", "weight", "initial", "config", "run_dir"]]],
    memory: Float[ArrayLike, "dim num_memory"],
    path: str = "output", filename: str = "cosine_similarity_per_param.html",
) -> None:
    """パラメータ毎にcosの時間変化をプロット

    list[dict]を受け取って、成形し、パラメータ毎の類似度をプロットする

    Args:
        multirun_data (list[dict]): パラメータ毎の結果
        memory (Float[ArrayLike, "dim num_memory"]): 記憶
        path (str): 出力先のディレクトリ。
        filename (str): 出力ファイル名。

    """
    # df = (
    #     # history列に多重配列、残りの列はパラメータ
    #     results_to_dataframe(multirun_data, loading_data="history")
    #     # 各要素に多重配列が入っているため、そのまま処理
    #     .with_columns(
    #         calc_cos(pl.col("history"), memory.T).alias("cos"),
    #     )
    #     .with_columns(
    #         array2df(pl.col("cos")).alias("cos_df"),
    #     )
    #     # .explode("cos_df")   # 多重配列を展開
    # )
    # print(df)

    # fig = px.line(
    #     df,
    #     x="cos_df.t",
    #     y="cos_df.value",
    #     color="particle",
    #     facet_col="memory",
    #     line_group="particle",
    #     markers=False,
    #     title="Cosine Similarity over Time per Memory (faceted)",
    # )
    # fig.update_yaxes(title_text="cosine similarity", range=[-1.0, 1.0])
    # fig.update_xaxes(title_text="t")
    # fig.update_layout(
    #     legend_title_text="particle",
    #     margin={"l": 40, "r": 40, "t": 60, "b": 40},
    # )

    # # 出力
    # assert Path(path).is_dir(), f"出力ディレクトリが存在しません: {path}"
    # fig.write_html(str(Path(path) / filename))
    # fig.show()
