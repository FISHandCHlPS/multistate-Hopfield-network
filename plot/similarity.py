"""類似度でソートし、上位の時系列推移をプロットする。"""
from pathlib import Path
from typing import Literal

import numpy as np
import plotly.express as px
import polars as pl
from jaxtyping import ArrayLike, Float
from matplotlib import cm
from matplotlib.colors import Normalize

from plot.evaluation import calc_cos
from plot.loader import extract_data, extract_parameters
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


SPEED = np.array([
    0.20502764, 0.10907875, 0.23376891, 0.23414722, 0.18664654, 0.11215612,
    0.17957862, 0.2656873,  0.25192973, 0.22420937, 0.5778988,  0.38830575,
    0.48060235, 0.44633207, 0.7669708,  0.538585,   0.5703915,  0.34581333,
    0.4458383,  0.5966047,
])


def plot_cos(
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
    sim_matrix = calc_cos(history_images, memory_images)  # 類似度 (T, N, M)

    df = (
        array2df(sim_matrix, column_names=["t", "particle", "memory"])
        # .with_columns(
        #     pl.col("particle")
        #     .cast(pl.Int64).map_elements(lambda x: SPEED[x])
        #     .alias("speed"),
        # )
    )
    # 正規化とカラーマップ適用
    norm = Normalize(vmin=SPEED.min(), vmax=SPEED.max())
    cmap = cm.get_cmap("viridis")
    colors_rgba = cmap(norm(SPEED))
    # RGB文字列形式に変換
    colors_rgb = [f"rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})" for c in colors_rgba]

    # サブプロットで各記憶ごとに可視化（粒子ごとに色分け）
    fig = px.line(
        df,
        x="t",
        y="value",
        color="particle",
        facet_col="memory",
        line_group="particle",
        markers=False,
        color_discrete_sequence=colors_rgb,
        title="Cosine Similarity over Time per Memory",
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


def plot_cos_trajectory(
    history_images: np.ndarray, memory_images: np.ndarray,
    path: str = "output", filename: str = "cosine_trajectory.html",
) -> None:
    """各記憶に対するコサイン類似度の時間変化を可視化

    軌跡の線と開始地点の点を表示する

    Args:
        history_images (np.ndarray): 形状 (T, N, D)。時刻 T、粒子 N、次元 D の履歴。
        memory_images (np.ndarray): 形状 (D, M)。記憶 M、本数はサブプロットの数になる。
        path (str): 出力先のディレクトリ。
        filename (str): 出力ファイル名。

    """
    sim_matrix = calc_cos(history_images, memory_images)  # 類似度 (T, N, M)
    diff_sim = sim_matrix[..., 2] - sim_matrix[..., 1]  # 縦軸
    memory_sim = sim_matrix[..., 0]     # 横軸

    df_memory = (
        array2df(memory_sim, column_names=["t", "particle"])
        .sort(by="t")
        .with_columns(pl.col("value").alias("value_x"))
    )
    df_diff = (
        array2df(diff_sim, column_names=["t", "particle"])
        .sort(by="t")
        .with_columns(pl.col("value").alias("value_y"))
    )
    df = df_memory.join(df_diff, on=["t", "particle"])

    # 軌跡の線を表示
    fig = px.line(
        df,
        x="value_x",
        y="value_y",
        color="particle",
        line_group="particle",
        markers=False,
        title="Cosine Similarity over Time",
    )

    # 開始時点（t=0）のデータのみを抽出
    df_start = df.filter(pl.col("t") == 0)
    # 開始地点の点を追加
    fig.add_trace(
        px.scatter(
            df_start,
            x="value_x",
            y="value_y",
            color="particle",
        ).data[0],
    )

    fig.update_yaxes(title_text="cosine of memory B - C", range=[-1.2, 1.2])
    fig.update_xaxes(title_text="cosine of memory A", range=[-1.1, 1.1])
    fig.update_layout(
        legend_title_text="particle",
        margin={"l": 40, "r": 40, "t": 60, "b": 40},
    )

    # 出力
    assert Path(path).is_dir(), f"出力ディレクトリが存在しません: {path}"
    fig.write_html(str(Path(path) / filename))
    fig.show()


def plot_cos_multirun(
    multirun_data: list[dict[Literal["history", "weight", "initial", "config", "run_dir"]]],
    memory: Float[ArrayLike, "dim num_memory"],
) -> None:
    """パラメータ毎にcosの時間変化をプロット

    list[dict]を受け取って、成形し、パラメータ毎の類似度をプロットする

    Args:
        multirun_data (list[dict]): パラメータ毎の結果
        memory (Float[ArrayLike, "dim num_memory"]): 記憶

    """
    params = extract_parameters(multirun_data)  # パラメータ
    params_filtered = params.filter(pl.col("beta").is_between(2.5, 3.5)).rename({"index": "filter_idx"})
    filter_idx = params_filtered.select("filter_idx").to_numpy().flatten()

    # フィルタリングしたパラメータの履歴
    filtered_data = [multirun_data[i] for i in filter_idx]
    history = extract_data(filtered_data, loading_data="history")
    sim_matrix = calc_cos(history, memory)  # 類似度 (run, step, particles, n_memory)
    data_df = array2df(
        sim_matrix,
        column_names=["index", "t", "particles_idx", "memory_idx"],
    )  # 新たにインデックスが作成される
    params_filtered = params_filtered.with_row_index("index")   # 上に合わせてインデックスを振り直す

    df_with_params = (
        data_df.join(params_filtered, on="index")
        .drop("index")
        .with_columns(
            pl.when(pl.col("particles_idx") < 15)
            .then(0)
            .otherwise(1)
            .alias("speed"),
        )
    )

    # サブプロットで各記憶ごとに可視化（粒子ごとに色分け）
    def plot(df: pl.DataFrame, title: str) -> None:
        fig = px.line(
            df,
            x="t",
            y="value",
            color="speed",
            facet_row="beta",
            facet_col="gamma",
            line_group="particles_idx",
            markers=False,
            title=title,
        )

        fig.update_yaxes(title_text="cosine similarity", range=[-1.0, 1.0])
        fig.update_xaxes(title_text="t")
        fig.update_layout(
            legend_title_text="particles_idx",
            margin={"l": 40, "r": 40, "t": 60, "b": 40},
        )
        fig.show()

    for i in range(3):
        df_ploting = df_with_params.filter(pl.col("memory_idx") == i)
        plot(df_ploting, title = f"cosine similarity of memory {i}")


