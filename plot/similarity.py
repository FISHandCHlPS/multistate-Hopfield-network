"""
類似度でソートし、上位の時系列推移をプロットする。
"""
import numpy as np
import plotly.express as px
import pandas as pd
from pathlib import Path
from plot.utils import array2df

def calc_cos(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    コサイン類似度を計算する。

    Args:
        X (np.ndarray): (T, N, D)
        Y (np.ndarray): (D, M)

    Returns:
        cos_matrix (np.ndarray): (T, N, M)
    """
    X_norm = X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-10)
    Y_norm = Y / (np.linalg.norm(Y, axis=-1, keepdims=True) + 1e-10)
    cos_matrix = X_norm @ Y_norm.T  # (T, N, M)
    return cos_matrix


def calc_psnr(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    PSNRを計算する。

    Args:
        X (np.ndarray): (T, N, D)
        Y (np.ndarray): (M, D)

    Returns:
        psnr_matrix (np.ndarray): (T, N, M)
    """
    MAX_I = 1.0
    mse = ((X[..., None, :] - Y[None, :, :]) ** 2).mean(axis=-1)  # (T, N, M)
    psnr_matrix = 10 * np.log10(MAX_I ** 2 / (mse + 1e-10))  # (T, N, M)
    return psnr_matrix


def sort_history_by_similarity(
    history_images: np.ndarray,
    memory_images: np.ndarray,
    top_k: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """
    履歴画像群と比較画像群との類似度を計算し、スコアの高い順に履歴画像を並び替える。

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
    """
    上位2つの類似度をx, y軸として、各粒子ごとに時系列の軌跡をプロットする。

    Args:
        history_images (np.ndarray): (T, N, D)
        memory_images (np.ndarray): (M, D)
    """
    T, N, D = history_images.shape
    topk_scores, _ = sort_history_by_similarity(history_images, memory_images, top_k=2)

    # データをロング形式のDataFrameに整形
    df = array2df(topk_scores)

    fig = px.line(
        df,
        x='top1',
        y='top2',
        animation_frame='t',
        color='particle',
        markers=True,
        line_group='particle',
        hover_data=['t'],
        title='粒子ごとの類似度軌跡（top1 vs top2）'
    )
    fig.update_layout(
        xaxis_title='top1類似度',
        yaxis_title='top2類似度',
        width=900,
        height=600
    )
    fig.show()
    fig.write_html("./output/similarity_trajectory.html")


def plot_cos_sim(
    history_images: np.ndarray,
    memory_images: np.ndarray,
    path: str = "output",
    filename: str = "cosine_similarity.html",
) -> None:
    """
    各記憶（`memory_images` の各ベクトル）に対するコサイン類似度の時間変化を、
    Plotly のサブプロット（facet）でまとめて可視化する。

    Args:
        history_images (np.ndarray): 形状 (T, N, D)。時刻 T、粒子 N、次元 D の履歴。
        memory_images (np.ndarray): 形状 (D, M)。記憶 M、本数はサブプロットの数になる。
        path (str): 出力ディレクトリ。
        filename (str): 出力 HTML ファイル名。
    """
    # 類似度 (T, N, M)
    sim_matrix = calc_cos(history_images, memory_images.T)

    T, N, M = sim_matrix.shape

    # dataframe列名: t, particle, memory, similarity
    mi = pd.MultiIndex.from_product(
        [range(T), range(N), range(M)], names=["t", "particle", "memory"]
    )
    df = (
        pd.Series(sim_matrix.reshape(-1), index=mi, name="similarity")
        .reset_index()
    )

    # サブプロットで各記憶ごとに可視化（粒子ごとに色分け）
    fig = px.line(
        df,
        x="t",
        y="similarity",
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
        margin=dict(l=40, r=40, t=60, b=40),
    )

    # 出力
    assert Path(path).is_dir(), f"出力ディレクトリが存在しません: {path}"
    fig.write_html(str(Path(path) / filename))
    fig.show()

