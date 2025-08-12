"""
類似度でソートし、上位の時系列推移をプロットする。
"""
import numpy as np
import plotly.express as px
import pandas as pd


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
    history_norm = history_images / (np.linalg.norm(history_images, axis=2, keepdims=True) + 1e-10)
    memory_norm = memory_images / (np.linalg.norm(memory_images, axis=1, keepdims=True) + 1e-10)
    cos_matrix = history_norm @ memory_norm.T  # (T, N, M)

    # PSNR行列の計算（全履歴・全粒子・全比較画像）
    MAX_I = 1.0
    mse = ((history_images[:, :, None, :] - memory_images[None, None, :, :]) ** 2).mean(axis=3)  # (T, N, M)
    psnr_matrix = 10 * np.log10(MAX_I ** 2 / (mse + 1e-10))  # (T, N, M)

    sim_matrix = cos_matrix
    #sim_matrix = psnr_matrix

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
    records = []
    for t in range(T):
        for n in range(N):
            records.append({
                't': t,
                'particle': f'粒子{n}',
                'top1': topk_scores[t, n, 0],
                'top2': topk_scores[t, n, 1]
            })

    df = pd.DataFrame(records)
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
