import numpy as np
import plotly.express as px
import pandas as pd
from sklearn.decomposition import PCA
import os


def to_uint8(img):
    """
    正規化してuint8に変換
    """
    img = np.asarray(img)
    if img.min() < 0 or img.max() > 1:
        img = (img - img.min()) / (img.max() - img.min())
    img = (img * 255).astype(np.uint8)
    return img

def plot_particle_image_slider(history):
    """
    plotly.expressで、時刻・粒子ごとにhistoryの各ベクトルを32x32画像として切り替え表示
    history: (steps+1, num_particles, 1024)
    """
    steps, num_particles, dim = history.shape
    xs = np.asarray(history)

    images = []
    t_list = []
    p_list = []
    img = xs#to_uint8(xs)
    for t in range(steps):
        for i in range(num_particles):
            images.append(img[t, i].reshape(32, 32))
            t_list.append(t)
            p_list.append(i)
    images = np.stack(images, axis=0)  # (num_images, 32, 32)

    fig = px.imshow(
        images,
        animation_frame=0,
        labels={"animation_frame": "index"},
        binary_string=False,
        color_continuous_scale="gray"
    )
    steps_labels = [f"t={t},p={p}" for t, p in zip(t_list, p_list)]
    for k, step in enumerate(fig.layout.sliders[0]['steps']):
        step['label'] = steps_labels[k]

    fig.update_layout(title="Particle Images (t, particle)")
    fig.write_html("./output/recall_images.html")
    fig.show()


def plot_img(image):
    fig = px.imshow(
        image,
        color_continuous_scale="gray"
    )
    fig.update_layout(title="memory image")
    fig.write_html("./output/memory.html")
    fig.show()


def plot_pca_images(image_array: np.ndarray) -> None:
    """
    画像配列をPCAで次元削減し、散布図としてプロットする。

    Args:
        image_array (np.ndarray): 画像配列 (画像数, 画像1枚の次元数)

    Returns:
        None
    """
    pca = PCA()
    reduced = pca.fit_transform(image_array)
    explained_var = pca.explained_variance_ratio_
    print(f"寄与率: {explained_var}")
    print(f"第二主成分までの累積寄与率: {np.sum(explained_var[:2])}")
    df = pd.DataFrame(reduced, columns=[f'PC{i+1}' for i in range(reduced.shape[1])])
    fig = px.scatter(df, x='PC1', y='PC2', title='PCA of Images', width=800, height=600)
    fig.write_html("./output/pca.html")
    fig.show()


def plot_pca_images_trajectory(history: np.ndarray) -> None:
    """
    PCAし、粒子ごとの軌跡を可視化する。

    Args:
        history (np.ndarray): 画像配列 (steps, num_particles, 画像次元)

    Returns:
        None
    """
    steps, num_particles, dim = history.shape
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(history.reshape(-1, dim))
    explained_var = pca.explained_variance_ratio_
    print(f"第二主成分までの累積寄与率: {np.sum(explained_var[:2])}")
    
    records = []
    for t in range(steps):
        for p in range(num_particles):
            records.append({
                't': t,
                'particle': p,
                'PC1': reduced[t*num_particles + p, 0],
                'PC2': reduced[t*num_particles + p, 1],
            })
    df = pd.DataFrame(records)

    fig = px.scatter(
        df,
        x='PC1',
        y='PC2',
        color='particle',
        animation_group='particle',
        animation_frame='t',
        range_x=[-1, 1],
        range_y=[-1, 1],
        title='PCA Trajectory of Particles',
        width=800,
        height=600
    )
    fig.write_html("./output/pca_trajectory.html")
    fig.show()


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
    sim_matrix = history_norm @ memory_norm.T  # (T, N, M)

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


def test_plot_similarity_trajectory() -> None:
    """
    plot_similarity_trajectory関数のテスト用関数。
    ダミーデータ（steps=5, num_particles=3, 1024次元, memory=4枚）で呼び出し動作を確認。
    """
    steps = 5
    num_particles = 3
    dim = 1024
    num_memory = 4
    np.random.seed(42)
    history = np.random.rand(steps, num_particles, dim)
    memory = np.random.rand(num_memory, dim)
    print('test history.shape:', history.shape)
    print('test memory.shape:', memory.shape)
    plot_similarity_trajectory(history, memory)


def test_plot_particle_image_slider():
    """
    plot_particle_image_sliderのテスト用関数。
    ダミーデータ（steps=3, num_particles=2, 1024次元）で呼び出し動作を確認。
    """
    steps = 3
    num_particles = 2
    dim = 1024
    np.random.seed(0)
    history = np.random.rand(steps+1, num_particles, dim)
    print('test history.shape:', history.shape)
    plot_particle_image_slider(history)


def test_plot_pca_images() -> None:
    """
    plot_pca_images関数のテスト用関数。
    ダミーデータ（10枚、各画像1024次元）で呼び出し動作を確認。
    """
    num_images = 10
    dim = 1024
    np.random.seed(0)
    dummy_images = np.random.rand(5, num_images, dim)
    plot_pca_images_trajectory(dummy_images)


# テスト実行用
if __name__ == "__main__":
    # test_plot_particle_image_slider()
    # test_plot_pca_images()
    test_plot_similarity_trajectory()
