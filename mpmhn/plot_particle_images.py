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


def plot_pca_images(image_array: np.ndarray, n_components: int = 2) -> None:
    """
    画像配列をPCAで次元削減し、散布図としてプロットする。

    Args:
        image_array (np.ndarray): 画像配列 (画像数, 画像1枚の次元数)
        n_components (int): 主成分数（デフォルト2）

    Returns:
        None
    """
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(image_array)
    df = pd.DataFrame(reduced, columns=[f'PC{i+1}' for i in range(n_components)])
    fig = px.scatter(df, x='PC1', y='PC2', title='PCA of Images', width=800, height=600)
    fig.write_html("./output/pca.html")
    fig.show()


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
    dummy_images = np.random.rand(num_images, dim)
    plot_pca_images(dummy_images)


# テスト実行用
if __name__ == "__main__":
    # test_plot_particle_image_slider()
    test_plot_pca_images()
