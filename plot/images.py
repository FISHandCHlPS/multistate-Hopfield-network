import numpy as np
import plotly.express as px
import pandas as pd
from sklearn.decomposition import PCA

def plot_images_trajectory(history: np.ndarray, path: str='output', filename: str='recall_images.html') -> None:
    """
    plotly.expressで、時刻・粒子ごとにhistoryの各ベクトルを32x32画像として切り替え表示
    history: (steps+1, num_particles, 1024)
    """
    steps, num_particles, dim = history.shape
    xs = history.reshape(steps, num_particles, 32, 32)

    fig = px.imshow(
        xs,
        animation_frame=0,
        facet_col=1,
        facet_col_wrap=4,
        binary_string=True,
        labels={"animation_frame": "t", "facet_col": "particle"},
    )

    fig.update_layout(title="Particle Images (t, particle)")
    fig.write_html(f"{path}/{filename}")
    fig.show()


def plot_image(image: np.ndarray, path: str='output', filename: str='memory.html') -> None:
    """
    画像をplotlyで表示する。

    Args:
        image (np.ndarray): 画像配列 (画像数, 画像1枚の次元数)
        path (str, optional): 出力先のパス。Defaults to 'output'。
        filename (str, optional): 出力ファイル名。Defaults to 'memory.html'。
    """
    fig = px.imshow(
        image,
        binary_string=True
    )
    fig.update_layout(title="memory image")
    fig.write_html(f"{path}/{filename}")
    fig.show()
