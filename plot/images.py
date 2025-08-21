import numpy as np
import plotly.express as px


def plot_images_trajectory(history: np.ndarray, path: str='output', filename: str='recall_images.html', interval: int=1) -> None:
    """
    plotly.expressで、時刻・粒子ごとにhistoryの各ベクトルを32x32画像として切り替え表示
    history: (steps+1, num_particles, 1024)
    """
    steps, num_particles, dim = history.shape
    xs = history.reshape(steps, num_particles, 32, 32)

    fig = px.imshow(
        xs[::interval],
        animation_frame=0,
        facet_col=1,
        facet_col_wrap=5,
        facet_col_spacing=0.1,
        binary_string=True,
        labels={"animation_frame": "t", "facet_col": "particle"},
        title="Particle Images (t, particle)",
        width=1000,
        aspect="auto"
    )

    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
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
        binary_string=True,
        title="memory image"
    )
    fig.write_html(f"{path}/{filename}")
    fig.show()
