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
        aspect="auto",
    )

    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.write_html(f"{path}/{filename}")
    fig.show()


def plot_images(
    images: np.ndarray,
    title: str = "Particle Images",
    path: str="output",
    filename: str="images.html",
) -> None:
    """
    粒子ごとにimagesの各ベクトルを32x32画像として表示

    images: (num_particles, 1024)
    """
    if images.ndim == 1:  # 単一画像の場合
        images = images[None, ...]
    
    if images.ndim == 3:  # 履歴の場合
        images = images[-1, ...]   # 最後の粒子のみ
    
    num_particles, dim = images.shape
    assert dim == 1024, f"dim {dim} と 1024 が一致しません"
    
    _images = images.reshape(num_particles, 32, 32)

    fig = px.imshow(
        _images,
        facet_col=0,
        facet_col_wrap=5,
        facet_col_spacing=0.1,
        binary_string=True,
        labels={"facet_col": "particle"},
        title=title,
        width=1000,
        aspect="auto",
    )
    fig.write_html(f"{path}/{filename}")
    fig.show()
