"""PCAによる画像の可視化"""
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA


def plot_pca_feature(
    history: np.ndarray,
    k: int = 6,
    img_shape: tuple[int, int] = (32, 32),
    path: str = "output", filename: str = "pca_feature.html",
) -> None:
    """PCAの主成分ベクトル（=画像）をk個可視化する。

    Args:
        history (np.ndarray): 履歴配列 (steps, num_particles, 画像次元)
        k (int, optional): 可視化する主成分の個数。Defaults to 4.
        img_shape (tuple[int, int], optional): 画像の(H, W)。Defaults to (32, 32)
        path (str, optional): 出力先のディレクトリ。Defaults to "output".
        filename (str, optional): 出力ファイル名。Defaults to "pca_feature.html".

    """
    _, _, dim = history.shape
    h, w = img_shape
    if h * w != dim:
        raise ValueError(f"img_shape {img_shape} の画素数 {h*w} と特徴次元 {dim} が一致しません")

    pca = PCA(n_components=k)
    pca.fit(history.reshape(-1, dim))  # 履歴全体で学習
    features = pca.components_  # (k, dim)

    images = features.reshape(k, h, w)

    fig = px.imshow(
        images,
        facet_col=0,
        facet_col_wrap=3,
        binary_string=True,
        labels={"facet_col": "PC"},
    )
    fig.update_layout(title=f"PCA Components (k={k}) as Images")
    fig.write_html(f"{path}/{filename}")
    fig.show()


def plot_pca_trajectory(
    history: np.ndarray,
    path: str="output", filename: str="pca_trajectory.html",
) -> None:
    """画像群をPCAし、粒子ごとの軌跡を可視化する。

    Args:
        history (np.ndarray): 画像配列 (steps, num_particles, 画像次元)
        path (str, optional): 出力先のディレクトリ。Defaults to "output".
        filename (str, optional): 出力ファイル名。Defaults to "pca_trajectory.html".

    """
    steps, num_particles, dim = history.shape
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(history.reshape(-1, dim))
    explained_var = pca.explained_variance_ratio_
    print(f"第二主成分までの累積寄与率: {np.sum(explained_var[:2])}")

    df = pd.DataFrame(reduced, columns=[f"PC{i+1}" for i in range(2)])
    idx = pd.MultiIndex.from_product([
        range(steps),
        range(num_particles),
    ], names=["t", "particles"])
    df.index = idx
    df = df.reset_index()

    fig = px.scatter(
        df,
        x="PC1",
        y="PC2",
        color="particles",
        animation_group="particles",
        animation_frame="t",
        range_x=[-1, 1],
        range_y=[-1, 1],
        title="PCA Trajectory of Particles",
        width=800,
        height=600,
    )
    fig.write_html(f"{path}/{filename}")
    fig.show()


def plot_pca_ccr(
    history: np.ndarray, k: int = 20,
    path: str="output", filename: str="pca_ccr.html",
) -> None:
    """画像群をPCAし、粒子ごとの累積寄与率を可視化する。

    Args:
        history (np.ndarray): 画像配列 (steps, num_particles, 画像次元)
        k (int, optional): 可視化する主成分の個数。Defaults to 20.
        path (str, optional): 出力先のディレクトリ。Defaults to "output".
        filename (str, optional): 出力ファイル名。Defaults to "pca_ccr.html".

    """
    _, _, dim = history.shape
    pca = PCA(n_components=k)
    pca.fit(history.reshape(-1, dim))  # 履歴全体で学習
    explained_var = pca.explained_variance_ratio_
    ccr = np.cumsum(explained_var)

    df = pd.DataFrame(ccr, columns=["CCR"])
    fig = px.line(df, x=df.index, y="CCR", title="PCA CCR of Particles", width=800, height=600)
    fig.write_html(f"{path}/{filename}")
    fig.show()
