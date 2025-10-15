"""t-SNEによる画像の可視化"""
import plotly.express as px
import polars as pl
from jaxtyping import ArrayLike
from sklearn.manifold import TSNE

from plot.utils import get_flat_coord


def plot_tsne_trajectory(
    history: ArrayLike,
    path: str = "output",
    filename: str = "tsne_trajectory.html",
) -> None:
    """画像群をt-SNEし、粒子ごとの軌跡を可視化する。

    Args:
        history (ArrayLike): 履歴配列 (steps, num_particles, 画像次元)
        path (str, optional): 出力先のディレクトリ。Defaults to "output".
        filename (str, optional): 出力ファイル名。Defaults to "tsne_trajectory.html".

    """
    _, _, dim = history.shape
    tsne = TSNE(n_components=2, random_state=42)
    reduced = tsne.fit_transform(history.reshape(-1, dim))
    reduced_df = pl.DataFrame(reduced, schema=["value_0", "value_1"])

    # historyの座標を生成
    coord = get_flat_coord(history[..., 0])   # (steps, num_particles)のインデックスを生成
    coord_df = pl.DataFrame(coord, schema=["t", "particles"])

    # t-SNE結果を結合
    tsne_df = pl.concat([coord_df, reduced_df], how="horizontal")

    # 可視化
    fig = px.scatter(
        tsne_df,
        x="value_0",
        y="value_1",
        color="particles",
        animation_group="particles",
        animation_frame="t",
        range_x=[-60, 60],
        range_y=[-60, 60],
        title="t-SNE Trajectory of Particles",
        width=800,
        height=600,
    )
    fig.write_html(f"{path}/{filename}")
    fig.show()
