"""エントロピー可視化ユーティリティ"""

from pathlib import Path
from typing import Literal

import numpy as np
import plotly.express as px
from jaxtyping import ArrayLike, Float
from plotly import graph_objects as go
from scipy.interpolate import griddata
from scipy.spatial import QhullError

from plot.evaluation import calc_entropy
from plot.loader import eval_results

MIN_POINTS_FOR_LINEAR_INTERP = 4


def plot_entropy_time_series(
    history: Float[ArrayLike, "steps num_particles dim"],
    w: Float[ArrayLike, "dim n_memory"],
    path: str = "output",
    filename: str = "entropy_time_series.html",
    interval: int = 1,
) -> None:
    """履歴データに基づくエントロピーの時間変化をプロットする。

    Args:
        history (np.ndarray): (steps, num_particles, dim) 形式の粒子履歴。
        w (np.ndarray): (dim, n_memory) 形式の記憶ベクトル。
        path (str, optional): 出力先ディレクトリ。デフォルトは "output"。
        filename (str, optional): 出力ファイル名。デフォルトは "entropy_time_series.html"。
        interval (int, optional): プロットするステップ間隔。デフォルトは 1。

    """
    if interval <= 0:
        message = "interval は正の整数で指定してください。"
        raise ValueError(message)

    entropies = np.asarray(calc_entropy(history, w))

    if entropies.ndim == 1:
        entropy_mean = entropies
    else:
        entropy_2d = entropies.reshape(-1, entropies.shape[-1])
        entropy_mean = entropy_2d.mean(axis=0)

    steps = np.arange(entropy_mean.shape[0])

    fig = px.line(
        x=steps[::interval],
        y=entropy_mean[::interval],
        labels={"x": "t", "y": "Entropy"},
        title="Entropy over Time",
    )
    fig.update_layout(
        xaxis_title="t",
        yaxis_title="Entropy",
        yaxis={"range": [0.5, 3.5]},
    )

    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_dir / filename)
    fig.show()


def plot_entropy_multirun(
    multirun_data: list[dict[Literal["history", "weight", "initial", "config", "run_dir"]]],
    memory: Float[ArrayLike, "dim num_memory"],
) -> None:
    """多ランダムシミュレーションのエントロピーをプロットする"""

    def mean_entropy(
        history: Float[ArrayLike, "trial steps num_particles dim"],
    ) -> Float[ArrayLike, " trial"]:
        """最後の50ステップの平均エントロピーを返す"""
        return np.mean(calc_entropy(history, memory)[..., -50:], axis=-1)

    eval_df = (
        eval_results(multirun_data, mean_entropy, column_name="entropy")
        .unnest("runtime")
        .select("eta", "noise_amount", "entropy")
    )
    eta_values = np.asarray(eval_df.get_column("eta").to_numpy(), dtype=float)
    noise_values = np.asarray(eval_df.get_column("noise_amount").to_numpy(), dtype=float)
    entropy_values = np.asarray(eval_df.get_column("entropy").to_numpy(), dtype=float)

    points = np.column_stack((eta_values, noise_values))
    fig = go.Figure()
    unique_points = np.unique(points, axis=0)
    can_interpolate = unique_points.shape[0] >= MIN_POINTS_FOR_LINEAR_INTERP

    if can_interpolate:
        eta_grid = np.linspace(
            float(np.min(eta_values)),
            float(np.max(eta_values)),
            60,
        )
        noise_grid = np.linspace(
            float(np.min(noise_values)),
            float(np.max(noise_values)),
            60,
        )
        grid_x, grid_y = np.meshgrid(eta_grid, noise_grid)
        try:
            linear_interp = griddata(points, entropy_values, (grid_x, grid_y), method="linear")
        except QhullError:
            can_interpolate = False
        else:
            nearest_interp = griddata(points, entropy_values, (grid_x, grid_y), method="nearest")
            if linear_interp is None:
                entropy_grid = np.asarray(nearest_interp, dtype=float)
            else:
                entropy_grid = np.asarray(
                    np.where(np.isnan(linear_interp), nearest_interp, linear_interp),
                    dtype=float,
                )

    if can_interpolate:
        fig.add_trace(
            go.Heatmap(
                x=eta_grid,
                y=noise_grid,
                z=entropy_grid,
                colorscale="Viridis",
                zsmooth="best",
                colorbar={"title": "Entropy"},
                opacity=0.7,
            ),
        )
    else:
        fig.add_annotation(
            text="データ点が不足しているため補間をスキップしました",
            xref="paper",
            yref="paper",
            x=0.5,
            y=1.08,
            showarrow=False,
        )
    fig.add_trace(
        go.Scatter(
            x=eta_values,
            y=noise_values,
            mode="markers",
            marker={
                "size": 10,
                "color": entropy_values,
                "colorscale": "Viridis",
                "showscale": False,
                "line": {"width": 0.5, "color": "white"},
            },
            hovertemplate="η=%{x:.3f}<br>ノイズ=%{y:.3f}<br>エントロピー=%{marker.color:.3f}<extra></extra>",
        ),
    )
    title_suffix = "（補間付き）" if can_interpolate else "（補間なし）"
    fig.update_layout(
        title=f"エントロピー分布{title_suffix}",
        xaxis_title="η",
        yaxis_title="ノイズ量",
    )
    fig.show()

