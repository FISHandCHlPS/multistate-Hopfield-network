"""エントロピー可視化ユーティリティ"""

from pathlib import Path
from typing import Literal

import numpy as np
import plotly.express as px
from jaxtyping import ArrayLike, Float

from plot.evaluation import calc_entropy
from plot.loader import eval_results


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

    df = eval_results(multirun_data, mean_entropy, column_name="entropy")

    

    fig = px.imshow(df, x="gamma", y="trial", z="entropy")
    fig.show()
