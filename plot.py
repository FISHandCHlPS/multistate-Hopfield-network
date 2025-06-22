"""
描画用関数の定義
"""

import pandas as pd
import plotly.express as px
from jax import numpy as jnp
from jax.typing import ArrayLike

def plotTrajectory(history: ArrayLike, num_particles: int, title="Trajectories"):
    """
    == Plotly Expressによる軌跡描画 ==
    history: (steps+1, num_particles, 2)
    各パーティクルの軌跡を色分けしてプロット
    """
    history = jnp.asarray(history)

    records = []
    steps = history.shape[0]
    for t in range(steps):
        for i in range(num_particles):
            records.append({
                "t": t,
                "particle": i,
                "x": float(history[t, i, 0]),
                "y": float(history[t, i, 1])
            })
    df = pd.DataFrame(records)
    fig = px.line(df, x="x", y="y", color="particle", line_group="particle", title=title)
    fig.update_layout(xaxis_title="X", yaxis_title="Y")
    fig.show()


def animationTrajectory(history: ArrayLike, num_particles: int, interval: int=10):
    """
    == Plotly Expressによるアニメーション ==
    history: (steps+1, num_particles, 2)
    """
    history = jnp.asarray(history)
    # TODO:3Dplotでエネルギーの上昇を可視化
    # history → DataFrameへ変換
    records = []
    for t in range(0, history.shape[0], interval):
        for i in range(num_particles):
            records.append({
                "t": t,
                "particle": i,
                "x": float(history[t, i, 0]),
                "y": float(history[t, i, 1])
            })
    df = pd.DataFrame(records)

    fig = px.scatter(
        df,
        x="x",
        y="y",
        animation_frame="t",
        animation_group="particle",
        color="particle",
        color_continuous_scale="Viridis",
        range_x=[float(jnp.min(history[...,0]))-1, float(jnp.max(history[...,0]))+1],   # 全ての点が描画できる範囲を指定
        range_y=[float(jnp.min(history[...,1]))-1, float(jnp.max(history[...,1]))+1],
        title="Particle Animation (Plotly Express)",
        width=600,
        height=600,
    )
    fig.show()