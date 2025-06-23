"""
描画用関数の定義
"""

import pandas as pd
import plotly.express as px
from jax import numpy as jnp
from jax.typing import ArrayLike
import numpy as np
import plotly.graph_objects as go

def plotEnergySurface(func, xmin=-10, xmax=10, ymin=-10, ymax=10, num=100):
    """
    エネルギー関数E(x, y)の3Dサーフェスプロット（plotly版）
    func: エネルギー関数
    xmin, xmax, ymin, ymax: 描画範囲
    num: 分割数
    """
    x = np.linspace(xmin, xmax, num)
    y = np.linspace(ymin, ymax, num)
    X, Y = np.meshgrid(x, y)
    XY = np.stack([X.ravel(), Y.ravel()], axis=1)
    Z = np.array([func(xy) for xy in XY])
    Z = Z.reshape(X.shape)

    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])
    fig.update_layout(
        title="Energy Surface (E(x, y))",
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='E(x, y)'
        ),
        width=800,
        height=600,
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig.show()



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
    history = jnp.asarray(history)  # (steps+1, num_particles, 2)
    # TODO:3Dplotでエネルギーの上昇を可視化
    # history → DataFrameへ変換
    # 間引き
    selected = history[::interval]  # (steps+1//interval, num_particles, 2)
    steps = selected.shape[0]
    particles = selected.shape[1]
    t_arr = jnp.arange(0, history.shape[0], interval)
    t_col = jnp.repeat(t_arr, particles)
    particle_col = jnp.tile(jnp.arange(particles), steps)
    x_col = selected[:, :, 0].ravel()
    y_col = selected[:, :, 1].ravel()
    df = pd.DataFrame({
        "t": t_col,
        "particle": particle_col,
        "x": x_col,
        "y": y_col
    })

    fig = px.scatter(
        df,
        x="x",
        y="y",
        animation_frame="t",
        animation_group="particle",
        color="particle",
        color_continuous_scale="Viridis",
        range_x=[float(jnp.min(history[0,:,0]))-1, float(jnp.max(history[0,:,0]))+1],   # 初期状態で全ての点が描画できる範囲を指定
        range_y=[float(jnp.min(history[0,:,1]))-1, float(jnp.max(history[0,:,1]))+1],
        title="Particle Animation (Plotly Express)",
        width=600,
        height=600,
    )
    fig.show()
