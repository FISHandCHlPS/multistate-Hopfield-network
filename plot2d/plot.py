"""
描画用関数の定義
"""
import pandas as pd
import plotly.express as px
import numpy as np
from numpy.typing import ArrayLike
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


def plotTrajectory(history: ArrayLike, title="Trajectories"):
    """
    == Plotly Expressによる軌跡描画 ==
    history: (steps+1, num_particles, 2)
    各パーティクルの軌跡を色分けしてプロット
    """
    history = np.asarray(history)
    df = arrayToDataFrame(history, interval=10)
    fig = px.line(df, x="x", y="y", color="particle", line_group="particle", title=title)
    fig.update_layout(xaxis_title="X", yaxis_title="Y")
    fig.show()


def animationTrajectory(history: ArrayLike, interval: int=10):
    """
    == Plotly Expressによるアニメーション ==
    history: (steps+1, num_particles, 2)
    """
    history = np.asarray(history)  # (steps+1, num_particles, 2)
    # TODO:3Dplotでエネルギーの上昇を可視化
    # history → DataFrameへ変換
    # 間引き
    df = arrayToDataFrame(history, interval)

    fig = px.scatter(
        df,
        x="x",
        y="y",
        animation_frame="t",
        animation_group="particle",
        color="particle",
        color_continuous_scale="Viridis",
        range_x=[float(np.min(history[:,:,0]))-1, float(np.max(history[:,:,0]))+1],   # 初期状態で全ての点が描画できる範囲を指定
        range_y=[float(np.min(history[:,:,1]))-1, float(np.max(history[:,:,1]))+1],
        title="Particle Animation (Plotly Express)",
        width=600,
        height=600,
    )
    fig.show()
    fig.write_html("animation.html")


def arrayToDataFrame(history: ArrayLike, interval: int=10):
    """
    history: (steps+1, num_particles, 2)
    配列に不正な値（NaN, inf, None）が含まれていないかチェックし、
    問題があれば例外を投げる。
    """
    selected = history[::interval]  # (steps+1//interval, num_particles, 2)
    steps = selected.shape[0]
    particles = selected.shape[1]
    t_arr = np.arange(0, history.shape[0], interval)
    t_col = np.repeat(t_arr, particles)
    particle_col = np.tile(np.arange(particles), steps)
    x_col = selected[:, :, 0].ravel()
    y_col = selected[:, :, 1].ravel()

    # 不正な値チェック
    def check_invalid(arr, name):
        arr_np = np.asarray(arr)
        if np.any(np.isnan(arr_np)):
            raise ValueError(f'{name} に NaN が含まれています')
        if np.any(np.isinf(arr_np)):
            raise ValueError(f'{name} に inf が含まれています')
        if arr_np.dtype == object and np.any([v is None for v in arr_np.ravel()]):
            raise ValueError(f'{name} に None が含まれています')

    check_invalid(x_col, "x_col")
    check_invalid(y_col, "y_col")

    df = pd.DataFrame({
        "t": t_col,
        "particle": particle_col,
        "x": x_col,
        "y": y_col
    })  
    return df