import jax
import jax.numpy as jnp
import jax.lax as lax
from bokeh.plotting import figure, show
from bokeh.palettes import Category10
from jax import Array
from jax.typing import ArrayLike
from typing import Any
from jax import random


# 初期値やパラメータ
num_particles = 100      # 並列化する粒子数
key = random.PRNGKey(42)
init_mean, init_std = 0.0, 5.0
initial = random.normal(key, shape=(num_particles, 2)) * init_std + init_mean
# 格子状の初期値
# initial = jnp.stack(jnp.meshgrid(jnp.arange(5, dtype=float), jnp.arange(2, 7, dtype=float), indexing='ij'), axis=-1).reshape(-1, 2, order='F')
learning_rate = 0.5     # 学習率
steps = 500             # 合計ステップ数


def E(x: ArrayLike) -> Array:
    """
    最小化したいエネルギー関数。
    x: jax.Array (shape=(n,)) またはスカラー
    return: スカラー
    """
    x = jnp.asarray(x)
    amplitude = 1.0
    mu = jnp.array([2,3], dtype=float)
    sigma = 5.0
    e = amplitude * jnp.exp( jnp.sum( -((x - mu)**2 / (2 * sigma**2)), axis=-1))
    return -e


def calc_force(x0: ArrayLike, x1: ArrayLike) -> Array:  # ([d], [d]) -> [d]
    """
    相互作用関数
    近づくほど強く反発する  斥力
    距離×力 x0に着目して、x1方向に働く
    """
    alpha = 1.0  # 斥力の強さ
    vec = x1 - x0
    force = 1 / (jnp.dot(vec, vec)**1 + 1e-10)
    return alpha * force * vec
# 他の全ての粒子との相互作用を計算
calc_force_v = jax.vmap(calc_force, in_axes=(None, 0), out_axes=0)  # ([d], [n,d]) -> [n,d]
# 全粒子同士の相互作用を計算
calc_force_all = jax.vmap(calc_force_v, in_axes=(0, None), out_axes=0)  # ([n,d], [n,d]) -> [n,d]

def total_force(x: ArrayLike) -> Array:
    """
    各粒子に働く合力を計算（自己相互作用を除く）
    """
    forces = calc_force_all(x, x)  # shape: (n, n, d)
    n = x.shape[0]
    mask = 1 - jnp.eye(n, dtype=bool)  # shape: (n, n)
    forces = forces * mask[:, :, None]  # 自己相互作用を除外
    return forces.sum(axis=1) / num_particles  # shape: (n, d)  平均斥力を計算


grad_E = jax.vmap(jax.grad(E))  # 導関数
# scan(vmap)形式：scanの中でvmapを使い、全粒子を一括更新
def step_fn(xs: Array, _: Any) -> tuple[Array, Array]:
    """
    scanで全粒子を一括更新するための関数。
    xs: 粒子の現在の状態( jax.Array, shape=(num_particles, 2) )
    _: ダミー入力（scan用、未使用）
    return: (新しいxs, 記録用xs) のタプル。
    """
    grad = grad_E(xs)   # shape: (粒子数、次元数)
    interaction = total_force(xs) # shape: (粒子数、次元数)
    # 勾配降下 + 斥力作用
    xs_new = xs - learning_rate * grad + interaction
    return xs_new, xs_new


# scanで全粒子まとめて更新
xs_init = initial
_, history = lax.scan(step_fn, xs_init, None, length=steps)     # history: (steps, num_particles, 2)
# 初期値も履歴に含める
history = jnp.concatenate([xs_init[None, :], history], axis=0)  # (steps+1, num_particles, 2)

# bokehでのプロット
# p = figure(title="Particle Trajectories in 2D", x_axis_label='x', y_axis_label='y', width=600, height=600)
# colors = Category10[10] if num_particles <= 10 else Category10[10] * ((num_particles // 10) + 1)

# for i in range(num_particles):
#     xs = history[:, i, 0]
#     ys = history[:, i, 1]
#     p.line(xs, ys, line_width=2, alpha=0.3, color=colors[i])#, legend_label=f"particle {i}")
#     #p.circle(xs, ys, size=4, color=colors[i], alpha=0.5)

# # p.legend.click_policy = "hide"
# show(p)


# == Plotly Expressによるアニメーション ==
import pandas as pd
import plotly.express as px

# history: (steps+1, num_particles, 2) → DataFrameへ変換
records = []
for t in range(0, history.shape[0], 10):
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
    range_x=[float(jnp.min(history[...,0]))-1, float(jnp.max(history[...,0]))+1],
    range_y=[float(jnp.min(history[...,1]))-1, float(jnp.max(history[...,1]))+1],
    title="Particle Animation (Plotly Express)",
    width=600,
    height=600,
)
fig.show()