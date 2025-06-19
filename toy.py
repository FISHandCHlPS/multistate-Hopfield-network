import jax
import jax.numpy as jnp
import jax.lax as lax
import matplotlib.pyplot as plt
from bokeh.plotting import figure, show, output_file
from bokeh.palettes import Category10
from jax import Array
from jax.typing import ArrayLike
from typing import Any

def E(x: ArrayLike) -> Array:
    """
    最小化したいエネルギー関数。
    x: jax.Array (shape=(n,)) またはスカラー
    return: スカラー
    """
    x = jnp.asarray(x)
    amplitude = 1.0
    mu = jnp.array([2,4], dtype=float)
    sigma = 1.0
    e = amplitude * jnp.exp(-((x - mu)**2 / (2 * sigma**2)))
    return jnp.sum(e)

# 初期値やパラメータ
num_particles = 25      # 並列化する粒子数
# range = jnp.arange(5, dtype=float)
# initial = jnp.stack( jnp.meshgrid([range.flatten(), range.flatten()]), axis=-1 )   # shape=(25, 2)
initial = jnp.stack(jnp.meshgrid(jnp.arange(5, dtype=float), jnp.arange(2, 7, dtype=float)), axis=-1).reshape(-1, 2, order='F')
#print(initial)
learning_rate = 0.1     # 学習率
steps = 1000             # 合計ステップ数
grad_E = jax.vmap(jax.grad(E))  # 導関数


# # 勾配降下法
# def update(x: ArrayLike) -> Array:
#     grad = grad_E(x)    # x: shape(並列数, 次元数)
#     return x + learning_rate * grad
# update_v = jax.vmap(update) #　並列化した更新関数


def calc_force(x0: ArrayLike, x1: ArrayLike) -> Array:  # ([d], [d]) -> [d]
    """
    相互作用関数
    近づくほど強く反発する  斥力
    距離×力 x0に働く
    """
    diff = x0 - x1
    return diff * jnp.dot(diff, diff)
# 他の全ての粒子との相互作用を計算
calc_force_v = jax.vmap(calc_force, in_axes=(None, 0), out_axes=0)  # ([d], [n,d]) -> [n,d]

def calc_force_all(x: ArrayLike) -> Array:
    """
    全粒子同士の相互作用を計算（shape: [n, n, d]）
    """
    return jax.vmap(calc_force_v, in_axes=(0, None), out_axes=0)(x, x)

def total_force(x: ArrayLike) -> Array:
    """
    各粒子に働く合力を計算（自己相互作用を除く）
    """
    forces = calc_force_all(x)  # shape: (n, n, d)
    n = x.shape[0]
    mask = 1 - jnp.eye(n, dtype=bool)  # shape: (n, n)
    forces = forces * mask[:, :, None]  # 自己相互作用を除外
    return forces.sum(axis=1)  # shape: (n, d)



# scan(vmap)形式：scanの中でvmapを使い、全粒子を一括更新
def step_fn(xs: Array, _: Any) -> tuple[Array, Array]:
    """
    scanで全粒子を一括更新するための関数。
    xs: 粒子の現在の状態( jax.Array, shape=(num_particles, 2) )
    _: ダミー入力（scan用、未使用）
    return: (新しいxs, 記録用xs) のタプル。
    """
    grad = grad_E(xs) # shape: (num_particles, 2)
    interaction = total_force(xs) # shape: (num_particles, 2)
    xs_new = xs + learning_rate * (grad + interaction)
    return xs_new, xs_new



# scanで全粒子まとめて更新
xs_init = initial
_, history = lax.scan(step_fn, xs_init, None, length=steps)
# history: (steps, num_particles)
# 初期値も履歴に含める
history = jnp.concatenate([xs_init[None, :], history], axis=0)  # (steps+1, num_particles)

p = figure(title="Particle Trajectories in 2D", x_axis_label='x', y_axis_label='y', width=600, height=600)
colors = Category10[10] if num_particles <= 10 else Category10[10] * ((num_particles // 10) + 1)

for i in range(num_particles):
    xs = history[:, i, 0]
    ys = history[:, i, 1]
    p.line(xs, ys, line_width=2, alpha=0.3, color=colors[i], legend_label=f"particle {i}")
    #p.circle(xs, ys, size=4, color=colors[i], alpha=0.5)

p.legend.click_policy = "hide"
show(p)