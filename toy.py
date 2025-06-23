import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import Array
from jax.typing import ArrayLike
from typing import Any
from jax import random
from plot import plotTrajectory, animationTrajectory, plotEnergySurface


# 初期値やパラメータ
num_particles = 200      # 並列化する粒子数
key = random.PRNGKey(42)
init_mean, init_std = 0.0, 10.0
initial = random.normal(key, shape=(num_particles, 2)) * init_std + init_mean
learning_rate = 2       # 学習率
alpha = 8               # 斥力の強さ
steps = 500             # 合計ステップ数


def E(x: ArrayLike) -> Array:
    """
    最小化したいエネルギー関数。
    x: jax.Array (shape=(d,)) またはスカラー
    return: スカラー
    """
    x = jnp.asarray(x)
    mu = jnp.array([[7,13], 
                    [-2,-4]], dtype=jnp.float32)
    sigma = 2.0
    dist = jnp.sqrt(jnp.sum((x - mu)**2, axis=1))
    e_i = (1 / (2 * sigma)) * jnp.exp(-dist / sigma)
    e = jnp.sum(e_i)
    return -e


def calc_force(x0: ArrayLike, x1: ArrayLike) -> Array:  # ([d], [d]) -> [d]
    """
    相互作用関数
    近づくほど強く反発する  斥力
    距離×力
    """
    vec = x0 - x1
    force = 1 / (jnp.dot(vec, vec)**2 + 1e-10)
    return alpha * force * vec
# 他の全ての粒子との相互作用を計算
calc_force_v = jax.vmap(calc_force, in_axes=(None, 0), out_axes=0)  # ([d], [n,d]) -> [n,d]
# 全粒子同士の相互作用を計算
calc_force_all = jax.vmap(calc_force_v, in_axes=(0, None), out_axes=0)  # ([n,d], [d]) -> [n,n,d]

def total_force(x: ArrayLike) -> Array:
    """
    各粒子に働く合力を計算（自己相互作用を除く）
    """
    forces = calc_force_all(x, x)  # shape: (n, n, d)
    n = x.shape[0]
    mask = 1 - jnp.eye(n, dtype=bool)  # shape: (n, n)
    forces = forces * mask[:, :, None]  # 自己相互作用を除外
    return forces.sum(axis=1) / num_particles  # shape: (n, d)  平均斥力を計算


grad_E = jax.vmap(jax.grad(E))  # ベクトル化された導関数
# scan(vmap)形式：scanの中でvmapを使い、全粒子を一括更新
def step_fn(xs: ArrayLike, _: Any) -> tuple[Array, Array]:
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


if __name__ == "__main__":
    # scanで全粒子まとめて更新
    xs_init = initial
    _, history = lax.scan(step_fn, xs_init, None, length=steps)     # history: (steps, num_particles, 2)
    # 初期値も履歴に含める
    history = jnp.concatenate([xs_init[None, :], history], axis=0)  # (steps+1, num_particles, 2)

    #plotTrajectory(history, num_particles)
    animationTrajectory(history, num_particles, 10)
    #plotEnergySurface(E, -20, 20, -20, 20, 100)

