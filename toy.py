import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import Array
from jax.typing import ArrayLike
from typing import Any
from jax import random
from plot import plotTrajectory, animationTrajectory, plotEnergySurface


# 初期値やパラメータ
num_particles = 100      # 並列化する粒子数
key = random.PRNGKey(42)
init_mean, init_std = 0.0, 10.0
initial = random.normal(key, shape=(num_particles, 2)) * init_std + init_mean
learning_rate = 2       # 学習率
alpha = 8               # 斥力の強さ
beta = 0.1              # 入力刺激の強さ
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
grad_E = jax.vmap(jax.grad(E))  # ベクトル化された導関数



def calc_force(x0: ArrayLike, x1: ArrayLike) -> Array:  # ([d], [d]) -> [d]
    """
    相互作用関数
    近づくほど強く反発する  斥力
    距離×力
    """
    vec = x0 - x1
    force = 1 / (jnp.dot(vec, vec)**2 + 1e-10)
    return force * vec
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



def stimulation_force(xs: ArrayLike, target: ArrayLike) -> Array:
    """
    入力刺激による力を計算。離れているほど大きな力。
    xs: 粒子の現在の状態 (jax.Array, shape=(num_particles, 2))
    target: 入力刺激座標 (jax.Array, shape=(2,))
    return: (num_particles, 2) の力ベクトル
    """
    vec = target - xs  # shape: (num_particles, 2)
    dist = jnp.linalg.norm(vec, axis=1, keepdims=True)  # shape: (num_particles, 1)
    return vec * dist



# scan(vmap)形式：scanの中でvmapを使い、全粒子を一括更新
def step_fn(xs: ArrayLike, target: ArrayLike) -> tuple[Array, Array]:
    """
    scanで全粒子を一括更新するための関数。
    xs: 粒子の現在の状態( jax.Array, shape=(num_particles, 2) )
    target: 入力刺激 (jax.Array, shape=(2,))
    return: (新しいxs, 記録用xs) のタプル。
    """
    grad = grad_E(xs)                # shape(粒子数、次元数): 勾配
    interaction = total_force(xs)    # shape(粒子数、次元数): 斥力
    stimulation = stimulation_force(xs, target)
    # 勾配降下 + 斥力作用 + 入力刺激
    xs_new = xs - learning_rate * grad + alpha * interaction + beta * stimulation
    #xs_new = xs + beta * stimulation
    return xs_new, xs_new



if __name__ == "__main__":
    target = jnp.array([5.0, 5.0])  # 一定の入力刺激
    xs_init = initial   # 初期値
    targets = jnp.tile(target, (steps, 1))  # shape: (steps, 2)
    _, history = lax.scan(step_fn, xs_init, targets)     # history: (steps, num_particles, 2)
    # 初期値も履歴に含める
    history = jnp.concatenate([xs_init[None, :], history], axis=0)  # (steps+1, num_particles, 2)

    plotTrajectory(history, num_particles)
    #animationTrajectory(history, num_particles, 20)
    #plotEnergySurface(E, -20, 20, -20, 20, 100)

