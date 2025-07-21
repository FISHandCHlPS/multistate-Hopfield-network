import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import Array
from jax.typing import ArrayLike
from jax import random
from plot import plotTrajectory, animationTrajectory, plotEnergySurface
from Energy import CMHN_Energy
from jax.tree_util import Partial


# 初期値やパラメータ
num_particles = 50      # 並列化する粒子数
key = random.PRNGKey(42)
init_mean, init_std = jnp.array([-0.0, 0.0]), 1
initial = random.normal(key, shape=(num_particles, 2)) * init_std + init_mean
learning_rate = 0.5       # 学習率
alpha = 1               # 斥力の強さ
beta = 0.1              # 入力刺激の強さ
steps = 50             # 合計ステップ数

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
#grad_E = jax.vmap(jax.grad(E))  # ベクトル化された導関数

W = jnp.array([[ 1,-1, 1],
               [-1, 1, 1]], dtype=jnp.float32)
beta = 5.0
E_CMHN = Partial(CMHN_Energy, W=W, beta=beta)   # 部分適用でxのみの関数に変換
grad_E = jax.vmap(jax.grad(E_CMHN))  # ベクトル化された導関数


def calc_force(x0: ArrayLike, x1: ArrayLike) -> Array:  # ([d], [d]) -> [d]
    """
    相互作用関数
    近づくほど強く反発する  斥力
    距離×力
    """
    vec = x0 - x1
    force = 1 / (jnp.dot(vec, vec)**2 + 1e-5)
    force = jnp.clip(force, 0.0, 10.0)
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
    return forces.sum(axis=1) / n  # shape: (n, d)  平均斥力を計算



def stimulation_force(xs: ArrayLike, target: ArrayLike) -> Array:
    """
    入力刺激による力を計算。離れているほど大きな力。NaNは無視する。
    xs: 粒子の現在の状態 (jax.Array, shape=(num_particles, 2))
    target: 入力刺激座標 (jax.Array, shape=(2,))
    return: (num_particles, 2) の力ベクトル
    """
    xs = jnp.asarray(xs)
    target = jnp.asarray(target)
    vec = target[None, :] - xs
    return jnp.nan_to_num(vec, nan=0.0)



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
    stimulation = stimulation_force(xs, target=target)
    # 勾配降下 + 斥力作用 + 入力刺激
    xs_new = xs - learning_rate * grad + alpha * interaction# + beta * stimulation
    return xs_new, xs_new



if __name__ == "__main__":
    target = jnp.array([3.0, 3.0])  # 一定の入力刺激
    stimulus = jnp.concatenate([jnp.tile(target, (50, 1)), jnp.full((steps-50, 2), jnp.nan)], axis=0)   # 50ステップ目まで刺激を与える
    xs_init = initial   # 初期値

    _, history = lax.scan(step_fn, xs_init, stimulus)     # history: (steps, num_particles, 2)
    # 初期値も履歴に含める
    history = jnp.concatenate([xs_init[None, :], history], axis=0)  # (steps+1, num_particles, 2)
    print('computed')

    #plotTrajectory(history)
    animationTrajectory(history, 1)
    #plotEnergySurface(CMHN_Energy, -5, 5, -5, 5, 20)

