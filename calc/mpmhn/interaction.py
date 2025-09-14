
"""
interaction.py

粒子同士の相互作用を計算する
"""
import jax
import jax.numpy as jnp
from jax import Array
from jax.tree_util import Partial
from jax.typing import ArrayLike
from jaxtyping import Float


@jax.jit
def calc_force(
    x0: Float[ArrayLike, " d"],
    x1: Float[ArrayLike, " d"],
    exponent: float = 1.0,
    f_max: float = 10.0,
) -> Float[Array, " d"]:
    """
    相互作用関数

    近づくほど強く反発する (斥力)
    距離×力
    """
    vec = x0 - x1
    force = 1 / (jnp.dot(vec, vec)**(exponent/2) + 1e-10)
    force = jnp.clip(force, 0.0, f_max)
    return force * vec


def total_force(
    x: Float[ArrayLike, "n d"],
    exponent: float = 1.0,
    f_max: float = 10.0,
) -> Float[Array, "n d"]:
    """各粒子に働く合力を計算（自己相互作用を除く）"""
    # 引数を固定
    calc_force_p = Partial(calc_force, exponent=exponent, f_max=f_max)

    # 他の全ての粒子との相互作用を計算  ([d], [n,d]) -> [n,d]
    calc_force_v = jax.vmap(calc_force_p, in_axes=(None, 0), out_axes=0)

    # 全粒子同士の相互作用を計算  ([n,d], [d]) -> [n,n,d]
    calc_force_all = jax.vmap(calc_force_v, in_axes=(0, None), out_axes=0)

    # 粒子毎の合力を計算
    forces = calc_force_all(x, x)  # shape: (n, n, d)
    n = x.shape[0]
    mask = 1 - jnp.eye(n, dtype=bool)  # shape: (n, n)
    forces = forces * mask[:, :, None]  # 自己相互作用を除外

    return jnp.sum(forces, axis=1) / n  # shape: (n, d)  平均斥力を計算
