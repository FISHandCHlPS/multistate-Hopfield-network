import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from jaxtyping import Float


@jax.jit
def stimulation_force(
    xs: Float[ArrayLike, "num_particles dim"],
    target: Float[ArrayLike, " dim"],
) -> Float[Array, "num_particles dim"]:
    """
    入力刺激による力を計算。離れているほど大きな力。NaNは無視する。

    xs: 粒子の現在の状態 (jax.Array, shape=(num_particles, dim))
    target: 入力刺激座標 (jax.Array, shape=(dim,))
    return: (num_particles, dim) の力ベクトル
    """
    xs = jnp.asarray(xs)
    target = jnp.asarray(target)
    vec = target[None, :] - xs
    return jnp.nan_to_num(vec, nan=0.0)
