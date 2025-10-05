"""Energy.py

JAX を用いたさまざまなホップフィールドネットワークモデルのエネルギー関数を提供します。
"""
import jax.numpy as jnp
from jaxtyping import Float
from numpy.typing import ArrayLike


def cmhn_energy(
    x: Float[ArrayLike, " d"],
    w: Float[ArrayLike, "d n"],
    beta: float = 1.0,
) -> float:
    """CMHNモデルのエネルギーを計算します。

    Args:
        x (ArrayLike): 連続値の状態ベクトル
        w (ArrayLike): 重み行列
        beta (float): 逆温度パラメータ

    Returns:
        float: エネルギー値(スカラー)

    """
    def log_sum_exp(x: float) -> float:
        """数値的に安定なlog-sum-exp"""
        x_max = jnp.max(x)
        return x_max + jnp.log(jnp.sum(jnp.exp(x - x_max)))

    n = w.shape[1]
    m = jnp.max(jnp.linalg.norm(w, axis=0))
    e = -log_sum_exp(beta * w.T @ x)/beta + 1/2 * jnp.dot(x, x)
    c = 1/beta * jnp.log(n) + 1/2 * m**2
    return e + c


def dam_energy(
    x: Float[ArrayLike, " d"],
    w: Float[ArrayLike, "d n"],
    ex: int = 2,
) -> float:
    """DAMモデルのエネルギーを計算します。

    Args:
        x (ArrayLike): 離散値の状態ベクトル
        w (ArrayLike): 重み行列
        ex (int): 指数
    Returns:
        float: エネルギー値(スカラー)

    """
    return -jnp.sum(jnp.power(w.T @ x, ex)) / ex



if __name__ == "__main__":
    #from ..plot2d.plot import plotEnergySurface
    w = jnp.array( [[ 1,-1, 1],
                    [-1, 1, 1]] )
    beta = 1.0
    ex = 2

    #plotEnergySurface(cmhn_energy, -5, 5, -5, 5, 20)
    #plotEnergySurface(dam_energy, -10, 10, -10, 10, 100)
