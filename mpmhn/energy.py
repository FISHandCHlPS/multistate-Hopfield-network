"""
Energy.py

このモジュールは、JAX を用いたさまざまなホップフィールドネットワークモデルのエネルギー関数を提供します。
"""
import jax.numpy as jnp
from numpy.typing import ArrayLike
from jax import Array


def CMHN_Energy(x: ArrayLike, W: ArrayLike, beta: float = 1.0) -> Array:
    """
    CMHNモデルのエネルギーを計算します。

    Args:
        x (ArrayLike): 連続値の状態ベクトル
        W (ArrayLike): 重み行列
        beta (float): 逆温度パラメータ

    Returns:
        Array: エネルギー値(スカラー)
    """
    def log_sum_exp(x):
        """数値的に安定なlog-sum-exp"""
        x_max = jnp.max(x)
        return x_max + jnp.log(jnp.sum(jnp.exp(x - x_max)))

    N = W.shape[1]
    M = jnp.max(jnp.linalg.norm(W, axis=0))
    E = -log_sum_exp(beta * W.T @ x)/beta + 1/2 * jnp.dot(x, x) + 1/beta * jnp.log(N) + 1/2 * M**2
    return E


def DAM_Energy(x: ArrayLike, W: ArrayLike, n: int = 2) -> Array:
    """
    DAMモデルのエネルギーを計算します。

    Args:
        x (ArrayLike): 離散値の状態ベクトル
        W (ArrayLike): 重み行列
        n (int): 指数
    Returns:
        Array: エネルギー値(スカラー)
    """
    E = -jnp.sum(jnp.power(W.T @ x, n))
    return E


if __name__ == "__main__":
    #from ..plot2d.plot import plotEnergySurface
    W = jnp.array([[ 1,-1, 1],
                   [-1, 1, 1]])
    beta = 1.0
    n = 2

    #plotEnergySurface(CMHN_Energy, -5, 5, -5, 5, 20)
    #plotEnergySurface(DAM_Energy, -10, 10, -10, 10, 100)