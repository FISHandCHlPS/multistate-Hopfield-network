"""
MHN.py: Modern Hopfield Network の簡易実装

- JAX を用いたモダンホップフィールドネットワークの実装
- W: 重み行列 (固定)
- update: 状態ベクトルの更新関数
"""

from jax import Array
from jax.typing import ArrayLike
import jax.numpy as jnp
import jax.nn
import jax.lax

# 重み行列: shape = (d, 2)
W: Array = jnp.array([
    [1, -1, 1, -1],
    [-1, 1, -1, 1]
]).T

def update(x: ArrayLike) -> Array:
    """
    入力状態ベクトル x を1ステップ更新する。

    Args:
        x (ArrayLike): 入力状態ベクトル (shape: (d,))
    Returns:
        Array: 更新後の状態ベクトル (shape: (d,))
    """
    x = jnp.asarray(x)
    x_new = W @ jax.nn.softmax( W.T @ x )
    return x_new

if __name__ == "__main__":
    # あるパターンにノイズを加えて入力
    x: Array = jnp.array([1, -1, -1, 1], dtype=jnp.float32)  # 1bit反転

    # jax.lax.scanを使った更新ステップ
    def step(carry, _):
        x_new = update(carry)
        return x_new, x_new
    x, _ = jax.lax.scan(step, x, length=10)

    print("state:", x)