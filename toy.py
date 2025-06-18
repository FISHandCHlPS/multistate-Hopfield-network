import jax
import jax.numpy as jnp
import jax.lax as lax
import matplotlib.pyplot as plt
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
    return (x-4) ** 2

# 初期値やパラメータ
num_particles = 20      # 並列化する粒子数
initial = jnp.arange(-10, 10, dtype=float)  # shape=(num_particles)
learning_rate = 0.1     # 学習率
steps = 100             # 合計ステップ数
v_grad_E = jax.vmap( jax.grad(E) )    # 並列化した導関数

# 勾配降下法
def update(x: ArrayLike) -> Array:
    grad = v_grad_E(x).T    # x: shape(並列数, 次元数)
    # 距離×力
    # (x-A) A^T x
    ( x[:, None] - grad ) * grad.T @ x
    return x - learning_rate * grad
update_v = jax.vmap(update) #　並列化した更新関数


# scan(vmap)形式：scanの中でvmapを使い、全粒子を一括更新
def step_fn(xs: Array, _: Any) -> tuple[Array, Array]:
    """
    scanで全粒子を一括更新するための関数。
    xs: 粒子の現在の状態（jax.Array, shape=(num_particles, 2)）
    _: ダミー入力（scan用、未使用）
    return: (新しいxs, 記録用xs) のタプル。
    """
    xs_new = update_v(xs)
    return xs_new, xs_new

# scanで全粒子まとめて更新
xs_init = initial
_, history = lax.scan(step_fn, xs_init, None, length=steps)
# history: (steps, num_particles)
# 初期値も履歴に含める
history = jnp.concatenate([xs_init[None, :], history], axis=0)  # (steps+1, num_particles)
print(history[0])

# プロット: 各粒子のx-E(x)軌跡を描画
for i in range(num_particles):
    xs = history[:, i]
    Es = jnp.array([E(x) for x in xs])
    plt.plot(xs, Es, marker='o', label=f'particle {i}')
plt.xlabel('x')
plt.ylabel('E(x)')
plt.title('Gradient Descent: Trajectory in x-E(x) space')
plt.show()