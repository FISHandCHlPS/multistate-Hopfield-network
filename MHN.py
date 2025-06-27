"""
MHN.py: Modern Hopfield Network の簡易実装

- JAX を用いたモダンホップフィールドネットワークの実装
- update: 状態ベクトルの更新関数
"""
import time
from jax import Array
from jax.typing import ArrayLike
import jax.numpy as jnp
import jax.nn
import jax.lax
import jax.random
import matplotlib.pyplot as plt
from cifar100 import get_cifar100


def flatten_images(images: Array) -> Array:
    """
    (N, 1, H, W) 形式の画像配列を (N, H*W) にflatten。
    Args:
        images (Array): CIFAR-100画像配列 (N, 1, 32, 32)
    Returns:
        Array: flatten後の画像配列 (N, 1024)
    """
    return images.reshape(images.shape[0], -1)


def update(x: ArrayLike, W: Array) -> Array:
    """
    入力状態ベクトル x を1ステップ更新する（記憶Wに基づくMHN更新）。
    Args:
        x (ArrayLike): 入力状態ベクトル (shape: (d,))
        W (Array): 記憶パターン集合 (shape: (d, N_mem))
    Returns:
        Array: 更新後の状態ベクトル (shape: (d,))
    """
    x = jnp.asarray(x)
    x_new = W @ jax.nn.softmax( 1000 * W.T @ x, axis=0)
    # test_s = jax.nn.softmax( 1000 * W.T @ x )
    # jax.debug.print('test_s: {test_s}', test_s = test_s)
    return x_new


if __name__ == "__main__":
    """
    CIFAR-100画像を記憶としたMHNの動作例：
    1. 画像記憶（W）を構築
    2. ランダムな画像1枚にノイズを加えて入力
    3. MHNダイナミクスで復元挙動を確認
    """
    # 1. CIFAR-100画像を取得
    images, labels, class_names = get_cifar100()  # images: (100, 1, 32, 32)
    W = flatten_images(images).T  # (1024, 100) 記憶集合
    norm_W = jnp.linalg.norm(W, axis=0) # (100,)
    W = W / norm_W[None, :]     # 記憶のノルムを揃える

    # 2. テスト画像（例: 0番目）にノイズを加える
    key_img = 0
    x_true = W[:,key_img]  # 正解画像ベクトル
    # ノイズ: 各画素に正規分布ノイズを加える（平均0, 標準偏差0.1）
    seed = int(time.time())
    rng = jax.random.PRNGKey(seed)
    noise = jax.random.normal(rng, shape=x_true.shape) * 0.005
    x_noisy = x_true + noise
    x_noisy = jnp.clip(x_noisy, 0.0, 1.0)

    # 3. MHNダイナミクスで復元
    def step(carry, _):
        x_new = update(carry, W)
        return x_new, x_new
    x, traj = jax.lax.scan(step, x_noisy, None, length=100)


    # 4. 結果表示
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(x_true.reshape(32, 32), cmap='gray')
    axes[0].set_title('Original')
    axes[1].imshow(x_noisy.reshape(32, 32), cmap='gray')
    axes[1].set_title('Noisy Input')
    axes[2].imshow(x.reshape(32, 32), cmap='gray')
    axes[2].set_title('MHN Output')
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    #print("state:", x)