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
import random
from .cifar100 import get_cifar100


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
    x_new = W @ jax.nn.softmax( 1000 * W.T @ x)
    return x_new
update_v = jax.vmap(update, in_axes=(0, None), out_axes=0)


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

    # 2. すべての画像にノイズを加えて比較画像を一覧表示
    num_imgs = W.shape[1]
    img_size = int(jnp.sqrt(W.shape[0]))  # 32
    # 各画像ごとに異なる乱数キーでノイズ生成
    seed = int(time.time() * 1e6) % (2**32)
    rng = jax.random.PRNGKey(seed)
    noise = jax.random.normal(rng, shape=W.shape) * 0.01  # (1024, 100)
    x_noisy = (W + noise).T   # (100, 1024)
    x_noisy = jnp.clip(x_noisy, 0.0, 1.0)

    # 3. MHNダイナミクスで復元
    def step(carry, _):
        x_new = update_v(carry, W)
        return x_new, x_new
    x, traj = jax.lax.scan(step, x_noisy, None, length=10)
    traj = jnp.vstack([x_noisy[None, ...], traj])  # 初期状態も含める shape(11, 100, 1024)


    # 4. 結果表示（3つのランダムな画像を表示）
    random.seed(seed)  # 再現性のためにseedを指定
    indices = random.sample(range(num_imgs), 3)
    fig, axes = plt.subplots(3, 3, figsize=(6, 6))  # 小さめに
    for row, idx in enumerate(indices):
        axes[row, 0].imshow(W[:, idx].reshape(32, 32), cmap='gray')
        axes[row, 0].set_title(f'Original #{idx}', fontsize=10)
        axes[row, 1].imshow(x_noisy[idx].reshape(32, 32), cmap='gray')
        axes[row, 1].set_title('Noisy Input', fontsize=10)
        axes[row, 2].imshow(x[idx].reshape(32, 32), cmap='gray')
        axes[row, 2].set_title('MHN Output', fontsize=10)
        for col in range(3):
            axes[row, col].axis('off')
    plt.tight_layout(pad=0.2, w_pad=0.2, h_pad=0.2)  # 余白を詰める
    plt.show()

    # 類似度の時系列推移をプロット
    sim = jnp.sum(traj * W.T, axis=2) / jnp.linalg.norm(traj, axis=2)

    plt.figure(figsize=(6, 6))
    plt.plot(sim)
    plt.xlabel('Step')
    plt.ylabel('Cosine Similarity')
    plt.title('Similarity to Original')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    #print("state:", x)