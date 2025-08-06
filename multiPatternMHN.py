"""
multiPatternMHN.py

高次元における提案記憶モデル
"""
import jax
import jax.numpy as jnp
from jax import Array, random, lax
from jax.typing import ArrayLike
from mpmhn.energy import CMHN_Energy
from mpmhn.interaction import total_force
from jax.tree_util import Partial
from mhn.cifar100 import get_cifar100
from mpmhn.plot_particle_images import plot_particle_image_slider, plot_img, plot_pca_images

# 粒子数やパラメータ設定
num_particles = 10
key = random.PRNGKey(42)
learning_rate = 1 # 学習率
gamma = 0.1   # 斥力の強さ
c = 1.0  # 斥力指数
beta = 100.0  # CMHNの逆温度
steps = 20 # 合計ステップ数

# CIFAR画像を読み込み
images, labels, class_names = get_cifar100()  # images: (100, 1, 32, 32)
images_flat = images.reshape(images.shape[0], -1)  # (100, 1024)
images_flat = images_flat / jnp.linalg.norm(images_flat, axis=1, keepdims=True)  # 各画像を正規化

# 画像ベクトルを列ベクトルとして100個並べた行列を重みとする（1024, 100）
W = images_flat.T  # (1024, 100)

# 初期値: ランダムに選んだ画像＋ノイズ
img_key, noise_key = random.split(key)
random_index = 0#random.choice(img_key, images_flat.shape[0])
base_img = images_flat[random_index]  # (1024)
noise = random.normal(noise_key, shape=(num_particles, base_img.shape[0])) * 0.001  # ノイズ強度は適宜調整
initial = base_img + noise  # (num_particles, 1024)

# CMHNのエネルギー関数
E_CMHN = Partial(CMHN_Energy, W=W, beta=beta)
grad_E = jax.vmap(jax.grad(E_CMHN))  # ベクトル化された導関数


def step_fn(xs: ArrayLike, _=None) -> tuple[Array, Array]:
    """
    scanで全粒子を一括更新するための関数。
    xs: 粒子の現在の状態( jax.Array, shape=(num_particles, dim) )
    return: (新しいxs, 記録用xs) のタプル。
    """
    grad = grad_E(xs)                # shape(粒子数、次元数): 勾配
    interaction = total_force(xs, exponent=c, f_max=10.0)    # shape(粒子数、次元数): 斥力
    # 勾配降下 + 斥力作用
    xs_new = xs - learning_rate * grad + gamma * interaction
    return xs_new, xs_new


if __name__ == "__main__":
    xs_init = initial   # 初期値

    _, history = lax.scan(step_fn, xs_init, length=steps)     # history: (steps, num_particles, dim)
    # 初期値も履歴に含める
    history = jnp.concatenate([xs_init[None, :], history], axis=0)  # (steps+1, num_particles, dim)
    print('computed')


    plot_pca_images(history[0, :, :])
    plot_pca_images(history[-1, :, :])
    # plot_img(base_img.reshape(32, 32))
    # plot_img(initial[0].reshape(32, 32))
    # plot_particle_image_slider(history)