"""
multiPatternMHN.py

高次元における提案記憶モデル
"""
import jax
import jax.numpy as jnp
from jax import Array, random, lax
from jax.typing import ArrayLike
from jax.tree_util import Partial
import hydra
from omegaconf import DictConfig

from calc.mpmhn import CMHN_Energy, total_force
from resource.cifar100 import get_cifar100
from plot.images import plot_images_trajectory, plot_image
from plot.pca import plot_pca_feature, plot_pca_trajectory, plot_pca_ccr
from plot.similarity import plot_cos_sim


def load_Weights():
    """
    重みを生成する。

    Returns:
        W: 重み行列( jax.Array, shape=(dim, num_patterns) )
    """
    # CIFAR画像を読み込み
    images, labels, class_names = get_cifar100()  # images: (100, 1, 32, 32)
    images_flat = images.reshape(images.shape[0], -1)  # (100, 1024)
    images_flat = images_flat / jnp.linalg.norm(images_flat, axis=1, keepdims=True)  # 各画像を正規化

    # 画像ベクトルを列ベクトルとして100個並べた行列を重みとする(1024, 100)
    W = images_flat.T
    return W


def create_initial(W: ArrayLike, num_particles: int = 1, seed: int = 42):
    """
    初期値を生成する。

    Args:
        W (ArrayLike): 重み行列
        num_particles (int): 粒子数
        seed (int): 乱数シード

    Returns:
        initial: 初期値( jax.Array, shape=(num_particles, dim) )
    """

    # 初期値: ランダムに選んだ画像＋ノイズ
    img_key, noise_key = random.split(random.PRNGKey(seed))
    random_index = 0#random.choice(img_key, images_flat.shape[0])
    base_img = W[:, random_index]  # (1024)
    noise = random.normal(noise_key, shape=(num_particles, base_img.shape[0])) * 0.05  # ノイズ強度は適宜調整
    initial = base_img + noise  # (num_particles, 1024)
    return initial


@hydra.main(config_path="config", config_name="mpmhn", version_base=None)
def run(cfg: DictConfig) -> Array:
    """
    実行関数。
    """
    weight = load_Weights()  # 重み生成
    weight = weight[:, :3]
    initial = create_initial(weight, num_particles=cfg.num_particles, seed=cfg.seed)    # 初期値

    # CMHNのエネルギー関数
    E_CMHN = Partial(CMHN_Energy, W=weight, beta=cfg.beta)   # 3つの記憶のみ
    grad_E = jax.vmap(jax.grad(E_CMHN))  # ベクトル化された導関数

    # scan更新用
    # xs: 粒子の現在の状態( jax.Array, shape=(num_particles, dim) )
    def step_fn(xs: ArrayLike, _=None) -> tuple[Array, Array]:
        grad = grad_E(xs)   # shape(粒子数、次元数): 勾配
        interaction = total_force(xs, exponent=cfg.c, f_max=cfg.f_max)    # shape(粒子数、次元数): 斥力
        # 勾配降下 + 斥力作用
        xs_new = xs - cfg.learning_rate * grad + cfg.gamma * interaction
        return xs_new, xs_new   # (新しいxs, 記録用xs) のタプル
    # scanでシミュレーション
    _, history = lax.scan(step_fn, initial, length=cfg.steps)     # history: (steps, num_particles, dim)
    # 初期値も履歴に含める
    history = jnp.concatenate([initial[None, :], history], axis=0)  # (steps+1, num_particles, dim)

    # 結果保存
    output_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    jnp.savez(output_path + '/result.npz', history=history, weight=weight, initial=initial)

    # 結果表示
    # plot_image(weight[:, 0].reshape(32, 32))
    # plot_image(initial[0].reshape(32, 32))
    plot_images_trajectory(history, interval=2)
    
    # plot_pca_feature(history)
    # plot_pca_trajectory(history)
    # plot_pca_ccr(history)

    plot_cos_sim(history, weight)


if __name__ == "__main__":
    run()

