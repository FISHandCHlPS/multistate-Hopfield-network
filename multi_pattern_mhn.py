"""
multiPatternMHN.py

高次元における提案記憶モデル
"""
from resource.cifar100 import get_cifar100

import hydra
import jax
import jax.numpy as jnp
from jax import Array, lax, random
from jax.tree_util import Partial
from jax.typing import ArrayLike
from omegaconf import DictConfig

from calc.mpmhn import cmhn_energy, stimulation_force, total_force
from plot.images import plot_images_trajectory
from plot.similarity import plot_cos_sim


def load_weights() -> Array:
    """
    cifar100から重みを生成する。

    Returns:
        W: 重み行列( jax.Array, shape=(dim, num_patterns) )

    """
    # CIFAR画像を読み込み
    images, labels, class_names = get_cifar100()  # images: (100, 1, 32, 32)
    images_flat = images.reshape(images.shape[0], -1)  # (100, 1024)
    # 各画像を正規化
    images_flat = images_flat / jnp.linalg.norm(images_flat, axis=1, keepdims=True)

    # 画像ベクトルを列ベクトルとして100個並べた行列を重みとする(1024, 100)
    return images_flat.T


def create_initial(w: ArrayLike, num_particles: int = 1, seed: int = 42) -> Array:
    """
    初期値を生成する。

    Args:
        w (ArrayLike): 重み行列
        num_particles (int): 粒子数
        seed (int): 乱数シード

    Returns:
        initial: 初期値( jax.Array, shape=(num_particles, dim) )

    """
    # 初期値: ランダムに選んだ画像+ノイズ
    img_key, noise_key = random.split(random.PRNGKey(seed))
    random_index = 0#random.choice(img_key, images_flat.shape[0])
    base_img = w[:, random_index]  # (1024)
    noise = random.normal(
        key=noise_key,
        shape=(num_particles, base_img.shape[0]),
        dtype=jnp.float32,
    ) * 1  # ノイズ強度は適宜調整
    noise = noise / jnp.linalg.norm(noise, axis=1, keepdims=True)   # 正規化
    return base_img + noise  # (num_particles, 1024)


def add_stimulation(xs: ArrayLike, start: int, end: int, stimuli: ArrayLike) -> Array:
    """入力刺激を生成する。"""
    assert xs.shape[1] == stimuli.shape[0]
    xs = jnp.asarray(xs)
    stimuli = jnp.asarray(stimuli)
    return xs.at[start:end, :].set(stimuli)


@hydra.main(config_path="config", config_name="mpmhn", version_base=None)
def run(cfg: DictConfig) -> Array:
    """実行関数"""
    lr = cfg.learning_rate

    # 学習率を粒子数分の配列にする
    lr = jnp.linspace(1.0, 0.1, cfg.num_particles)
    lr = jnp.atleast_2d(lr).T

    weight = load_weights()  # 重み生成
    weight = weight[:, :3]
    initial = create_initial(
        w=weight,
        num_particles=cfg.num_particles,
        seed=cfg.seed,
    )    # 初期値

    # CMHNのエネルギー関数
    e_cmhn = Partial(cmhn_energy, w=weight, beta=cfg.beta)   # 3つの記憶のみ
    grad_e = jax.vmap(jax.grad(e_cmhn))  # ベクトル化された導関数

    # 入力刺激
    stimulus = jnp.zeros((cfg.steps, initial.shape[1]))  # (steps, dim)
    stimulus = add_stimulation(stimulus, 20, 30, weight[:, 0])  # (steps, dim)

    # scan更新用
    # xs: 粒子の現在の状態( jax.Array, shape=(num_particles, dim) )
    def step_fn(xs: ArrayLike, target:ArrayLike) -> tuple[Array, Array]:
        grad = grad_e(xs)   # shape(粒子数、次元数): 勾配
        interaction = total_force(  # shape(粒子数、次元数):斥力
            xs,
            exponent=cfg.c,
            f_max=cfg.f_max,
        )
        stimulation = stimulation_force(xs, target=target)  # shape(粒子数、次元数): 入力刺激
        # 入力刺激に画像適当な画像を入れ、時間変化を見る

        # 勾配降下 + 斥力作用 + 入力刺激
        xs_new = xs - lr * grad + cfg.gamma * interaction + 0.1 * stimulation
        return xs_new, xs_new   # (新しいxs, 記録用xs) のタプル
    # scanでシミュレーション
    _, history = lax.scan(step_fn, initial, xs=stimulus) # history: (steps, num_particles, dim)
    # 初期値も履歴に含める
    history = jnp.concatenate([initial[None, :], history], axis=0)  # (steps+1, num_particles, dim)

    # 結果保存
    output_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    jnp.save(output_path + "/history.npy", history)
    jnp.save(output_path + "/weight.npy", weight)
    jnp.save(output_path + "/initial.npy", initial)

    # ノルムを1に正規化
    history /= jnp.linalg.norm(history, axis=-1, keepdims=True)
    # historyを[0, 1]の範囲に正規化
    # history = (history - jnp.min(history)) / (jnp.max(history) - jnp.min(history))

    # 結果表示
    plot_images_trajectory(history, interval=5, path=output_path)
    plot_cos_sim(history, weight, path=output_path)


if __name__ == "__main__":
    run()
