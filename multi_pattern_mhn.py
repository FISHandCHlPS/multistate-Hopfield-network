"""multiPatternMHN.py

高次元における提案記憶モデル
"""
from resource.cifar100 import get_cifar100
from typing import Any

import hydra
import jax
import jax.numpy as jnp
import plotly.io as pio
from jax import lax, random
from jax.tree_util import Partial
from jaxtyping import Array, ArrayLike, Float
from omegaconf import DictConfig, OmegaConf, open_dict

from calc.mpmhn.energy import cmhn_energy
from calc.mpmhn.interaction import total_force
from calc.mpmhn.stimulation import stimulation_force
from plot.entropy import plot_entropy_time_series

# from plot.images import plot_mean_image
#from plot.images import plot_images_trajectory
from plot.similarity import plot_cos

# from plot.tsne import plot_tsne_trajectory

# VSCode用のレンダラーを設定
pio.renderers.default = "browser"


def load_weights() -> Float[Array, "dim num_patterns"]:
    """cifar100から重みを生成する。

    Returns:
        W: 重み行列( jax.Array, shape=(dim, num_patterns) )

    """
    # CIFAR画像を読み込み
    images, _, _ = get_cifar100()  # images: (100, 1, 32, 32)
    images_flat = images.reshape(images.shape[0], -1)  # (100, 1024)
    # 各画像を正規化
    images_flat = images_flat / jnp.linalg.norm(images_flat, axis=1, keepdims=True)

    # 画像ベクトルを列ベクトルとして100個並べた行列を重みとする(1024, 100)
    return images_flat.T


def normal_noise(shape: tuple[int, ...], std: float, key: jax.Array) -> jax.Array:
    """ノイズを生成する。"""
    return random.normal(key=key, shape=shape, dtype=jnp.float32) * std


def create_initial(
    w: Float[ArrayLike, "dim num_patterns"],
    num_particles: int = 1,
    seed: int = 42,
    base_idx: int | list[int] = 0,
    std: float = 1.0,
) -> Float[Array, "num_particles dim"]:
    """初期値を生成する。

    Args:
        w (ArrayLike): 重み行列
        num_particles (int): 粒子数
        seed (int): 乱数シード

    Returns:
        initial: 初期値( jax.Array, shape=(num_particles, dim) )

    """
    # 初期値: ランダムに選んだ画像+ノイズ
    img_key, noise_key = random.split(random.PRNGKey(seed))
    # random_index = 0 # random.choice(img_key, images_flat.shape[0])
    if isinstance(base_idx, int):
        base_idx = [base_idx]
    base_img = jnp.sum(w[:, base_idx], axis=1)  # (1024)
    noise = normal_noise(
        shape=(num_particles, base_img.shape[0]),
        std=std,
        key=noise_key,
    )
    #noisy_img = base_img + noise  # (num_particles, 1024)
    noisy_img = noise
    return noisy_img / jnp.linalg.norm(noisy_img, axis=1, keepdims=True)   # 正規化


def add_stimulation(
    xs: Float[ArrayLike, "num_particles dim"], start: int, end: int,
    stimuli: Float[ArrayLike, "dim num_particles"],
) -> Float[Array, "num_particles dim"]:
    """刺激を入れる"""
    assert xs.shape[1] == stimuli.shape[0]
    xs = jnp.asarray(xs)
    stimuli = jnp.asarray(stimuli)
    return xs.at[start:end, :].set(stimuli)


@hydra.main(config_path="config", config_name="mpmhn", version_base=None)
def run(cfg: DictConfig) -> Float[Array, "num_particles dim"]:
    """実行関数"""
    rand_key = random.PRNGKey(cfg.seed)

    # 学習率を粒子毎にガウス分布から生成
    lr_configs = [
        # {"mean": 0.3, "std": 0.3, "count": 15},
        # {"mean": 0.7, "std": 0.3, "count": 5},
        {"a": 10.0, "scale": 0.02, "count": 10},
        {"a": 20.0, "scale": 0.03, "count": 10},
    ]

    def generate_lr(lr_config: dict, key: jax.Array) -> jax.Array:
        """設定から学習率を生成"""
        a = lr_config["a"]
        scale = lr_config["scale"]
        count = lr_config["count"]

        return random.gamma(key=key, a=a, shape=(count,)) * scale

    # 設定リストから学習率を生成して結合
    rand_key, *keys = random.split(rand_key, len(lr_configs)+1)
    lr_batches = []

    for lr_config, key in zip(lr_configs, keys, strict=True):
        lr_batch = generate_lr(lr_config, key)
        lr_batches.append(lr_batch)

    lr = jnp.concatenate(lr_batches)

    lr_2d = jnp.atleast_2d(lr).T

    weight = load_weights()  # 重み生成
    weight = weight[:, :3]

    # CMHNのエネルギー関数
    e_cmhn = Partial(cmhn_energy, w=weight, beta=cfg.beta)   # 3つの記憶のみ
    grad_e = jax.vmap(jax.grad(e_cmhn))  # ベクトル化された導関数

    def simulate(initial: Float[ArrayLike, "num_particles dim"]):
        """シミュレーション"""
        # 入力刺激
        stimulus = jnp.full((cfg.steps, initial.shape[1]), jnp.nan)  # (steps, dim)
        key, subkey = random.split(rand_key)
        noisy_img = weight[:, 0] + normal_noise(weight.shape[0], cfg.stimulus.std, subkey)
        noisy_img = noisy_img / jnp.linalg.norm(noisy_img)
        stimulus = add_stimulation(stimulus, cfg.stimulus.start, cfg.stimulus.end, noisy_img)
        # for i in range(5):
        #     stimulus = add_stimulation(stimulus, 10+i*2, 11+i*2, weight[:, 0])  # (steps, dim)

        def step_fn(
            xs: Float[ArrayLike, "num_particles dim"],
            target: Float[ArrayLike, "dim num_particles"],
        ) -> tuple[Float[Array, "num_particles dim"], Float[Array, "num_particles dim"]]:
            """scan更新用関数"""
            grad = grad_e(xs)   # shape(粒子数、次元数): 勾配
            interaction = total_force(  # shape(粒子数、次元数):斥力
                xs,
                exponent=cfg.c,
                f_max=cfg.f_max,
            )
            # 入力刺激に画像適当な画像を入れ、時間変化を見る
            stimulation = stimulation_force(xs, target=target)  # shape(粒子数、次元数): 入力刺激

            # 勾配降下 + 斥力作用 + 入力刺激
            xs_new = xs - lr_2d * grad + cfg.gamma * interaction + 0.4 * stimulation
            return xs_new, xs_new   # (新しいxs, 記録用xs) のタプル

        # scanでシミュレーション
        _, history = lax.scan(step_fn, initial, xs=stimulus) # history: (steps, num_particles, dim)
        # 初期値も履歴に含める
        history = jnp.concatenate([initial[None, :], history], axis=0)  # (steps+1, num_particles, dim)
        # ノルムを1に正規化
        history /= jnp.linalg.norm(history, axis=-1, keepdims=True)


        # 結果保存
        output_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

        save_data = {
            "lr": lr.tolist(),  # 学習率
        }
        save_settings(save_data, cfg, output_path + "/settings.yaml")
        jnp.save(output_path + "/history.npy", history)
        jnp.save(output_path + "/weight.npy", weight)
        jnp.save(output_path + "/initial.npy", initial)

        # 結果表示
        # plot_images_trajectory(history, interval=5, path=output_path)
        plot_cos(history, weight, path=output_path)
        # plot_cos_trajectory(history, weight, path=output_path)
        # plot_tsne_trajectory(history, path=output_path)
        # plot_mean_image(history, interval=30)
        plot_entropy_time_series(history, weight, path=output_path)

    # シミュレーション
    initial = create_initial(
        w=weight,
        num_particles=cfg.num_particles,
        seed=cfg.seed,
        base_idx=0,
        std=0.1,
    )   # 初期値
    simulate(initial)

    # mean_initial = create_initial(
    #     w=weight,
    #     num_particles=cfg.num_particles,
    #     seed=cfg.seed,
    #     base_idx=[0, 1, 2],
    #     std=0.1,
    # )   # 初期値
    # simulate(mean_initial)


def save_settings(parameters: dict[str, Any], cfg: DictConfig, yaml_path: str) -> None:
    """設定を保存する"""
    with open_dict(cfg):
        cfg.runtime = parameters
    OmegaConf.save(cfg, yaml_path, resolve=True)




if __name__ == "__main__":
    run()
