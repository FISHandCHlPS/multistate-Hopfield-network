"""multiPatternMHN.py

高次元における提案記憶モデル
"""
from collections.abc import Callable
from resource.cifar100 import get_cifar100

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
from plot.similarity import plot_cos

# VSCode用のレンダラーを設定
pio.renderers.default = "browser"


def load_weights() -> Float[Array, "dim num_patterns"]:
    """cifar100から重みを生成する"""
    images, _, _ = get_cifar100()  # CIFAR画像を読み込み
    images_flat = images.reshape(images.shape[0], -1)
    images_flat = images_flat / jnp.linalg.norm(images_flat, axis=1, keepdims=True)

    return images_flat.T    # 数式に合わせる shape(1024, 100)


def normal_noise(shape: tuple[int, ...], std: float, key: jax.Array) -> jax.Array:
    """ノイズを生成する"""
    return random.normal(key=key, shape=shape, dtype=jnp.float32) * std


def create_initial(
    w: Float[ArrayLike, "dim num_patterns"],
    num_particles: int = 1,
    seed: int = 42,
    base_idx: int | list[int] = 0,
    std: float = 1.0,
) -> Float[Array, "num_particles dim"]:
    """初期値を生成する"""
    img_key, noise_key = random.split(random.PRNGKey(seed))
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


def prepare_stimulus(
    cfg: DictConfig,
    weight: Float[ArrayLike, "dim num_patterns"],
    dim: int,
    rand_key: Array,
) -> tuple[Array, Array]:
    """入力刺激を準備する"""
    key, subkey = random.split(rand_key)
    stimulus = jnp.full((cfg.steps, dim), jnp.nan)  # 初期化
    noisy_img = weight[:, cfg.stimulus.target] + normal_noise(
        (weight.shape[0],), cfg.stimulus.std, subkey,
    )
    noisy_img = noisy_img / jnp.linalg.norm(noisy_img)
    stimulus = stimulus.at[cfg.stimulus.start : cfg.stimulus.end, :].set(noisy_img)
    return stimulus, key


def generate_learning_rates(
    lr_configs: list[dict], rand_key: Array,
) -> tuple[Array, Array]:
    """設定から学習率を生成"""
    rand_key, *keys = random.split(rand_key, len(lr_configs) + 1)
    lr_batches = []

    for conf, key in zip(lr_configs, keys, strict=True):
        a = conf["a"]
        scale = conf["scale"]
        count = conf["count"]

        lr_batches.append(random.gamma(key=key, a=a, shape=(count,)) * scale)

    return jnp.concatenate(lr_batches), rand_key


def simulate(
    initial: Float[ArrayLike, "num_particles dim"],
    lr_2d: Float[ArrayLike, "num_particles 1"],
    grad_e: Callable[[ArrayLike], Array],
    stimulus: Float[ArrayLike, "steps dim"],
    cfg: DictConfig,
) -> Float[Array, "steps num_particles dim"]:
    """シミュレーションを実行し、状態の履歴を返す"""

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
        # 入力刺激に適当な画像を入れ、時間変化を見る
        stimulation = stimulation_force(xs, target=target)  # shape(粒子数、次元数): 入力刺激

        # 勾配降下 + 斥力作用 + 入力刺激
        xs_new = xs - lr_2d * grad + cfg.gamma * interaction + cfg.eta * stimulation
        return xs_new, xs_new   # (新しいxs, 記録用xs) のタプル

    # scanでシミュレーション
    _, history = lax.scan(step_fn, initial, xs=stimulus) # history: (steps, num_particles, dim)
    # 初期値も履歴に含める
    history = jnp.concatenate([initial[None, :], history], axis=0)  # (steps+1, num_particles, dim)
    # ノルムを1に正規化
    history /= jnp.linalg.norm(history, axis=-1, keepdims=True)
    return history


def save_results(
    output_path: str,
    history: Float[Array, "steps num_particles dim"],
    weight: Float[Array, "dim num_patterns"],
    initial: Float[Array, "num_particles dim"],
    lr: Float[Array, " num_particles"],
    cfg: DictConfig,
) -> None:
    """シミュレーション結果と設定を保存する"""
    save_data = {
        "lr": lr.tolist(),  # 学習率
    }
    with open_dict(cfg):
        cfg.runtime = save_data
    OmegaConf.save(cfg, f"{output_path}/settings.yaml", resolve=True)

    jnp.save(f"{output_path}/history.npy", history)
    jnp.save(f"{output_path}/weight.npy", weight)
    jnp.save(f"{output_path}/initial.npy", initial)


def plot_results(
    output_path: str,
    history: Float[Array, "steps num_particles dim"],
    weight: Float[Array, "dim num_patterns"],
) -> None:
    """結果をプロットする"""
    plot_cos(history, weight, path=output_path)
    plot_entropy_time_series(history, weight, path=output_path)


@hydra.main(config_path="config", config_name="mpmhn", version_base=None)
def run(cfg: DictConfig) -> None:
    """実行関数"""
    rand_key = random.PRNGKey(cfg.seed)

    # 学習率を粒子毎にガウス分布から生成
    lr, rand_key = generate_learning_rates(list(cfg.lr_configs), rand_key)
    lr_2d = jnp.atleast_2d(lr).T

    # 重み
    weight = load_weights()
    if cfg.num_patterns is not None:
        weight = weight[:, : cfg.num_patterns]

    # CMHNのエネルギー関数
    e_cmhn = Partial(cmhn_energy, w=weight, beta=cfg.beta)
    grad_e = jax.vmap(jax.grad(e_cmhn))

    # 初期値生成
    initial = create_initial(
        w=weight,
        num_particles=cfg.num_particles,
        seed=cfg.seed,
        base_idx=0,
        std=0.1,
    )

    stimulus, rand_key = prepare_stimulus(
        cfg, weight, initial.shape[1], rand_key,
    )

    history = simulate(initial, lr_2d, grad_e, stimulus, cfg)

    output_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    save_results(output_path, history, weight, initial, lr, cfg)
    plot_results(output_path, history, weight)

    # mean_initial = create_initial(
    #     w=weight,
    #     num_particles=cfg.num_particles,
    #     seed=cfg.seed,
    #     base_idx=[0, 1, 2],
    #     std=0.1,
    # )   # 初期値
    # simulate(mean_initial)


if __name__ == "__main__":
    run()
