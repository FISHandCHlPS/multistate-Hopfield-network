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
from jaxtyping import Array, ArrayLike, Float, Key
from omegaconf import DictConfig, OmegaConf, open_dict

from calc.mpmhn.energy import cmhn_energy
from calc.mpmhn.interaction import total_force
from calc.mpmhn.stimulation import stimulation_force
from plot.entropy import plot_entropy_time_series
from plot.similarity import plot_cos

# VSCode用のレンダラーを設定
pio.renderers.default = "browser"


def load_weights() -> Float[Array, "dim memory_size"]:
    """cifar100から重みを生成する"""
    images, _, _ = get_cifar100()  # CIFAR画像を読み込み
    images_flat = images.reshape(images.shape[0], -1)
    images_flat = images_flat / jnp.linalg.norm(images_flat, axis=1, keepdims=True)

    return images_flat.T    # 数式に合わせる shape(1024, 100)


def normal_noise(shape: tuple[int, ...], std: float, key: Key[Array, ""]) -> Array:
    """ノイズを生成する"""
    return random.normal(key=key, shape=shape, dtype=jnp.float32) * std


def create_initial(
    w: Float[ArrayLike, "dim memory_size"],
    sample_size: int,
    key: Key[Array, ""],
    base_idx: int | list[int] = 0,
    std: float = 1.0,
) -> Float[Array, "sample_size dim"]:
    """初期値を生成する"""
    if isinstance(base_idx, int):
        base_idx = [base_idx]
    base_img = jnp.sum(w[:, base_idx], axis=1)  # (1024)
    noise = normal_noise(
        shape=(sample_size, base_img.shape[0]),
        std=std,
        key=key,
    )
    #noisy_img = base_img + noise  # (sample_size, 1024)
    noisy_img = noise
    return noisy_img / jnp.linalg.norm(noisy_img, axis=1, keepdims=True)   # 正規化


def prepare_stimulus(
    cfg: DictConfig,
    weight: Float[ArrayLike, "dim memory_size"],
    key: Key[Array, ""],
) -> tuple[Float[Array, "steps dim"], float]:
    """入力刺激を準備する"""
    stimulus = jnp.full((cfg.steps, weight.shape[0]), jnp.nan)  # 初期化
    target_img = weight[:, cfg.stimulus.target]
    noisy_img = target_img + normal_noise(
        (weight.shape[0],), cfg.stimulus.std, key,
    )
    noisy_img = noisy_img / jnp.linalg.norm(noisy_img)

    # 正しい記憶からのずれを計算
    noise_amount = jnp.dot(noisy_img, target_img) / (
        jnp.linalg.norm(noisy_img) * jnp.linalg.norm(target_img)
    )

    stimulus = stimulus.at[cfg.stimulus.start : cfg.stimulus.end, :].set(noisy_img)

    return stimulus, noise_amount


def generate_learning_rates(
    lr_configs: list[dict], rand_key: Key[Array, ""],
) -> Float[Array, " sample_size"]:
    """設定から学習率を生成"""
    keys = random.split(rand_key, len(lr_configs))
    lr_batches = []

    for conf, key in zip(lr_configs, keys, strict=True):
        a = conf["a"]
        scale = conf["scale"]
        count = conf["count"]

        lr_batches.append(random.gamma(key=key, a=a, shape=(count,)) * scale)

    return jnp.concatenate(lr_batches)


def simulate(
    initial: Float[ArrayLike, "sample_size dim"],
    lr_2d: Float[ArrayLike, "sample_size 1"],
    grad_e: Callable[[ArrayLike], Array],
    stimulus: Float[ArrayLike, "steps dim"],
    cfg: DictConfig,
) -> Float[Array, "steps sample_size dim"]:
    """シミュレーションを実行し、状態の履歴を返す"""

    def step_fn(
        xs: Float[ArrayLike, "sample_size dim"],
        target: Float[ArrayLike, "dim sample_size"],
    ) -> tuple[Float[Array, "sample_size dim"], Float[Array, "sample_size dim"]]:
        """更新関数"""
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
    _, history = lax.scan(step_fn, initial, xs=stimulus) # history: (steps, sample_size, dim)
    # 初期値も履歴に含める
    history = jnp.concatenate([initial[None, :], history], axis=0)  # (steps+1, sample_size, dim)
    # ノルムを1に正規化
    history /= jnp.linalg.norm(history, axis=-1, keepdims=True)
    return history


def save_results(
    output_path: str,
    history: Float[Array, "steps sample_size dim"],
    weight: Float[Array, "dim memory_size"],
    initial: Float[Array, "sample_size dim"],
    params: dict[str, ArrayLike],
    cfg: DictConfig,
) -> None:
    """シミュレーション結果と設定を保存する"""
    save_data = params
    with open_dict(cfg):
        cfg.runtime = save_data
    OmegaConf.save(cfg, f"{output_path}/settings.yaml", resolve=True)

    jnp.save(f"{output_path}/history.npy", history)
    jnp.save(f"{output_path}/weight.npy", weight)
    jnp.save(f"{output_path}/initial.npy", initial)


def plot_results(
    output_path: str,
    history: Float[Array, "steps sample_size dim"],
    weight: Float[Array, "dim memory_size"],
) -> None:
    """結果をプロットする"""
    plot_cos(history, weight, path=output_path)
    plot_entropy_time_series(history, weight, path=output_path)


@hydra.main(config_path="config", config_name="mpmhn", version_base=None)
def run(cfg: DictConfig) -> None:
    """実行関数"""
    rang_key = random.PRNGKey(cfg.seed)

    # 学習率を生成
    rang_key, sub_key = random.split(rang_key)
    lr = generate_learning_rates(list(cfg.lr_configs), sub_key)
    lr_2d = jnp.atleast_2d(lr).T

    # 重み
    weight = load_weights()
    if cfg.memory_size is not None:
        weight = weight[:, : cfg.memory_size]

    # CMHNのエネルギー関数
    e = Partial(cmhn_energy, w=weight, beta=cfg.beta)
    grad_e = jax.vmap(jax.grad(e))

    # 初期値生成
    rang_key, sub_key = random.split(rang_key)
    initial = create_initial(
        w=weight,
        sample_size=cfg.sample_size,
        key=sub_key,
        base_idx=0,     # [0, 1, 2],
        std=0.1,
    )

    rang_key, sub_key = random.split(rang_key)
    stimulus, noise_amount = prepare_stimulus(
        cfg, weight, sub_key,
    )

    history = simulate(initial, lr_2d, grad_e, stimulus, cfg)

    output_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    params = {
        "lr": lr.tolist(),
        "noise_amount": float(noise_amount),
    }
    save_results(output_path, history, weight, initial, params, cfg)
    plot_results(output_path, history, weight)


if __name__ == "__main__":
    run()
