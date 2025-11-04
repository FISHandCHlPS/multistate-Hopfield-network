import numpy as np
from jaxtyping import Array, ArrayLike, Float


def calc_cos(
    x: Float[ArrayLike, "... dim"], y: Float[ArrayLike, "dim n_memory"],
) -> Float[Array, "... n_memory"]:
    """コサイン類似度を計算する。

    Args:
        x (np.ndarray): (..., dim)
        y (np.ndarray): (dim, n_memory)

    Returns:
        cos_matrix (np.ndarray): (..., n_memory)

    """
    x = np.asarray(x)
    y = np.asarray(y)
    x_norm = x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10)
    y_norm = y / (np.linalg.norm(y, axis=-2, keepdims=True) + 1e-10)
    return x_norm @ y_norm  # (..., n_memory)


def calc_psnr(
    x: Float[ArrayLike, "... dim"], y: Float[ArrayLike, "dim n_memory"],
) -> Float[Array, "... n_memory"]:
    """PSNRを計算する。

    Args:
        x (np.ndarray): (..., dim)
        y (np.ndarray): (n_memory, dim)

    Returns:
        psnr_matrix (np.ndarray): (..., n_memory)

    """
    x = np.asarray(x)
    y = np.asarray(y)
    max_i = 1.0
    mse = ((x[..., None, :] - y[None, :, :]) ** 2).mean(axis=-1)  # (..., n_memory)
    return 10 * np.log10(max_i ** 2 / (mse + 1e-10))  # (..., n_memory)


def calc_timechange(history: Float[ArrayLike, "... steps n_particles dim"]) -> tuple[Array, Array]:
    """時間変化量の平均と分散を計算する"""
    history = np.asarray(history)

    # 時間変化ベクトル (t-1, n, d)
    diff_vec = np.abs(history[..., 1:, :, :] - history[..., :-1, :, :])
    diff_vec_norm = np.linalg.norm(diff_vec, axis=-1)  # 時間変化ベクトルの長さ (t-1, n)
    timechange = np.mean(diff_vec_norm, axis=-2)  # 粒子ごとの平均時間変化 (n,)
    mean = np.mean(timechange, axis=-1)
    var = np.var(timechange, axis=-1)

    return mean, var


def calc_entropy(
    history: Float[ArrayLike, "... steps n_particles dim"],
    w: Float[ArrayLike, "dim n_memory"],
) -> Float[Array, "... steps"]:
    """エントロピーを計算する"""
    history = np.asarray(history)
    w = np.asarray(w)

    # コサイン類似度
    sim_matrix = calc_cos(history, w)  # (..., steps, n_particles, n_memory)

    # 一番類似度が高い記憶インデックス
    max_sim_indices = np.argmax(sim_matrix, axis=-1)  # (..., steps, n_particles)

    category = np.eye(history.shape[-2], dtype=float)
    counts = np.sum(
        category[max_sim_indices],
        axis=-2,
    )  # (..., steps, n_memory)
    prob = counts / history.shape[-2]  # (..., steps, n_memory)
    # エントロピーを計算    $$H(X) = -\sum_i p(x_i) \log_2 p(x_i)$$
    return - np.sum(prob * np.log2(prob + 1e-10), axis=-1)  # (..., steps)
