import numpy as np
from jaxtyping import Array, ArrayLike, Float


def calc_cos(
    x: Float[ArrayLike, "step num_particles dim"], y: Float[ArrayLike, "dim num_memory"],
) -> Float[Array, "step num_particles num_memory"]:
    """コサイン類似度を計算する。

    Args:
        x (np.ndarray): (T, N, D)
        y (np.ndarray): (D, M)

    Returns:
        cos_matrix (np.ndarray): (T, N, M)

    """
    x = np.asarray(x)
    y = np.asarray(y)
    x_norm = x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10)
    y_norm = y / (np.linalg.norm(y, axis=-1, keepdims=True) + 1e-10)
    return x_norm @ y_norm.T  # (T, N, M)


def calc_psnr(
    x: Float[ArrayLike, "step num_particles dim"], y: Float[ArrayLike, "num_memory dim"],
) -> Float[Array, "step num_particles num_memory"]:
    """PSNRを計算する。

    Args:
        x (np.ndarray): (T, N, D)
        y (np.ndarray): (M, D)

    Returns:
        psnr_matrix (np.ndarray): (T, N, M)

    """
    x = np.asarray(x)
    y = np.asarray(y)
    max_i = 1.0
    mse = ((x[..., None, :] - y[None, :, :]) ** 2).mean(axis=-1)  # (T, N, M)
    return 10 * np.log10(max_i ** 2 / (mse + 1e-10))  # (T, N, M)


def calc_timechange(history: Float[ArrayLike, "step num_particles dim"]) -> float:
    """粒子の平均時間変化量を計算する"""
    history = np.asarray(history)

    diff_vec = np.abs(history[1:, ...] - history[:-1, ...])  # 時間変化ベクトル (t-1, n, d)
    diff_vec_norm = np.linalg.norm(diff_vec, axis=2)  # 時間変化量 (t-1, n)

    return np.mean(diff_vec_norm)  # float



# def calc_variance(history: ArrayLike, w: ArrayLike) -> float:
#     """収束後の平均分散を計算する"""
#     history = np.asarray(history)
#     w = np.asarray(w)
#     # t, n, d = history.shape
#     history = history[-1:, ...]  # (1, n, d)
#     # wとのコサイン類似度の分散を計算する
#     history_norm = history / (np.linalg.norm(history, axis=-1, keepdims=True) + 1e-10)
#     w_norm = w / (np.linalg.norm(w, axis=-1, keepdims=True) + 1e-10)
#     cos_matrix = history_norm @ w_norm.T  # (1, n, d) @ (d, M) -> (1, n, M)

#     return cos_matrix.var(axis=1).mean()  # float

