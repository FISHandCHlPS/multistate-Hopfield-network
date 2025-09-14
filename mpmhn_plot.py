from collections.abc import Sequence
from pathlib import Path

from jax.typing import ArrayLike

from plot.loader import results_loader
from plot.multi_run.plot import plot_multi_run  # 類似度分散の可視化
from plot.pca import plot_pca_ccr, plot_pca_feature, plot_pca_trajectory
from plot.similarity import plot_cos_sim, plot_similarity_trajectory

"""データを読み込んでプロットする。"""


# 実行時に編集して使う定数（CLI を廃止し、ここで指定）
DEFAULT_ROOT: str = "output/multi_pattern_mhn/run"
SELECT_DATE: str | None = None  # 例: "2025-09-06"。指定しない場合は None
SELECT_TIME: str | None = None  # 例: "08-33-08"。指定しない場合は None

# 生成する可視化（必要に応じて編集）
DEFAULT_PLOTS: list[str] = [
    "similarity_variance",  # 類似度分散（推奨）
    # "cosine_similarity",
    # "similarity_trajectory",
    # "pca_feature",
    # "pca_trajectory",
    # "pca_ccr",
]

# 可視化の共通パラメータ
DEFAULT_IMG_SHAPE: tuple[int, int] = (32, 32)
DEFAULT_PCA_K: int = 6
DEFAULT_CCR_K: int = 20


def visualize_run(
    run: dict[str, ArrayLike],
    plots: Sequence[str],
    k: int,
    ccr_k: int,
    img_shape: tuple[int, int],
    reset_index: bool = False,
) -> None:
    """
    1ラン分の結果を可視化する。

    Args:
        run: multirun_loader が返す1要素の辞書
        plots: 生成する可視化の種類
        k: PCA可視化の主成分数
        ccr_k: 累積寄与率の主成分数
        img_shape: 画像形状 (H, W)
        reset_index: Trueの場合、historyのindexをリセットする

    """
    history: ArrayLike = run["history"]
    weight: ArrayLike = run["weight"]
    run_dir = run["run_dir"]

    out_dir = Path(run_dir)

    if reset_index:
        history = history.reset_index(drop=True)

    if "pca_feature" in plots:
        try:
            plot_pca_feature(history, k=k, img_shape=img_shape, path=str(out_dir), filename="pca_feature.html")
        except Exception as e:
            print(f"[WARN] PCA feature 可視化に失敗: {run_dir}: {e}")

    if "pca_trajectory" in plots:
        try:
            plot_pca_trajectory(history, path=str(out_dir), filename="pca_trajectory.html")
        except Exception as e:
            print(f"[WARN] PCA trajectory 可視化に失敗: {run_dir}: {e}")

    if "pca_ccr" in plots:
        try:
            plot_pca_ccr(history, k=ccr_k, path=str(out_dir), filename="pca_ccr.html")
        except Exception as e:
            print(f"[WARN] PCA CCR 可視化に失敗: {run_dir}: {e}")

    if "cosine_similarity" in plots:
        try:
            # plot_cos_sim は (D, M) 形状を想定して内部で転置する実装のため weight をそのまま渡す
            plot_cos_sim(history, weight, path=str(out_dir), filename="cosine_similarity.html")
        except Exception as e:
            print(f"[WARN] Cosine similarity 可視化に失敗: {run_dir}: {e}")

    if "similarity_trajectory" in plots:
        try:
            # similarity_trajectory は (M, D) 形状を想定
            plot_similarity_trajectory(history, weight.T)
        except Exception as e:
            print(f"[WARN] Similarity trajectory 可視化に失敗: {run_dir}: {e}")

    if "similarity_variance" in plots:
        try:
            plot_multi_run(
                history,
                weight,
                path=str(out_dir),
                bar_filename="similarity_variance_bar.html",
                time_filename="similarity_variance_time.html",
            )
        except Exception as e:
            print(f"[WARN] Similarity variance 可視化に失敗: {run_dir}: {e}")


def select_single_run(root: str, date: str, time: str) -> dict[str, ArrayLike]:
    """
    Run ルート配下から、指定した日付・時刻のランディレクトリを読み込み 1 件分の結果を返す。

    Args:
        root (str): run ディレクトリのルート。例: "output/multi_pattern_mhn/run"
        date (str): 日付ディレクトリ。例: "2025-09-06"
        time (str): 時刻ディレクトリ。例: "08-33-08"

    Returns:
        dict[str, Any]: results_loader が返す辞書 1 件分

    """
    target = Path(root) / date / time
    results = results_loader(root=str(target), isSort=False)
    if not results:
        msg = f"指定のパスに結果が見つかりません: {target}"
        raise FileNotFoundError(msg)
    return results[0]


def visualize_by_datetime(
    root: str,
    date: str,
    time: str,
    plots: Sequence[str],
    k: int,
    ccr_k: int,
    img_shape: tuple[int, int],
) -> None:
    """
    Run ディレクトリから特定の日付・時刻の結果を 1 件可視化する。

    Args:
        root (str): run ディレクトリのルート。
        date (str): 日付ディレクトリ名。
        time (str): 時刻ディレクトリ名。
        plots (Sequence[str]): 生成する可視化の種類。
        k (int): PCA 可視化の主成分数。
        ccr_k (int): 累積寄与率の主成分数。
        img_shape (tuple[int, int]): 画像形状 (H, W)。

    """
    run = select_single_run(root, date, time)
    visualize_run(run, plots, k=k, ccr_k=ccr_k, img_shape=img_shape)


def main() -> None:
    """シンプル実行のエントリポイント。モジュール先頭の定数を編集して可視化対象を切り替える"""
    root: str = DEFAULT_ROOT
    plots: list[str] = list(DEFAULT_PLOTS)
    img_shape: tuple[int, int] = DEFAULT_IMG_SHAPE
    k: int = DEFAULT_PCA_K
    ccr_k: int = DEFAULT_CCR_K

    # 特定日付・時刻が指定されている場合はその 1 件のみ可視化
    if SELECT_DATE is not None and SELECT_TIME is not None:
        print(f"[INFO] 指定ランを可視化: {root}/{SELECT_DATE}/{SELECT_TIME}")
        visualize_by_datetime(
            root=root,
            date=SELECT_DATE,
            time=SELECT_TIME,
            plots=plots,
            k=k,
            ccr_k=ccr_k,
            img_shape=img_shape,
        )
        return

    # それ以外は root 配下を再帰的に読み込んで順次可視化
    results = results_loader(root=root, isSort=True)
    if not results:
        print(f"[INFO] 対象ディレクトリに結果が見つかりません: {root}")
        return
    print(f"[INFO] 読み込み件数: {len(results)} ルート: {root}")
    for i, run in enumerate(results[:20], 1):
        print(f"[INFO] ({i}/{len(results)}) 可視化中: {run['run_dir']}")
        visualize_run(run, plots, k=k, ccr_k=ccr_k, img_shape=img_shape)


if __name__ == "__main__":
    main()
