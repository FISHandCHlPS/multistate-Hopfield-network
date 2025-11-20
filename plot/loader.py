from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl
from jaxtyping import ArrayLike
from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

USE_MEMMAP = True  # True にするとメモリマップで読み込む。一部のデータのみ使いたい場合にTrue


def results_loader(
    root: str | Path = "output/multi_pattern_mhn/run",
) -> list[dict[str, Any]]:
    """ハイドラの実行結果を読み込むユーティリティ。

    Hydra の実行で生成されたディレクトリ配下を再帰的に探索し、
    各ランディレクトリの配列と `.hydra/config.yaml` を読み込んで返す。

    Args:
        root: 探索ルートディレクトリ（`run` でも `multirun` でも可）

    Returns:
        list[dict[str, Any]]: 以下のキーを持つ辞書のリスト
            'run_dir': ランディレクトリのパス (str)
            'history': 履歴配列 (np.ndarray|np.memmap, 形状: (T, N, D))
            'weight': 重み配列 (np.ndarray|np.memmap, 形状: (D, M))
            'initial': 初期値配列 (np.ndarray|np.memmap, 形状: (N, D))
            'config': Hydra 設定 (dict) | 読み込み失敗時は None

    """
    # root ディレクトリが無ければ空リストを返す
    root_path = Path(root)
    if not root_path.exists():
        return []

    results: list[dict[str, Any]] = []  # 結果を格納するリスト

    # .npy の3点セットを探索（history.npy を基準に走査）
    for hist_path in root_path.rglob("history.npy"):
        run_dir = hist_path.parent
        w_path = run_dir / "weight.npy"
        i_path = run_dir / "initial.npy"
        if not (w_path.exists() and i_path.exists()):
            continue    # 欠けている場合はスキップ

        mm_mode = "r" if USE_MEMMAP else None
        try:
            history = np.load(hist_path, mmap_mode=mm_mode)
            weight = np.load(w_path, mmap_mode=mm_mode)
            initial = np.load(i_path, mmap_mode=mm_mode)
        except (OSError, ValueError) as e:
            print(f"[WARN] .npy 読み込み失敗: {run_dir}")
            print(f"        reason: {e}")
            continue

        cfg_dict: dict[str, Any] | None = None
        cfg_path = run_dir / "settings.yaml"
        if cfg_path.exists():
            try:
                cfg = OmegaConf.load(cfg_path)
                cfg_dict = OmegaConf.to_container(cfg, resolve=True)
                if not isinstance(cfg_dict, dict):
                    cfg_dict = {"config": cfg_dict}
            except (OmegaConfBaseException, OSError) as e:
                cfg_dict = None
                print(f"[WARN] settings.yaml 読み込み失敗: {cfg_path} -> {e}")

        results.append({
            "run_dir": str(run_dir),
            "history": history,
            "weight": weight,
            "initial": initial,
            "config": cfg_dict,
        })

    return results


def extract_data(
    results: list[dict[str]], loading_data: Literal["history", "weight", "initial"],
) -> np.ndarray:
    """マルチラン結果から指定データを抽出して, 配列に変換する

    list[dict[str, Any]] -> dict[str, list] -> np.ndarray

    Returns:
        np.ndarray(num_run, *data_shape): 抽出したデータ

    """
    _list = (
        pl.DataFrame(results)
        .select(loading_data)
        .to_dict(as_series=False)[loading_data]
    )
    return np.asarray(_list)


def extract_parameters(results: list[dict[str]]) -> pl.DataFrame:
    """マルチラン結果からパラメータを抽出して, DataFrameに変換する"""
    return (
        pl.DataFrame(results)   # dfに変換
        .select("config")
        .with_row_index(name="trial")  # index列を追加
        .unnest("config")  # 展開してパラメータ毎の列を作成
    )


def eval_results(
    results: list[dict[str]], eval_func: Callable[[ArrayLike], ArrayLike],
    column_name: str = "eval", batch_size: int = 100,
) -> pl.DataFrame:
    """マルチラン結果に評価値を追加する"""
    v = []
    for i in range(0, len(results), batch_size):
        batch_results = results[i:i+batch_size]
        data = extract_data(batch_results, "history")
        v.append(pl.DataFrame(eval_func(data), schema=[column_name]))

    eval_df = pl.concat(v).with_row_index(name="trial")  # 評価値のDataFrame
    params = extract_parameters(results)  # パラメータのDataFrame
    return params.join(eval_df, on="trial").drop("trial")


if __name__ == "__main__":
    # results = results_loader(root="output/multi_pattern_mhn/run")
    results = results_loader(root="output/multi_pattern_mhn/multirun")
    #print(results_to_dataframe(results, loading_data="history"))
