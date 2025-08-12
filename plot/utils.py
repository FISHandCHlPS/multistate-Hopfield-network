import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from omegaconf import OmegaConf


def array2df(arr: np.ndarray) -> pd.DataFrame:
    """
    history to dataframe
    arr: (T, num_particles, dim)
    
    Returns:
        pd.DataFrame: (T*num_particles, dim) に整形したデータフレーム
    """
    df = pd.DataFrame(
        arr.reshape(-1, arr.shape[-1]),  # (T*num_particles, dim)に変形
        index=pd.MultiIndex.from_product([
            range(arr.shape[0]), 
            range(arr.shape[1])
        ], names=['t', 'particles']),
        columns=[f'dim_{i}' for i in range(arr.shape[-1])]
    )
    return df


def multirun_loader(
    multirun_root: Union[str, Path] = "output/multi_pattern_mhn/multirun",
    file_name: str = "history.npy",
    isSort: bool = False,
) -> List[Dict[str, Any]]:
    """
    パラメータごとに multirun の結果 (history.npy) を全て読み込む。

    Hydra の multirun 実行で生成されたディレクトリ配下を再帰的に探索し、
    各ランディレクトリに保存された `history.npy` を読み込んで返す。
    併せて、`.hydra/config.yaml` を読み込んで各ランの設定も辞書として付与する。

    Returns:
        List[Dict[str, Any]]: 以下のキーを持つ辞書のリスト
            - 'run_dir': ランディレクトリのパス (str)
            - 'history': 履歴配列 (np.ndarray, 形状: (T, N, D))
            - 'config': Hydra 設定 (dict) | 読み込み失敗時は None
    """
    root_path = Path(multirun_root)
    if not root_path.exists():
        return []

    results: List[Dict[str, Any]] = []

    # multirun 配下の全ての history.npy を探索
    for history_file in root_path.rglob(file_name):
        try:
            history = np.load(history_file)
        except Exception:
            # 壊れたファイル等はスキップ
            continue

        run_dir = history_file.parent
        cfg_dict: Optional[Dict[str, Any]] = None

        # 各ランの .hydra/config.yaml を読み取り (存在すれば)
        hydra_cfg = run_dir / ".hydra" / "config.yaml"
        if hydra_cfg.exists():
            try:
                cfg = OmegaConf.load(hydra_cfg)
                cfg_dict = OmegaConf.to_container(cfg, resolve=True)  # type: ignore
                if not isinstance(cfg_dict, dict):
                    cfg_dict = {"config": cfg_dict}
            except Exception:
                cfg_dict = None

        results.append({
            "run_dir": str(run_dir),
            "history": history,
            "config": cfg_dict,
        })

    # 開始時間が新しい順にソート (パス文字列の辞書順で近似)
    if isSort:
        results.sort(key=lambda x: x["run_dir"], reverse=True)
    return results


if __name__ == "__main__":
    results = multirun_loader(multirun_root="output/multi_pattern_mhn/multirun")
    print(results)