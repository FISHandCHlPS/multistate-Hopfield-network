from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from omegaconf import OmegaConf


def results_loader(
    root: Union[str, Path] = "output/multi_pattern_mhn/run",
    file_name: str = "result.npz",
    isSort: bool = False,
    mmap: bool = True,
) -> List[Dict[str, Any]]:
    """
    ハイドラの実行結果を読み込むユーティリティ。

    優先: .npy の3点セットをメモリマップで読み込み（`history.npy`, `weight.npy`, `initial.npy`）。
    互換: 見つからない場合は `result.npz` をフォールバックで読み込む。

    Hydra の実行で生成されたディレクトリ配下を再帰的に探索し、
    各ランディレクトリの配列と `.hydra/config.yaml` を読み込んで返す。

    Args:
        root: 探索ルートディレクトリ（`run` でも `multirun` でも可）
        file_name: フォールバック用の npz ファイル名（既定: result.npz）
        isSort: 結果をパスの降順でソートするか
        mmap: .npy 読み込み時にメモリマップを有効化するか

    Returns:
        List[Dict[str, Any]]: 以下のキーを持つ辞書のリスト
            - 'run_dir': ランディレクトリのパス (str)
            - 'history': 履歴配列 (np.ndarray|np.memmap, 形状: (T, N, D))
            - 'weight': 重み配列 (np.ndarray|np.memmap, 形状: (D, M))
            - 'initial': 初期値配列 (np.ndarray|np.memmap, 形状: (N, D))
            - 'config': Hydra 設定 (dict) | 読み込み失敗時は None
    """
    root_path = Path(root)
    if not root_path.exists():
        return []

    results: list[dict[str, Any]] = []

    # 1) .npy の3点セットを探索（history.npy を基準に走査）
    for hist_path in root_path.rglob("history.npy"):
        run_dir = hist_path.parent
        w_path = run_dir / "weight.npy"
        i_path = run_dir / "initial.npy"
        if not (w_path.exists() and i_path.exists()):
            # 欠けている場合はスキップ（後の npz フォールバックで拾う可能性あり）
            continue

        try:
            mmap_mode = "r" if mmap else None
            history = np.load(hist_path, mmap_mode=mmap_mode)
            weight = np.load(w_path, mmap_mode=mmap_mode)
            initial = np.load(i_path, mmap_mode=mmap_mode)
        except Exception:
            print(f"[WARN] .npy 読み込み失敗: {run_dir}")
            continue

        cfg_dict: Optional[Dict[str, Any]] = None
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
            "weight": weight,
            "initial": initial,
            "config": cfg_dict,
        })

    # 2) フォールバック: result.npz を探索（既に .npy で登録済みの run_dir は重複回避）
    registered = {r["run_dir"] for r in results}
    for npz_path in root_path.rglob(file_name):
        run_dir = str(npz_path.parent)
        if run_dir in registered:
            continue
        try:
            res = np.load(npz_path)
            history = res["history"]
            weight = res["weight"]
            initial = res["initial"]
        except Exception as e:
            print(f"[WARN] npz 読み込み失敗: {npz_path}: {e}")
            continue

        cfg_dict: Optional[Dict[str, Any]] = None
        hydra_cfg = Path(run_dir) / ".hydra" / "config.yaml"
        if hydra_cfg.exists():
            try:
                cfg = OmegaConf.load(hydra_cfg)
                cfg_dict = OmegaConf.to_container(cfg, resolve=True)  # type: ignore
                if not isinstance(cfg_dict, dict):
                    cfg_dict = {"config": cfg_dict}
            except Exception:
                cfg_dict = None

        results.append({
            "run_dir": run_dir,
            "history": history,
            "weight": weight,
            "initial": initial,
            "config": cfg_dict,
        })

    # 開始時間が新しい順にソート (パス文字列の辞書順で近似)
    if isSort:
        results.sort(key=lambda x: x["run_dir"], reverse=True)
    return results  # List[Dict[str, Any]]


if __name__ == "__main__":
    results = results_loader(root="output/multi_pattern_mhn/run", mmap=True)
    print(results[0]["history"].shape)
    print(results[0]["weight"].shape)
    print(results[0]["initial"].shape)
    print(results[0]["config"])
    print(results[0]["run_dir"])
