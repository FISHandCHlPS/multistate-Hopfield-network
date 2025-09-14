import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from omegaconf import OmegaConf


def array2df(arr: np.ndarray, reset_index: bool = True) -> pd.DataFrame:
    """
    history to dataframe

    Args:
        arr: (T, num_particles, dim)
        reset_index: Trueの場合、indexをリセットする
    
    Returns:
        pd.DataFrame: (T*num_particles, dim) に整形したデータフレーム
    """
    df = pd.DataFrame(
        arr.reshape(-1, arr.shape[-1]),  # (T*num_particles, dim)に変形
        index=pd.MultiIndex.from_product([
            range(arr.shape[0]), 
            range(arr.shape[1])
        ], names=['t', 'particles']),
        columns=[f'entry_{i}' for i in range(arr.shape[-1])]
    )
    if reset_index: # マルチインデックス解除
        df = df.reset_index()
    return df


if __name__ == "__main__":
    results = multirun_loader(multirun_root="output/multi_pattern_mhn/multirun")
    print(results[0]["history"].shape)