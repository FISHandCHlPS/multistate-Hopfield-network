"""
hydra_test.py

Hydraを使った実行スクリプト
"""
import hydra
from omegaconf import DictConfig
import numpy as np

@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print(cfg)
    np.save("./seed.npy", np.array([cfg.seed]))

if __name__ == "__main__":
    main()