"""
hydra_test.py

Hydraを使った実行スクリプト
"""
import hydra
from omegaconf import DictConfig
import numpy as np

@hydra.main(config_path="config", config_name="test", version_base=None)
def main(cfg: DictConfig) -> None:
    output_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print(output_path)
    np.save(output_path + '/seed.npy', np.array([cfg.seed]))

if __name__ == "__main__":
    main()