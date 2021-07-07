from omegaconf import DictConfig, OmegaConf
import hydra
from birds import datasets

@hydra.main(config_path="conf2", config_name="config")
def run(cfg: DictConfig):

    datasets.preprocess(cfg)
    print('done')

if __name__ == "__main__":
    run()