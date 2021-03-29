from omegaconf import DictConfig, OmegaConf
import hydra
import baseline_GBT, baseline_LocalMLP

@hydra.main(config_path="conf", config_name="config")
def run(cfg: DictConfig):
    if cfg.model.name == 'GBT':
        baseline_GBT.train(cfg)
    elif cfg.model.name == 'LocalMLP':
        baseline_LocalMLP.train(cfg)

if __name__ == "__main__":
    run()