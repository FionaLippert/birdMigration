from omegaconf import DictConfig, OmegaConf
import hydra
import baseline_GBT, baseline_LocalMLP, baseline_LocalLSTM

@hydra.main(config_path="conf", config_name="config")
def run(cfg: DictConfig):
    if cfg.model.name == 'GBT':
        baseline_GBT.train(cfg)
    elif cfg.model.name == 'LocalMLP':
        baseline_LocalMLP.train(cfg)
    elif cfg.model.name == 'LocalLSTM':
        baseline_LocalLSTM.train(cfg)

if __name__ == "__main__":
    run()