from omegaconf import DictConfig, OmegaConf
import hydra
import os.path as osp
import os
import baseline_GBT, baseline_LocalMLP, baseline_LocalLSTM

@hydra.main(config_path="conf", config_name="config")
def run(cfg: DictConfig):
    # directory to which outputs will be written
    output_dir = osp.join(cfg.settings.root, 'results', cfg.datasource.name, cfg.action.name,
                          cfg.model.name, cfg.experiment)
    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(output_dir, 'log.txt')
    log = open(log_file, 'w')

    try:
        if cfg.model.name == 'GBT':
            baseline_GBT.run(cfg, output_dir, log)
        elif cfg.model.name == 'LocalMLP':
            baseline_LocalMLP.run(cfg, output_dir, log)
        elif cfg.model.name == 'LocalLSTM':
            baseline_LocalLSTM.run(cfg, output_dir, log)
    except Exception as e:
        print(e, file=log)

    log.flush()
    log.close()

if __name__ == "__main__":
    run()