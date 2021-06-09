from omegaconf import DictConfig, OmegaConf
import hydra
import os.path as osp
import os
import traceback
import run_NNs_2

@hydra.main(config_path="conf2", config_name="config")
def run(cfg: DictConfig):

    # directory to which outputs will be written
    output_dir = osp.join(cfg.root, 'results', cfg.datasource.name, cfg.action.name,
                          cfg.model.name, cfg.experiment)
    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(output_dir, 'log.txt')
    log = open(log_file, 'w')

    try:
        run_NNs_2.run(cfg, output_dir, log)
    except Exception:
        print(traceback.format_exc(), file=log)

    log.flush()
    log.close()

if __name__ == "__main__":
    run()