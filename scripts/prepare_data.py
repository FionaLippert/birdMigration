from omegaconf import DictConfig, OmegaConf
import hydra
from birds import datasets
import os.path as osp
import traceback

@hydra.main(config_path="conf2", config_name="config")
def run(cfg: DictConfig):
    log_file = osp.join(cfg.root, cfg.output_dir, 'preprocessing_log.txt')
    log = open(log_file, 'w')

    try:
        datasets.preprocess(cfg)
        print('done', file=log)
    except Exception:
        print(traceback.format_exc(), file=log)

    log.flush()
    log.close()

if __name__ == "__main__":
    run()
