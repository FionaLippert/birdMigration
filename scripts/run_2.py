from omegaconf import DictConfig, OmegaConf
import hydra
import os.path as osp
import os
import traceback
import run_NNs_2

@hydra.main(config_path="conf2", config_name="config")
def run(cfg: DictConfig):

    os.makedirs(cfg.output_dir, exist_ok=True)

    log_file = os.path.join(cfg.output_dir, 'log.txt')
    log = open(log_file, 'w')

    try:
        run_NNs_2.train(cfg, cfg.output_dir, log)
        run_NNs_2.test(cfg, cfg.output_dir, log)
    except Exception:
        print(traceback.format_exc(), file=log)

    log.flush()
    log.close()

if __name__ == "__main__":
    run()