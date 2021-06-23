from omegaconf import DictConfig, OmegaConf
import hydra
import os.path as osp
import os
import traceback
import run_NNs_2, run_GAM_2, run_GBT_2, run_HA_2

@hydra.main(config_path="conf2", config_name="config")
def run(cfg: DictConfig):

    out = osp.join(cfg.output_dir, f'job_{cfg.get("job_id", 0)}')
    print(f'output directory: {out}')
    os.makedirs(out, exist_ok=True)

    log_file = os.path.join(out, 'log.txt')
    log = open(log_file, 'w')

    try:
        if cfg.model.name == 'GBT':
            run_GBT_2.train(cfg, out, log)
            run_GBT_2.test(cfg, out, log)
        elif cfg.model.name == 'GAM':
            run_GAM_2.train(cfg, out, log)
            run_GAM_2.test(cfg, out, log)
        elif cfg.model.name == 'HA':
            run_HA_2.train(cfg, out, log)
            run_HA_2.test(cfg, out, log)
        else:
            run_NNs_2.train(cfg, out, log)
            run_NNs_2.test(cfg, out, log)
    except Exception:
        print(traceback.format_exc(), file=log)
    print('flush log')
    log.flush()
    log.close()
    print('done')

if __name__ == "__main__":
    run()