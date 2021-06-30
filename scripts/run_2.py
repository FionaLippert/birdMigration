from omegaconf import DictConfig, OmegaConf
import hydra
import os.path as osp
import os
import traceback
import run_NNs_2, run_GAM_2, run_GBT_2, run_HA_2

@hydra.main(config_path="conf2", config_name="config")
def run(cfg: DictConfig):

    out = osp.join(cfg.output_dir, cfg.get('sub_dir', ''))
    print(f'output directory: {out}')
    os.makedirs(out, exist_ok=True)

    log_file = os.path.join(out, 'log.txt')
    log = open(log_file, 'w+')

    action = cfg.get('action', 'training+testing')

    try:
        if cfg.model.name == 'GBT':
            if 'training' in action: run_GBT_2.train(cfg, out, log)
            if 'testing' in action: run_GBT_2.test(cfg, out, log)
        elif cfg.model.name == 'GAM':
            if 'training' in action: run_GAM_2.train(cfg, out, log)
            if 'testing' in action: run_GAM_2.test(cfg, out, log)
        elif cfg.model.name == 'HA':
            if 'training' in action: run_HA_2.train(cfg, out, log)
            if 'testing' in action: run_HA_2.test(cfg, out, log)
        else:
            if 'training' in action: run_NNs_2.train(cfg, out, log)
            if 'testing' in action:
                run_NNs_2.test(cfg, out, log)
                cfg['use_nights'] = False
                run_NNs_2.test(cfg, out, log, ext='_no_nights')
    except Exception:
        print(traceback.format_exc(), file=log)
    print('flush log')
    log.flush()
    log.close()
    print('done')

if __name__ == "__main__":
    run()