from omegaconf import DictConfig, OmegaConf
import hydra
import os.path as osp
import os
import traceback
import run_NNs, run_GAM, run_GBT, run_HA


@hydra.main(config_path="conf", config_name="config")
def run(cfg: DictConfig):

    print(f'hydra working directory: {os.getcwd()}')
    out = osp.join(cfg.output_dir, cfg.get('sub_dir', ''))
    print(f'output directory: {out}')
    os.makedirs(out, exist_ok=True)

    log_file = osp.join(out, 'log.txt')
    print(f'log file: {osp.abspath(log_file)}')
    log = open(log_file, 'w+')

    action = cfg.action

    try:
        if cfg.model.name == 'GBT':
            if 'cv' in action: run_GBT.cross_validation(cfg, out, log)
            if 'train' in action: run_GBT.train(cfg, out, log)
            if 'test' in action: run_GBT.test(cfg, out, log)
        elif cfg.model.name == 'GAM':
            if 'train' in action: run_GAM.train(cfg, out, log)
            if 'test' in action: run_GAM.test(cfg, out, log)
        elif cfg.model.name == 'HA':
            if 'train' in action: run_HA.train(cfg, out, log)
            if 'test' in action: run_HA.test(cfg, out, log)
        else:
            if 'cv' in action: run_NNs.run_cross_validation(cfg, out, log)
            if 'train' in action: run_NNs.run_training(cfg, out, log)
            if 'test' in action:
                cfg['fixed_t0'] = True
                run_NNs.run_testing(cfg, out, log, ext='_fixedT0')
                cfg['fixed_t0'] = False
                run_NNs.run_testing(cfg, out, log)
    except Exception:
        print(f'Error occurred! See {osp.abspath(log_file)} for more details.')
        print(traceback.format_exc(), file=log)
    print('flush log')
    log.flush()
    log.close()
    print('done')

if __name__ == "__main__":
    run()
