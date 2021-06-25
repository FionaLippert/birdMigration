from omegaconf import DictConfig, OmegaConf
import hydra
import os.path as osp
import os
import traceback
import dataloader

@hydra.main(config_path="conf2", config_name="config")
def run(cfg: DictConfig):

    data_root = osp.join(cfg.root, 'data')

    log_file = os.path.join(out, 'log.txt')
    log = open(log_file, 'w')

    try:
        normalization = dataloader.Normalization(cfg.datasource.training_years,
                                                 cfg.datasource.name, data_root=data_root, **cfg)
        print('load data')
        # load training data
        data = [dataloader.RadarData(year, seq_len, **cfg,
                                     data_root=data_root,
                                     data_source=cfg.datasource.name,
                                     use_buffers=cfg.datasource.use_buffers,
                                     normalization=normalization,
                                     env_vars=cfg.datasource.env_vars,
                                     compute_fluxes=cfg.model.get('compute_fluxes', False))
                for year in cfg.datasource.training_years]
    except Exception:
        print(traceback.format_exc(), file=log)
    print('flush log')
    log.flush()
    log.close()
    print('done')

if __name__ == "__main__":
    run()