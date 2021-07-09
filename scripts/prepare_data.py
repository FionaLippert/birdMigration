from omegaconf import DictConfig, OmegaConf
import hydra
import os
import os.path as osp
from birds import datasets

@hydra.main(config_path="conf2", config_name="config")
def preprocess(cfg: DictConfig):
    data_root = osp.join(cfg.root, 'data')
    raw_dir = osp.join(data_root, 'raw')
    years = cfg.datasource.training_years + [cfg.datasource.test_year]
    for year in years:
        dir = osp.join(data_root, 'preprocessed', cfg.t_unit,
                       f'{cfg.edge_type}_dummy_radars={cfg.n_dummy_radars}_exclude={cfg.exclude}',
                        cfg.datasource.name, cfg.season, str(year))
        if not osp.isdir(dir):
            # load all features and organize them into dataframes
            os.makedirs(dir)
            datasets.prepare_features(dir, raw_dir, str(year), cfg.datasource.name, **cfg)
        else:
            print(f'Data has already been processed. '
                  f'To rerun preprocessing, remove the following directory: \n {dir}')

if __name__ == "__main__":
    preprocess()