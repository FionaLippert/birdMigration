from birds import datasets, era5interface
from omegaconf import DictConfig, OmegaConf
import hydra
import os.path as osp
import os
import geopandas as gpd


@hydra.main(config_path="conf", config_name="config")
def run(cfg: DictConfig):

    years = cfg.datasource.years
    print('preprocess data for years', years)
    data_root = osp.join(cfg.device.root, 'data')

    for year in years:
        target_dir = osp.join(data_root, 'preprocessed',
                              f'{cfg.t_unit}_{cfg.model.edge_type}_ndummy={cfg.model.n_dummy_radars}',
                              cfg.datasource.name, cfg.season, str(year))
        if not osp.isdir(target_dir):
            # load all features and organize them into dataframes
            print(f'year {year}: start preprocessing')
            os.makedirs(target_dir, exist_ok=True)
            datasets.prepare_features(target_dir, osp.join(data_root, 'raw'), str(year), cfg.datasource.name,
                             random_seed=cfg.seed, edge_type=cfg.model.edge_type,
                             n_dummy_radars=cfg.model.n_dummy_radars, **cfg)
        else:
            print(f'year {year}: nothing to be done')

if __name__ == "__main__":
    run()
