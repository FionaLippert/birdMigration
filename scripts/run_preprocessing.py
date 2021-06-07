from birds import datasets
from omegaconf import DictConfig, OmegaConf
import os.path as osp
import os



@hydra.main(config_path="conf", config_name="config")
def run(cfg: DictConfig):

    years = set(cfg.datasource.training_years + [cfg.datasource.test_year, cfg.datasource.validation_year])
    print('preprocess data for years', years)
    for year in years:
        dir = osp.join(cfg.root, 'preprocessed', cfg.t_unit, f'{cfg.edge_type}_dummy_radars={cfg.n_dummy_radars}_exclude={cfg.exclude}',
                        cfg.data_source.name, cfg.season, year)
        if not osp.isdir(dir):
            # load all features and organize them into dataframes
            os.makedirs(dir)
            datasets.prepare_features(dir, osp.join(cfg.root, 'raw'), cfg.data_source.name, cfg.season, str(year),
                             random_seed=cfg.seed, max_distance=cfg.max_distance,
                             t_unit=cfg.t_unit, edge_type=cfg.edge_type,
                             n_dummy_radars=cfg.n_dummy_radars, exclude=cfg.exclude)

if __name__ == "__main__":
    run()