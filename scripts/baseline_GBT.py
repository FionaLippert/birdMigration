from birds import GBT, datasets, utils
import torch
from omegaconf import DictConfig, OmegaConf
import hydra
import itertools as it
import pickle5 as pickle
import os.path as osp
import os
import json
import numpy as np
import ruamel.yaml
# data=json.loads(argv[1])


#@hydra.main(config_path="conf", config_name="config")
def train(cfg: DictConfig):
    assert cfg.model.name == 'GBT'
    assert cfg.action.name == 'training'

    data_root = osp.join(cfg.settings.root, 'data')
    seed = cfg.settings.seed
    repeats = cfg.settings.repeats
    season = cfg.settings.season
    ts = cfg.action.timesteps
    ds = cfg.datasource.name
    use_buffers = cfg.datasource.use_buffers
    bird_scale = cfg.datasource.bird_scale
    hps = cfg.model.hyperparameters

    # directory to which outputs will be written
    output_dir = osp.join(cfg.settings.root, 'results', ds, cfg.action.name, cfg.model.name, cfg.experiment)
    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(output_dir, 'log.txt')
    log = open(log_file, 'w')

    # hyperparameters to use
    if cfg.action.grid_search:
        hp_space = it.product(*[settings.search_space for settings in hps.values()])
    else:
        hp_space = [[settings.default for settings in hps.values()]]
    param_names = [key for key in cfg.model.hyperparameters]

    # load datasets
    train_data = [datasets.RadarData(data_root, str(year), season, ts,
                                     data_source=ds, use_buffers=use_buffers,
                                     bird_scale=bird_scale) for year in cfg.datasource.training_years]
    train_data = torch.utils.data.ConcatDataset(train_data)
    X_train, y_train = GBT.prepare_data(train_data, timesteps=ts)

    val_data = datasets.RadarData(data_root, str(cfg.datasource.validation_year), season, ts,
                                  data_source=ds, use_buffers=use_buffers, bird_scale=bird_scale)
    if cfg.datasource.validation_year == cfg.datasource.test_year:
        val_data, _ = utils.val_test_split(val_data, cfg.datasource.test_val_split, seed)
    X_val, y_val, mask_val = GBT.prepare_data(val_data, timesteps=ts, return_mask=True)


    best_val_loss = np.inf
    best_hp_settings = None
    for vals in hp_space:
        hp_settings = dict(zip(param_names, vals))
        print('-' * 40)
        print(f'hyperparameter settings: {hp_settings}')

        sub_dir = osp.join(output_dir, json.dumps(hp_settings))
        os.makedirs(sub_dir, exist_ok=True)

        val_losses = np.zeros(repeats)
        for r in range(repeats):
            print(f'train model [trial {r}]')
            gbt = GBT.fit_GBT(X_train, y_train, **hp_settings, seed=(seed+r))

            with open(osp.join(sub_dir, f'model_{r}.pkl'), 'wb') as f:
                pickle.dump(gbt, f, pickle.HIGHEST_PROTOCOL)

            y_hat = gbt.predict(X_val)
            val_losses[r] = utils.MSE(y_hat, y_val, mask_val)

        if val_losses.mean() < best_val_loss:
            best_val_loss = val_losses.mean()
            best_hp_settings = hp_settings

            print(f'best settings so far with settings {hp_settings}', file=log)
            print(f'validation loss = {best_val_loss}', file=log)
            print('---------------------', file=log)

        log.flush()


    print('saving best settings as default', file=log)
    # use ruamel.yaml to not overwrite comments in the original yaml
    yaml = ruamel.yaml.YAML()
    fp = osp.join(cfg.settings.root, 'scripts', 'conf', 'model', f'{cfg.model.name}.yaml')
    with open(fp, 'r') as f:
        model_config = yaml.load(f)
    for key, val in best_hp_settings.items():
        model_config['hyperparameters'][key]['default'] = val
    with open(fp, 'w') as f:
        yaml.dump(model_config, f)


    # save complete config to output dir
    with open(osp.join(output_dir, f'config.yaml'), 'w') as f:
        OmegaConf.save(config=cfg, f=f)

    log.flush()
    log.close()


if __name__ == "__main__":
    train()