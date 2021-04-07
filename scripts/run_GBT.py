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
import pandas as pd
# data=json.loads(argv[1])


#@hydra.main(config_path="conf", config_name="config")
def train(cfg: DictConfig, output_dir: str, log):
    assert cfg.model.name == 'GBT'
    assert cfg.action.name == 'training'

    data_root = osp.join(cfg.root, 'data')
    ts = cfg.model.timesteps
    hps = cfg.model.hyperparameters

    # hyperparameters to use
    if cfg.action.grid_search:
        hp_space = it.product(*[settings.search_space for settings in hps.values()])
    else:
        hp_space = [[settings.default for settings in hps.values()]]
    param_names = [key for key in cfg.model.hyperparameters]

    # initialize normalizer
    normalization = datasets.Normalization(data_root, cfg.datasource.training_years, cfg.season,
                                           cfg.datasource.name, seed=cfg.seed)

    # load datasets
    train_data = [datasets.RadarData(data_root, str(year), cfg.season, ts,
                                     data_source=cfg.datasource.name, use_buffers=cfg.datasource.use_buffers,
                                     normalization=normalization) for year in cfg.datasource.training_years]
    train_data = torch.utils.data.ConcatDataset(train_data)
    X_train, y_train = GBT.prepare_data(train_data, timesteps=ts)

    val_data = datasets.RadarData(data_root, str(cfg.datasource.validation_year), cfg.season, ts,
                                  data_source=cfg.datasource.name, use_buffers=cfg.datasource.use_buffers,
                                  normalization=normalization)
    if cfg.datasource.validation_year == cfg.datasource.test_year:
        val_data, _ = utils.val_test_split(val_data, cfg.datasource.test_val_split, cfg.seed)
    X_val, y_val, mask_val = GBT.prepare_data(val_data, timesteps=ts, return_mask=True)

    with open(osp.join(output_dir, 'normalization.pkl'), 'wb') as f:
        pickle.dump(normalization, f)
    cfg.datasource.bird_scale = float(normalization.max('birds'))


    best_val_loss = np.inf
    best_hp_settings = None
    for vals in hp_space:
        hp_settings = dict(zip(param_names, vals))
        print('-' * 40)
        print(f'hyperparameter settings: {hp_settings}')

        sub_dir = osp.join(output_dir, json.dumps(hp_settings))
        os.makedirs(sub_dir, exist_ok=True)

        val_losses = np.zeros(cfg.repeats)
        for r in range(cfg.repeats):
            print(f'train model [trial {r}]')
            gbt = GBT.fit_GBT(X_train, y_train, **hp_settings, seed=(cfg.seed+r), tol=cfg.model.tolerance)

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
        np.save(osp.join(sub_dir, 'validation_losses.npy'), val_losses)


    print('saving best settings as default', file=log)
    # use ruamel.yaml to not overwrite comments in the original yaml
    yaml = ruamel.yaml.YAML()
    fp = osp.join(cfg.root, 'scripts', 'conf', 'model', f'{cfg.model.name}.yaml')
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

def test(cfg: DictConfig, output_dir: str, log):
    assert cfg.model.name == 'GBT'
    assert cfg.action.name == 'testing'

    data_root = osp.join(cfg.root, 'data')

    # load model settings
    hp_settings = {key: settings.default for key, settings in cfg.model.hyperparameters.items()}

    # directory to which outputs will be written
    output_dir = osp.join(output_dir, json.dumps(hp_settings))
    os.makedirs(output_dir, exist_ok=True)

    # directory from which model is loaded
    train_dir = osp.join(cfg.root, 'results', cfg.datasource.name, 'training',
                         cfg.model.name, cfg.experiment)
    model_dir = osp.join(train_dir, json.dumps(hp_settings))

    # load normalizer
    with open(osp.join(train_dir, 'normalization.pkl'), 'rb') as f:
        normalization = pickle.load(f)
    cfg.datasource.bird_scale = float(normalization.max('birds'))

    # load test data
    test_data = datasets.RadarData(data_root, str(cfg.datasource.test_year),
                                   cfg.season, cfg.model.timesteps,
                                   data_source=cfg.datasource.name,
                                   use_buffers=cfg.datasource.use_buffers,
                                   normalization=normalization)
    # load additional data
    time = test_data.info['timepoints']
    radars = test_data.info['radars']
    radar_index = {idx: name for idx, name in enumerate(radars)}

    if cfg.datasource.validation_year == cfg.datasource.test_year:
        _, test_data = utils.val_test_split(test_data, cfg.datasource.test_val_split, cfg.seed)
    X_test, y_test, mask_test = GBT.prepare_data_nights_and_radars(test_data,
                                    timesteps=cfg.model.timesteps, return_mask=True)


    # load models and predict
    gt, prediction, night, radar, seqID, tidx, datetime, trial = [], [], [], [], [], [], [], []
    for r in range(cfg.repeats):
        with open(osp.join(model_dir, f'model_{r}.pkl'), 'rb') as f:
            model = pickle.load(f)

        for nidx, data in enumerate(test_data):
            y = data.y * cfg.datasource.bird_scale
            _tidx = data.tidx
            local_night = data.local_night

            for ridx, name in radar_index.items():
                y_hat = model.predict(X_test[nidx, :, ridx]) * cfg.datasource.bird_scale
                gt.append(y[ridx, :])
                prediction.append(y_hat)
                night.append(local_night[ridx, :])
                radar.append([name] * y.shape[1])
                seqID.append([nidx] * y.shape[1])
                tidx.append(_tidx)
                datetime.append(time[_tidx])
                trial.append([r] * y.shape[1])

    # create dataframe containing all results
    df = pd.DataFrame(dict(
        gt=np.concatenate(gt),
        prediction = np.concatenate(prediction),
        night=np.concatenate(night),
        radar = np.concatenate(radar),
        seqID = np.concatenate(seqID),
        tidx = np.concatenate(tidx),
        datetime = np.concatenate(datetime),
        trial = np.concatenate(trial)
    ))
    df.to_csv(osp.join(output_dir, 'results.csv'))

    print(f'successfully saved results to {osp.join(output_dir, "results.csv")}', file=log)
    log.flush()


def run(cfg: DictConfig, output_dir: str, log):
    if cfg.action.name == 'training':
        train(cfg, output_dir, log)
    elif cfg.action.name == 'testing':
        test(cfg, output_dir, log)


if __name__ == "__main__":
    train()