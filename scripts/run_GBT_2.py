from birds import GBT, dataloader, utils
import torch
from torch.utils.data import random_split
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

    data_root = osp.join(cfg.root, 'data')
    seq_len = cfg.model.horizon

    print('normalize features')
    # initialize normalizer
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

    n_nodes = len(data[0].info['radars'])
    data = torch.utils.data.ConcatDataset(data)
    n_data = len(data)

    # split data into training and validation set
    n_val = max(1, int(cfg.datasource.val_train_split * n_data))
    n_train = n_data - n_val

    print('------------------------------------------------------')
    print('-------------------- data sets -----------------------')
    print(f'total number of sequences = {n_data}')
    print(f'number of training sequences = {n_train}')
    print(f'number of validation sequences = {n_val}')

    train_data, val_data = random_split(data, (n_train, n_val), generator=torch.Generator().manual_seed(cfg.seed))
    train_data = torch.utils.data.ConcatDataset(train_data)
    X_train, y_train, mask_train = GBT.prepare_data(train_data, timesteps=seq_len, mask_daytime=False,
                                                    use_acc_vars=cfg.model.use_acc_vars)

    X_val, y_val, mask_val = GBT.prepare_data(val_data, timesteps=seq_len, mask_daytime=False,
                                              use_acc_vars=cfg.model.use_acc_vars)

    with open(osp.join(output_dir, 'normalization.pkl'), 'wb') as f:
        pickle.dump(normalization, f)
    if cfg.root_transform == 0:
        cfg.datasource.bird_scale = float(normalization.max('birds_km2'))
    else:
        cfg.datasource.bird_scale = float(normalization.root_max('birds_km2', cfg.root_transform))

    print('------------------ model settings --------------------')
    print(cfg.model)
    print('------------------------------------------------------')


    print(f'train model')
    gbt = GBT.fit_GBT(X_train[mask_train], y_train[mask_train], **cfg.model,
                      seed=(cfg.seed + cfg.get('job_id', 0)))

    with open(osp.join(output_dir, f'model.pkl'), 'wb') as f:
        pickle.dump(gbt, f, pickle.HIGHEST_PROTOCOL)

    y_hat = gbt.predict(X_val)
    val_loss = utils.MSE(y_hat, y_val, mask_val)

    print(f'validation loss = {val_loss}', file=log)

    with open(osp.join(output_dir, f'config.yaml'), 'w') as f:
        OmegaConf.save(config=cfg, f=f)

    log.flush()

def test(cfg: DictConfig, output_dir: str, log, model_dir=None):
    assert cfg.model.name == 'GBT'

    data_root = osp.join(cfg.root, 'data')
    seq_len = cfg.model.test_horizon
    if model_dir is None: model_dir = output_dir

    # load normalizer
    with open(osp.join(model_dir, 'normalization.pkl'), 'rb') as f:
        normalization = pickle.load(f)
    if cfg.root_transform == 0:
        cfg.datasource.bird_scale = float(normalization.max('birds_km2'))
    else:
        cfg.datasource.bird_scale = float(normalization.root_max('birds_km2', cfg.root_transform))

    # load test data
    test_data = dataloader.RadarData(str(cfg.datasource.test_year), seq_len, **cfg,
                                     data_root=data_root,
                                     data_source=cfg.datasource.name,
                                     use_buffers=cfg.datasource.use_buffers,
                                     normalization=normalization,
                                     env_vars=cfg.datasource.env_vars,
                                     compute_fluxes=False
                                     )
    # load additional data
    time = test_data.info['timepoints']
    radars = test_data.info['radars']
    radar_index = {idx: name for idx, name in enumerate(radars)}

    X_test, y_test, mask_test = GBT.prepare_data_nights_and_radars(test_data,
                                    timesteps=cfg.model.test_horizon, mask_daytime=False,
                                    use_acc_vars=cfg.model.use_acc_vars)


    # load models and predict
    results = dict(gt_km2=[], prediction_km2=[], night=[], radar=[], seqID=[],
                   tidx=[], datetime=[], trial=[], horizon=[], missing=[])

    with open(osp.join(model_dir, f'model.pkl'), 'rb') as f:
        model = pickle.load(f)

    for nidx, data in enumerate(test_data):
        y = data.y * cfg.datasource.bird_scale
        _tidx = data.tidx
        local_night = data.local_night
        missing = data.missing

        if cfg.root_transform > 0:
            y = np.power(y, cfg.root_transform)

        for ridx, name in radar_index.items():
            y_hat = model.predict(X_test[nidx, :, ridx]) * cfg.datasource.bird_scale
            if cfg.root_transform > 0:
                y_hat = np.power(y_hat, cfg.root_transform)
            results['gt_km2'].append(y[ridx, :])
            results['prediction_km2'].append(y_hat)
            results['night'].append(local_night[ridx, :])
            results['radar'].append([name] * y.shape[1])
            results['seqID'].append([nidx] * y.shape[1])
            results['tidx'].append(_tidx)
            results['datetime'].append(time[_tidx])
            results['trial'].append([cfg.get('job_id', 0)] * y.shape[1])
            results['horizon'].append(np.arange(y.shape[1]))
            results['missing'].append(missing[ridx, :])

    # create dataframe containing all results
    for k, v in results.items():
        results[k] = np.concatenate(v)
    results['residual_km2'] = results['gt_km2'] - results['prediction_km2']
    df = pd.DataFrame(results)
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