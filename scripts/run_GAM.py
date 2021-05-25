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
from pygam import PoissonGAM, te
# data=json.loads(argv[1])
from matplotlib import pyplot as plt


#@hydra.main(config_path="conf", config_name="config")
def train(cfg: DictConfig, output_dir: str, log):
    assert cfg.model.name == 'GAM'
    assert cfg.action.name == 'training'

    data_root = osp.join(cfg.root, 'data')
    ts = cfg.model.horizon

    # initialize normalizer
    normalization = datasets.Normalization(data_root, cfg.datasource.training_years, cfg.season,
                                           cfg.datasource.name, seed=cfg.seed,
                                   max_distance=cfg.max_distance,
                                   t_unit=cfg.t_unit, edge_type=cfg.edge_type)

    # load datasets
    train_data_list = [datasets.RadarData(data_root, str(year), cfg.season, ts,
                                     data_source=cfg.datasource.name,
                                     use_buffers=cfg.datasource.use_buffers,
                                     normalization=normalization,
                                     env_vars=[],
                                     root_transform=0,
                                     missing_data_threshold=cfg.missing_data_threshold,
                                          edge_type=cfg.edge_type,
                                          max_distance=cfg.max_distance,
                                          use_nights=cfg.use_nights
                                          )
                  for year in cfg.datasource.training_years]


    with open(osp.join(output_dir, 'normalization.pkl'), 'wb') as f:
        pickle.dump(normalization, f)

    cfg.datasource.bird_scale = float(normalization.max('birds_km2'))

    all_X = []
    all_y = []
    all_masks = []
    all_mappings = []
    for idx, data in enumerate(train_data_list):
        X_train, y_train, mask_train = GBT.prepare_data_gam(data, timesteps=ts, mask_daytime=False)
        all_X.append(X_train)
        all_y.append(y_train)
        all_masks.append(mask_train)
        radars = ['nldbl-nlhrw' if r in ['nldbl', 'nlhrw'] else r for r in data.info['radars']]
        m = {name: jdx for jdx, name in enumerate(radars)}
        all_mappings.append(m)

    for r in all_mappings[0].keys():
        X_r = []
        y_r = []
        for i, mapping in enumerate(all_mappings):
            ridx = mapping[r]
            X_r.append(all_X[i][all_masks[i][:, ridx], ridx]) # shape (time, features)
            y_r.append(all_y[i][all_masks[i][:, ridx], ridx]) # shape (time)
        X_r = np.concatenate(X_r, axis=0)
        y_r = np.concatenate(y_r, axis=0)


        # fit GAM with poisson distribution and log link
        print(f'fitting GAM for radar {r}')
        gam = PoissonGAM(te(0, 1, 2))
        gam.fit(X_r, y_r)

        with open(osp.join(output_dir, f'model_{r}.pkl'), 'wb') as f:
            pickle.dump(gam, f)


    with open(osp.join(output_dir, f'config.yaml'), 'w') as f:
        OmegaConf.save(config=cfg, f=f)

    log.flush()

def test(cfg: DictConfig, output_dir: str, log):
    assert cfg.model.name == 'GAM'
    assert cfg.action.name == 'testing'

    data_root = osp.join(cfg.root, 'data')

    train_dir = osp.join(cfg.root, 'results', cfg.datasource.name, 'training',
                         cfg.model.name, cfg.experiment)
    yaml = ruamel.yaml.YAML()
    fp = osp.join(train_dir, 'config.yaml')
    with open(fp, 'r') as f:
        model_cfg = yaml.load(f)

    # load normalizer
    with open(osp.join(train_dir, 'normalization.pkl'), 'rb') as f:
        normalization = pickle.load(f)

    cfg.datasource.bird_scale = float(normalization.max('birds_km2'))

    # load test data
    test_data = datasets.RadarData(data_root, str(cfg.datasource.test_year),
                                   cfg.season, cfg.model.horizon,
                                   data_source=cfg.datasource.name,
                                   use_buffers=cfg.datasource.use_buffers,
                                   normalization=normalization,
                                   env_vars=[],
                                   root_transform=0,
                                   missing_data_threshold=cfg.missing_data_threshold,
                                   edge_type=cfg.edge_type,
                                   max_distance=cfg.max_distance,
                                   t_unit=cfg.t_unit,
                                   use_nights=cfg.use_nights
                                   )
    # load additional data
    time = test_data.info['timepoints']
    radars = test_data.info['radars']
    radar_index = {idx: name for idx, name in enumerate(radars)}

    if cfg.datasource.validation_year == cfg.datasource.test_year:
        _, test_data = utils.val_test_split(test_data, cfg.datasource.val_test_split, cfg.seed)
    X_test, y_test, mask_test = GBT.prepare_data_nights_and_radars_gam(test_data,
                                    timesteps=cfg.model.horizon, mask_daytime=False)


    # load models and predict
    results = dict(gt_km2=[], prediction_km2=[], night=[], radar=[], seqID=[],
                   tidx=[], datetime=[], trial=[], horizon=[], missing=[], constant_prediction=[])

    for nidx, data in enumerate(test_data):
        y = data.y * cfg.datasource.bird_scale
        _tidx = data.tidx
        local_night = data.local_night
        missing = data.missing

        if cfg.root_transform > 0:
            y = np.power(y, cfg.root_transform)

        for ridx, name in radar_index.items():
            if name in ['nlhrw', 'nldbl']: name = 'nldbl-nlhrw'
            with open(osp.join(train_dir, f'model_{name}.pkl'), 'rb') as f:
                model = pickle.load(f)
            y_hat = model.predict(X_test[nidx, :, ridx]) * cfg.datasource.bird_scale
            if cfg.root_transform > 0:
                y_hat = np.power(y_hat, cfg.root_transform)

            results['gt_km2'].append(y[ridx, :])
            results['prediction_km2'].append(y_hat)
            results['constant_prediction'].append([y[ridx, 0]] * y.shape[1])
            results['night'].append(local_night[ridx, :])
            results['radar'].append([name] * y.shape[1])
            results['seqID'].append([nidx] * y.shape[1])
            results['tidx'].append(_tidx)
            results['datetime'].append(time[_tidx])
            results['trial'].append([0] * y.shape[1])
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