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


#@hydra.main(config_path="conf", config_name="config")
def train(cfg: DictConfig, output_dir: str, log):
    assert cfg.model.name == 'HA'
    assert cfg.action.name == 'training'

    data_root = osp.join(cfg.root, 'data')
    ts = cfg.model.timesteps

    # initialize normalizer
    normalization = datasets.Normalization(data_root, cfg.datasource.training_years, cfg.season,
                                           cfg.datasource.name, seed=cfg.seed, edge_type=cfg.edge_type,
                                   max_distance=cfg.max_distance,
                                   t_unit=cfg.t_unit)

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
                                          t_unit=cfg.t_unit
                                          )
                  for year in cfg.datasource.training_years]

    with open(osp.join(output_dir, 'normalization.pkl'), 'wb') as f:
        pickle.dump(normalization, f)

    cfg.datasource.bird_scale = float(normalization.max('birds'))

    all_y = []
    all_masks = []
    all_mappings = []
    for idx, data in enumerate(train_data_list):
        _, y_train, mask_train = GBT.prepare_data_gam(data, timesteps=ts, return_mask=True)
        all_y.append(y_train)
        all_masks.append(mask_train)
        radars = ['nldbl-nlhrw' if r in ['nldbl', 'nlhrw'] else r for r in data.info['radars']]
        m = {name: idx for idx, name in enumerate(radars)}
        all_mappings.append(m)

    ha = dict()
    for r in all_mappings[0].keys():
        y_r = []
        for i, mapping in enumerate(all_mappings):
            ridx = mapping[r]
            mask = all_masks[i][:, ridx]
            y_r.append(all_y[i][mask, ridx])
        y_r = np.concatenate(y_r, axis=0)

        ha[r] = y_r.mean()

    with open(osp.join(output_dir, f'HAs.pkl'), 'wb') as f:
        pickle.dump(ha, f)


    with open(osp.join(output_dir, f'config.yaml'), 'w') as f:
        OmegaConf.save(config=cfg, f=f)

    log.flush()

def test(cfg: DictConfig, output_dir: str, log):
    assert cfg.model.name == 'HA'
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

    cfg.datasource.bird_scale = float(normalization.max('birds'))

    # load test data
    test_data = datasets.RadarData(data_root, str(cfg.datasource.test_year),
                                   cfg.season, cfg.model.timesteps,
                                   data_source=cfg.datasource.name,
                                   use_buffers=cfg.datasource.use_buffers,
                                   normalization=normalization,
                                   env_vars=[],
                                   root_transform=0,
                                   missing_data_threshold=cfg.missing_data_threshold,
                                   edge_type=cfg.edge_type,
                                   max_distance=cfg.max_distance,
                                   t_unit=cfg.t_unit
                                   )
    # load additional data
    time = test_data.info['timepoints']
    radars = test_data.info['radars']
    radar_index = {idx: name for idx, name in enumerate(radars)}

    if cfg.datasource.validation_year == cfg.datasource.test_year:
        _, test_data = utils.val_test_split(test_data, cfg.datasource.val_test_split, cfg.seed)
    _, y_test, mask_test = GBT.prepare_data_nights_and_radars_gam(test_data,
                                    timesteps=cfg.model.timesteps, return_mask=True)


    # load models and predict
    results = dict(gt=[], prediction=[], night=[], radar=[], seqID=[],
                   tidx=[], datetime=[], trial=[], horizon=[], missing=[])

    with open(osp.join(train_dir, f'HAs.pkl'), 'rb') as f:
        ha = pickle.load(f)

    for nidx, data in enumerate(test_data):
        y = data.y * cfg.datasource.bird_scale
        _tidx = data.tidx
        local_night = data.local_night
        missing = data.missing

        if cfg.root_transform > 0:
            y = np.power(y, cfg.root_transform)

        for ridx, name in radar_index.items():
            if name in ['nlhrw', 'nldbl']: name = 'nldbl-nlhrw'
            y_hat = ha[name] * cfg.datasource.bird_scale
            if cfg.root_transform > 0:
                y_hat = np.power(y_hat, cfg.root_transform)

            results['gt'].append(y[ridx, :])
            results['prediction'].append([y_hat] * y.shape[1])
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
        results[k] = np.concatenate(v, axis=0)
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