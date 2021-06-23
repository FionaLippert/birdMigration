from birds import GBT, dataloader, utils
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
    assert cfg.model.name == 'HA'

    data_root = osp.join(cfg.root, 'data')
    ts = cfg.model.horizon

    print('normalize features')
    normalization = dataloader.Normalization(cfg.datasource.training_years, cfg.datasource.name,
                                             data_root=data_root, **cfg)

    print('load data')
    data_list = [dataloader.RadarData(year, ts, **cfg,
                                      data_root=data_root,
                                      data_source=cfg.datasource.name,
                                      use_buffers=cfg.datasource.use_buffers,
                                      normalization=normalization,
                                      env_vars=cfg.datasource.env_vars,
                                      compute_fluxes=False)
                 for year in cfg.datasource.training_years]

    with open(osp.join(output_dir, 'normalization.pkl'), 'wb') as f:
        pickle.dump(normalization, f)

    cfg.datasource.bird_scale = float(normalization.max('birds_km2'))

    all_y = []
    all_masks = []
    all_mappings = []
    for idx, data in enumerate(data_list):
        _, y_train, mask_train = GBT.prepare_data_gam(data, timesteps=ts, mask_daytime=True)
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

def test(cfg: DictConfig, output_dir: str, log, model_dir=None):
    assert cfg.model.name == 'HA'

    data_root = osp.join(cfg.root, 'data')
    if model_dir is None: model_dir = output_dir

    # load normalizer
    with open(osp.join(model_dir, 'normalization.pkl'), 'rb') as f:
        normalization = pickle.load(f)

    cfg.datasource.bird_scale = float(normalization.max('birds_km2'))

    # load test data
    test_data = dataloader.RadarData(str(cfg.datasource.test_year), cfg.model.test_horizon, **cfg,
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

    _, y_test, mask_test = GBT.prepare_data_nights_and_radars_gam(test_data,
                                    timesteps=cfg.model.test_horizon, mask_daytime=True)


    # load models and predict
    results = dict(gt_km2=[], prediction_km2=[], night=[], radar=[], seqID=[],
                   tidx=[], datetime=[], trial=[], horizon=[], missing=[])

    with open(osp.join(model_dir, f'HAs.pkl'), 'rb') as f:
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

            y_hat = np.ones(y.shape[1]) * y_hat * local_night[ridx, :].detach().numpy()

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
        results[k] = np.concatenate(v, axis=0)
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