from birds import GBT, datasets, utils
from birds.graphNN import *
import torch
from torch.optim import lr_scheduler
from torch_geometric.data import DataLoader
from omegaconf import DictConfig, OmegaConf
import hydra
import itertools as it
import pickle5 as pickle
import os.path as osp
import os
import json
import numpy as np
import ruamel.yaml
import yaml
from matplotlib import pyplot as plt
import pandas as pd

# map model name to implementation
MODEL_MAPPING = {'LocalMLP': LocalMLP,
                 'LocalLSTM': LocalLSTM,
                 'GraphLSTM': BirdDynamicsGraphLSTM}


# @hydra.main(config_path="conf", config_name="config")
def train(cfg: DictConfig, output_dir: str, log):
    assert cfg.model.name in MODEL_MAPPING
    assert cfg.action.name == 'training'

    Model = MODEL_MAPPING[cfg.model.name]

    data_root = osp.join(cfg.root, 'data')
    ts = cfg.model.timesteps
    hps = cfg.model.hyperparameters
    epochs = cfg.model.epochs
    fixed_boundary = cfg.model.get('fixed_boundary', False)

    device = 'cuda:0' if (cfg.cuda and torch.cuda.is_available()) else 'cpu'

    # hyperparameters to use
    if cfg.action.grid_search:
        hp_space = it.product(*[settings.search_space for settings in hps.values()])
    else:
        hp_space = [[settings.default for settings in hps.values()]]
    param_names = [key for key in cfg.model.hyperparameters]


    # initialize normalizer
    normalization = datasets.Normalization(data_root, cfg.datasource.training_years, cfg.season,
                                  cfg.datasource.name, seed=cfg.seed)

    # load training data
    train_data = [datasets.RadarData(data_root, year, cfg.season, ts,
                                     data_source=cfg.datasource.name, use_buffers=cfg.datasource.use_buffers,
                                     normalization=normalization)
                  for year in cfg.datasource.training_years]
    boundary = [ridx for ridx, b in train_data[0].info['boundaries'].items() if b]
    train_data = torch.utils.data.ConcatDataset(train_data)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)

    cfg.datasource.bird_scale = float(normalization.max('birds'))
    with open(osp.join(output_dir, 'config.yaml'), 'w') as f:
        OmegaConf.save(config=cfg, f=f)
    with open(osp.join(output_dir, 'normalization.pkl'), 'wb') as f:
        pickle.dump(normalization, f)

    # load validation data
    val_data = datasets.RadarData(data_root, str(cfg.datasource.validation_year), cfg.season, ts,
                                  data_source=cfg.datasource.name, use_buffers=cfg.datasource.use_buffers,
                                  normalization=normalization)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)
    if cfg.datasource.validation_year == cfg.datasource.test_year:
        val_loader, _ = utils.val_test_split(val_loader, cfg.datasource.test_val_split, cfg.seed)



    best_val_loss = np.inf
    best_hp_settings = None
    for vals in hp_space:
        hp_settings = dict(zip(param_names, vals))
        print('-' * 40)
        print(f'hyperparameter settings: {hp_settings}')

        sub_dir = osp.join(output_dir, json.dumps(hp_settings))
        os.makedirs(sub_dir, exist_ok=True)

        val_losses = np.ones(cfg.repeats) * np.inf
        training_curves = np.ones((cfg.repeats, epochs)) * np.nan
        val_curves = np.ones((cfg.repeats, epochs)) * np.nan
        for r in range(cfg.repeats):

            print(f'train model [trial {r}]')
            model = Model(**hp_settings, timesteps=ts, seed=(cfg.seed + r),
                          fixed_boundary=boundary if fixed_boundary else [])

            params = model.parameters()
            optimizer = torch.optim.Adam(params, lr=hp_settings['lr'])
            scheduler = lr_scheduler.StepLR(optimizer, step_size=hp_settings['lr_decay'])

            tf = 1.0 # initialize teacher forcing (is ignored for LocalMLP)
            for epoch in range(epochs):
                loss = train_dynamics(model, train_loader, optimizer, utils.MSE, device, teacher_forcing=tf)
                training_curves[r, epoch] = loss / len(train_data)
                print(f'epoch {epoch + 1}: loss = {training_curves[r, epoch]}')

                model.eval()
                val_loss = test_dynamics(model, val_loader, ts, utils.MSE, device, bird_scale=1)
                val_loss = val_loss[torch.isfinite(val_loss)].mean()  # TODO isfinite needed?
                val_curves[r, epoch] = val_loss
                print(f'epoch {epoch + 1}: val loss = {val_loss}')

                if val_loss < val_losses[r]:
                    # save best model so far
                    torch.save(model.cpu(), osp.join(sub_dir, f'model_{r}.pkl'))
                    val_losses[r] = val_loss

                scheduler.step()
                tf = tf * hp_settings.get('teacher_forcing_gamma', 0)


        if val_losses.mean() < best_val_loss:
            best_val_loss = val_losses.mean()
            best_hp_settings = hp_settings

            print(f'best settings so far with settings {hp_settings}', file=log)
            print(f'validation loss = {best_val_loss}', file=log)
            print('---------------------', file=log)

        log.flush()

        # save training and validation curves
        np.save(osp.join(sub_dir, 'training_curves.npy'), training_curves)
        np.save(osp.join(sub_dir, 'validation_curves.npy'), val_curves)
        np.save(osp.join(sub_dir, 'validation_losses.npy'), val_losses)

        # plotting
        fig, ax = plt.subplots()
        train_line = ax.plot(range(1, epochs + 1), training_curves.mean(0), label='training')
        ax.fill_between(range(1, epochs + 1), training_curves.mean(0) - training_curves.std(0),
                        training_curves.mean(0) + training_curves.std(0), alpha=0.2,
                        color=train_line[0].get_color())
        val_line = ax.plot(range(1, epochs + 1), val_curves.mean(0), label='validation')
        ax.fill_between(range(1, epochs + 1), val_curves.mean(0) - val_curves.std(0),
                        val_curves.mean(0) + val_curves.std(0), alpha=0.2,
                        color=val_line[0].get_color())
        ax.set(xlabel='epoch', ylabel='MSE', yscale='log', xscale='log')
        plt.legend()
        fig.savefig(osp.join(sub_dir, f'training_validation_curves.png'), bbox_inches='tight')

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
    assert cfg.model.name in MODEL_MAPPING
    assert cfg.action.name == 'testing'

    data_root = osp.join(cfg.root, 'data')
    device = 'cuda:0' if (cfg.cuda and torch.cuda.is_available()) else 'cpu'
    fixed_boundary = cfg.model.get('fixed_boundary', False)

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
    with open(osp.join(train_dir, 'normalization.pkl'), 'wb') as f:
        normalization = pickle.load(f)

    # load test data
    test_data = datasets.RadarData(data_root, str(cfg.datasource.test_year),
                                   cfg.season, cfg.model.timesteps,
                                   data_source=cfg.datasource.name,
                                   use_buffers=cfg.datasource.use_buffers,
                                   normalization=normalization)
    boundary = [ridx for ridx, b in test_data.info['boundaries'].items() if b]
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    if cfg.datasource.validation_year == cfg.datasource.test_year:
        _, test_loader = utils.val_test_split(test_loader, cfg.datasource.test_val_split, cfg.seed)

    # load additional data
    time = test_data.info['timepoints']
    radars = test_data.info['radars']
    radar_index = {idx: name for idx, name in enumerate(radars)}

    # load models and predict
    gt, prediction, night, radar, seqID, tidx, datetime, trial = [[]] * 8
    for r in range(cfg.repeats):
        model = torch.load(osp.join(model_dir, f'model_{r}.pkl'))

        # adjust model settings for testing
        model.timesteps = cfg.model.timesteps
        if fixed_boundary:
            model.fixed_boundary = boundary

        model.to(device)
        model.eval()

        for nidx, data in enumerate(test_loader):
            data = data.to(device)
            y_hat = model(data).cpu() * cfg.datasource.bird_scale
            y = data.y.cpu() * cfg.datasource.bird_scale
            _tidx = data.tidx.cpu()
            local_night = data.local_night.cpu()

            for ridx, name in radar_index.items():
                gt.append(y[ridx, :])
                prediction.append(y_hat[ridx, :])
                night.append(local_night[ridx, :])
                radar.append([name] * y.shape[1])
                seqID.append([nidx] * y.shape[1])
                tidx.append(_tidx)
                datetime.append(time[_tidx])
                trial.append([r] * y.shape[1])

    # create dataframe containing all results
    df = pd.DataFrame(dict(
        gt=torch.cat(gt).detach().numpy(),
        prediction = torch.cat(prediction).detach().numpy(),
        night=torch.cat(night),
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

