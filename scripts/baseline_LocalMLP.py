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
from matplotlib import pyplot as plt


#@hydra.main(config_path="conf", config_name="config")
def train(cfg: DictConfig):
    assert cfg.model.name == 'LocalMLP'
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
    epochs = cfg.model.epochs

    cuda = (not cfg.settings.cuda and torch.cuda.is_available())

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
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)

    val_data = datasets.RadarData(data_root, str(cfg.datasource.validation_year), season, ts,
                                  data_source=ds, use_buffers=use_buffers, bird_scale=bird_scale)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)
    if cfg.datasource.validation_year == cfg.datasource.test_year:
        val_loader, _ = utils.val_test_split(val_loader, cfg.datasource.test_val_split, seed)


    best_val_loss = np.inf
    best_hp_settings = None
    for vals in hp_space:
        hp_settings = dict(zip(param_names, vals))
        print('-' * 40)
        print(f'hyperparameter settings: {hp_settings}')

        sub_dir = osp.join(output_dir, json.dumps(hp_settings))
        os.makedirs(sub_dir, exist_ok=True)

        val_losses = np.ones(repeats) * np.inf
        training_curves = np.ones((repeats, epochs)) * np.nan
        val_curves = np.ones((repeats, epochs)) * np.nan
        for r in range(repeats):
            print(f'train model [trial {r}]')
            model = LocalMLP(**hp_settings, timesteps=ts, seed=(seed+r))

            params = model.parameters()
            optimizer = torch.optim.Adam(params, lr=hp_settings['lr'])
            scheduler = lr_scheduler.StepLR(optimizer, step_size=hp_settings['lr_decay'])

            for epoch in range(epochs):
                loss = train_dynamics(model, train_loader, optimizer, utils.MSE, cuda)
                training_curves[r, epoch] = loss / len(train_data)
                print(f'epoch {epoch + 1}: loss = {training_curves[r, epoch]}')

                model.eval()
                val_loss = test_dynamics(model, val_loader, ts, utils.MSE, cuda, bird_scale=1)
                val_loss = val_loss[torch.isfinite(val_loss)].mean() # TODO isfinite needed?
                val_curves[r, epoch] = val_loss
                print(f'epoch {epoch + 1}: val loss = {val_loss}')

                if epoch > 0 and np.abs(val_curves[r, epoch] - val_curves[r, epoch-1]) < cfg.action.tolerance:
                    # stop early
                    print(f'Stopped after epoch {epoch + 1}, because improvement was smaller than tolerance', file=log)
                    break


            if val_loss < val_losses[r]:
                # save best model so far
                torch.save(model.cpu(), osp.join(sub_dir, f'model_{r}.pkl'))
                val_losses[r] = val_loss

            scheduler.step()

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