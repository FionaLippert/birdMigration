from birds import dataloader, utils
from birds.graphNN import *
import torch
from torch.optim import lr_scheduler
from torch_geometric.data import DataLoader, DataListLoader
from torch_geometric.utils import to_dense_adj
from torch_geometric.nn import DataParallel
from omegaconf import DictConfig, OmegaConf
import itertools as it
import pickle5 as pickle
import os.path as osp
import os
import json
import numpy as np
import ruamel.yaml
import pandas as pd

# map model name to implementation
MODEL_MAPPING = {'LocalMLP': LocalMLP,
                 'LSTM': LSTM,
                 'LocalLSTM': LocalLSTM,
                 'GraphLSTM': BirdDynamicsGraphLSTM,
                 'GraphLSTM_transformed': BirdDynamicsGraphLSTM_transformed,
                 'BirdFluxGraphLSTM': BirdFluxGraphLSTM,
                 'BirdFluxGraphLSTM2': BirdFluxGraphLSTM2,
                 'testFluxMLP': testFluxMLP,
                 #'BirdFluxGroundGraphLSTM': BirdFluxGroundGraphLSTM,
                 #'BlackBoxGraphLSTM': BlackBoxGraphLSTM,
                 'AttentionGraphLSTM': AttentionGraphLSTM}




# @hydra.main(config_path="conf", config_name="config")
def train(cfg: DictConfig, output_dir: str, log):
    assert cfg.model.name in MODEL_MAPPING
    assert cfg.action.name == 'training'

    torch.autograd.set_detect_anomaly(True)

    Model = MODEL_MAPPING[cfg.model.name]

    data_root = osp.join(cfg.root, 'data')

    device = 'cuda:0' if (cfg.cuda and torch.cuda.is_available()) else 'cpu'
    batch_size = cfg.model.batch_size
    epochs = cfg.model.epochs
    seq_len = cfg.model.get('context', 0) + cfg.model.horizon


    print('normalize features')
    # initialize normalizer
    normalization = dataloader.Normalization(data_root, cfg.datasource.training_years, cfg.season,
                                  cfg.datasource.name, **cfg)
    print('load training data')
    # load training data
    train_data = [dataloader.RadarData(data_root, year, cfg.season, seq_len, **cfg,
                                     data_source=cfg.datasource.name,
                                     use_buffers=cfg.datasource.use_buffers,
                                     normalization=normalization,
                                     env_vars=cfg.datasource.env_vars,
                                     compute_fluxes=cfg.model.get('compute_fluxes', False))
                  for year in cfg.datasource.training_years]
    # boundary = [ridx for ridx, b in train_data[0].info['boundaries'].items() if b]
    n_nodes = len(train_data[0].info['radars'])
    train_data = torch.utils.data.ConcatDataset(train_data)

    if cfg.use_nights:
        print(f'training set size = {len(train_data)}')
        args = dict(batch_size=batch_size, shuffle=True)
    else:
        n_exclude = int((1 - cfg.data_perc) * len(train_data))
        print(f'training set size = {len(train_data) - n_exclude}')
        rng = np.random.default_rng(cfg.seed)
        exclude_indices = torch.from_numpy(rng.choice(len(train_data), size=n_exclude, replace=False))
        args = dict(batch_size=batch_size, shuffle=True, exclude_keys=list(exclude_indices))

    train_loader = DataLoader(train_data, **args)


    print('loaded training data')

    if cfg.edge_type == 'voronoi':
        if cfg.datasource.use_buffers:
            input_col = 'birds_from_buffer'
        else:
            input_col = 'birds'
    else:
        input_col = 'birds_km2'

    cfg.datasource.bird_scale = float(normalization.max(input_col))
    with open(osp.join(output_dir, 'config.yaml'), 'w') as f:
        OmegaConf.save(config=cfg, f=f)
    with open(osp.join(output_dir, 'normalization.pkl'), 'wb') as f:
        pickle.dump(normalization, f)

    if cfg.root_transform == 0:
        cfg.datasource.bird_scale = float(normalization.max(input_col))
    else:
        cfg.datasource.bird_scale = float(normalization.root_max(input_col, cfg.root_transform))

    print('load val data')
    # load validation data
    val_data = dataloader.RadarData(data_root, str(cfg.datasource.validation_year),
                                  cfg.season, seq_len, **cfg,
                                  data_source=cfg.datasource.name,
                                  use_buffers=cfg.datasource.use_buffers,
                                  normalization=normalization,
                                  env_vars=cfg.datasource.env_vars,
                                  compute_fluxes=cfg.model.get('compute_fluxes', False),
                                  use_nights=True
                                  )
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)
    if cfg.datasource.validation_year == cfg.datasource.test_year:
        val_loader, _ = utils.val_test_split(val_loader, cfg.datasource.val_test_split, cfg.seed)
    print('loaded val data')

    if cfg.model.get('root_transformed_loss', False):
        loss_func = utils.MSE_root_transformed
    else:
        loss_func = utils.MSE

    print('-------------- hyperparamter settings ----------------')
    print(cfg.model)
    print('------------------------------------------------------')


    os.makedirs(output_dir, exist_ok=True)

    val_losses = np.ones(cfg.repeats) * np.inf
    training_curves = np.ones((cfg.repeats, epochs)) * np.nan
    val_curves = np.ones((cfg.repeats, epochs)) * np.nan
    for r in range(cfg.repeats):

        print(f'train model [trial {r}]')
        print(cfg.datasource.env_vars)
        model = Model(**cfg.model, timesteps=cfg.model.horizon, seed=(cfg.seed + r),
                      n_env=2+len(cfg.datasource.env_vars),
                      n_nodes=n_nodes,
                      edge_type=cfg.edge_type)

        states_path = cfg.model.get('load_states_from', '')
        if osp.isfile(states_path):
            model.load_state_dict(torch.load(states_path))

        # if n_devices > 1:
        #     model = DataParallel(model)
        model = model.to(device)

        params = model.parameters()
        optimizer = torch.optim.Adam(params, lr=cfg.model.lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.model.lr_decay)

        model = model.to(device)

        tf = 1.0 # initialize teacher forcing (is ignored for LocalMLP)
        for epoch in range(epochs):
            print('model on GPU?', next(model.parameters()).is_cuda)
            if 'BirdFluxGraphLSTM' in cfg.model.name:
                loss = train_fluxes(model, train_loader, optimizer, loss_func, device,
                                    conservation_constraint=cfg.model.get('conservation_constraint', 0),
                                    teacher_forcing=tf, daymask=cfg.model.get('force_zeros', 0),
                                    boundary_constraint_only=cfg.model.get('boundary_constraint_only', 0))
            elif cfg.model.name == 'testFluxMLP':
                loss = train_testFluxMLP(model, train_loader, optimizer, loss_func, device)
            else:
                loss = train_dynamics(model, train_loader, optimizer, loss_func, device, teacher_forcing=tf,
                                  daymask=cfg.model.get('force_zeros', 0))
            training_curves[r, epoch] = loss / len(train_data)
            print(f'epoch {epoch + 1}: loss = {training_curves[r, epoch]}')

            val_loss = test_dynamics(model, val_loader, loss_func, device, bird_scale=1,
                                     daymask=cfg.model.get('force_zeros', 0)).cpu()
            val_loss = val_loss[torch.isfinite(val_loss)].mean()  # TODO isfinite needed?
            val_curves[r, epoch] = val_loss
            print(f'epoch {epoch + 1}: val loss = {val_loss}')

            if val_loss <= val_losses[r]:
                # save best model so far
                print('best model so far; save to disk ...')
                torch.save(model.state_dict(), osp.join(output_dir, f'model_{r}.pkl'))
                val_losses[r] = val_loss

            scheduler.step()
            tf = tf * cfg.model.get('teacher_forcing_gamma', 0)

            # plotting
            utils.plot_training_curves(training_curves, val_curves, output_dir, log=True)
            utils.plot_training_curves(training_curves, val_curves, output_dir, log=False)


        print(f'validation loss run {r} = {val_losses[r]}', file=log)

    log.flush()

    # save training and validation curves
    np.save(osp.join(output_dir, 'training_curves.npy'), training_curves)
    np.save(osp.join(output_dir, 'validation_curves.npy'), val_curves)
    np.save(osp.join(output_dir, 'validation_losses.npy'), val_losses)

    # plotting
    utils.plot_training_curves(training_curves, val_curves, output_dir, log=True)
    utils.plot_training_curves(training_curves, val_curves, output_dir, log=False)

    with open(osp.join(output_dir, f'config.yaml'), 'w') as f:
        OmegaConf.save(config=cfg, f=f)

    log.flush()


def test(cfg: DictConfig, output_dir: str, log):
    assert cfg.model.name in MODEL_MAPPING
    assert cfg.action.name == 'testing'

    Model = MODEL_MAPPING[cfg.model.name]

    data_root = osp.join(cfg.root, 'data')
    device = 'cuda:0' if (cfg.cuda and torch.cuda.is_available()) else 'cpu'
    fixed_boundary = cfg.model.get('fixed_boundary', False)
    birds_per_km2 = cfg.get('birds_per_km2', False)

    context = cfg.model.get('context', 0)
    seq_len = context + cfg.model.horizon
    seq_shift = context // 24

    compute_fluxes = cfg.model.get('compute_fluxes', False)
    p_std = cfg.model.get('perturbation_std', 0)
    p_mean = cfg.model.get('perturbation_mean', 0)

    # load best settings from grid search (or setting used for regular training)
    train_dir = osp.join(cfg.root, 'results', cfg.datasource.name, 'training',
                         cfg.model.name, cfg.experiment)
    yaml = ruamel.yaml.YAML()
    fp = osp.join(train_dir, 'config.yaml')
    with open(fp, 'r') as f:
        model_cfg = yaml.load(f)


    if cfg.edge_type == 'voronoi':
        if cfg.datasource.use_buffers:
            input_col = 'birds_from_buffer'
        else:
            input_col = 'birds'
    else:
        input_col = 'birds_km2'

    # load normalizer
    with open(osp.join(train_dir, 'normalization.pkl'), 'rb') as f:
        normalization = pickle.load(f)
    if cfg.root_transform == 0:
        cfg.datasource.bird_scale = float(normalization.max(input_col))
    else:
        cfg.datasource.bird_scale = float(normalization.root_max(input_col, cfg.root_transform))

    # load test data
    test_data = dataloader.RadarData(data_root, str(cfg.datasource.test_year),
                                   cfg.season, seq_len,
                                   data_source=cfg.datasource.name,
                                   use_buffers=cfg.datasource.use_buffers,
                                   normalization=normalization,
                                   env_vars=cfg.datasource.env_vars,
                                   root_transform=cfg.root_transform,
                                   missing_data_threshold=cfg.missing_data_threshold,
                                   edge_type=cfg.edge_type,
                                   max_distance=cfg.max_distance,
                                   t_unit=cfg.t_unit,
                                   n_dummy_radars=cfg.n_dummy_radars,
                                   exclude=cfg.exclude,
                                   compute_fluxes=compute_fluxes,
                                   use_nights=True,
                                   birds_per_km2=birds_per_km2
                                   )
    n_nodes = len(test_data.info['radars'])
    # boundary = [ridx for ridx, b in test_data.info['boundaries'].items() if b]
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    if cfg.datasource.validation_year == cfg.datasource.test_year:
        _, test_loader = utils.val_test_split(test_loader, cfg.datasource.val_test_split, cfg.seed)

    # load additional data
    time = test_data.info['timepoints']
    radars = test_data.info['radars']
    areas = np.ones(len(radars)) if input_col == 'birds_km2' else test_data.info['areas']
    radar_index = {idx: name for idx, name in enumerate(radars)}

    # load models and predict
    results = dict(gt=[], gt_km2=[], prediction=[], prediction_km2=[], night=[], radar=[], seqID=[],
                   tidx=[], datetime=[], trial=[], horizon=[], missing=[])
    if cfg.model.name in ['GraphLSTM', 'BirdFluxGraphLSTM', 'BirdFluxGraphLSTM2']:
        results['fluxes'] = []
        results['local_deltas'] = []
    if cfg.model.name in ['BirdFluxGraphLSTM', 'BirdFluxGraphLSTM2']:
        results['influxes'] = []
        results['outfluxes'] = []

    for r in range(cfg.repeats):

        try:
            model = Model(**model_cfg.model, seed=(cfg.seed + r),
                  n_env=2 + len(cfg.datasource.env_vars),
                  n_nodes=n_nodes,
                  edge_type=cfg.edge_type)

            model.load_state_dict(torch.load(osp.join(model_dir, f'model_{r}.pkl')))
        except Exception:
            model = torch.load(osp.join(model_dir, f'model_{r}.pkl'))


        # adjust model settings for testing
        model.horizon = cfg.model.horizon
        if cfg.model.get('fixed_boundary', 0):
            model.fixed_boundary = True
            model.perturbation_mean = p_mean
            model.perturbation_std = p_std

        model.to(device)
        model.eval()

        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data)

        local_fluxes = {}
        radar_fluxes = {}
        radar_mtr = {}
        attention_weights = {}
        # attention_weights_state = {}

        for nidx, data in enumerate(test_loader):
            nidx += seq_shift
            data = data.to(device)
            y_hat = model(data).cpu().detach() * cfg.datasource.bird_scale
            y = data.y.cpu() * cfg.datasource.bird_scale

            if cfg.root_transform > 0:
                # transform back
                y = torch.pow(y, cfg.root_transform)
                y_hat = torch.pow(y_hat, cfg.root_transform)

            _tidx = data.tidx[context:].cpu()
            local_night = data.local_night.cpu()
            missing = data.missing.cpu()

            if cfg.model.name == 'GraphLSTM':
                fluxes = model.fluxes.detach().cpu()
                local_deltas = model.local_deltas.detach().cpu()
            elif cfg.model.name in ['BirdFluxGraphLSTM', 'BirdFluxGraphLSTM2', 'testFluxMLP']:
                local_fluxes[nidx] = to_dense_adj(data.edge_index, edge_attr=model.local_fluxes).view(
                                    data.num_nodes, data.num_nodes, -1).detach().cpu()
                radar_fluxes[nidx] = to_dense_adj(data.edge_index, edge_attr=data.fluxes).view(
                    data.num_nodes, data.num_nodes, -1).detach().cpu()
                radar_mtr[nidx] = to_dense_adj(data.edge_index, edge_attr=data.mtr).view(
                    data.num_nodes, data.num_nodes, -1).detach().cpu()
                fluxes = (local_fluxes[nidx]  - local_fluxes[nidx].permute(1, 0, 2)).sum(1)

                influxes = local_fluxes[nidx].sum(1)
                outfluxes = local_fluxes[nidx].permute(1, 0, 2).sum(1)
                local_deltas = model.local_deltas.detach().cpu()
            elif cfg.model.name == 'AttentionGraphLSTM':
                attention_weights[nidx] = to_dense_adj(data.edge_index, edge_attr=model.alphas_s).view(
                                    data.num_nodes, data.num_nodes, -1).detach().cpu()
                # attention_weights_state[nidx] = to_dense_adj(data.edge_index, edge_attr=model.alphas2).view(
                #     data.num_nodes, data.num_nodes, -1).cpu()

            for ridx, name in radar_index.items():
                results['gt'].append(y[ridx, context:])
                results['prediction'].append(y_hat[ridx, :])
                results['gt_km2'].append(y[ridx, context:] / areas[ridx])
                results['prediction_km2'].append(y_hat[ridx, :] / areas[ridx])
                results['night'].append(local_night[ridx, context:])
                results['radar'].append([name] * y_hat.shape[1])
                results['seqID'].append([nidx] * y_hat.shape[1])
                results['tidx'].append(_tidx)
                results['datetime'].append(time[_tidx])
                results['trial'].append([r] * y_hat.shape[1])
                results['horizon'].append(np.arange(y_hat.shape[1]))
                results['missing'].append(missing[ridx, context:])

                if cfg.model.name in ['GraphLSTM', 'BirdFluxGraphLSTM', 'BirdFluxGraphLSTM2']:
                    results['fluxes'].append(fluxes[ridx].view(-1))
                    results['local_deltas'].append(local_deltas[ridx].view(-1))
                if cfg.model.name in ['BirdFluxGraphLSTM', 'BirdFluxGraphLSTM2']:
                    results['influxes'].append(influxes[ridx].view(-1))
                    results['outfluxes'].append(outfluxes[ridx].view(-1))


        if cfg.model.name in ['BirdFluxGraphLSTM', 'BirdFluxGraphLSTM2', 'testFluxMLP']:
            with open(osp.join(output_dir, f'local_fluxes_{r}.pickle'), 'wb') as f:
                pickle.dump(local_fluxes, f, pickle.HIGHEST_PROTOCOL)
            with open(osp.join(output_dir, f'radar_fluxes_{r}.pickle'), 'wb') as f:
                pickle.dump(radar_fluxes, f, pickle.HIGHEST_PROTOCOL)
            with open(osp.join(output_dir, f'radar_mtr_{r}.pickle'), 'wb') as f:
                pickle.dump(radar_mtr, f, pickle.HIGHEST_PROTOCOL)
        if cfg.model.name == 'AttentionGraphLSTM':
            with open(osp.join(output_dir, f'attention_weights_{r}.pickle'), 'wb') as f:
                pickle.dump(attention_weights, f, pickle.HIGHEST_PROTOCOL)
            # with open(osp.join(output_dir, f'attention_weights_state_{r}.pickle'), 'wb') as f:
            #     pickle.dump(attention_weights_state, f, pickle.HIGHEST_PROTOCOL)

        del data, model

    with open(osp.join(output_dir, f'radar_index.pickle'), 'wb') as f:
        pickle.dump(radar_index, f, pickle.HIGHEST_PROTOCOL)

    # create dataframe containing all results
    for k, v in results.items():
        if torch.is_tensor(v[0]):
            results[k] = torch.cat(v).numpy()
        else:
            results[k] = np.concatenate(v)
    results['residual'] = results['gt'] - results['prediction']
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
