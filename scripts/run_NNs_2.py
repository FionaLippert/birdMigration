from birds import dataloader, utils
from birds.graphNN import *
import torch
from torch.utils.data import random_split, Subset
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
                 'LocalLSTM': LocalLSTM,
                 'BirdFluxGraphLSTM': FluxGraphLSTM,
                 'AttentionGraphLSTM': AttentionGraphLSTM}


def run_training(cfg: DictConfig, output_dir: str, log):
    assert cfg.model.name in MODEL_MAPPING

    if cfg.debugging: torch.autograd.set_detect_anomaly(True)

    Model = MODEL_MAPPING[cfg.model.name]

    device = 'cuda' if (cfg.device.cuda and torch.cuda.is_available()) else 'cpu'
    seed = cfg.seed + cfg.get('job_id', 0)

    data = setup_training(cfg, output_dir)
    n_data = len(data)

    # split data into training and validation set
    n_val = max(1, int(cfg.datasource.val_train_split * n_data))
    n_train = n_data - n_val

    if cfg.verbose:
        print('------------------------------------------------------')
        print('-------------------- data sets -----------------------')
        print(f'total number of sequences = {n_data}')
        print(f'number of training sequences = {n_train}')
        print(f'number of validation sequences = {n_val}')

    train_data, val_data = random_split(data, (n_train, n_val), generator=torch.Generator().manual_seed(cfg.seed))
    train_loader = DataLoader(train_data, batch_size=cfg.model.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=True)

    if cfg.model.edge_type == 'voronoi':
        n_edge_attr = 4
    else:
        n_edge_attr = 3

    if cfg.model.get('root_transformed_loss', False):
        loss_func = utils.MSE_root_transformed
    elif cfg.model.get('weighted_loss', False):
        loss_func = utils.MSE_weighted
    else:
        loss_func = utils.MSE

    if cfg.verbose:
        print('------------------ model settings --------------------')
        print(cfg.model)
        print('------------------------------------------------------')

    best_val_loss = np.inf
    training_curve = np.ones((1, cfg.model.epochs)) * np.nan
    val_curve = np.ones((1, cfg.model.epochs)) * np.nan

    if cfg.verbose: print(f'environmental variables: {cfg.datasource.env_vars}')

    model = Model(n_env=len(cfg.datasource.env_vars), coord_dim=2, n_edge_attr=n_edge_attr,
                  seed=seed, **cfg.model)

    states_path = cfg.model.get('load_states_from', '')
    if osp.isfile(states_path):
        model.load_state_dict(torch.load(states_path))

    model = model.to(device)
    params = model.parameters()
    optimizer = torch.optim.Adam(params, lr=cfg.model.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.model.lr_decay, gamma=cfg.model.get('lr_gamma', 0.1))

    for p in model.parameters():
        p.register_hook(lambda grad: torch.clamp(grad, -1.0, 1.0))


    tf = 1.0 # initialize teacher forcing (is ignored for LocalMLP)
    all_tf = np.zeros(cfg.model.epochs)
    all_lr = np.zeros(cfg.model.epochs)
    avg_loss = np.inf

    for epoch in range(cfg.model.epochs):
        all_tf[epoch] = tf
        all_lr[epoch] = optimizer.param_groups[0]["lr"]

        #for name, param in model.named_parameters():
        #    if param.requires_grad:
        #        print(name, param.data, param.grad)

        loss = train(model, train_loader, optimizer, loss_func, device, teacher_forcing=tf, **cfg.model)
        training_curve[0, epoch] = loss / n_train

        val_loss = test(model, val_loader, loss_func, device, **cfg.model).cpu()
        val_loss = val_loss[torch.isfinite(val_loss)].mean()
        val_curve[0, epoch] = val_loss

        if cfg.verbose:
            print(f'epoch {epoch + 1}: loss = {training_curve[0, epoch]}')
            print(f'epoch {epoch + 1}: val loss = {val_loss}')

        if val_loss <= best_val_loss:
            if cfg.verbose: print('best model so far; save to disk ...')
            torch.save(model.state_dict(), osp.join(output_dir, f'best_model.pkl'))
            best_val_loss = val_loss

        if cfg.early_stopping and (epoch + 1) % cfg.stopping_period == 0:
            # every 5 epochs, check for convergence of validation loss
            l = val_curve[0, (epoch - (cfg.stopping_period - 1)) : (epoch + 1)].mean()
            if (avg_loss - l) > cfg.model.stopping_criterion:
                # loss decayed significantly, continue training
                avg_loss = l
            else:
                # loss converged sufficiently, stop training
                break

        tf = tf * cfg.model.get('teacher_forcing_gamma', 0)
        scheduler.step()

    torch.save(model.state_dict(), osp.join(output_dir, 'final_model.pkl'))

    print(f'validation loss = {best_val_loss}', file=log)
    log.flush()

    # save training and validation curves
    np.save(osp.join(output_dir, 'training_curves.npy'), training_curve)
    np.save(osp.join(output_dir, 'validation_curves.npy'), val_curve)
    np.save(osp.join(output_dir, 'learning_rates.npy'), all_lr)
    np.save(osp.join(output_dir, 'teacher_forcing.npy'), all_tf)

    # plotting
    utils.plot_training_curves(training_curve, val_curve, output_dir, log=True)
    utils.plot_training_curves(training_curve, val_curve, output_dir, log=False)

    with open(osp.join(output_dir, f'config.yaml'), 'w') as f:
        OmegaConf.save(config=cfg, f=f)

    log.flush()

def run_cross_validation(cfg: DictConfig, output_dir: str, log):
    assert cfg.model.name in MODEL_MAPPING
    assert cfg.action.name == 'cv'

    if cfg.debugging: torch.autograd.set_detect_anomaly(True)

    Model = MODEL_MAPPING[cfg.model.name]

    device = 'cuda' if (cfg.device.cuda and torch.cuda.is_available()) else 'cpu'
    epochs = cfg.model.epochs
    n_folds = cfg.action.n_folds
    seed = cfg.seed + cfg.get('job_id', 0)

    data = setup_training(cfg, output_dir)
    n_data = len(data)

    if cfg.model.edge_type == 'voronoi':
        n_edge_attr = 4
    else:
        n_edge_attr = 3

    if cfg.model.get('root_transformed_loss', False):
        loss_func = utils.MSE_root_transformed
    elif cfg.model.get('weighted_loss', False):
        loss_func = utils.MSE_weighted
    else:
        loss_func = utils.MSE

    if cfg.verbose:
        print('------------------ model settings --------------------')
        print(cfg.model)
        print(f'environmental variables: {cfg.datasource.env_vars}')

    cv_folds = np.array_split(np.arange(n_data), n_folds)

    if cfg.verbose: print(f'--- run cross-validation with {n_folds} folds ---')

    training_curves = np.ones((n_folds, epochs)) * np.nan
    val_curves = np.ones((n_folds, epochs)) * np.nan
    best_val_losses = np.ones(n_folds) * np.nan
    best_epochs = np.zeros(n_folds)

    for f in range(n_folds):
        if cfg.verbose: print(f'------------------- fold = {f} ----------------------')

        subdir = osp.join(output_dir, f'cv_fold_{f}')
        os.makedirs(subdir, exist_ok=True)

        # split into training and validation set
        val_data = Subset(data, cv_folds[f].tolist())
        train_idx = np.concatenate([cv_folds[i] for i in range(n_folds) if i!=f]).tolist()
        n_train = len(train_idx)
        train_data = Subset(data, train_idx) # everything else
        train_loader = DataLoader(train_data, batch_size=cfg.model.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=1, shuffle=True)

        model = Model(n_env=len(cfg.datasource.env_vars), coord_dim=2, n_edge_attr=n_edge_attr,
                      seed=seed, **cfg.model)

        states_path = cfg.model.get('load_states_from', '')
        if osp.isfile(states_path):
            model.load_state_dict(torch.load(states_path))

        model = model.to(device)
        params = model.parameters()
        optimizer = torch.optim.Adam(params, lr=cfg.model.lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.model.lr_decay, gamma=cfg.model.get('lr_gamma', 0.1))
        best_val_loss = np.inf
        avg_loss = np.inf

        for p in model.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -1.0, 1.0))

        tf = 1.0 # initialize teacher forcing (is ignored for LocalMLP)
        all_tf = np.zeros(epochs)
        all_lr = np.zeros(epochs)
        for epoch in range(epochs):
            all_tf[epoch] = tf
            all_lr[epoch] = optimizer.param_groups[0]["lr"]

            loss = train(model, train_loader, optimizer, loss_func, device, teacher_forcing=tf, **cfg.model)
            training_curves[f, epoch] = loss / n_train

            val_loss = test(model, val_loader, loss_func, device, **cfg.model).cpu()
            val_loss = val_loss[torch.isfinite(val_loss)].mean()
            val_curves[f, epoch] = val_loss

            if cfg.verbose:
                print(f'epoch {epoch + 1}: loss = {training_curves[f, epoch]}')
                print(f'epoch {epoch + 1}: val loss = {val_loss}')

            if val_loss <= best_val_loss:
                if cfg.verbose: print('best model so far; save to disk ...')
                torch.save(model.state_dict(), osp.join(subdir, f'best_model.pkl'))
                best_val_loss = val_loss
                best_epochs[f] = epoch

            if cfg.early_stopping and (epoch + 1) % cfg.stopping_period == 0:
                # every X epochs, check for convergence of validation loss
                l = val_curves[f, (epoch - (cfg.stopping_period - 1)): (epoch + 1)].mean()
                if (avg_loss - l) > cfg.model.stopping_criterion:
                    # loss decayed significantly, continue training
                    avg_loss = l
                else:
                    # loss converged sufficiently, stop training
                    val_curves[f, epoch:] = l
                    break

            tf = tf * cfg.model.get('teacher_forcing_gamma', 0)
            scheduler.step()

        torch.save(model.state_dict(), osp.join(subdir, 'final_model.pkl'))

        print(f'fold {f}: final validation loss = {val_curves[f, -1]}', file=log)
        best_val_losses[f] = best_val_loss

        log.flush()

        # update training and validation curves
        np.save(osp.join(subdir, 'training_curves.npy'), training_curves)
        np.save(osp.join(subdir, 'validation_curves.npy'), val_curves)
        np.save(osp.join(subdir, 'learning_rates.npy'), all_lr)
        np.save(osp.join(subdir, 'teacher_forcing.npy'), all_tf)

        # plotting
        utils.plot_training_curves(training_curves, val_curves, subdir, log=True)
        utils.plot_training_curves(training_curves, val_curves, subdir, log=False)

    print(f'average validation loss = {val_curves[:, -1].mean()}', file=log)

    summary = pd.DataFrame({'fold': range(n_folds),
                            'final_val_loss': val_curves[:, -1],
                            'best_val_loss': best_val_losses,
                            'best_epoch': best_epochs})
    summary.to_csv(osp.join(output_dir, 'summary.csv'))


    with open(osp.join(output_dir, f'config.yaml'), 'w') as f:
        OmegaConf.save(config=cfg, f=f)

    log.flush()

def setup_training(cfg: DictConfig, output_dir: str):

    seq_len = cfg.model.get('context', 0) + cfg.model.horizon
    seed = cfg.seed + cfg.get('job_id', 0)

    preprocessed_dirname = f'{cfg.model.edge_type}_dummy_radars={cfg.model.n_dummy_radars}_exclude={cfg.exclude}'
    processed_dirname = f'buffers={cfg.datasource.use_buffers}_root_transform={cfg.root_transform}_use_nights={cfg.use_nights}_' \
                        f'edges={cfg.model.edge_type}_birds_km2={cfg.model.birds_per_km2}_' \
                        f'dummy_radars={cfg.model.n_dummy_radars}_t_unit={cfg.t_unit}_exclude={cfg.exclude}'

    # initialize normalizer
    training_years = set(cfg.datasource.years) - set([cfg.datasource.test_year])
    normalization = dataloader.Normalization(training_years, cfg.datasource.name,
                                             cfg.device.data_dir, preprocessed_dirname, **cfg)
    # load training and validation data
    data = [dataloader.RadarData(year, seq_len, preprocessed_dirname, processed_dirname,
                                 **cfg, **cfg.model,
                                 data_root=cfg.device.data_dir,
                                 data_source=cfg.datasource.name,
                                 normalization=normalization,
                                 env_vars=cfg.datasource.env_vars,
                                 )
            for year in training_years]

    data = torch.utils.data.ConcatDataset(data)

    if cfg.model.edge_type == 'voronoi':
        if cfg.datasource.use_buffers:
            input_col = 'birds_from_buffer'
        else:
            input_col = 'birds'
    else:
        input_col = 'birds_km2'

    cfg.datasource.bird_scale = float(normalization.max(input_col))
    cfg.model_seed = seed
    with open(osp.join(output_dir, 'config.yaml'), 'w') as f:
        OmegaConf.save(config=cfg, f=f)
    with open(osp.join(output_dir, 'normalization.pkl'), 'wb') as f:
        pickle.dump(normalization, f)

    if cfg.root_transform == 0:
        cfg.datasource.bird_scale = float(normalization.max(input_col))
    else:
        cfg.datasource.bird_scale = float(normalization.root_max(input_col, cfg.root_transform))

    return data


def run_testing(cfg: DictConfig, output_dir: str, log, ext=''):
    assert cfg.model.name in MODEL_MAPPING

    Model = MODEL_MAPPING[cfg.model.name]

    preprocessed_dirname = f'{cfg.model.edge_type}_dummy_radars={cfg.model.n_dummy_radars}_exclude={cfg.exclude}'
    processed_dirname = f'buffers={cfg.datasource.use_buffers}_root_transform={cfg.root_transform}_use_nights={cfg.use_nights}_' \
                        f'edges={cfg.model.edge_type}_birds_km2={cfg.model.birds_per_km2}_' \
                        f'dummy_radars={cfg.model.n_dummy_radars}_t_unit={cfg.t_unit}_exclude={cfg.exclude}'

    model_dir = cfg.get('model_dir', output_dir)

    device = 'cuda' if (cfg.device.cuda and torch.cuda.is_available()) else 'cpu'

    context = cfg.model.get('context', 0)
    seq_len = context + cfg.model.test_horizon
    seq_shift = context // 24

    if cfg.model.edge_type == 'voronoi':
        n_edge_attr = 4
        if cfg.datasource.use_buffers:
            input_col = 'birds_from_buffer'
        else:
            input_col = 'birds'
    else:
        n_edge_attr = 3
        input_col = 'birds_km2'

    # load training config
    yaml = ruamel.yaml.YAML()
    fp = osp.join(model_dir, 'config.yaml')
    with open(fp, 'r') as f:
        model_cfg = yaml.load(f)

    # load normalizer
    with open(osp.join(model_dir, 'normalization.pkl'), 'rb') as f:
        normalization = pickle.load(f)
    if cfg.root_transform == 0:
        cfg.datasource.bird_scale = float(normalization.max(input_col))
    else:
        cfg.datasource.bird_scale = float(normalization.root_max(input_col, cfg.root_transform))

    # load test data
    test_data = dataloader.RadarData(str(cfg.datasource.test_year), seq_len,
                                    preprocessed_dirname, processed_dirname,
                                    **cfg, **cfg.model,
                                    data_root=cfg.device.data_dir,
                                    data_source=cfg.datasource.name,
                                    normalization=normalization,
                                    env_vars=cfg.datasource.env_vars,
                                    )
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    # load additional data
    time = test_data.info['timepoints']
    radars = test_data.info['radars']
    areas = np.ones(len(radars)) if input_col == 'birds_km2' else test_data.info['areas']
    radar_index = {idx: name for idx, name in enumerate(radars)}

    # load models and predict
    results = dict(gt=[], gt_km2=[], prediction=[], prediction_km2=[], night=[], radar=[], seqID=[],
                   tidx=[], datetime=[], horizon=[], missing=[], trial=[])
    if 'Flux' in cfg.model.name:
        results['flux'] = []
        results['source/sink'] = []
        results['influx'] = []
        results['outflux'] = []


    model = Model(n_env=len(cfg.datasource.env_vars), coord_dim=2, n_edge_attr=n_edge_attr,
                  seed=model_cfg['seed'], **model_cfg['model'])
    model.load_state_dict(torch.load(osp.join(model_dir, f'final_model.pkl')))

    # adjust model settings for testing
    model.horizon = cfg.model.test_horizon
    if cfg.model.get('fixed_boundary', 0):
        model.fixed_boundary = True

    model.to(device)
    model.eval()

    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.data)

    edge_fluxes = {}
    radar_fluxes = {}
    radar_mtr = {}
    attention_weights = {}

    enc_att = hasattr(model, 'node_lstm') and cfg.model.get('use_encoder', False)

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

        if enc_att:
            attention_weights[nidx] = torch.stack(model.node_lstm.alphas, dim=-1).detach().cpu()

        if 'Flux' in cfg.model.name:
            # fluxes along edges
            edge_fluxes[nidx] = to_dense_adj(data.edge_index, edge_attr=model.edge_fluxes).view(
                                data.num_nodes, data.num_nodes, -1).detach().cpu()
            radar_fluxes[nidx] = to_dense_adj(data.edge_index, edge_attr=data.fluxes).view(
                data.num_nodes, data.num_nodes, -1).detach().cpu()
            radar_mtr[nidx] = to_dense_adj(data.edge_index, edge_attr=data.mtr).view(
                data.num_nodes, data.num_nodes, -1).detach().cpu()
            # net fluxes per node
            fluxes = model.node_fluxes.detach().cpu()
            influxes = edge_fluxes[nidx].sum(1)
            outfluxes = edge_fluxes[nidx].permute(1, 0, 2).sum(1)
            node_deltas = model.node_deltas.detach().cpu()

        elif cfg.model.name == 'AttentionGraphLSTM':
            attention_weights[nidx] = to_dense_adj(data.edge_index, edge_attr=model.alphas_s).view(
                                data.num_nodes, data.num_nodes, -1).detach().cpu()

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
            results['trial'].append([cfg.get('job_id', 0)] * y_hat.shape[1])
            results['horizon'].append(np.arange(y_hat.shape[1]))
            results['missing'].append(missing[ridx, context:])

            if 'Flux' in cfg.model.name:
                results['flux'].append(fluxes[ridx].view(-1))
                results['source/sink'].append(node_deltas[ridx].view(-1))
                results['influx'].append(influxes[ridx].view(-1))
                results['outflux'].append(outfluxes[ridx].view(-1))

    if 'Flux' in cfg.model.name:
        with open(osp.join(output_dir, f'model_fluxes{ext}.pickle'), 'wb') as f:
            pickle.dump(edge_fluxes, f, pickle.HIGHEST_PROTOCOL)
        with open(osp.join(output_dir, f'radar_fluxes{ext}.pickle'), 'wb') as f:
            pickle.dump(radar_fluxes, f, pickle.HIGHEST_PROTOCOL)
        with open(osp.join(output_dir, f'radar_mtr{ext}.pickle'), 'wb') as f:
            pickle.dump(radar_mtr, f, pickle.HIGHEST_PROTOCOL)
    if enc_att or cfg.model.name == 'AttentionGraphLSTM':
        with open(osp.join(output_dir, f'attention_weights{ext}.pickle'), 'wb') as f:
            pickle.dump(attention_weights, f, pickle.HIGHEST_PROTOCOL)


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
    df.to_csv(osp.join(output_dir, f'results{ext}.csv'))

    print(f'successfully saved results to {osp.join(output_dir, f"results{ext}.csv")}', file=log)
    log.flush()


def run(cfg: DictConfig, output_dir: str, log):
    if 'cv' in cfg.action.name:
        run_cross_validation(cfg, output_dir, log)
    if 'train' in cfg.action.name:
        run_training(cfg, output_dir, log)
    if 'test' in cfg.action.name:
        run_testing(cfg, output_dir, log)
