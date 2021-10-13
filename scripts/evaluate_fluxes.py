import numpy as np
import os
import os.path as osp
import pandas as pd
import geopandas as gpd
import pickle5 as pickle
from yaml import Loader, load
import itertools as it


def load_cv_results(result_dir, ext='', trials=1):

    result_list = []
    for t in range(1, trials+1):
        file = osp.join(result_dir, f'trial_{t}', f'results{ext}.csv')
        if osp.isfile(file):
            df = pd.read_csv(file)
            df['trial'] = t
            result_list.append(df)

            cfg_file = osp.join(result_dir, f'trial_{t}', 'config.yaml')
            with open(cfg_file) as f:
                cfg = load(f, Loader=Loader)

    results = pd.concat(result_list)

    return results, cfg

def load_model_fluxes(result_dir, ext='', trials=1):

    fluxes = {}
    for t in range(1, trials + 1):
        file = osp.join(result_dir, f'trial_{t}', f'model_fluxes{ext}.pickle')

        with open(file, 'rb') as f:
            fluxes[t] = pickle.load(f)

    return fluxes


def compute_mse(model, experiment, results, groupby='trial', threshold=0, km2=True):
    ext = '_km2' if km2 else ''

    results[f'squared_error{ext}'] = results[f'residual{ext}'].pow(2)
    df = results.query(f'missing == 0 & gt{ext} >= {threshold}') # & night == 1')
    mse = df.groupby(groupby)[f'squared_error{ext}'].aggregate(np.mean) #.apply(np.sqrt)
    mse = mse.reset_index(name='mse')
    mse['model'] = model
    mse['experiment'] = experiment

    return mse


def compute_pcc(model, experiment, results, groupby='trial', threshold=0, km2=True):
    ext = '_km2' if km2 else ''

    df = results.query(f'missing == 0 & gt{ext} >= {threshold}').dropna()
    pcc = df.groupby(groupby)[[f'gt{ext}', f'prediction{ext}']].corr().iloc[0::2, -1]
    pcc = pcc.reset_index()
    pcc['pcc'] = pcc[f'prediction{ext}']
    pcc['model'] = model
    pcc['experiment'] = experiment

    return pcc

def compute_bin(model, experiment, results, groupby='trial', threshold=0, km2=True):
    ext = '_km2' if km2 else ''

    df = results.query(f'missing == 0').dropna()
    df['tp'] = (df[f'prediction{ext}'] > threshold) & (df[f'gt{ext}'] > threshold)
    df['fp'] = (df[f'prediction{ext}'] > threshold) & (df[f'gt{ext}'] < threshold)
    df['fn'] = (df[f'prediction{ext}'] < threshold) & (df[f'gt{ext}'] > threshold)
    df['tn'] = (df[f'prediction{ext}'] < threshold) & (df[f'gt{ext}'] < threshold)

    bin = df.groupby(groupby).aggregate(sum).reset_index()
    bin['accuracy'] = (bin.tp + bin.tn) / (bin.tp + bin.fp + bin.tn + bin.fn)
    bin['precision'] = bin.tp / (bin.tp + bin.fp)
    bin['sensitivity'] = bin.tp / (bin.tp + bin.fn)
    bin['specificity'] = bin.tn / (bin.tn + bin.fp)
    bin['fscore'] = 2 / ((1/bin.precision) + (1/bin.sensitivity))

    bin = bin.reset_index()
    bin['model'] = model
    bin['experiment'] = experiment

    return bin



if __name__ == "__main__":

    models = { 'FluxGraphLSTM': ['test_new_weight_func'] }

    trials = 5
    year = 2017
    season = 'fall'

    ext = ''
    datasource = 'abm'
    base_dir = '/home/fiona/birdMigration/results'
    base_dir = f'/media/flipper/Seagate Basic/PhD/paper_1/results/{datasource}'
    #base_dir = '/media/flipper/Seagate Basic/PhD/paper_1/results/abm'
    #base_dir = '/home/flipper/birdMigration/results/radar'

    result_dir = osp.join(base_dir, 'results', datasource)

    if datasource == 'abm':
        data_dir = osp.join(base_dir, 'data', 'raw', 'abm')
        dep = np.load(osp.join(data_dir, 'departing_birds.npy'))
        land = np.load(osp.join(data_dir, 'landing_birds.npy'))
        delta = dep - land

        with open(osp.join(base_dir, 'data', 'raw', 'abm', 'time.pkl', 'rb')) as f:
            abm_time = pickle.load(f)
        time_dict = {t: idx for idx, t in enumerate(abm_time)}

        voronoi = gpd.read_file(osp.join(base_dir, 'data', 'preprocessed', '1H_voronoi_ndummy=25',
                                         'abm', season, year, 'voronoi.shp'))
        radar_dict = voronoi.radar.to_dict()
        radar_dict = {v: k for k, v in radar_dict.items()}

        def get_abm_delta(datetime, radar, bird_scale=1):
            tidx = time_dict[pd.Timestamp(datetime)]
            ridx = radar_dict[radar]
            return delta[tidx, ridx] / bird_scale

        inner_radars = voronoi.query('boundary == 0').radar.values
        boundary_idx = voronoi.query('boundary == 1').index.values

        gt_fluxes = np.load(osp.join(data_dir, 'outfluxes.npy'))


    for m, dirs in models.items():
        print(f'evaluate model components for {m}')
        for d in dirs:
            result_dir = osp.join(base_dir, m, f'test_{year}', d)
            results, cfg = load_cv_results(result_dir, ext=ext, trials=trials)
            model_fluxes = load_model_fluxes(result_dir, ext=ext, trials=trials)
            bird_scale = cfg.datasource.bird_scale
            output_dir = osp.join(result_dir, 'performance_evaluation')
            os.makedirs(output_dir, exist_ok=True)

            results['abm_delta'] = results.apply(
                lambda row: get_abm_delta(row.datetime, row.radar, bird_scale), axis=1)

            # corr per radar
            gr = results[results.radar.isin(inner_radars)].dropna().groupby(['radar', 'trial'])
            corr = gr[['abm_delta', 'source/sink']].corr().iloc[0::2, -1].reset_index()
            corr.to_csv(osp.join(output_dir, f'delta_corr_per_radar{ext}.csv'))

            # corr per gt bin
            gr = results[results.radar.isin(inner_radars)].dropna().groupby(['seqID', 'radar', 'trial'])
            gr['activity'] = gr['gt'].transform(np.nanmean)
            df = gr.reset_index()
            df['activity_bin'] = pd.cut(df['activity'].values, bins=np.arange(0, df.activity.max()+200, 200))
            gr = df.groupby(['activity_bin', 'trial'])
            corr = gr[['abm_delta', 'source/sink']].corr().iloc[0::2, -1].reset_index()
            corr.to_csv(osp.join(output_dir, f'delta_corr_per_activity_bin{ext}.csv'))

            # corr per hour
            gr = results[results.radar.isin(inner_radars)].dropna().groupby(['horizon', 'trial'])
            corr = gr[['abm_delta', 'source/sink']].corr().iloc[0::2, -1].reset_index()
            corr.to_csv(osp.join(output_dir, f'delta_corr_per_hour{ext}.csv'))


            # rearange abm fluxes to match model fluxes
            gt_flux_dict = {}
            for group, df in results.groupby('seqID'):
                time = df.datetime.unique()
                gt_flux_dict[group] = np.stack([gt_fluxes[time_dict[pd.Timestamp(t)]] for t in time], axis=-1)

            gt_flux = np.stack([f[..., cfg.model.context: cfg.model.context + cfg.model.test_horizon] for
                        f in gt_flux_dict.values()], axis=-1)
            # exclude "self-fluxes"
            for i in range(gt_flux.shape[0]):
                np.fill_diagonal(gt_flux[i, i], np.nan)

            # exclude boundary to boundary fluxes
            for i, j in it.product(boundary_idx, repeat=2):
                gt_flux[i, j] = np.nan

            # aggregate fluxes per sequence
            gt_flux_per_seq = gt_flux.sum(2)
            #gt_flux[boundary_idx, boundary_idx,:] = np.ones(len(boundary_idx), len(boundary_idx)) * np.nan

            # net fluxes
            gt_net_flux = gt_flux - np.moveaxis(gt_flux, 0, 1)
            gt_net_flux_per_seq = gt_flux_per_seq - np.moveaxis(gt_flux_per_seq, 0, 1)

            overall_corr = {}
            corr_per_radar = {}
            corr_per_hour = {}
            for t, model_flux_t in model_fluxes.items():
                model_flux_t = np.stack([f.detach().numpy() for f in model_flux_t.values()], axis=-1)
                model_net_flux_t = model_flux_t - np.moveaxis(model_flux_t, 0, 1)
                model_flux_per_seq_t = model_flux_per_seq_t = model_flux_t.sum(2)
                model_net_flux_per_seq_t = model_flux_per_seq_t - np.moveaxis(model_flux_per_seq_t, 0, 1)

                mask = np.isfinite(gt_net_flux_per_seq)
                overall_corr[t] = np.corrcoef(gt_net_flux_per_seq[mask].flatten(),
                                              model_net_flux_per_seq_t[mask].flatten())

                corr_per_radar[t] = {}
                for r, ridx in radar_dict.items():
                    mask = np.isfinite(gt_net_flux_per_seq[ridx])
                    corr_per_radar[t][r] = np.corrcoef(gt_net_flux_per_seq[ridx][mask].flatten(),
                                                       model_net_flux_per_seq_t[ridx][mask].flatten())

                corr_per_hour[t] = {}
                for h in range(cfg.model.test_horizon):
                    mask = np.isfinite(gt_net_flux[:, :, h])
                    corr_per_hour[t][h] = np.corrcoef(gt_net_flux[:, :, h][mask].flatten(),
                                                       model_net_flux_t[:, :, h][mask].flatten())







