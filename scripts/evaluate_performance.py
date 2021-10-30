import numpy as np
import scipy.stats as stats
import os
import os.path as osp
import pandas as pd
import geopandas as gpd
import itertools as it
from yaml import Loader, load


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

def compute_residual_corr(results, radar_df, km2=True):
    ext = '_km2' if km2 else ''

    # radars = results[model].radar.unique()
    #radars = radar_df.query('observed == 1').sort_values(by=['lat'], ascending=False).radar.values
    radars = [r for r in results.radar.unique() if not 'boundary' in r]
    
    corr = []
    radars1 = []
    radars2 = []
    for r1, r2 in it.product(radars, repeat=2):
        data1 = results.query(f'radar == "{r1}"')[f'residual{ext}'].to_numpy()
        data2 = results.query(f'radar == "{r2}"')[f'residual{ext}'].to_numpy()

        mask = np.logical_and(np.isfinite(data1), np.isfinite(data2))
        r, p = stats.pearsonr(data1[mask], data2[mask])
        radars1.append(r1)
        radars2.append(r2)
        corr.append(r)

    corr = pd.DataFrame(list(zip(radars1, radars2, corr)), columns=['radar1', 'radar2', 'corr'])

    return corr


if __name__ == "__main__":

    models = {  'FluxGraphLSTM': ['test_new_weight_func_split_delta']#, 'test_new_weight_func_no_dropout'],
                #'LocalLSTM': ['final_evaluation'],
                #'LocalMLP': ['final_evaluation_importance_sampling'],
                #'GAM': ['final_evaluation_new'],
                #'HA': ['final_evaluation_new'],
                #'GBT': ['final_evaluation_new', 'final_evaluation_new_importance_sampling',
                #        'final_evaluation_new_acc', 'final_evaluation_new_importance_sampling_acc']
             }

    trials = 5
    year = 2017
    season = 'fall'
    thresholds = [0.05, 0.1, 0.2] #[0, 20, 40]
    #ext = '_fixedT0'
    ext = ''
    #base_dir = '/home/fiona/birdMigration/results/abm'
    #base_dir = '/media/flipper/Seagate Basic/PhD/paper_1/results/radar'
    #base_dir = '/media/flipper/Seagate Basic/PhD/paper_1/results/abm'
    #base_dir = '/media/flipper/Seagate Basic/PhD/paper_1/results/abm'
    base_dir = '/home/flipper/birdMigration'
    datasource = 'abm'
    data_dir = osp.join(base_dir, 'data', 'preprocessed', '1H_none_ndummy=0', datasource, season, str(year))

    for m, dirs in models.items():
        print(f'evaluate {m}')

        radar_df = gpd.read_file(osp.join(data_dir, 'voronoi.shp'))
        for d in dirs:

            result_dir = osp.join(base_dir, 'results', datasource, m, f'test_{year}', d)
            results, cfg = load_cv_results(result_dir, ext=ext, trials=trials)
            if 'Flux' in m:
                _, cfg = load_cv_results(osp.join(base_dir, 'results', datasource, 'GBT', f'test_{year}', 'final_evaluation_new'), ext=ext, trials=1)
            output_dir = osp.join(result_dir, 'performance_evaluation')
            os.makedirs(output_dir, exist_ok=True)

            # compute spatial correlation of residuals
            #corr = compute_residual_corr(results, radar_df, km2=True)
            #corr.to_csv(osp.join(output_dir, f'spatial_corr{ext}.csv'))
            
            # compute mean squared error
            mse_per_fold = compute_mse(m, d, results, groupby='trial', threshold=0, km2=True)
            mse_per_fold.to_csv(osp.join(output_dir, f'mse{ext}.csv'))
            
            mse_per_hour = compute_mse(m, d, results, groupby=['horizon', 'trial'], threshold=0, km2=True)
            mse_per_hour.to_csv(osp.join(output_dir, f'mse_per_hour{ext}.csv'))
            
            mse_per_radar = compute_mse(m, d, results, groupby=['horizon', 'trial', 'radar'], threshold=0, km2=True)
            mse_per_radar.to_csv(osp.join(output_dir, f'mse_per_radar{ext}.csv'))


            # compute pearson correlation coefficient
            pcc_per_fold = compute_pcc(m, d, results, groupby='trial', threshold=0, km2=True)
            pcc_per_fold.to_csv(osp.join(output_dir, f'pcc{ext}.csv'))
            
            pcc_per_hour = compute_pcc(m, d, results, groupby=['horizon', 'trial'], threshold=0, km2=True)
            pcc_per_hour.to_csv(osp.join(output_dir, f'pcc_per_hour{ext}.csv'))

            pcc_per_radar = compute_pcc(m, d, results, groupby=['horizon', 'trial', 'radar'], threshold=0, km2=True)
            pcc_per_radar.to_csv(osp.join(output_dir, f'pcc_per_radar{ext}.csv'))

            # compute binary classification measures
            #thresholds = [cfg.datasource.bird_scale * f for f in [0.05, 0.1, 0.2]]
            for f in thresholds:
                thr = cfg['datasource']['bird_scale'] * f
                bin_per_fold = compute_bin(m, d, results, groupby='trial', threshold=thr, km2=True)
                bin_per_fold.to_csv(osp.join(output_dir, f'bin_thr{int(f*100)}{ext}.csv'))

                bin_per_hour = compute_bin(m, d, results, groupby=['horizon', 'trial'], threshold=thr, km2=True)
                bin_per_hour.to_csv(osp.join(output_dir, f'bin_per_hour_thr{int(f*100)}{ext}.csv'))

                bin_per_radar = compute_bin(m, d, results, groupby=['horizon', 'trial', 'radar'], threshold=thr, km2=True)
                bin_per_radar.to_csv(osp.join(output_dir, f'bin_per_radar_thr{int(f*100)}{ext}.csv'))

