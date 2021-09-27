import numpy as np
import os
import os.path as osp
import pandas as pd


def load_cv_results(model, base_dir, years, ext=''):

    model_dir = osp.join(base_dir, f'nested_cv_{model}')
    result_list = []
    for fold, y in enumerate(years):
        file = osp.join(model_dir, f'test_{y}', 'final_evaluation', 'trial_1', f'results{ext}.csv')
        df = pd.read_csv(file)
        df['fold'] = fold
        result_list.append(df)

    results = pd.concat(result_list)

    print(f'successfully loaded results for {model}')

    return results


def compute_rmse(model, results, groupby='fold', threshold=0, km2=True):
    ext = '_km2' if km2 else ''

    results[f'squared_error{ext}'] = results[f'residual{ext}'].pow(2)
    df = results.query(f'missing == 0 & gt{ext} >= {threshold} & night == 1')
    rmse = df.groupby(groupby)[f'squared_error{ext}'].aggregate(np.mean).apply(np.sqrt)
    rmse = rmse.reset_index(name='rmse')
    rmse['model'] = model

    return rmse

if __name__ == "__main__":

    models = ['GAM', 'LocalMLP', 'LocalLSTM', 'FluxGraphLSTM']
    #models = ['FluxGraphLSTM']
    years = [2015, 2016, 2017]
    thresholds = [0, 20, 40]
    ext = '_fixedT0'
    #ext = ''
    base_dir = '/home/fiona/birdMigration/results'
    for m in models:
        results = load_cv_results(m, base_dir, years, ext=ext)
        output_dir = osp.join(base_dir, f'nested_cv_{m}', 'performance_evaluation')
        os.makedirs(output_dir, exist_ok=True)

        for thr in thresholds:
            rmse_per_fold = compute_rmse(m, results, groupby='fold', threshold=thr, km2=True)
            rmse_per_fold.to_csv(osp.join(output_dir, f'rmse_per_fold_thr{thr}{ext}.csv'))

        thr = 0
        rmse_per_hour = compute_rmse(m, results, groupby=['horizon', 'fold'], threshold=thr, km2=True)
        rmse_per_hour.to_csv(osp.join(output_dir, f'rmse_per_hour_thr{thr}{ext}.csv'))