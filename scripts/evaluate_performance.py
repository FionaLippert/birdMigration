import numpy as np
import os
import os.path as osp
import pandas as pd


def load_cv_results(result_dir, ext='', trials=1):

    result_list = []
    for t in range(1, trials+1):
        file = osp.join(result_dir, f'trial_{t}', f'results{ext}.csv')
        if osp.isfile(file):
            df = pd.read_csv(file)
            df['trial'] = t
            result_list.append(df)

    results = pd.concat(result_list)

    return results


def compute_rmse(model, experiment, results, groupby='trial', threshold=0, km2=True):
    ext = '_km2' if km2 else ''

    results[f'squared_error{ext}'] = results[f'residual{ext}'].pow(2)
    df = results.query(f'missing == 0 & gt{ext} >= {threshold} & night == 1')
    rmse = df.groupby(groupby)[f'squared_error{ext}'].aggregate(np.mean).apply(np.sqrt)
    rmse = rmse.reset_index(name='rmse')
    rmse['model'] = model
    rmse['experiment'] = experiment

    return rmse

if __name__ == "__main__":

    #models = ['GAM', 'LocalMLP', 'LocalLSTM', 'FluxGraphLSTM']

    models = {'FluxGraphLSTM': ['final_evaluation_0', 'final_evaluation_1', 'final_evaluation_2']}
    # models = {
    #             'LocalLSTM': ['final_evaluation', 'final_evaluation_no_encoder'],
    #             'LocalMLP': ['final_evaluation'],
    #             'GAM': ['final_evaluation'],
    #             'GBT': ['final_evaluation']
    #          }

    models = {
                'GAM': ['final_evaluation'],
                'GBT': ['final_evaluation']
             }

    trials = 5
    year = 2017
    thresholds = [0, 20, 40]
    ext = '_fixedT0'
    ext = ''
    base_dir = '/home/fiona/birdMigration/results'
    base_dir = '/media/flipper/Seagate Basic/PhD/paper_1/results/radar'
    base_dir = '/media/flipper/Seagate Basic/PhD/paper_1/results/abm'
    for m, dirs in models.items():
        for d in dirs:
            result_dir = osp.join(base_dir, m, f'test_{year}', d)
            results = load_cv_results(result_dir, ext=ext, trials=trials)
            output_dir = osp.join(result_dir, 'performance_evaluation')
            os.makedirs(output_dir, exist_ok=True)

            for thr in thresholds:
                rmse_per_fold = compute_rmse(m, d, results, groupby='trial', threshold=thr, km2=True)
                rmse_per_fold.to_csv(osp.join(output_dir, f'rmse_thr{thr}{ext}.csv'))

                rmse_per_hour = compute_rmse(m, d, results, groupby=['horizon', 'trial'], threshold=thr, km2=True)
                rmse_per_hour.to_csv(osp.join(output_dir, f'rmse_per_hour_thr{thr}{ext}.csv'))
