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

    #models = ['GAM', 'LocalMLP', 'LocalLSTM', 'FluxGraphLSTM']

    models = {  'FluxGraphLSTM': ['test_new_weight_func'],# 'test_new_weight_func_split_delta', 'test_new_weight_func_no_dropout'],
                #'LocalLSTM': ['final_evaluation', 'final_evaluation_no_encoder'],
                #'LocalMLP': ['final_evaluation'],
                #'GAM': ['final_evaluation'],
                'GBT': ['final_evaluation', 'final_evaluation_importance_sampling']
             }

    trials = 1
    year = 2017
    thresholds = [0, 0.01] #[0, 20, 40]
    ext = '_fixedT0'
    ext = ''
    base_dir = '/home/fiona/birdMigration/results'
    base_dir = '/media/flipper/Seagate Basic/PhD/paper_1/results/radar'
    base_dir = '/media/flipper/Seagate Basic/PhD/paper_1/results/abm'
    #base_dir = '/media/flipper/Seagate Basic/PhD/paper_1/results/abm'
    #base_dir = '/home/flipper/birdMigration/results/radar'

    for m, dirs in models.items():
        print(f'evaluate {m}')
        for d in dirs:
            result_dir = osp.join(base_dir, m, f'test_{year}', d)
            results = load_cv_results(result_dir, ext=ext, trials=trials)
            output_dir = osp.join(result_dir, 'performance_evaluation')
            os.makedirs(output_dir, exist_ok=True)

            for thr in thresholds:
                mse_per_fold = compute_mse(m, d, results, groupby='trial', threshold=thr, km2=True)
                mse_per_fold.to_csv(osp.join(output_dir, f'mse_thr{thr}{ext}.csv'))

                mse_per_hour = compute_mse(m, d, results, groupby=['horizon', 'trial'], threshold=thr, km2=True)
                mse_per_hour.to_csv(osp.join(output_dir, f'mse_per_hour_thr{thr}{ext}.csv'))

                pcc_per_fold = compute_pcc(m, d, results, groupby='trial', threshold=thr, km2=True)
                pcc_per_fold.to_csv(osp.join(output_dir, f'pcc_thr{thr}{ext}.csv'))

                pcc_per_hour = compute_pcc(m, d, results, groupby=['horizon', 'trial'], threshold=thr, km2=True)
                pcc_per_hour.to_csv(osp.join(output_dir, f'pcc_per_hour_thr{thr}{ext}.csv'))

                if thr > 0:
                    bin_per_fold = compute_bin(m, d, results, groupby='trial', threshold=thr, km2=True)
                    bin_per_fold.to_csv(osp.join(output_dir, f'bin_thr{thr}{ext}.csv'))

                    bin_per_hour = compute_bin(m, d, results, groupby=['horizon', 'trial'], threshold=thr, km2=True)
                    bin_per_hour.to_csv(osp.join(output_dir, f'bin_per_hour_thr{thr}{ext}.csv'))
