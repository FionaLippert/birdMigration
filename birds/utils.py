import numpy as np
from matplotlib import pyplot as plt
import os.path as osp
import torch
import warnings

def val_test_split(dataloader, val_ratio, random_seed):
    rng = np.random.default_rng(random_seed)
    N = len(dataloader)
    n_val = int(N * val_ratio)
    #val_idx = rng.choice(np.arange(N, dtype=int), n_val, replace=False)
    val_idx = np.arange(n_val)
    val_loader = [list(dataloader)[i] for i in val_idx]
    test_loader = [list(dataloader)[i] for i in range(N) if i not in val_idx]

    return val_loader, test_loader

def MSE(output, gt, mask):
    # errors = (output - gt)**2
    # errors = errors[mask]
    # mse = errors.mean()

    #print(output.shape, gt.shape, torch.sum(mask))
    diff = torch.abs(output - gt)
    # print(torch.isnan(output).sum(), torch.isnan(gt).sum(), torch.isnan(diff).sum())
    diff2 = torch.square(diff)
    mse = torch.sum(diff2 * mask) / torch.sum(mask)
    return mse

# def MSE_weighted(output, gt, mask):
#     # errors = (output - gt)**2
#     # errors = errors[mask]
#     # mse = errors.mean()
#
#     #print(output.shape, gt.shape, torch.sum(mask))
#     diff = torch.abs(output - gt)
#     # print(torch.isnan(output).sum(), torch.isnan(gt).sum(), torch.isnan(diff).sum())
#     diff2 = torch.square(diff)
#     weight = torch.square(gt)
#     mse = torch.sum(diff2 * mask) / torch.sum(mask)
#     return mse

def MSE_root_transformed(output, gt, mask, root=3):
    errors = (torch.pow(output.relu(), 1/root) - torch.pow(gt, 1/root))**2
    errors = errors[mask]
    mse = errors.mean()
    return mse

def plot_training_curves(training_curves, val_curves, dir, log=True):
    epochs = training_curves.shape[1]
    fig, ax = plt.subplots()
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', message='Mean of empty slice')
        warnings.filterwarnings(action='ignore', message='Degrees of freedom <= 0 for slice.')
        train_line = ax.plot(range(1, epochs + 1), np.nanmean(training_curves, 0), label='training')
        ax.fill_between(range(1, epochs + 1), np.nanmean(training_curves, 0) - np.nanstd(training_curves, 0),
                        np.nanmean(training_curves, 0) + np.nanstd(training_curves, 0), alpha=0.2,
                        color=train_line[0].get_color())
        val_line = ax.plot(range(1, epochs + 1), np.nanmean(val_curves, 0), label='validation')
        ax.fill_between(range(1, epochs + 1), np.nanmean(val_curves, 0) - np.nanstd(val_curves, 0),
                        np.nanmean(val_curves, 0) + np.nanstd(val_curves, 0), alpha=0.2,
                        color=val_line[0].get_color())
    ax.set(xlabel='epoch', ylabel='MSE')
    if log: ax.set(yscale='log', xscale='log')
    plt.legend()
    fig.savefig(osp.join(dir, f'training_validation_curves_log={log}.png'), bbox_inches='tight')
    plt.close(fig)