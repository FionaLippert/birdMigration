import numpy as np

def val_test_split(dataloader, val_ratio, random_seed):
    rng = np.random.default_rng(random_seed)
    N = len(dataloader)
    print(N)
    n_val = int(N * val_ratio)
    #val_idx = rng.choice(np.arange(N, dtype=int), n_val, replace=False)
    val_idx = np.arange(n_val)
    print(val_idx)
    val_loader = [list(dataloader)[i] for i in val_idx]
    test_loader = [list(dataloader)[i] for i in range(N) if i not in val_idx]

    print(len(test_loader))

    return val_loader, test_loader

def MSE(output, gt, mask):
    errors = (output - gt)**2
    errors = errors[mask]
    mse = errors.mean()
    return mse