from sklearn.ensemble import GradientBoostingRegressor
import numpy as np


def prepare_data(dataset, timesteps, return_mask=False):

    X = []
    y = []
    mask = []
    for seq in dataset:
        for t in range(timesteps+1):
            features = np.concatenate([seq.coords.detach().numpy(),
                                       seq.areas.view(-1,1).detach().numpy(),
                                       seq.env[..., t].detach().numpy()], axis=1) # shape (nodes, features)
            X.append(features)
            y.append(seq.y[:, t])
            mask.append(seq.local_night[:, t] & ~seq.missing[:, t])
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    mask = np.concatenate(mask, axis=0)
    if return_mask:
        return X, y, mask
    else:
        return X, y

def prepare_data_gam(dataset, timesteps, return_mask=False):

    X = []
    y = []
    mask = []
    for seq in dataset:
        for t in range(timesteps+1):
            features = np.concatenate([seq.day_of_year.detach().numpy(),
                                       seq.env[..., t].detach().numpy()], axis=1) # shape (nodes, features)
            X.append(features)
            y.append(seq.y[:, t])
            mask.append(seq.local_night[:, t] & ~seq.missing[:, t])
    X = np.stack(X, axis=0)
    y = np.stack(y, axis=0)
    mask = np.stack(mask, axis=0)
    if return_mask:
        return X, y, mask
    else:
        return X, y


def prepare_data_nights(dataset, timesteps):

    X = []
    y = []
    for seq in dataset:
        X_night = []
        y_night = []
        for t in range(timesteps+1):
            features = np.concatenate([seq.coords.detach().numpy(),
                                       seq.areas.view(-1,1).detach().numpy(),
                                       seq.env[..., t].detach().numpy()], axis=1) # shape (nodes, features)
            X_night.append(features)
            y_night.append(seq.y[:, t])
        X.append(np.stack(X_night, axis=0))  # shape (timesteps, nodes, features)
        y.append(np.stack(y_night, axis=0))  # shape (timesteps, nodes)

    X = np.concatenate(X, axis=1)  # shape (timesteps, samples, features)
    y = np.concatenate(y, axis=1)  # shape (timesteps, samples)
    return X, y

def prepare_data_nights_and_radars(dataset, timesteps, return_mask=False):

    X = []
    y = []
    mask = []
    for seq in dataset:
        X_night = []
        y_night = []
        mask_night = []
        for t in range(timesteps+1):
            features = np.concatenate([seq.coords.detach().numpy(),
                                       seq.areas.view(-1,1).detach().numpy(),
                                       seq.env[..., t].detach().numpy()], axis=1) # shape (nodes, features)
            X_night.append(features)
            y_night.append(seq.y[:, t])
            mask_night.append(seq.local_night[:, t])
        X.append(np.stack(X_night, axis=0)) # shape (timesteps, nodes, features)
        y.append(np.stack(y_night, axis=0)) # shape (timesteps, nodes)
        mask.append(np.stack(mask_night, axis=0))  # shape (timesteps, nodes)

    X = np.stack(X, axis=0) # shape (nights, timesteps, nodes, features)
    y = np.stack(y, axis=0) # shape (nights, timesteps, nodes)
    mask = np.stack(mask, axis=0)  # shape (nights, timesteps, nodes)
    if return_mask:
        return X, y, mask
    else:
        return X, y


def fit_GBT(X, y, n_estimators=100, lr=0.05, max_depth=5, seed=1234, tol=1e-6):

    gbt = GradientBoostingRegressor(random_state=seed, n_estimators=n_estimators, learning_rate=lr,
                                    max_depth=max_depth, tol=tol, n_iter_no_change=10)
    gbt.fit(X, y)
    return gbt

def predict_GBT(gbt, dataset, timesteps):
    X, y = prepare_data(dataset, timesteps)
    y_hat = gbt.predict(X)
    return y, y_hat
