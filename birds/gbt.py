import numpy as np


def prepare_data(dataset, timesteps, mask_daytime=False, use_acc_vars=False):

    X = []
    y = []
    mask = []
    for seq in dataset:
        for t in range(timesteps):
            features = [seq.coords.detach().numpy(),
                                       seq.areas.view(-1,1).detach().numpy(),
                                       seq.env[..., t].detach().numpy()]
            if use_acc_vars:
                features.append(seq.acc[..., t].detach().numpy())
            features = np.concatenate(features, axis=1) # shape (nodes, features)
            X.append(features)
            y.append(seq.y[:, t])
            if mask_daytime:
                mask.append(seq.local_night[:, t] & ~seq.missing[:, t])
            else:
                mask.append(~seq.missing[:, t])
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    mask = np.concatenate(mask, axis=0)

    return X, y, mask


def prepare_data_gam(dataset, timesteps, mask_daytime=False):

    X = []
    y = []
    mask = []
    for seq in dataset:
        for t in range(timesteps):
            env = seq.env[:, -2:, t].detach().numpy()  # shape (nodes, features) where features are solarpos and solarpos_dt
            doy = np.ones((env.shape[0], 1)) * seq.day_of_year[t].detach().numpy()
            features = np.concatenate([env, doy], axis=-1)

            X.append(features)
            y.append(seq.y[:, t])
            if mask_daytime:
                mask.append(seq.local_night[:, t] & ~seq.missing[:, t])
            else:
                mask.append(~seq.missing[:, t])
    X = np.stack(X, axis=0)
    y = np.stack(y, axis=0)
    mask = np.stack(mask, axis=0)

    return X, y, mask


def prepare_data_nights_and_radars(dataset, context, horizon, mask_daytime=False, use_acc_vars=False):

    X = []
    y = []
    mask = []
    for seq in dataset:
        X_night = []
        y_night = []
        mask_night = []
        for t in range(context, context+horizon):

            features = [seq.coords.detach().numpy(),
                 seq.areas.view(-1, 1).detach().numpy(),
                 seq.env[..., t].detach().numpy()]
            if use_acc_vars:
                features.append(seq.acc[..., t].detach().numpy())
            features = np.concatenate(features, axis=1) # shape (nodes, features)
            X_night.append(features)
            y_night.append(seq.y[:, t])
            if mask_daytime:
                mask_night.append(seq.local_night[:, t] & ~seq.missing[:, t])
            else:
                mask_night.append(~seq.missing[:, t])
        X.append(np.stack(X_night, axis=0)) # shape (timesteps, nodes, features)
        y.append(np.stack(y_night, axis=0)) # shape (timesteps, nodes)
        mask.append(np.stack(mask_night, axis=0))  # shape (timesteps, nodes)

    X = np.stack(X, axis=0) # shape (nights, timesteps, nodes, features)
    y = np.stack(y, axis=0) # shape (nights, timesteps, nodes)
    mask = np.stack(mask, axis=0)  # shape (nights, timesteps, nodes)

    return X, y, mask


def prepare_data_nights_and_radars_gam(dataset, context, horizon, mask_daytime=False):

    X = []
    y = []
    mask = []
    for seq in dataset:
        X_night = []
        y_night = []
        mask_night = []
        for t in range(context, context+horizon):
            env = seq.env[:, -2:, t].detach().numpy()  # shape (nodes, features)
            doy = np.ones((env.shape[0], 1)) * seq.day_of_year[t].detach().numpy()
            features = np.concatenate([env, doy], axis=-1)

            X_night.append(features)
            y_night.append(seq.y[:, t])
            if mask_daytime:
                mask_night.append(seq.local_night[:, t] & ~seq.missing[:, t])
            else:
                mask_night.append(~seq.missing[:, t])
        X.append(np.stack(X_night, axis=0)) # shape (timesteps, nodes, features)
        y.append(np.stack(y_night, axis=0)) # shape (timesteps, nodes)
        mask.append(np.stack(mask_night, axis=0))  # shape (timesteps, nodes)

    X = np.stack(X, axis=0) # shape (nights, timesteps, nodes, features)
    y = np.stack(y, axis=0) # shape (nights, timesteps, nodes)
    mask = np.stack(mask, axis=0)  # shape (nights, timesteps, nodes)

    return X, y, mask
