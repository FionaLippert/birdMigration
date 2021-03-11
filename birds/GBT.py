from sklearn.ensemble import GradientBoostingRegressor
from birds.graphNN import *
from torch_geometric.data import DataLoader


def prepare_data(split, root, year, season, timesteps, data_source, bird_scale):
    dataset = RadarData(root, split, year, season, timesteps, data_source=data_source, bird_scale=bird_scale)
    #dataloader = DataLoader(dataset, batch_size=1)
    X = []
    y = []
    for seq in dataset:
        for t in range(timesteps+1):
            features = np.concatenate([seq.coords.detach().numpy(),
                                       seq.areas.view(-1,1).detach().numpy(),
                                       seq.env[..., t].detach().numpy()], axis=1) # shape (nodes, features)
            X.append(features)
            y.append(seq.y[:, t])
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    return X, y

def prepare_data_nights(split, root, year, season, timesteps, data_source, bird_scale):
    dataset = RadarData(root, split, year, season, timesteps, data_source=data_source, bird_scale=bird_scale)
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

def prepare_data_nights_and_radars(split, root, year, season, timesteps, data_source, bird_scale):
    dataset = RadarData(root, split, year, season, timesteps, data_source=data_source, bird_scale=bird_scale)
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
        X.append(np.stack(X_night, axis=0)) # shape (timesteps, nodes, features)
        y.append(np.stack(y_night, axis=0)) # shape (timesteps, nodes)

    X = np.stack(X, axis=0) # shape (nights, timesteps, nodes, features)
    y = np.stack(y, axis=0) # shape (nights, timesteps, nodes)
    return X, y

def fit_GBT(root, years, season, timesteps, data_source, bird_scale, seed=1234):
    X = []
    y = []
    for year in years:
        X_year, y_year = prepare_data('train', root, year, season, timesteps, data_source, bird_scale)
        X.append(X_year)
        y.append(y_year)
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    reg = GradientBoostingRegressor(random_state=seed, n_estimators=100, learning_rate=0.05,
                                    max_depth=5, tol=0.00001, n_iter_no_change=200)
    reg.fit(X, y)
    return reg

def predict_GBT(gbt, root, year, season, timesteps, data_source, bird_scale):
    X, y = prepare_data('test', root, year, season, timesteps, data_source, bird_scale)
    y_hat = gbt.predict(X)
    return y, y_hat
