from sklearn.ensemble import GradientBoostingRegressor
from graphNN import *
from torch_geometric.data import DataLoader


def prepare_data(root, year, season, timesteps, data_source, bird_scale):
    dataset = RadarData(root, year, season, timesteps, data_source=data_source, bird_scale=bird_scale)
    #dataloader = DataLoader(dataset, batch_size=1)
    X = []
    y = []
    for seq in dataset:
        for t in range(1, timesteps):
            features = np.concatenate([seq.coords.detach().numpy(),
                                       seq.areas.view(-1,1).detach().numpy(),
                                       seq.env[..., t].detach().numpy()], axis=1) # shape (nodes, features)
            X.append(features)
            y.append(seq.y[:, t-1])
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    return X, y

root = '/home/fiona/birdMigration/data'
bird_scale = 2000
X, y = prepare_data(root, '2016', 'fall', 6, 'abm', bird_scale)
reg = GradientBoostingRegressor(random_state=0)
reg.fit(X, y)

X, y = prepare_data(root, '2015', 'fall', 6, 'abm', bird_scale)
y_hat = reg.predict(X)

print(np.sqrt(np.square(y * bird_scale - y_hat * bird_scale)).mean())