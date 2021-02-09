from scipy.optimize import minimize, Bounds
#import numpy as np
import networkx as nx
import pandas as pd
import itertools as it
import autograd.numpy as np  # Thinly-wrapped numpy
from autograd import grad
from autograd.misc.flatten import flatten
#from torch.utils import data
import os
import datahandling
import spatial
import era5interface
from matplotlib import pyplot as plt

class DataLoader():
    def __init__(self, split, season, timesteps,
                 environment_vars = [],
                 vpi_path='/home/fiona/radar_data/vpi/night_only',
                 wind_path=None):

        #super(VP, self).__init__()

        self.split = split
        if split == 'test':
            year = '2015'
        else:
            year = '2016'

        if season == 'spring':
            start_date = '03-15 18:00:00'
            end_date = '05-15 10:00:00'
        elif season == 'fall':
            start_date = '08-15 18:00:00'
            end_date = '09-15 10:00:00'

        self.timesteps = timesteps

        data_dir = os.path.join(vpi_path, f'{year}0801T0000_to_{year}1130T2359')
        vid, self.radars, self.t_range = datahandling.load_data(data_dir, 'vid',
                                                           f'{year}-{start_date}', f'{year}-{end_date}',
                                                           t_unit='1H', mask_days=True)

        data = np.stack([ds.data.flatten() for ds in vid.values()])
        check = np.isfinite(data).all(axis=0)
        self.dft = pd.DataFrame({'check': np.append(np.logical_and(check[:-1], check[1:]), False),
                            'tidx': range(len(self.t_range))}, index=self.t_range)


        if self.timesteps > 1:
            # group into nights
            groups = [list(g) for k, g in it.groupby(enumerate(self.dft.check), key=lambda x: x[-1])]
            self.nights = [[item[0] for item in g] for g in groups if g[0][1] and len(g) > self.timesteps]

        self.data = self.reshape_data(data)

        self.max = np.nanmax(data)
        self.data /= self.max

        if len(environment_vars) > 0:
            self.environment = {}
            if 'wind_u' in environment_vars and 'wind_v' in environment_vars:
                # load wind data
                wind = era5interface.extract_points(datahandling.load_radars(data_dir).values(),
                                                    os.path.join(wind_path, year, season, 'wind_850.nc'))
                self.environment['wind_u'] = np.stack([ds.u.sel(time=self.t_range).data.flatten() for ds in wind.values()])
                self.environment['wind_v'] = np.stack([ds.v.sel(time=self.t_range).data.flatten() for ds in wind.values()])

            if 'solarpos' in environment_vars:
                # load sun data
                solarpos, _, _ = datahandling.load_data(data_dir, 'solarpos',
                                                                       f'{year}-{start_date}', f'{year}-{end_date}',
                                                                       t_unit='1H', mask_days=True)
                self.environment['solarpos'] = np.stack([ds.data.flatten() for ds in solarpos.values()])

            if 'x' in environment_vars and 'y' in environment_vars:
                # load x and y locations of radars (in local crs for easier computation of distances)
                space = spatial.Spatial(self.radars)
                x, y = zip(*space.pts2coords(space.pts_local))
                self.environment['x'] = np.stack([x for _ in self.t_range], axis=1)  # shape (nodes, time)
                self.environment['y'] = np.stack([y for _ in self.t_range], axis=1)  # shape (nodes, time)

            # normalize environment data
            self.environment = { var : self.normalize_data(values) for var, values in self.environment.items() }

            # reshape environment data
            self.environment = {var: self.reshape_data(values) for var, values in self.environment.items()}


    def reshape_data(self, data):
        if hasattr(self, 'nights'):
            return np.stack([data[:, night[1:self.timesteps+1]] for night in self.nights], axis=-1)
        else:
            return data[:,self.dft.check]

    def normalize_data(self, data):
        max = np.nanmax(data)
        return data / max


    def __len__(self):
        return self.data.shape[-1]

    def __getitem__(self, idx):
        return self.data[::, idx]


class ParamWrapper:
    def __init__(self, adj, num_targets):
        self.idx = np.where(adj)
        self.adj = adj
        self.shape = adj.shape
        self.num_params = self.idx[0].size
        self.num_targets = num_targets
        self.num_nodes = adj.shape[0]

    def zero_embedding(self, w, i, idx):
        w_i = []
        for j in range(self.num_nodes):
            if self.adj[i, j] >= 1:
                w_i.append(w[idx])
                idx += 1
            else:
                w_i.append(0)
        return w_i, idx

    def wrap(self, w, concat=False):
        idx = 0
        W_targets = []
        for i in range(self.num_targets):
            w_i, idx = self.zero_embedding(w, i, idx)
            W_targets.append(w_i)
        W_targets = np.vstack(W_targets)  # dimensions: (targets, sources)

        W_boundary = []
        for i in range(self.num_targets, self.num_nodes):
            w_i, idx = self.zero_embedding(w, i, idx)
            W_boundary.append(w_i)
        W_boundary = np.vstack(W_boundary)

        if concat:
            return np.concatenate([W_targets, W_boundary])
        else:
            return W_targets, W_boundary

    def unwrap(self, W_targets, W_boundary):
        W = np.concatenate([W_targets, W_boundary])
        return W[self.idx]



class Optimizer:
    def __init__(self, G, data, environment=None):
        self.G = G
        self.N = len(G)

        self.targets = [n for n, data in G.nodes(data=True) if not data.get('boundary')]
        self.boundary = [n for n, data in G.nodes(data=True) if data.get('boundary')]

        self.adj = nx.to_numpy_matrix(G, nodelist=np.concatenate([self.targets, self.boundary]))
        self.wrapper = ParamWrapper(self.adj, len(self.targets))

        assert(data.ndim in [2, 3])

        self.data = np.concatenate([data[self.targets], data[self.boundary]])

        if data.ndim == 2:
            # assume data has shape (nodes, time)
            #self.X = np.concatenate([data[self.targets, :-1], data[self.boundary, :-1]])
            self.X = self.data[:, :-1]
            self.Y = data[self.targets, 1:]
            self.timesteps = 1
            self.T = self.Y.shape[1]


        elif data.ndim == 3:
            # assume data has shape (nodes, time, night)
            #self.X = np.concatenate([data[self.targets, 0], data[self.boundary, 0]])
            self.X = self.data[:, 0]
            self.Y = data[self.targets, 1:, :]
            self.N_nights = data.shape[2]
            self.timesteps = data.shape[1] - 1 # first timestep is needed as initial state

        if environment is not None:
            self.environment = {f : np.concatenate([d[self.targets], d[self.boundary]]) for f, d in environment.items()}
            self.edge_features = self.construct_edge_features()  # has shape (edges, features, time) or (edges, features, time, night)


    def construct_edge_features(self):

        source_features = ['wind_u', 'wind_v', 'solarpos', 'x', 'y']
        target_features = ['x', 'y']

        all_features = []
        for i in range(self.N):
            for j in range(self.N):
                # i is target, j is source
                if self.adj[i, j] >= 1:
                    # edge from j to i should have features
                    # - wind u and v at j
                    # - coords of i and j
                    # ...
                    target_data = np.stack([self.environment[f][i] for f in target_features])
                    source_data = np.stack([self.environment[f][j] for f in source_features])
                    offset = np.ones((1, *target_data.shape[1:]))
                    all_features.append(np.concatenate([target_data, source_data, offset], axis=0))

        all_features = np.stack(all_features)

        return all_features

    def minimize(self, use_cons=None, use_jac=True, ftol=0.001, maxiter=200):

        if hasattr(self, 'edge_features'):
            x0 = np.random.rand(self.edge_features.shape[1])
            res = minimize(self.loss_edge_features, x0,
                           jac=grad(objective) if use_jac else '2-point', method='SLSQP',
                           options={'ftol': ftol, 'disp': True, 'maxiter': maxiter})
        else:
            x0 = np.random.rand(self.wrapper.num_params)
            res = minimize(self.loss, x0, bounds=Bounds(0, 1),
                           constraints=self.mass_conservation() if use_cons else None,
                           jac=grad(objective) if use_jac else '2-point', method='SLSQP',
                           options={'ftol': ftol, 'disp': True, 'maxiter': maxiter})

        return res

    def mass_conservation(self):
        constraints = {'type': 'eq',
                       'fun': lambda w: np.ones(self.N) - np.sum(self.wrapper.wrap(w, concat=True), axis=0)}
        return constraints


    def loss(self, w):
        W_targets, W_boundary = self.wrapper.wrap(w)
        predictions = []
        state = self.X
        for t in range(self.timesteps):
            y_targets = np.dot(W_targets, state)
            predictions.append(y_targets)
            #y_boundary = np.dot(W_boundary, state)
            if self.timesteps > 1:
                y_boundary = self.data[len(self.targets):, t + 1]
                state = np.concatenate([y_targets, y_boundary])

        diff, _ = flatten(self.Y - np.stack(predictions, axis=1))
        loss = np.dot(diff.T, diff) / self.Y.size

        return loss


    def loss_edge_features_nights(self, w):

        predictions = []
        for n in range(self.N_nights):
            state = self.data[:, 0, n]
            for t in range(self.timesteps):
                # compute fluxes for this timestep
                flux_targets, flux_boundary = self.fluxes(w, t, n)
                y_targets = np.dot(flux_targets, state)
                predictions.append(y_targets)
                #y_boundary = np.dot(flux_boundary, state)
                y_boundary = self.data[len(self.targets):, t+1, n]
                state = np.concatenate([y_targets, y_boundary])

        diff, _ = flatten(self.Y - np.stack(predictions, axis=1))
        loss = np.dot(diff.T, diff) / self.Y.size

        all_fluxes = self.all_target_fluxes(w)
        penalty = self.mass_conservation_penalty(all_fluxes)
        #loss += 0.0001 * penalty

        return loss

    def loss_edge_features_time(self, w):
        state = self.X  # shape (nodes, time)
        predictions = []
        for t in range(self.T):
            # compute fluxes for this timestep
            flux_targets, flux_boundary = self.fluxes(w, t)
            y_targets = np.dot(flux_targets, state[:,t])
            predictions.append(y_targets)
            y_boundary = np.dot(flux_boundary, state[:,t])
            #state = np.concatenate([y_targets, y_boundary])

        diff, _ = flatten(self.Y - np.stack(predictions, axis=1))
        loss = np.dot(diff.T, diff) / self.Y.size

        return loss

    def loss_edge_features(self, w):
        state = self.X  # shape (nodes, time)

        # compute all fluxes for target nodes
        target_fluxes = self.all_target_fluxes(w)  # shape (targets, sources, time)
        #print(target_fluxes.shape)
        y_targets = np.stack([np.dot(target_fluxes[:,:, t], state[:,t]) for t in range(self.T)], axis=1)
        #print(state.T.shape)
        #print(y_targets.shape)

        diff, _ = flatten(self.Y - y_targets)
        loss = np.dot(diff.T, diff) / self.Y.size

        #all_fluxes = self.all_fluxes(w)
        all_fluxes = self.all_target_fluxes(w)
        penalty = self.mass_conservation_penalty(all_fluxes)
        loss += 0.0002 * penalty

        return loss


    def mass_conservation_penalty(self, fluxes):
        return np.sum(np.abs(np.sum(fluxes, axis=0) - np.ones((self.N, self.T))))

    def fluxes(self, w, time, night=None, concat=False):
        if self.timesteps > 1:
            fluxes = np.dot(self.edge_features[:, :, time, night], w)  # (edges, features) x (features) --> (edges)
        else:
            fluxes = np.dot(self.edge_features[:, :, time], w)  # (edges, features) x (features) --> (edges)
        fluxes = sigmoid(fluxes)
        return self.wrapper.wrap(fluxes, concat)


    def all_target_fluxes(self, w):
        fluxes = np.dot(np.moveaxis(self.edge_features, 1, -1), w)  # move feature axis to end and perform dot product
        #if self.edge_features.ndim == 4:
        #    wrapped_fluxes = np.stack([self.wrapper.wrap(fluxes[:,:,time, night], concat=True)])
        #else:
        fluxes = sigmoid(fluxes)
        wrapped_fluxes = np.stack([self.wrapper.wrap(fluxes[:,time], concat=False)[0] for time in range(self.T)], axis=-1) # shape (targets, sources, time)
        return wrapped_fluxes

    def all_fluxes(self, w):
        fluxes = np.dot(np.moveaxis(self.edge_features, 1, -1), w)  # move feature axis to end and perform dot product
        fluxes = sigmoid(fluxes)
        wrapped_fluxes = np.stack([self.wrapper.wrap(fluxes[:,time], concat=True) for time in range(self.T)], axis=-1) # shape (nodes, nodes, time)
        return wrapped_fluxes

def sigmoid(x):
    return 1/(1+np.exp(-x))


if __name__ == '__main__':
    ts = 6
    dl = DataLoader('train', 'fall', timesteps=ts,
                            #environment_vars=['wind_u', 'wind_v', 'solarpos', 'x', 'y'],
                            #wind_path='/home/fiona/environmental_data/era5'
                    )

    space = spatial.Spatial(dl.radars)
    voronoi, G = space.voronoi()
    G = space.subgraph('type', 'measured') # graph without sink nodes

    #opt = Optimizer(G, dl.data, dl.environment)
    opt = Optimizer(G, dl.data)
    res = opt.minimize(use_cons=True, use_jac=False, maxiter=200, ftol=1e-10)

    #print(opt.all_target_fluxes(res.x))
    #
    W = opt.wrapper.wrap(res.x, concat=True)
    print(W.sum(axis=0))
    print(W)

    dl_1ts = DataLoader('train', 'fall', timesteps=1,
                            #environment_vars=['wind_u', 'wind_v', 'solarpos', 'x', 'y'],
                            #wind_path='/home/fiona/environmental_data/era5'
                    )
    opt_1ts = Optimizer(G, dl_1ts.data)
    res_1ts = opt_1ts.minimize(use_cons=True, use_jac=False, maxiter=200, ftol=1e-10)
    W = opt.wrapper.wrap(res_1ts.x, concat=True)
    print(W.sum(axis=0))
    print(W)

    dl_env = DataLoader('train', 'fall', timesteps=1,
                        environment_vars=['wind_u', 'wind_v', 'solarpos', 'x', 'y'],
                        wind_path='/home/fiona/environmental_data/era5'
                        )
    opt_env = Optimizer(G, dl_env.data, dl_env.environment)
    res_env = opt_env.minimize(use_cons=True, use_jac=False, maxiter=200, ftol=1e-08)

    all_fluxes = opt_env.fluxes(res_env.x, 10, concat=True)
    print(np.sum(all_fluxes, axis=0))
    print(all_fluxes)
    print(opt_env.edge_features[:,:,10])
    print(res_env.x)


    dl_test = DataLoader('test', 'fall', timesteps=ts,
                         environment_vars=['wind_u', 'wind_v', 'solarpos', 'x', 'y'],
                         wind_path='/home/fiona/environmental_data/era5'
                         )
    n_targets = len(opt.targets)
    predictions = np.ones((dl_test.t_range.size, n_targets)) * np.nan
    predictions_1ts = np.ones((dl_test.t_range.size, n_targets)) * np.nan
    predictions_env = np.ones((dl_test.t_range.size, n_targets)) * np.nan
    gt = np.zeros((dl_test.t_range.size, n_targets))

    opt_test = Optimizer(G, dl_test.data, dl_test.environment)
    X = opt_test.X
    Y = opt_test.Y

    #X = np.concatenate([dl_test.data[opt.targets, 0], dl_test.data[opt.boundary, 0]])   # shape (nodes, nights)
    #Y = dl_test.data[opt.targets, 1:]  # shape (nodes, time, nights)
    #print(Y.shape)

    W_targets, W_boundary = opt.wrapper.wrap(res.x)
    W_targets_1ts, W_boundary_1ts = opt_1ts.wrapper.wrap(res_1ts.x)

    errors, errors_1ts, errors_env = [], [], []
    for nidx, night in enumerate(dl_test.nights):
        t0 = night[1]
        predictions[t0] = X[:n_targets, nidx]
        predictions_1ts[t0] = X[:n_targets, nidx]
        predictions_env[t0] = X[:n_targets, nidx]
        gt[t0] = X[:n_targets, nidx]
        state = X[:, nidx]
        state_1ts = X[:, nidx]
        state_env = X[:, nidx]
        #print('length of night: ', dl_test.t_range[t0], dl_test.t_range[t0+ts-2])
        for dt in range(1, ts-1):
            # ground truth
            gt[t0 + dt] = Y[:, dt, nidx]

            # trained with X timesteps
            y_targets = np.dot(W_targets, state)
            predictions[t0 + dt] = y_targets
            y_boundary = np.dot(W_boundary, state)
            state = np.concatenate([y_targets, y_boundary])

            # trained with 1 timestep
            y_targets = np.dot(W_targets_1ts, state_1ts)
            predictions_1ts[t0 + dt] = y_targets
            y_boundary = np.dot(W_boundary_1ts, state_1ts)
            state_1ts = np.concatenate([y_targets, y_boundary])

            # with environment data
            flux_targets, flux_boundary = opt_test.fluxes(res_env.x, dt, nidx)
            y_targets = np.dot(flux_targets, state_env)
            predictions_env[t0 + dt] = y_targets
            y_boundary = np.dot(flux_boundary, state_env)
            state_env = np.concatenate([y_targets, y_boundary])


        diff = predictions[t0:t0 + ts] - gt[t0:t0 + ts]
        diff_1ts = predictions_1ts[t0:t0 + ts] - gt[t0:t0 + ts]
        diff_env = predictions_env[t0:t0 + ts] - gt[t0:t0 + ts]
        errors.append(np.mean(np.square(diff), axis=1))
        errors_1ts.append(np.mean(np.square(diff_1ts), axis=1))
        errors_env.append(np.mean(np.square(diff_env), axis=1))

    errors = np.stack(errors)
    errors_1ts = np.stack(errors_1ts)
    errors_env = np.stack(errors_env)


    #dir = '../predictions_no_weather'
    dir = '../predictions_with_weather'
    os.makedirs(dir, exist_ok=True)

    radar_names = list(dl_test.radars.values())
    for i, r in enumerate(opt.targets):
        fig, ax = plt.subplots(figsize=(15,4))
        ax.plot(dl_test.t_range, gt[:, i], color='gray', alpha=0.6, label='ground truth')
        ax.scatter(dl_test.t_range, predictions[:,i], s=30, facecolors='none', edgecolors='red', alpha=0.6,
                   label=f'trained on {opt.timesteps} timesteps')
        ax.scatter(dl_test.t_range, predictions_1ts[:, i], s=30, facecolors='none', edgecolors='blue', alpha=0.6,
                   label='trained on 1 timestep')
        ax.scatter(dl_test.t_range, predictions_env[:, i], s=30, facecolors='none', edgecolors='green', alpha=0.6,
                   label='environmental model trained on 1 timestep')
        ax.set_title(radar_names[r])
        fig.legend()
        fig.savefig(os.path.join(dir, f'{radar_names[r].split("/")[1]}.png'), bbox_inches='tight')


    fig, ax = plt.subplots()
    ax.plot(range(ts), np.mean(errors, axis=0), color='red', label=f'trained on {opt.timesteps} timesteps')
    ax.fill_between(range(ts), np.mean(errors, axis=0) - np.std(errors, axis=0),
                    np.mean(errors, axis=0) + np.std(errors, axis=0), color='red', alpha=0.2)
    ax.plot(range(ts), np.mean(errors_1ts, axis=0), color='blue', label='trained on 1 timestep')
    ax.fill_between(range(ts), np.mean(errors_1ts, axis=0) - np.std(errors_1ts, axis=0),
                    np.mean(errors_1ts, axis=0) + np.std(errors_1ts, axis=0), color='blue', alpha=0.2)
    #ax.plot(range(ts), np.mean(errors_env, axis=0), color='green', label='environmental model trained on 1 timestep')
    #ax.fill_between(range(ts), np.mean(errors_env, axis=0) - np.std(errors_env, axis=0),
    #                np.mean(errors_env, axis=0) + np.std(errors_env, axis=0), color='green', alpha=0.2)
    ax.set(xlabel='forecast horizon [hours]', ylabel='MSE')
    fig.legend()
    fig.savefig(os.path.join(dir, 'errors.png'), bbox_inches='tight')
