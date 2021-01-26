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

class VP():
    def __init__(self, split, start_date, end_date, path='/home/fiona/radar_data/vpi/night_only', use_nights=False):

        #super(VP, self).__init__()

        self.split = split
        if split == 'test':
            year = '2017'
        else:
            year = '2016'

        data_dir = os.path.join(path, f'{year}0801T0000_to_{year}1130T2359')
        vid, self.radars, t_range = datahandling.load_data(data_dir, 'vid',
                                                           f'{year}-{start_date}', f'{year}-{end_date}',
                                                           t_unit='1H', mask_days=True)

        data = np.stack([v.data.flatten() for k, v in vid.items()])
        check = np.isfinite(data).all(axis=0)
        self.dft = pd.DataFrame({'check': np.append(np.logical_and(check[:-1], check[1:]), False),
                            'tidx': range(len(t_range))}, index=t_range)

        if use_nights:
            # group into nights
            timesteps = 4
            groups = [list(g) for k, g in it.groupby(enumerate(self.dft.check), key=lambda x: x[-1])]
            self.nights = [[item[0] for item in g] for g in groups if g[0][1] and len(g) > timesteps]
            self.data = np.stack([data[:, night[1:timesteps+1]] for night in self.nights], axis=-1)
        else:
            self.data = data[:,self.dft.check]

        self.max = np.nanmax(data)
        self.data /= self.max

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
        W_targets = np.vstack(W_targets)

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
    def __init__(self, G, data):
        self.G = G
        self.N = len(G)

        self.targets = [n for n, data in G.nodes(data=True) if not data.get('boundary')]
        self.boundary = [n for n, data in G.nodes(data=True) if data.get('boundary')]

        self.adj = nx.to_numpy_matrix(G, nodelist=np.concatenate([self.targets, self.boundary]))
        self.wrapper = ParamWrapper(self.adj, len(self.targets))

        assert(data.ndim in [2, 3])

        if data.ndim == 2:
            # assume data has shape (nodes, time)
            self.X = np.concatenate([data[self.targets, :-1], data[self.boundary, :-1]])
            self.Y = data[self.targets, 1:]
            self.timesteps = 1
        elif data.ndim == 3:
            # assume data has shape (nodes, time, night)
            self.X = np.concatenate([data[self.targets, 0], data[self.boundary, 0]])
            self.Y = data[self.targets, 1:, :]
            self.N_nights = data.shape[2]
            self.timesteps = data.shape[1] - 1 # first timestep is needed as initial state


    def minimize(self, use_cons=None, use_jac=True, ftol=0.001, maxiter=200):

        bounds = Bounds(0, 1)
        x0 = np.random.rand(self.wrapper.num_params)
        objective = self.loss

        res = minimize(objective, x0, bounds=bounds,
                       constraints=self.mass_conservation() if use_cons else None,
                       jac=grad(objective) if use_jac else '2-point', method='SLSQP',
                       options={'ftol': ftol, 'disp': True, 'maxiter': maxiter})

        return res

    def mass_conservation(self):
        constraints = {'type': 'eq',
                #'fun': lambda x: np.array([1 - x[np.where(self.sources == i)[0]].sum() for i in self.G])}
                'fun': lambda w: np.ones(self.N) - np.sum(self.wrapper.wrap(w, concat=True), axis=0)}
        return constraints


    def loss(self, w):
        W_targets, W_boundary = self.wrapper.wrap(w)
        predictions = []
        state = self.X
        for t in range(self.timesteps):
            y_targets = np.dot(W_targets, state)
            predictions.append(y_targets)
            y_boundary = np.dot(W_boundary, state)
            state = np.concatenate([y_targets, y_boundary])

        diff, _ = flatten(self.Y - np.stack(predictions, axis=1))
        loss = np.dot(diff.T, diff) / self.Y.size

        return loss

    def loss_wind(self, w):




if __name__ == '__main__':
    dataloader = VP('train', '08-15 18:00:00', '09-15 10:00:00', use_nights=False)
    space = spatial.Spatial(dataloader.radars)
    adj, voronoi, G = space.voronoi()
    G = space.subgraph('type', 'measured') # graph without sink nodes

    opt = Optimizer(G, dataloader.data)
    res = opt.minimize(use_cons=True, use_jac=True, maxiter=500, ftol=0.0001)

    W = opt.wrapper.wrap(res.x, concat=True)
    print(W.sum(axis=0))
    print(W)
