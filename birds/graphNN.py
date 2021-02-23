import torch
from torch_geometric.data import Data, DataLoader, Dataset, InMemoryDataset
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, to_dense_adj
import numpy as np
import networkx as nx
import os.path as osp
import os
import pandas as pd
import pickle5 as pickle
import glob
from pvlib import solarposition
from matplotlib import pyplot as plt
import itertools as it
from datetime import datetime as dt

from birds import spatial, datahandling, era5interface, abm


class RadarData(InMemoryDataset):

    def __init__(self, root, year, season='fall', timesteps=1,
                 data_source='radar', start=None, end=None, transform=None, pre_transform=None):

        #self.split = split
        self.season = season
        self.year = year
        self.timesteps = timesteps
        self.data_source = data_source
        self.start = start
        self.end = end

        super(RadarData, self).__init__(root, transform, pre_transform)

        print('super done')

        self.data, self.slices = torch.load(self.processed_paths[0])

        with open(osp.join(self.processed_dir, self.info_file_name), 'rb') as f:
            self.info = pickle.load(f)

    @property
    def raw_file_names(self):
        return []

    @property
    def raw_dir(self):
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, 'processed', self.data_source, self.season, self.year)

    @property
    def processed_file_names(self):
        return [f'data_timesteps={self.timesteps}.pt']

    @property
    def info_file_name(self):
        return f'info_timesteps={self.timesteps}.pkl'

    def download(self):
        pass

    def process(self):

        solarpos, radars, t_range = datahandling.load_season(osp.join(self.raw_dir, 'radar'), self.season, self.year,
                                                             'solarpos')

        print('solarpos loaded')

        # construct graph
        space = spatial.Spatial(radars)
        cells, G = space.voronoi()
        G = nx.DiGraph(space.subgraph('type', 'measured'))  # graph without sink nodes
        edges = torch.tensor(list(G.edges()), dtype=torch.long)
        edge_index = edges.t().contiguous()

        print('edges loaded')

        if self.data_source == 'radar':
            data, _, _ = datahandling.load_season(osp.join(self.raw_dir, 'radar'), self.season, self.year, 'vid')

        elif self.data_source == 'abm':
            print('load abm data')
            abm_dir = osp.join(self.raw_dir, 'abm', self.season, self.year)
            files = glob.glob(os.path.join(abm_dir, '*.pkl'))
            traj = []
            states = []
            for file in files:
                with open(file, 'rb') as f:
                    result = pickle.load(f)
                traj.append(result['trajectories'])
                states.append(result['states'])
                abm_time = result['time']

            traj = np.concatenate(traj, axis=1)[1:, ...]  # ignore first timestep because bird response is shifted to the right by one timestep
            states = np.concatenate(states, axis=1)[1:, ...]
            abm_time = abm_time[:-1]
            T = states.shape[0]

            counts, cols = abm.aggregate(traj, states, cells, range(T), state=1)
            counts = counts.fillna(0)
            data = counts[cols].to_numpy()
            print(data.shape)

            # adjust time range of sun data to abm time range
            abm_time = abm_time.tz_convert('UTC') # make sure time zone is consistent
            abm_time_naive = abm_time.tz_localize(None) # remove time zone info

            start = np.where(t_range == abm_time_naive[0])[0][0]
            end = np.where(t_range == abm_time_naive[-2])[0][0]  # ignore last timestep because bird response is shifted to the right by one timestep

            t_range = t_range[start:end+1]
            solarpos = solarpos[:, start:end+1]
            print(solarpos.shape)
            # for i, (lon, lat) in enumerate(radars.keys()):
            #     print('solarposition: ', solarposition.get_solarposition(abm_time[10:20], lat, lon))
            #     print('solarpos: ', solarpos[i, 10:20])


        wind = era5interface.extract_points(os.path.join(self.raw_dir, 'env', self.season, self.year, 'wind_850.nc'),
                                            radars.keys(), t_range, vars=['u', 'v'])



        check = np.isfinite(solarpos).all(axis=0) # day/night mask
        dft = pd.DataFrame({'check': np.append(np.logical_and(check[:-1], check[1:]), False),
                            'tidx': range(len(t_range))}, index=t_range)


        # group into nights
        groups = [list(g) for k, g in it.groupby(enumerate(dft.check), key=lambda x: x[-1])]
        nights = [[item[0] for item in g] for g in groups if g[0][1]] # and len(g) > self.timesteps]

        def reshape(data, nights, mask):
            return np.stack([timeslice(data, night[0], mask) for night in nights], axis=-1)

        def timeslice(data, start_night, mask):
            data_night = data[:, start_night:]
            data_night = data_night[:, mask[start_night:]]
            if data_night.shape[1] > self.timesteps:
                data_night = data_night[:, 1:self.timesteps + 1]
            else:
                data_night = np.pad(data_night[:, 1:], ((0, 0), (0, 1+self.timesteps-data_night.shape[1])),
                                    constant_values=0)
            return data_night


        # def reshape(data, nights):
        #     return np.stack([data[:, night[1:self.timesteps + 1]] for night in nights], axis=-1)


        data = reshape(data, nights, dft.check)
        solarpos = reshape(solarpos, nights, dft.check)
        wind = {key: reshape(val, nights, dft.check) for key, val in wind.items()}

        #
        # else:
        #     # discard missing data
        #     data = data[:, dft.check]
        #     solarpos = solarpos[:, dft.check]
        #     wind = {key : val[:, dft.check] for key, val in wind.items()}


        def normalize(features, min=None, max=None):
            if min is None:
                min = np.min(features)
            if max is None:
                max = np.max(features)
            if type(features) is not np.ndarray:
                features = np.array(features)
            return (features - min) / (max - min)

        # compute total number of birds within each cell around radar
        if self.timesteps > 1:
            birds_per_cell = data * cells.geometry.area.to_numpy()[:, None, None]
        else:
            birds_per_cell = data * cells.geometry.area.to_numpy()[:, None]

        # normalize node data
        birds_per_cell = normalize(birds_per_cell, min=0)
        solarpos = normalize(solarpos)
        xcoords = normalize(cells.x)
        ycoords = normalize(cells.y)
        wind = {key: normalize(val) for key, val in wind.items()}

        # compute distances and angles between radars
        distances = normalize([distance(cells.x.iloc[j], cells.y.iloc[j],
                                        cells.x.iloc[i], cells.y.iloc[i]) for j, i in G.edges], min=0)
        angles = normalize([angle(cells.x.iloc[j], cells.y.iloc[j],
                                  cells.x.iloc[i], cells.y.iloc[i]) for j, i in G.edges], min=0, max=360)

        # write data to disk
        os.makedirs(self.processed_dir, exist_ok=True)

        data_list = [Data(x=torch.tensor(birds_per_cell[:, :-1, t], dtype=torch.float),
                          y=torch.tensor(birds_per_cell[:, 1:, t], dtype=torch.float),
                          coords=torch.stack([
                              torch.tensor(xcoords, dtype=torch.float),
                              torch.tensor(ycoords, dtype=torch.float)
                          ], dim=1),
                          env=torch.stack([
                              *[torch.tensor(w[..., t], dtype=torch.float) for w in wind.values()],
                              torch.tensor(solarpos[..., t], dtype=torch.float)
                          ], dim=1),
                          edge_index=edge_index,
                          edge_attr=torch.stack([
                              torch.tensor(distances, dtype=torch.float),
                              torch.tensor(angles, dtype=torch.float)
                          ], dim=1))
                     for t in range(data.shape[-1] - 1)]

        # else:
        #     data_list = [Data(x=torch.tensor(birds_per_cell[..., t], dtype=torch.float),
        #                       y=torch.tensor(birds_per_cell[..., t+1], dtype=torch.float),
        #                       coords=torch.stack([
        #                           torch.tensor(xcoords, dtype=torch.float),
        #                           torch.tensor(ycoords, dtype=torch.float)
        #                       ], dim=1),
        #                       env=torch.stack([
        #                           *[torch.tensor(w[..., t], dtype=torch.float) for w in wind.values()],
        #                           torch.tensor(solarpos[..., t], dtype=torch.float)
        #                       ], dim=1),
        #                       edge_index=edge_index,
        #                       edge_attr=torch.stack([
        #                           torch.tensor(distances, dtype=torch.float),
        #                           torch.tensor(angles, dtype=torch.float)
        #                       ], dim=1))
        #                  for t in range(data.shape[-1] - 1)]


        info = {'radars': list(radars.values()),
                 'timepoints': t_range,
                 'time_mask': dft.check,
                 'nights': nights,
                 'boundaries': cells['boundary'].to_dict()}
        with open(osp.join(self.processed_dir, self.info_file_name), 'wb') as f:
            pickle.dump(info, f)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

#class MLP

class BirdFlowTime(MessagePassing):

    def __init__(self, num_nodes, timesteps, embedding=0, model='linear', recurrent=True, norm=True):
        super(BirdFlowTime, self).__init__(aggr='add', node_dim=0) # inflows from neighbouring radars are aggregated by adding
        #self.edgeflow = torch.nn.Linear(in_channels, out_channels)
        # self.edgeflow = torch.nn.Sequential(torch.nn.Linear(in_channels, out_channels),
        #                 torch.nn.ReLU(),
        #                 torch.nn.Linear(out_channels, out_channels),
        #                 torch.nn.Sigmoid())

        in_channels = 9 + embedding
        hidden_channels = in_channels
        out_channels = 1

        in_channels_dep = 9
        hidden_channels_dep = in_channels_dep
        out_channels_dep = 1

        if model == 'linear':
            self.edgeflow = torch.nn.Linear(in_channels, out_channels)
        elif model == 'linear+sigmoid':
            self.edgeflow = torch.nn.Sequential(torch.nn.Linear(in_channels, out_channels),
                                                torch.nn.Sigmoid())
        else:
            self.edgeflow = torch.nn.Sequential(torch.nn.Linear(in_channels, hidden_channels),
                                                torch.nn.ReLU(),
                                                torch.nn.Linear(hidden_channels, out_channels),
                                                torch.nn.Sigmoid())

        self.departure = torch.nn.Sequential(torch.nn.Linear(in_channels_dep, hidden_channels_dep),
                                             torch.nn.ReLU(),
                                             torch.nn.Linear(hidden_channels_dep, out_channels_dep),
                                             torch.nn.Tanh())

        self.node_embedding = torch.nn.Embedding(num_nodes, embedding) if embedding > 0 else None
        self.timesteps = timesteps
        self.recurrent = recurrent
        self.norm = norm


    def forward(self, data):

        x = data.x[..., 0].view(-1, 1)
        coords = data.coords
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        embedding = torch.cat([self.node_embedding.weight]*data.num_graphs) if self.node_embedding is not None else None

        # normalize outflow from each source node using the inverse of its degree
        src, dst = edge_index
        deg = degree(src, x.size(0), dtype=x.dtype)
        deg_inv = deg.pow(-1)

        y_hat = []
        self.flows = []
        for t in range(self.timesteps - 1):
            if not self.recurrent:
                x = data.x[..., t].view(-1, 1)
            x = self.propagate(edge_index, x=x, norm=deg_inv, coords=coords, env=data.env[..., t],
                                   edge_attr=edge_attr, embedding=embedding)

            y_hat.append(x)

        return torch.cat(y_hat, dim=-1)


    def message(self, x_j, coords_i, coords_j, env_j, norm_j, edge_attr, embedding_j):
        # construct messages to node i for each edge (j,i)
        # can take any argument initially passed to propagate()
        # x_j are source features with shape [E, out_channels]

        if embedding_j is None:
            features = torch.cat([coords_i, coords_j, env_j, edge_attr], dim=1)
        else:
            features = torch.cat([coords_i, coords_j, env_j, edge_attr, embedding_j], dim=1)
        flow = self.edgeflow(features)

        if self.norm:
            flow = flow * norm_j.view(-1, 1)
        #print(flow.view(-1))

        self.flows.append(flow)

        return flow * x_j

    def update(self, aggr_out):
       # simply return aggregation (sum) of inflows computed by message()
       return aggr_out

    # def update(self, aggr_out, coords, env):
    #    # return aggregation (sum) of inflows computed by message() PLUS departure prediction
    #    features = torch.cat([coords, env], dim=1)
    #    departure = self.departure(features)
    #    return aggr_out + departure


class BirdFlow(MessagePassing):

    def __init__(self, in_channels, out_channels, num_nodes, embedding=0):
        super(BirdFlow, self).__init__(aggr='add', node_dim=0) # inflows from neighbouring radars are aggregated by adding
        #self.edgeflow = torch.nn.Linear(in_channels, out_channels)
        self.edgeflow = torch.nn.Sequential(torch.nn.Linear(in_channels, out_channels),
                        torch.nn.ReLU(),
                        torch.nn.Linear(out_channels, out_channels))
        self.node_embedding = torch.nn.Embedding(num_nodes, embedding) if embedding > 0 else None



    def forward(self, data):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        x = data.x.view(-1,1)
        coords = data.coords
        env = data.env
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        embedding = torch.cat([self.node_embedding.weight]*data.num_graphs) if self.node_embedding is not None else None

        # normalize outflow from each source node using the inverse of its degree
        src, dst = edge_index
        deg = degree(src, x.size(0), dtype=x.dtype)
        deg_inv = deg.pow(-1)
        y_hat = self.propagate(edge_index, x=x, norm=deg_inv, coords=coords, env=env,
                               edge_attr=edge_attr, embedding=embedding)

        return y_hat


    def message(self, x_j, coords_i, coords_j, env_j, norm_j, edge_attr, embedding_j):
        # construct messages to node i for each edge (j,i)
        # can take any argument initially passed to propagate()
        # x_j are source features with shape [E, out_channels]

        if embedding_j is None:
            features = torch.cat([coords_i, coords_j, env_j, edge_attr], dim=1)
        else:
            features = torch.cat([coords_i, coords_j, env_j, edge_attr, embedding_j], dim=1)
        flow = self.edgeflow(features)

        self.flow = flow

        return norm_j.view(-1,1) * flow * x_j

    #def update(self, aggr_out, x):
    #    # aggr_out has shape [N, out_channels]
    #    # simply return aggregation (sum) of inflows computed by message()
    #    print(aggr_out.shape)
    #    return aggr_out

def angle(x1, y1, x2, y2):
    y = y1 - y2
    x = x1 - x2
    rad = np.arctan2(y, x)
    deg = np.rad2deg(rad)
    deg = (deg + 360) % 360
    return deg

def distance(x1, y1, x2, y2):
    # for coord1 and coord2 given in local crs
    return np.linalg.norm(np.array([x1-x2, y1-y2])) / 10**3 # in kilometers

def MSE(output, gt):
    return torch.mean((output - gt)**2)

def train(model, train_loader, optimizer, boundaries, loss_func, device, conservation=True, aggr=False):
    model.train()
    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data) #.view(-1)
        gt = data.y.to(device)

        outfluxes = to_dense_adj(data.edge_index, edge_attr=torch.stack(model.flows, dim=-1)).view(data.num_nodes,
                                                                                                  data.num_nodes,
                                                                                                  -1).sum(1)
        outfluxes = torch.stack([outfluxes[node] for node in range(data.num_nodes) if not boundaries[node]])

        if aggr:
            output = output.view(data.num_nodes, -1, 2).sum(-1)
            gt = gt.view(data.num_nodes, -1, 2).sum(-1)

        if conservation:
            constraints = torch.mean((outfluxes - torch.ones(outfluxes.shape))**2)
            loss = loss_func(output, gt) + 0.01 * constraints
        else:
            loss = loss_func(output, gt)
        loss.backward()
        #loss_all += data.num_graphs * loss.item()
        loss_all += data.num_graphs * loss
        optimizer.step()

    print(f'outfluxes: {outfluxes}')
    #print(f'embedding: {model.node_embedding.weight.view(-1)}')

    return loss_all

def test(model, test_loader, timesteps, loss_func, device):
    model.eval()
    loss_all = []
    outfluxes = {}
    for tidx, data in enumerate(test_loader):
        data = data.to(device)
        output = model(data)#.view(-1)

        gt = data.y.to(device)

        outfluxes[tidx] = to_dense_adj(data.edge_index, edge_attr=torch.stack(model.flows, dim=-1)).view(data.num_nodes,
                                                                                                   data.num_nodes,
                                                                                                   -1) #.sum(1)
        #constraints = torch.mean((outfluxes - torch.ones(data.num_nodes)) ** 2)
        loss_all.append(torch.tensor([loss_func(output[:, t], gt[:, t]) for t in range(timesteps-1)]))
        #constraints_all.append(constraints)

    return torch.stack(loss_all), outfluxes #, torch.stack(constraints_all)



if __name__ == '__main__':

    # good results with: node embedding, degree normalization, multiple timesteps, outflow reg only for center nodes
    # constraints weight = 0.01
    # edge function: linear and sigmoid

    import argparse

    parser = argparse.ArgumentParser(description='GraphNN experiments')
    parser.add_argument('--root', type=str, default='/home/fiona/birdMigration', help='entry point to required data')
    args = parser.parse_args()

    root = osp.join(args.root, 'data')
    model_dir = osp.join(args.root, 'models')
    os.makedirs(model_dir, exist_ok=True)

    timesteps=5
    embedding = 0
    conservation = True
    epochs = 2
    recurrent = True
    norm = False

    loss_func = torch.nn.MSELoss()

    action = 'train'
    model_type = 'linear'

    if action == 'train':
        d1 = RadarData(root, 'train', '2015', 'fall', timesteps)
        d2 = RadarData(root, 'train', '2016', 'fall', timesteps)
        train_data = torch.utils.data.ConcatDataset([d1, d2])
        train_loader = DataLoader(train_data, batch_size=1, shuffle=True)

        #model = BirdFlow(9 + embedding, 1, train_data[0].num_nodes, 1)
        model = BirdFlowTime(train_data[0].num_nodes, timesteps, embedding, model_type, recurrent, norm)
        params = model.parameters()

        optimizer = torch.optim.Adam(params, lr=0.01)

        boundaries = d1.info['boundaries']
        for epoch in range(epochs):
            loss = train(model, train_loader, optimizer, boundaries, loss_func, 'cpu', conservation)
            print(f'epoch {epoch + 1}: loss = {loss/len(train_data)}')



