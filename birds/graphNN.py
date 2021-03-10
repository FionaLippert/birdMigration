import torch
from torch_geometric.data import Data, DataLoader, Dataset, InMemoryDataset
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, to_dense_adj
from sklearn.preprocessing import MinMaxScaler
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
                 data_source='radar', use_buffers=False, bird_scale = 2000, env_points=100, env_cells=False,
                 start=None, end=None, transform=None, pre_transform=None):

        #self.split = split
        self.season = season
        self.year = year
        self.timesteps = timesteps
        self.data_source = data_source
        self.start = start
        self.end = end
        self.bird_scale = bird_scale
        self.env_points = env_points # number of environment variable samples per radar cell
        self.use_buffers = use_buffers
        self.env_cells = env_cells

        super(RadarData, self).__init__(root, transform, pre_transform)

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

        print('load graph structure')

        if self.year in ['2015', '2016', '2017']:
            radars = datahandling.load_radars(osp.join(self.raw_dir, 'radar', self.season, self.year))
            print('radars available', len(radars))
        else:
            radars = datahandling.load_radars(osp.join(self.raw_dir, 'radar', self.season, '2017'))
            print('radars not available', len(radars))

        # solarpos, _, _ = datahandling.load_season(osp.join(self.raw_dir, 'radar'), self.season,
        #                                           self.year,
        #                                           'solarpos')

        # construct graph
        space = spatial.Spatial(radars)
        cells, G = space.voronoi()
        G = nx.DiGraph(space.subgraph('type', 'measured'))  # graph without sink nodes
        edges = torch.tensor(list(G.edges()), dtype=torch.long)
        edge_index = edges.t().contiguous()


        if self.data_source == 'radar':
            print('load radar data')
            data, _, t_range = datahandling.load_season(osp.join(self.raw_dir, 'radar'), self.season, self.year, 'vid', mask_days=False)

        elif self.data_source == 'abm':
            print('load abm data')
            abm_dir = osp.join(self.raw_dir, 'abm')
            data, abm_time = abm.load_season(abm_dir, self.season, self.year, cells)
            if self.use_buffers:
                radar_buffers = space.pts_local.buffer(25_000)
                buffer_data, _ = abm.load_season(abm_dir, self.season, self.year, radar_buffers)

            # adjust time range of sun data to abm time range
            t_range = abm_time.tz_convert('UTC').tz_localize(None)#[:-1] # remove time zone info


        solar_t_range = t_range.insert(-1, t_range[-1] + pd.Timedelta(t_range.freq)).tz_localize('UTC')
        solarpos = [solarposition.get_solarposition(solar_t_range, lat, lon).elevation for lon, lat in
                    radars.keys()]
        solarpos = np.stack(solarpos, axis=0)
        solarpos_change = solarpos[:, :-1] - solarpos[:, 1:]
        solarpos = solarpos[:, :-1]
        solarpos[solarpos >= -6] = np.nan  # mask nights


        print('load wind data')
        #
        # wind = era5interface.extract_points(os.path.join(self.raw_dir, 'env', self.season, self.year, 'wind_850.nc'),
        #                                     radars.keys(), t_range, vars=['u', 'v'])

        if self.env_cells:
            wind = era5interface.compute_cell_avg(
                os.path.join(self.raw_dir, 'env', self.season, self.year, 'pressure_level_850.nc'),
                cells.to_crs('epsg:4326').geometry, self.env_points, t_range, vars=['u', 'v'])
        else:
            wind = era5interface.extract_points(os.path.join(self.raw_dir, 'env', self.season, self.year, 'pressure_level_850.nc'),
                                            radars.keys(), t_range, vars=['u', 'v'])

        #print(wind)

        print('load sun data')



        check = np.isfinite(solarpos).all(axis=0) # day/night mask
        dft = pd.DataFrame({'check': np.append(np.logical_and(check[:-1], check[1:]), False),
                            'tidx': range(len(t_range))}, index=t_range)


        print('do further processing')

        # group into nights
        groups = [list(g) for k, g in it.groupby(enumerate(dft.check), key=lambda x: x[-1])]
        nights = [[item[0] for item in g] for g in groups if g[0][1]] # and len(g) > self.timesteps]

        def reshape(data, nights, mask):
            return np.stack([timeslice(data, night[0], mask) for night in nights], axis=-1)

        def timeslice(data, start_night, mask):
            data_night = data[:, start_night:]
            data_night = data_night[:, mask[start_night:]]
            #if data_night.shape[1] > self.timesteps:
            if data_night.shape[1] >= self.timesteps:
                #data_night = data_night[:, 1:self.timesteps + 1] # for radar data shift by 1 ts might be needed
                data_night = data_night[:, :self.timesteps]
            else:
                # data_night = np.pad(data_night[:, 1:], ((0, 0), (0, 1+self.timesteps-data_night.shape[1])),
                #                     constant_values=0)
                data_night = np.pad(data_night, ((0, 0), (0, self.timesteps - data_night.shape[1])),
                                    constant_values=0)
            return data_night


        # def reshape(data, nights):
        #     return np.stack([data[:, night[1:self.timesteps + 1]] for night in nights], axis=-1)


        data = reshape(data, nights, dft.check)
        if self.use_buffers:
            buffer_data = reshape(buffer_data, nights, dft.check)
        solarpos = reshape(solarpos, nights, dft.check)
        solarpos_change = reshape(solarpos_change, nights, dft.check)
        wind = {key: reshape(val, nights, dft.check) for key, val in wind.items()}

        wind = {'speed': np.sqrt(np.square(wind['u']) + np.square(wind['v'])),
                'direction': (abm.uv2deg(wind['u'], wind['v']) + 360) % 360
                }

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

        areas = cells.geometry.area.to_numpy()
        if self.use_buffers:
            buffer_areas = radar_buffers.area.to_numpy()

        # compute total number of birds within each cell around radar
        if self.timesteps > 1 and self.data_source == 'radar':
            birds_per_cell = data * areas[:, None, None]
        elif self.data_source == 'radar':
            birds_per_cell = data * areas[:, None]
        else:
            birds_per_cell = data
            if self.use_buffers and self.timesteps > 1:
                birds_per_cell_from_buffer = buffer_data / buffer_areas[:, None, None] * areas[:, None, None]
            elif self.use_buffers and self.timesteps == 1:
                birds_per_cell_from_buffer = buffer_data / buffer_areas[:, None] * areas[:, None]

        # normalize node data
        print('normalize radar data')
        #birds_per_cell = normalize(birds_per_cell, min=0)
        birds_per_cell = birds_per_cell / self.bird_scale
        if self.use_buffers:
            birds_per_cell_from_buffer = birds_per_cell_from_buffer / self.bird_scale
        print('normalize solarpos')
        solarpos = normalize(solarpos)
        #solarpos = solarpos / 360
        solarpos_change = normalize((solarpos_change))
        #solarpos_change = solarpos_change / 360

        print('normalize coords')
        xcoords = normalize(cells.x)
        ycoords = normalize(cells.y)
        print('normalize areas')
        areas = normalize(areas)
        print('normalize wind')
        wind = {key: normalize(val) for key, val in wind.items()}

        # compute distances and angles between radars
        distances = normalize([distance(cells.x.iloc[j], cells.y.iloc[j],
                                        cells.x.iloc[i], cells.y.iloc[i]) for j, i in G.edges], min=0)
        angles = normalize([angle(cells.x.iloc[j], cells.y.iloc[j],
                                  cells.x.iloc[i], cells.y.iloc[i]) for j, i in G.edges], min=0, max=360)

        # write data to disk
        os.makedirs(self.processed_dir, exist_ok=True)

        if self.use_buffers:
            x = birds_per_cell_from_buffer
        else:
            x = birds_per_cell
        data_list = [Data(x=torch.tensor(x[:, :-1, t], dtype=torch.float),
                          y=torch.tensor(x[:, 1:, t], dtype=torch.float),
                          coords=torch.stack([
                              torch.tensor(xcoords, dtype=torch.float),
                              torch.tensor(ycoords, dtype=torch.float)
                          ], dim=1),
                          areas=torch.tensor(areas, dtype=torch.float),
                          env=torch.stack([
                              *[torch.tensor(w[..., t], dtype=torch.float) for w in wind.values()],
                              torch.tensor(solarpos[..., t], dtype=torch.float),
                              torch.tensor(solarpos_change[..., t], dtype=torch.float)
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


class LSTM(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels):
        super(MLP, self).__init__()

        self.lstm = torch.nn.LSTM(in_channels, hidden_channels)
        self.hidden2birds = torch.nn.Linear(hidden_channels, out_channels)


    def forward(self, data):

        x = data.x[..., 0].view(-1, 1)

        # for details how to implement LSTM see
        # https://machinelearningmastery.com/multi-step-time-series-forecasting-long-short-term-memory-networks-python/
        return x


class MLP(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, timesteps, recurrent, seed=12345):
        super(MLP, self).__init__()

        torch.manual_seed(seed)

        self.lin1 = torch.nn.Linear(in_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)
        self.timesteps = timesteps
        self.recurrent = recurrent

    def forward(self, data):

        x = data.x[..., 0]

        y_hat = []
        for t in range(self.timesteps - 1):
            if not self.recurrent:
                x = data.x[..., t]

            features = torch.cat([x.flatten(), data.coords.flatten(), data.env[..., t].flatten()], dim=0)
            x = self.lin1(features)
            x = x.relu()
            #x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
            x = self.lin2(x)
            x = x.sigmoid()
            y_hat.append(x)

        return torch.stack(y_hat, dim=1)

class Departure(torch.nn.Module):
    # model the bird density departing within one radar cell based on cell properties and environmental conditions
    def __init__(self, in_channels, hidden_channels, out_channels, model='linear', seed=12345):
        super(Departure, self).__init__()

        torch.manual_seed(seed)

        if model == 'linear':
            self.model = torch.nn.Linear(in_channels, out_channels)
        else:
            self.model = torch.nn.Sequential(torch.nn.Linear(in_channels, hidden_channels),
                                                torch.nn.ReLU(),
                                                torch.nn.Linear(hidden_channels, out_channels),
                                                torch.nn.Sigmoid())

    def forward(self, data):
        features = torch.cat([data.coords, data.areas.view(-1, 1), data.env[..., 0]], dim=1)
        return self.model(features)


class BirdFlowTime(MessagePassing):

    def __init__(self, num_nodes, timesteps, hidden_dim=16, embedding=0, model='linear', norm=True,
                 use_departure=False, seed=12345, fix_boundary=[]):
        super(BirdFlowTime, self).__init__(aggr='add', node_dim=0) # inflows from neighbouring radars are aggregated by adding

        torch.manual_seed(seed)

        in_channels = 10 + embedding
        hidden_channels = hidden_dim #16 #2*in_channels #int(in_channels / 2)
        out_channels = 1

        in_channels_dep = 6
        hidden_channels_dep = int(in_channels_dep / 2)
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

        if use_departure:
            self.departure = torch.nn.Sequential(torch.nn.Linear(in_channels_dep, hidden_channels_dep),
                                                 torch.nn.ReLU(),
                                                 torch.nn.Linear(hidden_channels_dep, out_channels_dep),
                                                 torch.nn.Sigmoid())
            #                                      #torch.nn.Tanh())
            # self.departure = torch.nn.Sequential(torch.nn.Linear(in_channels_dep, out_channels_dep),
            #                                     torch.nn.Sigmoid())

        self.node_embedding = torch.nn.Embedding(num_nodes, embedding) if embedding > 0 else None
        self.timesteps = timesteps
        self.norm = norm
        self.use_departure = use_departure
        self.fix_boundary = fix_boundary


    def forward(self, data, teacher_forcing=0.0):
        # with teacher_forcing = 0.0 the model always uses previous predictions to make new predictions
        # with teacher_forcing = 1.0 the model always uses the ground truth to make new predictions

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
        if self.use_departure:
            features = torch.cat([coords, data.env[..., 0]], dim=1)
            x = self.departure(features)

            if len(self.fix_boundary) > 0:
                x[self.fix_boundary] = data[self.fix_boundary, 0]
        y_hat.append(x)

        self.flows = []
        self.abs_flows = []
        for t in range(self.timesteps - 1):
            r = torch.rand(1)
            if r < teacher_forcing:
                x = data.x[..., t].view(-1, 1)
            x = self.propagate(edge_index, x=x, norm=deg_inv, coords=coords, env=data.env[..., t],
                                   edge_attr=edge_attr, embedding=embedding)

            if len(self.fix_boundary) > 0:
                # use ground truth for boundary nodes
                x[self.fix_boundary, 0] = data.y[self.fix_boundary, t]

            y_hat.append(x)

        prediction = torch.cat(y_hat, dim=-1)
        return prediction


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

        # if self.use_departure:
        #     features = torch.cat([coords_j, env_j], dim=1)
        #     x_j += self.departure(features)

        abs_flow = flow * x_j
        self.abs_flows.append(abs_flow)

        return abs_flow


    # def update(self, aggr_out, coords, env):
    #    # return aggregation (sum) of inflows computed by message() PLUS departure prediction
    #    features = torch.cat([coords, env], dim=1)
    #    departure = self.departure(features)
    #    return aggr_out + departure



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

def train_fluxes(model, train_loader, optimizer, boundaries, loss_func, cuda, conservation=True, departure=False,
                 teacher_forcing=1.0):
    if cuda: model.cuda()
    model.train()
    loss_all = 0
    for data in train_loader:
        if cuda: data = data.to('cuda')
        optimizer.zero_grad()
        output = model(data, teacher_forcing) #.view(-1)

        gt = data.y
        if departure:
            gt = torch.cat([data.x[:,0].view(-1,1), gt], dim=1)
        else:
            output = output[:,1:]

        if conservation:
            outfluxes = to_dense_adj(data.edge_index, edge_attr=torch.stack(model.flows, dim=-1)).view(data.num_nodes,
                                                                                                       data.num_nodes,
                                                                                                       -1).sum(1)
            outfluxes = torch.stack([outfluxes[node] for node in range(data.num_nodes) if not boundaries[node]])
            target_fluxes = torch.ones(outfluxes.shape)
            if cuda: target_fluxes = target_fluxes.to('cuda')
            constraints = torch.mean((outfluxes - target_fluxes)**2)
            loss = loss_func(output, gt) + 0.01 * constraints
        else:
            loss = loss_func(output, gt)
        loss.backward()
        #loss_all += data.num_graphs * loss.item()
        loss_all += data.num_graphs * loss
        optimizer.step()

    return loss_all

def train_departure(model, train_loader, optimizer, loss_func, cuda):
    if cuda:
        model.to('cuda')
    model.train()
    loss_all = 0
    for data in train_loader:
        if cuda: data = data.to('cuda')
        optimizer.zero_grad()
        output = model(data).view(-1)
        gt = data.x[..., 0]
        loss = loss_func(output, gt)
        loss.backward()
        loss_all += data.num_graphs * loss
        optimizer.step()

    return loss_all

def test_fluxes(model, test_loader, timesteps, loss_func, cuda, get_outfluxes=True, bird_scale=2000, departure=False, fix_boundary=[]):
    if cuda:
        model.cuda()
    model.eval()
    loss_all = []
    outfluxes = {}
    outfluxes_abs = {}
    for tidx, data in enumerate(test_loader):
        if cuda: data = data.to('cuda')
        output = model(data) * bird_scale #.view(-1)

        gt = data.y
        if departure:
            gt = torch.cat([data.x[:, 0].view(-1, 1), gt], dim=1)
        else:
            output = output[:, 1:]
        gt = gt * bird_scale

        if len(fix_boundary) > 0:
            boundary_mask = np.ones(output.size(0))
            boundary_mask[fix_boundary] = 0
            output = output[boundary_mask]
            gt = gt[boundary_mask]

        if get_outfluxes:
            outfluxes[tidx] = to_dense_adj(data.edge_index, edge_attr=torch.stack(model.flows, dim=-1)).view(data.num_nodes,
                                                                                                   data.num_nodes,
                                                                                                   -1)
            outfluxes_abs[tidx] = to_dense_adj(data.edge_index, edge_attr=torch.stack(model.abs_flows, dim=-1)).view(
                data.num_nodes,
                data.num_nodes,
                -1)# .sum(1)
            if cuda:
                outfluxes[tidx] = outfluxes[tidx].cpu()
                outfluxes_abs[tidx] = outfluxes_abs[tidx].cpu()
            #constraints = torch.mean((outfluxes - torch.ones(data.num_nodes)) ** 2)
        loss_all.append(torch.tensor([loss_func(output[:, t], gt[:, t]) for t in range(timesteps-1)]))
        #loss_all.append(loss_func(output, gt))
        #constraints_all.append(constraints)

    if get_outfluxes:
        return torch.stack(loss_all), outfluxes , outfluxes_abs #, torch.stack(constraints_all)
    else:
        return torch.stack(loss_all)

def test_departure(model, test_loader, loss_func, cuda, bird_scale=2000):
    if cuda:
        model.cuda()
    model.eval()
    loss_all = []
    for tidx, data in enumerate(test_loader):
        if cuda: data = data.to('cuda')
        output = model(data).view(-1) * bird_scale

        gt = data.x[..., 0]
        gt = gt * bird_scale

        loss_all.append(loss_func(output, gt))

    return torch.stack(loss_all)



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



