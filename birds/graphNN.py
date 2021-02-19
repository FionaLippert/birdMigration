import torch
from torch_geometric.data import Data, DataLoader, Dataset, InMemoryDataset
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, to_dense_adj
import numpy as np
import networkx as nx
from birds import spatial, datahandling, era5interface
import os.path as osp
import os
import pandas as pd
import pickle5 as pickle
import glob
from matplotlib import pyplot as plt
import itertools as it
from datetime import datetime as dt


class RadarData(InMemoryDataset):

    def __init__(self, root, split, years, season='fall', timesteps=1,
                 data_source='radar', transform=None, pre_transform=None):

        self.split = split
        self.season = season
        self.years = years
        self.timesteps = timesteps
        self.data_source = data_source

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
        return osp.join(self.root, 'processed', self.season, f'timesteps={self.timesteps}')

    @property
    def processed_file_names(self):
        return [f'{self.split}_{self.data_source}_data.pt']

    @property
    def info_file_name(self):
        return f'{self.split}_{self.data_source}_data_info.pkl'

    def download(self):
        pass

    def process(self):
        data_list = []
        all_radars = []
        all_boundaries = []
        timepoints = []
        time_mask = []
        all_nights = []
        for i, year in enumerate(self.years):
            solarpos, radars, t_range = datahandling.load_season(osp.join(self.raw_dir, 'radar'), self.season, year,
                                                                 'solarpos')
            if self.data_source == 'radar':
                data, _, _ = datahandling.load_season(osp.join(self.raw_dir, 'radar'), self.season, year, 'vid')

            elif self.data_source == 'abm':
                abm_dir = osp.join(self.raw_dir, 'abm', self.season, year)
                print(abm_dir)
                files = glob.glob(os.path.join(abm_dir, '*.pkl'))
                print(files)
                data = []
                for file in files:
                    with open(file, 'rb') as f:
                        result = pickle.load(f)
                        data.append(result['counts'])
                        print(result['counts'].shape)

                data = np.stack(data, axis=-1).sum(-1).T
                print(data.shape)

                # adjust time range of sun data to abm time range
                print(t_range, result['time'].tz_localize(None))
                abm_time = result['time'].tz_localize(None) # remove time zone info
                start = np.where(t_range == abm_time[0])[0][0]
                end = np.where(t_range == abm_time[-1])[0][0]
                t_range = t_range[start:end+1]
                solarpos = solarpos[:, start:end+1]


            wind = era5interface.extract_points(os.path.join(self.raw_dir, 'env', self.season, year, 'wind_850.nc'),
                                                radars.keys(), t_range, vars=['u', 'v'])



            check = np.isfinite(solarpos).all(axis=0) # day/night mask
            dft = pd.DataFrame({'check': np.append(np.logical_and(check[:-1], check[1:]), False),
                                'tidx': range(len(t_range))}, index=t_range)

            timepoints.extend(t_range)
            all_radars.append(list(radars.values()))


            if self.timesteps > 1:
                # group into nights
                groups = [list(g) for k, g in it.groupby(enumerate(dft.check), key=lambda x: x[-1])]
                nights = [[item[0] for item in g] for g in groups if g[0][1]] # and len(g) > self.timesteps]
                all_nights.append(nights)

                def reshape(data, nights, mask):
                    return np.stack([timeslice(data, night[0], mask) for night in nights], axis=-1)

                def timeslice(data, start_night, mask):
                    data_night = data[:, start_night:]
                    data_night = data_night[:, mask[start_night:]]
                    if data_night.shape[1] > self.timesteps:
                        data_night = data_night[:, 1:self.timesteps + 1]
                        print(data_night.shape)
                    else:
                        data_night = np.pad(data_night[:, 1:], ((0, 0), (0, 1+self.timesteps-data_night.shape[1])),
                                            constant_values=0)
                        print(data_night.shape)
                    return data_night


                # def reshape(data, nights):
                #     return np.stack([data[:, night[1:self.timesteps + 1]] for night in nights], axis=-1)


                data = reshape(data, nights, dft.check)
                solarpos = reshape(solarpos, nights, dft.check)
                wind = {key: reshape(val, nights, dft.check) for key, val in wind.items()}


            else:
                # discard missing data
                time_mask.extend(dft.check)
                data = data[:, dft.check]
                solarpos = solarpos[:, dft.check]
                wind = {key : val[:, dft.check] for key, val in wind.items()}


            # construct graph
            space = spatial.Spatial(radars)
            cells, G = space.voronoi()
            G = nx.DiGraph(space.subgraph('type', 'measured'))  # graph without sink nodes

            print(cells['boundary'].to_dict())
            all_boundaries.append(cells['boundary'].to_dict())

            edges = torch.tensor(list(G.edges()), dtype=torch.long)
            edge_index = edges.t().contiguous()

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

            print(birds_per_cell.shape)

            # compute distances and angles between radars
            distances = normalize([distance(cells.x.iloc[j], cells.y.iloc[j],
                                            cells.x.iloc[i], cells.y.iloc[i]) for j, i in G.edges], min=0)
            angles = normalize([angle(cells.x.iloc[j], cells.y.iloc[j],
                                      cells.x.iloc[i], cells.y.iloc[i]) for j, i in G.edges], min=0, max=360)

            # write data to disk
            os.makedirs(self.processed_dir, exist_ok=True)


            if self.timesteps > 1:
                print(f'timesteps = {self.timesteps}')
                data_list.extend([Data(x=torch.tensor(birds_per_cell[..., :-1, t], dtype=torch.float),
                                  y=torch.tensor(birds_per_cell[..., 1:, t], dtype=torch.float),
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
                             for t in range(data.shape[-1] - 1)])

            else:
                data_list.extend([Data(x=torch.tensor(birds_per_cell[..., t], dtype=torch.float),
                                  y=torch.tensor(birds_per_cell[..., t+1], dtype=torch.float),
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
                             for t in range(data.shape[-1] - 1)])


        assert(all(r == all_radars[0] for r in all_radars))
        info = {'radars': all_radars[0],
                 'timepoints': timepoints,
                 'time_mask': time_mask,
                 'nights': all_nights,
                 'all_boundaries': all_boundaries}
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
        out_channels = 1
        if model == 'linear':
            self.edgeflow = torch.nn.Linear(in_channels, out_channels)
        elif model == 'linear+sigmoid':
            self.edgeflow = torch.nn.Sequential(torch.nn.Linear(in_channels, out_channels),
                                                torch.nn.Sigmoid())
        else:
            self.edgeflow = torch.nn.Sequential(torch.nn.Linear(in_channels, in_channels),
                                                                torch.nn.ReLU(),
                                                                torch.nn.Linear(in_channels, out_channels),
                                                                torch.nn.Sigmoid())
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

    #root = '/home/fiona/birdMigration/data'
    root = osp.join(args.root, 'data')
    #model_dir = '/home/fiona/birdMigration/models'
    model_dir = osp.join(args.root, 'models')
    os.makedirs(model_dir, exist_ok=True)

    timesteps=10
    embedding = 0
    conservation = True
    epochs = 100
    recurrent = False
    norm = False

    loss_func = torch.nn.MSELoss()

    action = 'train'
    model_type = 'linear'
    name = f'GNN_{model_type}_ts={timesteps}_embedding={embedding}_conservation={conservation}_epochs={epochs}_recurrent={recurrent}_norm={norm}.pt'

    if action == 'train':
        train_data = RadarData(root, 'train', ['2016'], 'fall', timesteps)
        train_loader = DataLoader(train_data, batch_size=1, shuffle=True)

        #model = BirdFlow(9 + embedding, 1, train_data[0].num_nodes, 1)
        model = BirdFlowTime(train_data[0].num_nodes, timesteps, embedding, model_type, recurrent, norm)
        params = model.parameters()

        optimizer = torch.optim.Adam(params, lr=0.01)

        boundaries = train_data.info['all_boundaries'][0]
        for epoch in range(epochs):
            loss = train(model, train_loader, optimizer, boundaries, loss_func, 'cpu', conservation)
            print(f'epoch {epoch + 1}: loss = {loss/len(train_data)}')

        torch.save(model, osp.join(model_dir, name))

    elif action == 'test':

        def load_model(name):
            model = torch.load(osp.join(model_dir, name))
            model.recurrent = True
            return model

        output_dir = osp.join(root, 'model_performance', f'recurrent={recurrent}_norm={norm}_embedding={embedding}')
        os.makedirs(output_dir, exist_ok=True)

        names = ['linear without conservation', 'linear', 'linear+sigmoid', 'mlp']
        models = [load_model(f'GNN_linear_ts={timesteps}_embedding={embedding}_conservation=False_epochs={epochs}_recurrent={recurrent}_norm={norm}.pt'),
                load_model(f'GNN_linear_ts={timesteps}_embedding={embedding}_conservation=True_epochs={epochs}_recurrent={recurrent}_norm={norm}.pt'),
                #load_model(f'GNN_linear_ts={timesteps}_embedding={embedding}_conservation=True_epochs={epochs}_recurrent={recurrent}_aggr=True.pt'),
                load_model(f'GNN_linear+sigmoid_ts={timesteps}_embedding={embedding}_conservation=True_epochs={epochs}_recurrent={recurrent}_norm={norm}.pt'),
                load_model(f'GNN_mlp_ts={timesteps}_embedding={embedding}_conservation=True_epochs={epochs}_recurrent={recurrent}_norm={norm}.pt')
        ]

        # name = f'GNN_{model_type}_ts={timesteps}_embedding={embedding}_conservation={conservation}_epochs={epochs}_recurrent={recurrent}.pt'
        # model = torch.load(osp.join(model_dir, name))

        # if not recurrent:
        #     model.recurrent = True

        test_data = RadarData(root, 'test', ['2015'], 'fall', timesteps)
        test_loader = DataLoader(test_data, batch_size=1)

        fig, ax = plt.subplots()
        for midx, model in enumerate(models):
            loss_all = test(model, test_loader, loss_func, 'cpu')
            mean_loss = loss_all.mean(0)
            std_loss = loss_all.std(0)
            line = ax.plot(range(1, timesteps), mean_loss, label=f'{names[midx]}')
            ax.fill_between(range(1, timesteps), mean_loss-std_loss, mean_loss+std_loss, alpha=0.2, color=line[0].get_color())
        ax.set_xlabel('timestep')
        ax.set_ylabel('MSE')
        ax.set_ylim(-0.005, 0.055)
        ax.set_xticks(range(1, timesteps))
        ax.legend()
        fig.savefig(os.path.join(output_dir, f'errors_recurrent={recurrent}.png'), bbox_inches='tight')
        plt.close(fig)


        dataset = test_data
        dataloader = DataLoader(dataset)

        time = dataset.info['timepoints']
        nights = dataset.info['nights'][0]

        for idx, radar in enumerate(dataset.info['radars']):
            gt = np.zeros(len(time))
            pred = []
            for _ in models:
                pred.append(np.ones(len(time)) * np.nan)
            #pred = [np.ones(len(time)) * np.nan] * len(models)
            #pred = np.ones(len(time)) * np.nan
            #pred = np.zeros(len(time))

            for nidx, data in enumerate(dataloader):
                gt[nights[nidx][1]] = data.x[idx, 0]
                gt[nights[nidx][2:timesteps+1]] = data.y[idx]
                for midx, model in enumerate(models):
                    y = model(data).detach().numpy()[idx]
                    pred[midx][nights[nidx][2:timesteps+1]] = y
                    pred[midx][nights[nidx][1]] = data.x[idx, 0]
            #gt[mask][1:] = [data.y[idx] for data in test_loader]

            fig, ax = plt.subplots(figsize=(20,4))
            ax.plot(time, gt, label='ground truth', c='gray', alpha=0.5)
            for midx, model_type in enumerate(names):
                line = ax.plot(time, pred[midx], ls='--', alpha=0.3)
                ax.scatter(time, pred[midx], s=30, facecolors='none', edgecolor=line[0].get_color(), label=f'prediction ({model_type})')

                outfluxes = to_dense_adj(data.edge_index, edge_attr=torch.stack(models[midx].flows, dim=-1)).view(
                    data.num_nodes,
                    data.num_nodes,
                    -1)
                print(idx, radar, model_type)
                for jdx, radar_j in enumerate(dataset.info['radars']):
                    print(radar, radar_j, outfluxes[idx][jdx])

            #ax.scatter(np.array(time)[mask][1:], [data.y[idx].detach().numpy() for data in dataloader], color='gray', alpha=0.6, label='ground truth')
            #ax.scatter(np.array(time)[mask][1:], [model(data).view(-1).detach().numpy()[idx] for data in dataloader],
            #           s=30, facecolors='none', edgecolors='red', alpha=0.6,
            #           label=f'trained on 1 timestep')
            ax.set_title(radar)
            ax.set_ylim(-0.05, 1.05)
            ax.set_ylabel('normalized bird density')
            fig.legend(loc='upper right', bbox_to_anchor=(0.77, 0.85))
            fig.savefig(os.path.join(output_dir, f'{radar.split("/")[1]}.png'), bbox_inches='tight')
            plt.close(fig)


