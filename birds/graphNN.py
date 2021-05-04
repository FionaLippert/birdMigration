import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader, Dataset, InMemoryDataset
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, to_dense_adj, dense_to_sparse
from torch_geometric_temporal.nn.recurrent import DCRNN
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import networkx as nx
import os.path as osp
import os
import pandas as pd
import geopandas as gpd
import pickle5 as pickle
import glob
from pvlib import solarposition
from matplotlib import pyplot as plt
import itertools as it
from datetime import datetime as dt


class LSTM(torch.nn.Module):
    """
   Standard LSTM taking all observed/predicted bird densities and environmental features as input to LSTM
   Args:
       in_channels (int): number of input features (node features x number of nodes)
       hidden_channels (int): number of units per hidden layer
       out_channels (int): number of nodes x number of outputs per node
       timesteps (int): length of forecasting horizon
   """
    def __init__(self, in_channels, hidden_channels, out_channels, timesteps, n_layers=1, dropout_p=0, seed=1234):
        super(LSTM, self).__init__()

        torch.manual_seed(seed)

        self.fc_in = torch.nn.Linear(in_channels, hidden_channels)
        self.lstm_layers = nn.ModuleList([torch.nn.LSTMCell(hidden_channels, hidden_channels) for l in range(n_layers)])
        self.fc_out = torch.nn.Linear(hidden_channels, out_channels)

        self.timesteps = timesteps
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers


    def forward(self, data, teacher_forcing=0):

        x = data.x[:, 0]
        # states = torch.zeros(1, self.hidden_channels).to(x.device)
        # hidden = torch.zeros(1, self.hidden_channels).to(x.device)
        h_t = [torch.zeros(1, self.hidden_channels).to(x.device) for l in range(self.n_layers)]
        c_t = [torch.zeros(1, self.hidden_channels).to(x.device) for l in range(self.n_layers)]

        #hidden = None

        y_hat = [x]
        for t in range(self.timesteps):
            r = torch.rand(1)
            if r < teacher_forcing:
                x = data.x[:, t]


            # use both bird prediction/observation and environmental features as input to LSTM
            inputs = torch.cat([data.coords.flatten(),
                                data.env[..., t+1].flatten(),
                                data.local_dusk[:, t].float().flatten(),
                                x], dim=0).view(1, -1)

            # multi-layer LSTM
            inputs = self.fc_in(inputs).relu()
            h_t[0], c_t[0] = self.lstm_layers[0](inputs, (h_t[0], c_t[0]))
            for l in range(1, self.n_layers):
                h_t[l], c_t[l] = self.lstm_layers[l](h_t[l-1], (h_t[l], c_t[l]))

            x = self.fc_out(h_t[-1]).sigmoid().view(-1)

            # for locations where it is night: set birds in the air to zero
            x = x * data.local_night[:, t+1]

            y_hat.append(x)

        return torch.stack(y_hat, dim=1)


class MLP(torch.nn.Module):
    """
    Standard MLP mapping concatenated features of all nodes at time t to migration intensities
    of all nodes at time t
    Args:
        in_channels (int): number of input features (node features x number of nodes)
        hidden_channels (int): number of units per hidden layer
        out_channels (int): number of nodes x number of outputs per node
        timesteps (int): length of forecasting horizon
    """
    def __init__(self, in_channels, hidden_channels, out_channels, timesteps, n_layers=1, dropout_p=0.5, seed=12345):
        super(MLP, self).__init__()

        torch.manual_seed(seed)

        self.fc_in = torch.nn.Linear(in_channels, hidden_channels)
        self.fc_hidden = nn.ModuleList([torch.nn.Linear(hidden_channels, hidden_channels) for _ in range(n_layers - 1)])
        self.fc_out = torch.nn.Linear(hidden_channels, out_channels)
        self.timesteps = timesteps
        self.dropout_p = dropout_p

    def forward(self, data):

        y_hat = []
        for t in range(self.timesteps + 1):

            features = torch.cat([data.coords.flatten(),
                                  data.env[..., t].flatten()], dim=0)
            x = self.fc_in(features)
            x = x.relu()
            x = torch.nn.functional.dropout(x, p=self.dropout_p, training=self.training)

            for l in self.fc_hidden:
                x = l(x)
                x = x.relu()
                x = torch.nn.functional.dropout(x, p=self.dropout_p, training=self.training)

            x = self.fc_out(x)
            x = x.sigmoid()

            # for locations where it is night: set birds in the air to zero
            x = x * data.local_night[:, t]

            y_hat.append(x)

        return torch.stack(y_hat, dim=1)


class LocalMLP(MessagePassing):

    def __init__(self, **kwargs):
        super(LocalMLP, self).__init__(aggr='add', node_dim=0)

        self.timesteps = kwargs.get('timesteps', 40)
        self.dropout_p = kwargs.get('dropout_p', 0)
        self.n_hidden = kwargs.get('n_hidden', 16)
        self.use_acc = kwargs.get('use_acc_vars', False)
        self.n_in = 4 + kwargs.get('n_env', 4) + self.use_acc * 2
        self.n_layers = kwargs.get('n_layers', 1)
        self.force_zeros = kwargs.get('force_zeros', False)

        torch.manual_seed(kwargs.get('seed', 1234))

        self.fc_in = torch.nn.Linear(self.n_in, self.n_hidden)
        self.fc_hidden = nn.ModuleList([torch.nn.Linear(self.n_hidden, self.n_hidden)
                                        for _ in range(self.n_layers - 1)])
        self.fc_out = torch.nn.Linear(self.n_hidden, 1)


    def forward(self, data, **kwargs):

        y_hat = []

        for t in range(self.timesteps + 1):

            x = self.propagate(data.edge_index, coords=data.coords, env=data.env[..., t], acc=data.acc[..., t],
                               areas=data.areas, edge_attr=data.edge_attr, night=data.local_night[:, t])

            if self.force_zeros:
                print('force birds in air to be zero')
                # for locations where it is night: set birds in the air to zero
                x = x * data.local_night[:, t].view(-1, 1)

            y_hat.append(x)

        prediction = torch.cat(y_hat, dim=-1)
        return prediction


    def message(self, edge_attr):
        # set all messages to 0 --> no spatial dependencies
        n_edges = edge_attr.size(0)
        msg = torch.zeros(n_edges).to(edge_attr.device)
        return msg


    def update(self, aggr_out, coords, env, areas, night, acc):
        # use only location-specific features to predict migration intensities
        if self.use_acc:
            features = torch.cat([coords, env, areas.view(-1, 1), night.float().view(-1, 1), acc], dim=1)
        else:
            features = torch.cat([coords, env, areas.view(-1, 1), night.float().view(-1, 1)], dim=1)
        x = self.fc_in(features).relu()
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        for l in self.fc_hidden:
            x = l(x).relu()
            x = F.dropout(x, p=self.dropout_p, training=self.training)

        x = self.fc_out(x)
        x = x.sigmoid()

        return x


class LocalLSTM(MessagePassing):

    def __init__(self, **kwargs):
        super(LocalLSTM, self).__init__(aggr='add', node_dim=0)

        self.timesteps = kwargs.get('timesteps', 40)
        self.dropout_p = kwargs.get('dropout_p', 0)
        self.n_hidden = kwargs.get('n_hidden', 16)
        self.n_in = 7 + kwargs.get('n_env', 4)
        self.n_layers = kwargs.get('n_layers', 1)
        self.predict_delta = kwargs.get('predict_delta', True)
        self.force_zeros = kwargs.get('force_zeros', True)

        torch.manual_seed(kwargs.get('seed', 1234))

        #self.fc_in = torch.nn.Linear(self.n_in, self.n_hidden)
        self.mlp_in = torch.nn.Sequential(torch.nn.Linear(self.n_in, self.n_hidden),
                                          torch.nn.Dropout(p=self.dropout_p),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(self.n_hidden, self.n_hidden))
        self.lstm_layers = nn.ModuleList([torch.nn.LSTMCell(self.n_hidden, self.n_hidden)
                                          for _ in range(self.n_layers)])
        #self.fc_out = torch.nn.Linear(self.n_hidden, self.n_out)
        self.mlp_out = torch.nn.Sequential(torch.nn.Linear(self.n_hidden, self.n_hidden),
                                          torch.nn.Dropout(p=self.dropout_p),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(self.n_hidden, 1))


    def forward(self, data, **kwargs):

        teacher_forcing = kwargs.get('teacher_forcing', 0)

        x = data.x[:, 0].view(-1, 1)

        # initialize lstm variables
        # hidden = Variable(torch.zeros(data.x.size(0), self.n_hidden)).to(x.device)
        # #states = Variable(torch.zeros(data.x.size(0), self.n_hidden)).to(x.device)
        # #states = torch.cat([x] * self.n_hidden, dim=1)
        # states = self.birds2hidden(x)
        # hidden = None
        h_t = [torch.zeros(data.x.size(0), self.n_hidden).to(x.device) for l in range(self.n_layers)]
        c_t = [torch.zeros(data.x.size(0), self.n_hidden).to(x.device) for l in range(self.n_layers)]

        y_hat = [x]

        for t in range(self.timesteps):
            if True: #torch.any(data.local_night[:, t+1] | data.local_dusk[:, t+1]):
                # at least for one radar station it is night or dusk
                r = torch.rand(1)
                if r < teacher_forcing:
                    # if data is available use ground truth, otherwise use model prediction
                    x = data.missing[..., t].view(-1, 1) * x + \
                        ~data.missing[..., t].view(-1, 1) * data.x[..., t].view(-1, 1)

                x, h_t, c_t = self.propagate(data.edge_index, x=x, coords=data.coords, areas=data.areas,
                                             h_t=h_t, c_t=c_t, edge_attr=data.edge_attr,
                                             dusk=data.local_dusk[:, t],
                                             dawn=data.local_dawn[:, t+1],
                                             env=data.env[..., t+1],
                                             night=data.local_night[:, t+1]
                                             )

            if self.force_zeros:
                # for locations where it is night: set birds in the air to zero
                x = x * data.local_night[:, t+1].view(-1, 1)

            y_hat.append(x)

        prediction = torch.cat(y_hat, dim=-1)
        return prediction


    def message(self, edge_attr):
        # set all messages to 0 --> no spatial dependencies
        n_edges = edge_attr.size(0)
        msg = torch.zeros(n_edges).to(edge_attr.device)
        return msg


    def update(self, aggr_out, coords, env, dusk, dawn, h_t, c_t, x, areas, night):

        inputs = torch.cat([x.view(-1, 1), coords, env, dusk.float().view(-1, 1),
                            dawn.float().view(-1, 1), areas.view(-1, 1), night.float().view(-1, 1)], dim=1)
        #inputs = self.fc_in(inputs).relu()
        inputs = self.mlp_in(inputs).relu()

        h_t[0], c_t[0] = self.lstm_layers[0](inputs, (h_t[0], c_t[0]))
        h_t[0] = F.dropout(h_t[0], p=self.dropout_p, training=self.training)
        c_t[0] = F.dropout(c_t[0], p=self.dropout_p, training=self.training)
        for l in range(1, self.n_layers):
            h_t[l], c_t[l] = self.lstm_layers[l](h_t[l - 1], (h_t[l], c_t[l]))

        #delta = self.fc_out(h_t[-1]).tanh()

        if self.predict_delta:
            delta = self.mlp_out(h_t[-1]).tanh()
            x = x + delta
        else:
            x = self.mlp_out(h_t[-1]).sigmoid()

        return x, h_t, c_t


class RecurrentGCN(torch.nn.Module):
    def __init__(self, timesteps, node_features, n_hidden=32, n_out=1, K=1):
        # doesn't take external features into account
        super(RecurrentGCN, self).__init__()
        self.recurrent = DCRNN(7, n_hidden, K, bias=True)
        self.linear = torch.nn.Linear(n_hidden, n_out)
        self.timesteps = timesteps

    def forward(self, data, teacher_forcing=0):
        x = data.x[:, 0].view(-1, 1)
        predictions = [x]
        for t in range(self.timesteps):
            # TODO try concatenating input features and prection x to also use weather info etc
            r = torch.rand(1)
            if r < teacher_forcing:
                x = data.x[:, t].view(-1, 1)

            input = torch.cat([x, data.env[..., t], data.coords], dim=1)
            x = self.recurrent(input, data.edge_index, data.edge_weight.float())
            x = F.relu(x)
            x = self.linear(x)

            # for locations where it is night: set birds in the air to zero
            x = x * data.local_night[:, t+1].view(-1, 1)

            predictions.append(x)

        predictions = torch.cat(predictions, dim=-1)
        return predictions


class BirdFlowGNN(MessagePassing):

    def __init__(self, num_nodes, timesteps, hidden_dim=16, embedding=0, model='linear', norm=True,
                 use_departure=False, seed=12345, fix_boundary=[], multinight=False, use_wind=True, dropout_p=0.5):
        super(BirdFlowGNN, self).__init__(aggr='add', node_dim=0) # inflows from neighbouring radars are aggregated by adding

        torch.manual_seed(seed)

        in_channels = 10 + embedding
        if not use_wind:
            in_channels -= 2
        hidden_channels = hidden_dim #16 #2*in_channels #int(in_channels / 2)
        out_channels = 1

        in_channels_dep = 7
        if not use_wind:
            in_channels_dep -= 2
        hidden_channels_dep = in_channels_dep #int(in_channels_dep / 2)
        out_channels_dep = 1

        if model == 'linear':
            self.edgeflow = torch.nn.Linear(in_channels, out_channels)
        elif model == 'linear+sigmoid':
            self.edgeflow = torch.nn.Sequential(torch.nn.Linear(in_channels, out_channels),
                                                torch.nn.Sigmoid())
            self.departure = torch.nn.Sequential(torch.nn.Linear(in_channels_dep, out_channels_dep),
                                                 torch.nn.Tanh())
        else:
            self.edgeflow = torch.nn.Sequential(torch.nn.Linear(in_channels, hidden_channels),
                                                torch.nn.Dropout(p=dropout_p),
                                                torch.nn.ReLU(),
                                                torch.nn.Linear(hidden_channels, out_channels),
                                                torch.nn.Sigmoid())
            self.departure = torch.nn.Sequential(torch.nn.Linear(in_channels_dep, hidden_channels_dep),
                                                 torch.nn.Dropout(p=dropout_p),
                                                 torch.nn.ReLU(),
                                                 torch.nn.Linear(hidden_channels_dep, out_channels_dep),
                                                 torch.nn.Tanh())


        self.node_embedding = torch.nn.Embedding(num_nodes, embedding) if embedding > 0 else None
        self.timesteps = timesteps
        self.norm = norm
        self.use_departure = use_departure
        self.fix_boundary = fix_boundary
        self.multinight = multinight
        self.use_wind = use_wind


    def forward(self, data, teacher_forcing=0.0):
        # with teacher_forcing = 0.0 the model always uses previous predictions to make new predictions
        # with teacher_forcing = 1.0 the model always uses the ground truth to make new predictions

        # measurement at t=0
        x = data.x[..., 0].view(-1, 1)

        # birds on the ground at t=0
        # TODO use additional aggregation network to estimate the number of birds aggregated on the ground
        #  based on environmental conditions in the past
        ground = torch.zeros_like(x)

        coords = data.coords
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        embedding = torch.cat([self.node_embedding.weight]*data.num_graphs) if self.node_embedding is not None else None

        # normalize outflow from each source node using the inverse of its degree
        src, dst = edge_index
        deg = degree(src, x.size(0), dtype=x.dtype)
        deg_inv = deg.pow(-1)

        y_hat = []
        y_hat.append(x)

        self.flows = []
        self.abs_flows = []
        for t in range(self.timesteps):
            r = torch.rand(1)
            if r < teacher_forcing:
                # if data is available use ground truth, otherwise use model prediction
                x = data.missing[..., t].view(-1, 1) * x + \
                    ~data.missing[..., t].view(-1, 1) * data.x[..., t].view(-1, 1)

            env = data.env[..., t]
            if not self.use_wind:
                env = env[:, 2:]

            x = self.propagate(edge_index, x=x, norm=deg_inv, coords=coords, env=env,
                               edge_attr=edge_attr, embedding=embedding, ground=ground,
                               local_dusk=data.local_dusk[:, t])

            if len(self.fix_boundary) > 0:
                # use ground truth for boundary nodes
                x[self.fix_boundary, 0] = data.y[self.fix_boundary, t]

            if self.multinight:
                # for locations where it is dawn: save birds to ground and set birds in the air to zero
                r = torch.rand(1)
                if r < teacher_forcing:
                    ground = ground + data.local_dawn[:, t+1].view(-1, 1) * data.x[..., t+1].view(-1, 1)
                else:
                    ground = ground + data.local_dawn[:, t+1].view(-1, 1) * x
                x = x * ~data.local_dawn[:, t].view(-1, 1)

                # TODO for radar data, birds can stay on the ground or depart later in the night, so
                #  at dusk birds on ground shouldn't be set to zero but predicted departing birds should be subtracted
                # for locations where it is dusk: set birds on ground to zero
                ground = ground * ~data.local_dusk[:, t].view(-1, 1)

            y_hat.append(x)

        prediction = torch.cat(y_hat, dim=-1)
        return prediction


    def message(self, x_j, coords_i, coords_j, env_j, edge_attr):
        # construct messages to node i for each edge (j,i)
        # can take any argument initially passed to propagate()
        # x_j are source features with shape [E, out_channels]

        features = torch.cat([coords_i, coords_j, env_j, edge_attr], dim=1)
        flow = self.edgeflow(features)

        # if self.norm:
        #     flow = flow * norm_j.view(-1, 1)

        self.flows.append(flow)

        abs_flow = flow * x_j
        self.abs_flows.append(abs_flow)

        return abs_flow


    def update(self, aggr_out, coords, env, ground, local_dusk):
        # return aggregation (sum) of inflows computed by message()
        # add departure prediction if local_dusk flag is True

        if self.multinight:
            features = torch.cat([coords, env, ground, local_dusk.view(-1, 1)], dim=1)
            departure = self.departure(features)
            #departure = departure * local_dusk.view(-1, 1) # only use departure model if it is local dusk
            pred = aggr_out + departure
        else:
            pred = aggr_out

        return pred



class BirdFlowGraphLSTM(MessagePassing):

    def __init__(self, **kwargs):
        super(BirdFlowGraphLSTM, self).__init__(aggr='add', node_dim=0) # inflows from neighbouring radars are aggregated by adding

        self.timesteps = kwargs.get('timesteps', 40)
        self.dropout_p = kwargs.get('dropout_p', 0)
        self.n_hidden = kwargs.get('n_hidden', 16)
        self.n_env = kwargs.get('n_env', 4)
        self.n_node_in = 6 + self.n_env
        self.n_edge_in = 9 + 2*self.n_env
        self.n_self_in = 5 + kwargs.get('n_env', 4)
        self.n_fc_layers = kwargs.get('n_fc_layers', 1)
        self.n_lstm_layers = kwargs.get('n_lstm_layers', 1)
        self.fixed_boundary = kwargs.get('fixed_boundary', [])
        self.force_zeros = kwargs.get('force_zeros', True)
        self.recurrent = kwargs.get('recurrent', True)

        self.use_encoder = kwargs.get('use_encoder', False)
        self.t_context = kwargs.get('t_context', 0)

        self.edge_type = kwargs.get('edge_type', 'voronoi')
        if self.edge_type == 'voronoi':
            self.n_edge_in += 1 # use face_length as additional feature
            self.n_node_in += 1 # use voronoi cell area as additional feature

        seed = kwargs.get('seed', 1234)
        torch.manual_seed(seed)

        if self.n_fc_layers < 1:
            self.edgeflow = torch.nn.Sequential(torch.nn.Linear(self.n_edge_in, 1),
                                                torch.nn.Sigmoid())
            self.selfflow = torch.nn.Sequential(torch.nn.Linear(self.n_self_in, 1),
                                                torch.nn.Sigmoid())

        else:
            self.fc_edge_in = torch.nn.Linear(self.n_edge_in, self.n_hidden)
            self.fc_edge_hidden = nn.ModuleList([torch.nn.Linear(self.n_hidden, self.n_hidden)
                                                 for _ in range(self.n_fc_layers - 1)])
            self.fc_edge_out = torch.nn.Linear(self.n_hidden, 1)

            self.fc_self_in = torch.nn.Linear(self.n_self_in, self.n_hidden)
            self.fc_self_hidden = nn.ModuleList([torch.nn.Linear(self.n_hidden, self.n_hidden)
                                                 for _ in range(self.n_fc_layers - 1)])
            self.fc_self_out = torch.nn.Linear(self.n_hidden, 1)

        self.node2hidden = torch.nn.Sequential(torch.nn.Linear(self.n_node_in, self.n_hidden),
                                               torch.nn.Dropout(p=self.dropout_p),
                                               torch.nn.ReLU(),
                                               torch.nn.Linear(self.n_hidden, self.n_hidden))

        self.lstm_layers = nn.ModuleList([nn.LSTMCell(self.n_hidden, self.n_hidden) for _ in range(self.n_lstm_layers)])

        self.hidden2delta = torch.nn.Sequential(torch.nn.Linear(self.n_hidden, self.n_hidden),
                                                torch.nn.Dropout(p=self.dropout_p),
                                                torch.nn.ReLU(),
                                                torch.nn.Linear(self.n_hidden, 1))

        if self.use_encoder:
            self.encoder = RecurrentEncoder(timesteps=self.t_context, n_env=self.n_env, n_hidden=self.n_hidden,
                                            n_lstm_layers=self.n_lstm_layers, seed=seed, dropout_p=self.dropout_p)



    def forward(self, data, teacher_forcing=0.0):
        # with teacher_forcing = 0.0 the model always uses previous predictions to make new predictions
        # with teacher_forcing = 1.0 the model always uses the ground truth to make new predictions

        y_hat = []

        # birds on the ground at t=0
        #ground = torch.zeros_like(x).to(x.device)

        # initialize lstm variables
        if self.use_encoder and self.recurrent:
            # push context timeseries through encoder to initialize decoder
            h_t, c_t = self.encoder(data)
            #x = torch.zeros(data.x.size(0)).to(data.x.device) # TODO eventually use this!?
            x = data.x[..., self.t_context].view(-1, 1)
            y_hat.append(x)

        elif self.recurrent:
            # start from scratch
            # measurement at t=0
            x = data.x[..., 0].view(-1, 1)
            y_hat.append(x)
            h_t = [torch.zeros(data.x.size(0), self.n_hidden).to(x.device) for l in range(self.n_lstm_layers)]
            c_t = [torch.zeros(data.x.size(0), self.n_hidden).to(x.device) for l in range(self.n_lstm_layers)]

        else:
            x = data.x[..., 0].view(-1, 1)
            y_hat.append(x)
            h_t = []
            c_t = []

        coords = data.coords
        edge_index = data.edge_index
        edge_attr = data.edge_attr


        self.flows = torch.zeros((edge_index.size(1), 1, self.timesteps+1)).to(x.device)
        self.abs_flows = torch.zeros((edge_index.size(1), 1, self.timesteps+1)).to(x.device)
        self.selfflows = torch.zeros((data.x.size(0), 1, self.timesteps+1)).to(x.device)
        self.abs_selfflows = torch.zeros((data.x.size(0), 1, self.timesteps+1)).to(x.device)
        self.deltas = torch.zeros((data.x.size(0), 1, self.timesteps+1)).to(x.device)
        self.inflows = torch.zeros((data.x.size(0), 1, self.timesteps + 1)).to(x.device)

        if self.use_encoder:
            forecast_horizon = range(self.t_context + 1, self.t_context + self.timesteps + 1)
        else:
            forecast_horizon = range(1, self.timesteps + 1)

        for t in forecast_horizon:

            if True: #torch.any(data.local_night[:, t+1] | data.local_dusk[:, t+1]):
                # at least for one radar station it is night or dusk

                r = torch.rand(1)
                if r < teacher_forcing:
                    # if data is available use ground truth, otherwise use model prediction
                    x = data.missing[..., t-1].view(-1, 1) * x + \
                        ~data.missing[..., t-1].view(-1, 1) * data.x[..., t-1].view(-1, 1)

                x, h_t, c_t = self.propagate(edge_index, x=x, coords=coords,
                                                    h_t=h_t, c_t=c_t, areas=data.areas,
                                                    edge_attr=edge_attr, #ground=ground,
                                                    dusk=data.local_dusk[:, t-1],
                                                    dawn=data.local_dawn[:, t],
                                                    env=data.env[..., t],
                                                    t=t-self.t_context,
                                             night=data.local_night[:, t])

                if len(self.fixed_boundary) > 0:
                    # use ground truth for boundary nodes
                    x[self.fixed_boundary, 0] = data.y[self.fixed_boundary, t]


            # for locations where it is dawn: save birds to ground and set birds in the air to zero
            # r = torch.rand(1)
            # if r < teacher_forcing:
            #     ground = ground + data.local_dawn[:, t+1].view(-1, 1) * data.x[..., t+1].view(-1, 1)
            # else:
            #     ground = ground + data.local_dawn[:, t+1].view(-1, 1) * x

            if self.force_zeros:
                x = x * data.local_night[:, t].view(-1, 1)

            # for locations where it is dusk: set birds on ground to zero
            # ground = ground * ~data.local_dusk[:, t].view(-1, 1)

            y_hat.append(x)

        prediction = torch.cat(y_hat, dim=-1)
        return prediction


    def message(self, x_j, coords_i, coords_j, env_i, env_j, edge_attr, t, night_j, dusk_j, dawn_j):
        # construct messages to node i for each edge (j,i)
        # can take any argument initially passed to propagate()
        # x_j are source features with shape [E, out_channels]

        features = torch.cat([coords_i, coords_j, env_i, env_j, edge_attr,
                              night_j.float().view(-1, 1), dusk_j.float().view(-1, 1), dawn_j.float().view(-1, 1)], dim=1)
        if self.n_fc_layers < 1:
            flow = self.edgeflow(features)
        else:
            flow = self.fc_edge_in(features).relu()
            flow = F.dropout(flow, p=self.dropout_p, training=self.training)

            for l in self.fc_edge_hidden:
                flow = l(flow).relu()
                flow = F.dropout(flow, p=self.dropout_p, training=self.training)

            flow = self.fc_edge_out(flow).sigmoid()

        #self.flows.append(flow)
        self.flows[..., t] = flow

        abs_flow = flow * x_j
        #self.abs_flows.append(abs_flow)
        self.abs_flows[..., t] = abs_flow

        return abs_flow


    def update(self, aggr_out, x, coords, env, dusk, dawn, areas, h_t, c_t, t, night):
        if self.recurrent:
            if self.edge_type == 'voronoi':
                inputs = torch.cat([x.view(-1, 1), coords, env, dawn.float().view(-1, 1), #ground.view(-1, 1),
                                    dusk.float().view(-1, 1), areas.view(-1, 1), night.float().view(-1, 1)], dim=1)
            else:
                inputs = torch.cat([x.view(-1, 1), coords, env, dawn.float().view(-1, 1),  # ground.view(-1, 1),
                                    dusk.float().view(-1, 1), night.float().view()], dim=1)
            inputs = self.node2hidden(inputs).relu()

            h_t[0], c_t[0] = self.lstm_layers[0](inputs, (h_t[0], c_t[0]))
            for l in range(1, self.n_lstm_layers):
                h_t[l], c_t[l] = self.lstm_layers[l](h_t[l - 1], (h_t[l], c_t[l]))

            delta = self.hidden2delta(h_t[-1]).tanh()
            #self.deltas.append(delta)
            self.deltas[..., t] = delta
        else:
            delta = 0

        features = torch.cat([coords, env, dusk.float().view(-1, 1), dawn.float().view(-1, 1),
                              night.float().view(-1, 1)], dim=1)
        if self.n_fc_layers < 1:
            selfflow = self.selfflow(features)
        else:
            selfflow = self.fc_self_in(features).relu()
            selfflow = F.dropout(selfflow, p=self.dropout_p, training=self.training)

            for l in self.fc_self_hidden:
                selfflow = l(selfflow).relu()
                selfflow = F.dropout(selfflow, p=self.dropout_p, training=self.training)

            selfflow = self.fc_edge_out(selfflow).sigmoid()

        #self.selfflows.append(selfflow)
        self.selfflows[..., t] = selfflow
        selfflow = x * selfflow
        #self.abs_selfflows.append(selfflow)
        self.abs_selfflows[..., t] = selfflow
        self.inflows[..., t] = aggr_out

        #departure = departure * local_dusk.view(-1, 1) # only use departure model if it is local dusk
        pred = selfflow + aggr_out #+ delta

        return pred, h_t, c_t

class BirdFluxGraphLSTM(MessagePassing):

    def __init__(self, **kwargs):
        super(BirdFluxGraphLSTM, self).__init__(aggr='add', node_dim=0)

        self.timesteps = kwargs.get('timesteps', 40)
        self.dropout_p = kwargs.get('dropout_p', 0)
        self.n_hidden = kwargs.get('n_hidden', 16)
        self.n_env = kwargs.get('n_env', 4)
        self.n_node_in = 6 + self.n_env
        self.n_edge_in = 14 + 2*self.n_env
        self.n_fc_layers = kwargs.get('n_fc_layers', 1)
        self.n_lstm_layers = kwargs.get('n_lstm_layers', 1)
        self.fixed_boundary = kwargs.get('fixed_boundary', [])
        self.force_zeros = kwargs.get('force_zeros', True)

        self.use_encoder = kwargs.get('use_encoder', False)
        self.t_context = kwargs.get('t_context', 0)

        self.edge_type = kwargs.get('edge_type', 'voronoi')
        if self.edge_type == 'voronoi':
            self.n_edge_in += 1 # use face_length as additional feature
            self.n_node_in += 2 # use voronoi cell area and boundary boolean as additional feature

        seed = kwargs.get('seed', 1234)
        torch.manual_seed(seed)


        self.fc_edge_in = torch.nn.Linear(self.n_edge_in, self.n_hidden)
        self.fc_edge_hidden = nn.ModuleList([torch.nn.Linear(self.n_hidden, self.n_hidden)
                                             for _ in range(self.n_fc_layers - 1)])
        self.fc_edge_out = torch.nn.Linear(self.n_hidden, 1)


        self.node2hidden = torch.nn.Sequential(torch.nn.Linear(self.n_node_in, self.n_hidden),
                                               torch.nn.Dropout(p=self.dropout_p),
                                               torch.nn.ReLU(),
                                               torch.nn.Linear(self.n_hidden, self.n_hidden))

        self.lstm_layers = nn.ModuleList([nn.LSTMCell(self.n_hidden, self.n_hidden) for _ in range(self.n_lstm_layers)])

        self.hidden2delta = torch.nn.Sequential(torch.nn.Linear(self.n_hidden, self.n_hidden),
                                                torch.nn.Dropout(p=self.dropout_p),
                                                torch.nn.ReLU(),
                                                torch.nn.Linear(self.n_hidden, 1))

        if self.use_encoder:
            self.encoder = RecurrentEncoder(timesteps=self.t_context, n_env=self.n_env, n_hidden=self.n_hidden,
                                            n_lstm_layers=self.n_lstm_layers, seed=seed, dropout_p=self.dropout_p)



    def forward(self, data, teacher_forcing=0.0):
        # with teacher_forcing = 0.0 the model always uses previous predictions to make new predictions
        # with teacher_forcing = 1.0 the model always uses the ground truth to make new predictions

        self.edges = data.edge_index
        #self.mask_forth = edges[0] < edges[1]
        #self.mask_back = edges[1] < edges[0]


        y_hat = []


        # initialize lstm variables
        if self.use_encoder:
            # push context timeseries through encoder to initialize decoder
            h_t, c_t = self.encoder(data)
            #x = torch.zeros(data.x.size(0)).to(data.x.device) # TODO eventually use this!?
            x = data.x[..., self.t_context].view(-1, 1)
            y_hat.append(x)

        else:
            # start from scratch
            # measurement at t=0
            x = data.x[..., 0].view(-1, 1)
            y_hat.append(x)
            h_t = [torch.zeros(data.x.size(0), self.n_hidden).to(x.device) for l in range(self.n_lstm_layers)]
            c_t = [torch.zeros(data.x.size(0), self.n_hidden).to(x.device) for l in range(self.n_lstm_layers)]


        coords = data.coords
        edge_index = data.edge_index
        edge_attr = data.edge_attr


        self.local_fluxes = torch.zeros((edge_index.size(1), 1, self.timesteps+1)).to(x.device)
        self.fluxes = torch.zeros((data.x.size(0), 1, self.timesteps + 1)).to(x.device)
        self.local_deltas = torch.zeros((data.x.size(0), 1, self.timesteps+1)).to(x.device)

        if self.use_encoder:
            forecast_horizon = range(self.t_context + 1, self.t_context + self.timesteps + 1)
        else:
            forecast_horizon = range(1, self.timesteps + 1)

        for t in forecast_horizon:

            r = torch.rand(1)
            if r < teacher_forcing:
                # if data is available use ground truth, otherwise use model prediction
                x = data.missing[..., t-1].view(-1, 1) * x + \
                    ~data.missing[..., t-1].view(-1, 1) * data.x[..., t-1].view(-1, 1)

            x, h_t, c_t = self.propagate(edge_index, x=x, coords=coords,
                                                h_t=h_t, c_t=c_t, areas=data.areas,
                                                edge_attr=edge_attr,
                                                dusk=data.local_dusk[:, t-1],
                                                dawn=data.local_dawn[:, t],
                                                env=data.env[..., t],
                                                t=t-self.t_context,
                                                boundary=data.boundary,
                                         night=data.local_night[:, t])

            if len(self.fixed_boundary) > 0:
                # use ground truth for boundary nodes
                x[self.fixed_boundary, 0] = data.y[self.fixed_boundary, t]

            if self.force_zeros:
                x = x * data.local_night[:, t].view(-1, 1)

            y_hat.append(x)

        prediction = torch.cat(y_hat, dim=-1)
        return prediction


    def message(self, x_i, x_j, coords_i, coords_j, env_i, env_j, edge_attr, t,
                night_i, night_j, dusk_i, dusk_j, dawn_i, dawn_j):
        # construct messages to node i for each edge (j,i)
        # can take any argument initially passed to propagate()
        # x_j are source features with shape [E, out_channels]


        features = [x_i.view(-1, 1), x_j.view(-1, 1), coords_i, coords_j, env_i, env_j, edge_attr,
                              night_i.float().view(-1, 1), night_j.float().view(-1, 1),
                              dusk_i.float().view(-1, 1), dusk_j.float().view(-1, 1),
                              dawn_i.float().view(-1, 1), dawn_j.float().view(-1, 1)]
        features = torch.cat(features, dim=1)


        flux = self.fc_edge_in(features).relu()
        flux = F.dropout(flux, p=self.dropout_p, training=self.training)

        for l in self.fc_edge_hidden:
            flux = l(flux).relu()
            flux = F.dropout(flux, p=self.dropout_p, training=self.training)

        flux = self.fc_edge_out(flux) #.tanh()

        # # enforce fluxes to be symmetric along edges
        # A_flux = to_dense_adj(self.edges, edge_attr=flux).squeeze()
        # A_flux = torch.triu(A_flux, diagonal=1) # values on diagonal are zero
        # A_flux = A_flux - A_flux.T
        # edge_index, flux = dense_to_sparse(A_flux)
        # flux = flux.view(-1, 1)
        # #flux[self.mask_back] = - flux[self.mask_forth]

        self.local_fluxes[..., t] = flux

        return flux


    def update(self, aggr_out, x, coords, env, dusk, dawn, areas, h_t, c_t, t, night, boundary):

        if self.edge_type == 'voronoi':
            inputs = torch.cat([x.view(-1, 1), coords, env, dawn.float().view(-1, 1), #ground.view(-1, 1),
                                dusk.float().view(-1, 1), areas.view(-1, 1), night.float().view(-1, 1),
                                boundary.float().view(-1, 1)], dim=1)
        else:
            inputs = torch.cat([x.view(-1, 1), coords, env, dawn.float().view(-1, 1),  # ground.view(-1, 1),
                                dusk.float().view(-1, 1), night.float().view()], dim=1)
        inputs = self.node2hidden(inputs).relu()

        h_t[0], c_t[0] = self.lstm_layers[0](inputs, (h_t[0], c_t[0]))
        for l in range(1, self.n_lstm_layers):
            h_t[l], c_t[l] = self.lstm_layers[l](h_t[l - 1], (h_t[l], c_t[l]))

        delta = self.hidden2delta(h_t[-1]).tanh()
        self.local_deltas[..., t] = delta

        self.fluxes[..., t] = aggr_out
        pred = x + delta + ~boundary.view(-1, 1) * aggr_out # take messages into account for inner cells only

        return pred, h_t, c_t


class RecurrentEncoder(torch.nn.Module):
    def __init__(self, **kwargs):
        super(RecurrentEncoder, self).__init__()

        self.timesteps = kwargs.get('timesteps', 12)
        self.n_in = 3 + kwargs.get('n_env', 4)
        self.n_hidden = kwargs.get('n_hidden', 16)
        self.n_lstm_layers = kwargs.get('n_layers_lstm', 1)
        self.dropout_p = kwargs.get('dropout_p', 0)

        torch.manual_seed(kwargs.get('seed', 1234))

        self.node2hidden = torch.nn.Sequential(torch.nn.Linear(self.n_in, self.n_hidden),
                                               torch.nn.Dropout(p=self.dropout_p),
                                               torch.nn.ReLU(),
                                               torch.nn.Linear(self.n_hidden, self.n_hidden))

        self.lstm_layers = nn.ModuleList([nn.LSTMCell(self.n_hidden, self.n_hidden) for _ in range(self.n_lstm_layers)])

    def forward(self, data):
        # initialize lstm variables
        h_t = [torch.zeros(data.x.size(0), self.n_hidden).to(data.x.device) for l in range(self.n_lstm_layers)]
        c_t = [torch.zeros(data.x.size(0), self.n_hidden).to(data.x.device) for l in range(self.n_lstm_layers)]

        for t in range(self.timesteps):

            inputs = torch.cat([data.env[..., t], data.coords, data.x[:, t].view(-1, 1)], dim=1)
            inputs = self.node2hidden(inputs).relu()
            h_t[0], c_t[0] = self.lstm_layers[0](inputs, (h_t[0], c_t[0]))
            for l in range(1, self.n_lstm_layers):
                h_t[l], c_t[l] = self.lstm_layers[l](h_t[l - 1], (h_t[l], c_t[l]))

        return h_t, c_t


class BirdDynamicsGraphLSTM(MessagePassing):

    def __init__(self, **kwargs):
        super(BirdDynamicsGraphLSTM, self).__init__(aggr='add', node_dim=0) # inflows from neighbouring radars are aggregated by adding

        self.timesteps = kwargs.get('timesteps', 40)
        self.dropout_p = kwargs.get('dropout_p', 0)
        self.n_hidden = kwargs.get('n_hidden', 16)
        self.n_env = kwargs.get('n_env', 4)
        self.n_node_in = 6 + self.n_env
        self.n_edge_in = 11 + 2 * self.n_env
        self.n_fc_layers = kwargs.get('n_fc_layers', 1)
        self.n_lstm_layers = kwargs.get('n_lstm_layers', 1)
        self.predict_delta = kwargs.get('predict_delta', True)
        self.fixed_boundary = kwargs.get('fixed_boundary', [])
        self.force_zeros = kwargs.get('force_zeros', [])
        self.use_encoder = kwargs.get('use_encoder', False)
        self.t_context = kwargs.get('t_context', 0)

        self.edge_type = kwargs.get('edge_type', 'voronoi')
        if self.edge_type == 'voronoi':
            print('Use Voronoi tessellation')
            self.n_edge_in += 1  # use face_length as additional feature
            self.n_node_in += 1  # use voronoi cell area as additional feature

        seed = kwargs.get('seed', 1234)
        torch.manual_seed(seed)


        self.fc_edge_in = torch.nn.Linear(self.n_edge_in, self.n_hidden)
        self.fc_edge_hidden = nn.ModuleList([torch.nn.Linear(self.n_hidden, self.n_hidden)
                                        for _ in range(self.n_fc_layers - 1)])
        self.fc_edge_out = torch.nn.Linear(self.n_hidden, self.n_hidden)

        # self.mlp_edge = torch.nn.Sequential(torch.nn.Linear(self.n_in_edge, self.n_hidden),
        #                                     torch.nn.Dropout(p=self.dropout_p),
        #                                     torch.nn.ReLU(),
        #                                     torch.nn.Linear(self.n_hidden, self.n_hidden))

        self.mlp_aggr = torch.nn.Sequential(torch.nn.Linear(self.n_hidden, self.n_hidden),
                                            torch.nn.Dropout(p=self.dropout_p),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(self.n_hidden, 1))

        self.node2hidden = torch.nn.Sequential(torch.nn.Linear(self.n_node_in, self.n_hidden),
                                               torch.nn.Dropout(p=self.dropout_p),
                                               torch.nn.ReLU(),
                                               torch.nn.Linear(self.n_hidden, self.n_hidden))

        self.lstm_layers = nn.ModuleList([nn.LSTMCell(self.n_hidden, self.n_hidden) for _ in range(self.n_lstm_layers)])

        self.hidden2delta = torch.nn.Sequential(torch.nn.Linear(self.n_hidden, self.n_hidden),
                                               torch.nn.Dropout(p=self.dropout_p),
                                               torch.nn.ReLU(),
                                               torch.nn.Linear(self.n_hidden, 1))

        if self.use_encoder:
            self.encoder = RecurrentEncoder(timesteps=self.t_context, n_env=self.n_env, n_hidden=self.n_hidden,
                                            n_lstm_layers=self.n_lstm_layers, seed=seed, dropout_p=self.dropout_p)



    def forward(self, data, teacher_forcing=0.0):
        # with teacher_forcing = 0.0 the model always uses previous predictions to make new predictions
        # with teacher_forcing = 1.0 the model always uses the ground truth to make new predictions

        y_hat = []

        # initialize lstm variables
        if self.use_encoder:
            # push context timeseries through encoder to initialize decoder
            h_t, c_t = self.encoder(data)
            #x = torch.zeros(data.x.size(0)).to(data.x.device) # TODO eventually use this!?
            x = data.x[..., self.t_context].view(-1, 1)
            y_hat.append(x)
        else:
            # start from scratch
            # measurement at t=0
            x = data.x[..., 0].view(-1, 1)
            y_hat.append(x)
            h_t = [torch.zeros(data.x.size(0), self.n_hidden).to(x.device) for l in range(self.n_lstm_layers)]
            c_t = [torch.zeros(data.x.size(0), self.n_hidden).to(x.device) for l in range(self.n_lstm_layers)]

        coords = data.coords
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        self.fluxes = torch.zeros((data.x.size(0), 1, self.timesteps + 1)).to(x.device)
        self.local_deltas = torch.zeros((data.x.size(0), 1, self.timesteps + 1)).to(x.device)

        if self.use_encoder:
            forecast_horizon = range(self.t_context + 1, self.t_context + self.timesteps + 1)
        else:
            forecast_horizon = range(1, self.timesteps + 1)

        for t in forecast_horizon:

            if True: #torch.any(data.local_night[:, t] | data.local_dusk[:, t]):
                # at least for one radar station it is night or dusk
                r = torch.rand(1)
                if r < teacher_forcing:
                    # if data is available use ground truth, otherwise use model prediction
                    x = data.missing[..., t-1].view(-1, 1) * x + \
                        ~data.missing[..., t-1].view(-1, 1) * data.x[..., t-1].view(-1, 1)


                x, h_t, c_t = self.propagate(edge_index, x=x, coords=coords,
                                   edge_attr=edge_attr, h_t=h_t, c_t=c_t, areas=data.areas,
                                   dusk=data.local_dusk[:, t-1],
                                   dawn=data.local_dawn[:, t],
                                   env=data.env[..., t],
                                   night=data.local_night[:, t],
                                   t=t-self.t_context)

                if len(self.fixed_boundary) > 0:
                    # use ground truth for boundary cells
                    x[self.fixed_boundary, 0] = data.y[self.fixed_boundary, t]

            if self.force_zeros:
                # for locations where it is dawn: set birds in the air to zero
                x = x * data.local_night[:, t].view(-1, 1)

            y_hat.append(x)

        prediction = torch.cat(y_hat, dim=-1)
        return prediction


    def message(self, x_i, x_j, coords_i, coords_j, env_i, env_j, edge_attr, dusk_j, dawn_j, night_j):
        # construct messages to node i for each edge (j,i)
        # can take any argument initially passed to propagate()
        # x_j are source features with shape [E, out_channels]

        features = torch.cat([x_i.view(-1, 1), x_j.view(-1, 1), coords_i, coords_j, env_i, env_j, edge_attr,
                              dusk_j.float().view(-1, 1), dawn_j.float().view(-1, 1), night_j.float().view(-1, 1)], dim=1)
        #msg = self.mlp_edge(features).relu()

        msg = self.fc_edge_in(features).relu()
        msg = F.dropout(msg, p=self.dropout_p, training=self.training)

        for l in self.fc_edge_hidden:
            msg = l(msg).relu()
            msg = F.dropout(msg, p=self.dropout_p, training=self.training)

        msg = self.fc_edge_out(msg) #.relu()

        return msg


    def update(self, aggr_out, x, coords, env, areas, dusk, dawn, h_t, c_t, t, night):

        # predict departure/landing
        if self.edge_type == 'voronoi':
            inputs = torch.cat([x.view(-1, 1), coords, env, dusk.float().view(-1, 1),
                                dawn.float().view(-1, 1), areas.view(-1, 1), night.float().view(-1, 1)], dim=1)
        else:
            inputs = torch.cat([x.view(-1, 1), coords, env, dusk.float().view(-1, 1),
                                dawn.float().view(-1, 1), night.float().view(-1, 1)], dim=1)
        inputs = self.node2hidden(inputs).relu()
        h_t[0], c_t[0] = self.lstm_layers[0](inputs, (h_t[0], c_t[0]))
        for l in range(1, self.n_lstm_layers):
            h_t[l], c_t[l] = self.lstm_layers[l](h_t[l - 1], (h_t[l], c_t[l]))
        delta = self.hidden2delta(h_t[-1]) #.tanh()

        if self.predict_delta:
            # combine messages from neighbors into single number representing total flux
            flux = self.mlp_aggr(aggr_out) #.tanh()
            pred = x + flux + delta
        else:
            # combine messages from neighbors into single number representing the new bird density
            birds = self.mlp_aggr(aggr_out).sigmoid()
            pred = birds + delta

        self.fluxes[..., t] = flux
        self.local_deltas[..., t] = delta

        return pred, h_t, c_t



class BirdDynamicsGraphLSTM_transformed(MessagePassing):

    def __init__(self, **kwargs):
        super(BirdDynamicsGraphLSTM_transformed, self).__init__(aggr='add', node_dim=0)

        self.timesteps = kwargs.get('timesteps', 40)
        self.dropout_p = kwargs.get('dropout_p', 0)
        self.n_hidden = kwargs.get('n_hidden', 16)
        self.n_node_in = 5 + kwargs.get('n_env', 4)
        self.n_edge_in = 8 + 2*kwargs.get('n_env', 4)
        self.n_fc_layers = kwargs.get('n_fc_layers', 1)
        self.n_lstm_layers = kwargs.get('n_lstm_layers', 1)
        self.fixed_boundary = kwargs.get('fixed_boundary', [])
        self.forced_zeros = kwargs.get('forced_zeros', [])

        self.edge_type = kwargs.get('edge_type', 'voronoi')
        if self.edge_type == 'voronoi':
            self.n_node_in += 1
            self.n_edge_in += 1

        torch.manual_seed(kwargs.get('seed', 1234))


        self.fc_edge_in = torch.nn.Linear(self.n_edge_in, self.n_hidden)
        self.fc_edge_hidden = nn.ModuleList([torch.nn.Linear(self.n_hidden, self.n_hidden)
                                        for _ in range(self.n_fc_layers - 1)])
        self.fc_edge_out = torch.nn.Linear(self.n_hidden, self.n_hidden)


        self.node2hidden = torch.nn.Sequential(torch.nn.Linear(self.n_node_in, self.n_hidden),
                                               torch.nn.Dropout(p=self.dropout_p),
                                               torch.nn.ReLU(),
                                               torch.nn.Linear(self.n_hidden, self.n_hidden))

        self.lstm_layers = nn.ModuleList([nn.LSTMCell(self.n_hidden, self.n_hidden) for _ in range(self.n_lstm_layers)])

        self.hidden2birds = torch.nn.Sequential(torch.nn.Linear(2*self.n_hidden, self.n_hidden),
                                               torch.nn.Dropout(p=self.dropout_p),
                                               torch.nn.ReLU(),
                                               torch.nn.Linear(self.n_hidden, 1))



    def forward(self, data, teacher_forcing=0.0):
        # with teacher_forcing = 0.0 the model always uses previous predictions to make new predictions
        # with teacher_forcing = 1.0 the model always uses the ground truth to make new predictions

        # measurement at t=0
        x = data.x[..., 0].view(-1, 1)

        # initialize lstm variables
        h_t = [torch.zeros(data.x.size(0), self.n_hidden).to(x.device) for l in range(self.n_lstm_layers)]
        c_t = [torch.zeros(data.x.size(0), self.n_hidden).to(x.device) for l in range(self.n_lstm_layers)]

        coords = data.coords
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        y_hat = []
        y_hat.append(x)

        for t in range(self.timesteps):

            if True: #torch.any(data.local_night[:, t+1] | data.local_dusk[:, t+1]):
                # at least for one radar station it is night or dusk
                r = torch.rand(1)
                if r < teacher_forcing:
                    # if data is available use ground truth, otherwise use model prediction
                    x = data.missing[..., t].view(-1, 1) * x + \
                        ~data.missing[..., t].view(-1, 1) * data.x[..., t].view(-1, 1)

                x, h_t, c_t = self.propagate(edge_index, x=x, coords=coords,
                                             edge_attr=edge_attr, h_t=h_t, c_t=c_t, areas=data.areas,
                                             dusk=data.local_dusk[:, t],
                                             dawn=data.local_dawn[:, t+1],
                                             env=data.env[..., t+1])

                if len(self.fixed_boundary) > 0:
                    # use ground truth for boundary cells
                    x[self.fixed_boundary, 0] = data.y[self.fixed_boundary, t+1]

            if self.forced_zeros:
                # for locations where it is dawn: set birds in the air to zero
                x = x * data.local_night[:, t+1].view(-1, 1)

            y_hat.append(x)

        prediction = torch.cat(y_hat, dim=-1)
        return prediction


    def message(self, x_i, x_j, coords_i, coords_j, env_i, env_j, edge_attr):
        # construct messages to node i for each edge (j,i)
        # can take any argument initially passed to propagate()
        # x_j are source features with shape [E, out_channels]

        features = torch.cat([x_i.view(-1, 1), x_j.view(-1, 1), coords_i, coords_j, env_i, env_j, edge_attr], dim=1)
        #msg = self.mlp_edge(features).relu()

        msg = self.fc_edge_in(features).relu()
        msg = F.dropout(msg, p=self.dropout_p, training=self.training)

        for l in self.fc_edge_hidden:
            msg = l(msg).relu()
            msg = F.dropout(msg, p=self.dropout_p, training=self.training)

        msg = self.fc_edge_out(msg).relu()

        return msg


    def update(self, aggr_out, x, coords, env, areas, dusk, dawn, h_t, c_t):

        # recurrent component
        if self.edge_type == 'voronoi':
            inputs = torch.cat([x.view(-1, 1), coords, env, dusk.float().view(-1, 1),
                                dawn.float().view(-1, 1), areas.view(-1, 1)], dim=1)
        else:
            inputs = torch.cat([x.view(-1, 1), coords, env, dusk.float().view(-1, 1),
                                dawn.float().view(-1, 1)], dim=1)
        inputs = self.node2hidden(inputs).relu()
        h_t[0], c_t[0] = self.lstm_layers[0](inputs, (h_t[0], c_t[0]))
        for l in range(1, self.n_lstm_layers):
            h_t[l], c_t[l] = self.lstm_layers[l](h_t[l - 1], (h_t[l], c_t[l]))

        # combine messages from neighbors and recurrent module into single number representing the new bird density
        pred = self.hidden2birds(torch.cat([aggr_out, h_t[-1]], dim=-1)).sigmoid()

        return pred, h_t, c_t



class BirdDynamicsGraphGRU(MessagePassing):

    def __init__(self, msg_n_in=17, node_n_in=8, n_out=1, n_hidden=16, timesteps=6,
                 seed=12345, multinight=False, use_wind=True, dropout_p=0):
        super(BirdDynamicsGraphGRU, self).__init__(aggr='add', node_dim=0) # inflows from neighbouring radars are aggregated by adding

        torch.manual_seed(seed)

        if not use_wind:
            msg_n_in -= 2

        if not use_wind:
            node_n_in -= 2

        self.msg_nn = nn.Linear(msg_n_in, n_hidden)

        self.hidden_r = nn.Linear(n_hidden, n_hidden, bias=False)
        self.hidden_i = nn.Linear(n_hidden, n_hidden, bias=False)
        self.hidden_h = nn.Linear(n_hidden, n_hidden, bias=False)

        self.input_r = nn.Linear(node_n_in, n_hidden, bias=True)
        self.input_i = nn.Linear(node_n_in, n_hidden, bias=True)
        self.input_n = nn.Linear(node_n_in, n_hidden, bias=True)

        self.out_fc1 = nn.Linear(n_hidden, n_hidden)
        self.out_fc2 = nn.Linear(n_hidden, n_hidden)
        self.out_fc3 = nn.Linear(n_hidden, n_out)


        self.timesteps = timesteps
        self.multinight = multinight
        self.use_wind = use_wind
        self.dropout_p = dropout_p
        self.n_hidden = n_hidden


    def forward(self, data, teacher_forcing=0.0):
        # with teacher_forcing = 0.0 the model always uses previous predictions to make new predictions
        # with teacher_forcing = 1.0 the model always uses the ground truth to make new predictions

        # measurement at t=0
        x = data.x[..., 0].view(-1, 1)

        # birds on ground at t=0
        ground = torch.zeros_like(x)

        # initialize hidden variables
        hidden = Variable(torch.zeros(data.x.size(0),  self.n_hidden))
        if x.is_cuda:
            hidden = hidden.cuda()


        coords = data.coords
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        y_hat = []
        y_hat.append(x)

        for t in range(self.timesteps):
            r = torch.rand(1)
            if r < teacher_forcing:
                x = data.x[..., t].view(-1, 1)

            env = data.env[..., t]
            if not self.use_wind:
                env = env[:, 2:]

            x, hidden = self.propagate(edge_index, x=x, coords=coords, env=env, hidden=hidden,
                                       dusk=data.local_dusk[:, t], edge_attr=edge_attr,
                                       local_night=data.local_night[:, t], areas=data.areas)


            if self.multinight:
                # for locations where it is dawn: save birds to ground and set birds in the air to zero
                # r = torch.rand(1)
                # if r < teacher_forcing:
                #     ground = ground + data.local_dawn[:, t+1].view(-1, 1) * data.x[..., t+1].view(-1, 1)
                # else:
                #     ground = ground + data.local_dawn[:, t+1].view(-1, 1) * x
                x = x * data.local_night[:, t+1].view(-1, 1)

                # TODO for radar data, birds can stay on the ground or depart later in the night, so
                #  at dusk birds on ground shouldn't be set to zero but predicted departing birds should be subtracted
                # for locations where it is dusk: set birds on ground to zero
                # ground = ground * ~data.local_dusk[:, t].view(-1, 1)

            y_hat.append(x)

        prediction = torch.cat(y_hat, dim=-1)
        return prediction


    def message(self, x_i, x_j, coords_i, coords_j, env_i, env_j, local_night_j, edge_attr):
        # construct messages to node i for each edge (j,i)
        # can take any argument initially passed to propagate()
        # x_j are source features with shape [E, out_channels]

        features = torch.cat([x_i.view(-1, 1), x_j.view(-1, 1), coords_i, coords_j, env_i, env_j,
                              local_night_j.view(-1, 1), edge_attr], dim=1)
        msg = self.msg_nn(features)

        return msg


    def update(self, aggr_out, x, coords, env, dusk, areas, hidden):

        inputs = torch.cat([coords, env, dusk.float().view(-1, 1), areas.view(-1, 1)], dim=1)

        # GRU-style gated aggregation
        r = torch.sigmoid(self.input_r(inputs) + self.hidden_r(aggr_out))
        i = torch.sigmoid(self.input_i(inputs) + self.hidden_i(aggr_out))
        n = torch.tanh(self.input_n(inputs) + r * self.hidden_h(aggr_out))
        hidden = (1 - i) * n + i * hidden

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(hidden)), p=self.dropout_p)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_p)
        pred = self.out_fc3(pred)

        # Predict bird difference
        pred = x + pred

        return pred, hidden



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



def train_flows(model, train_loader, optimizer, loss_func, device, boundaries, conservation_constraint=0.01,
                 teacher_forcing=1.0, daymask=True):
    model.to(device)
    model.train()
    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data, teacher_forcing) #.view(-1)
        gt = data.y

        outfluxes = to_dense_adj(data.edge_index, edge_attr=model.flows).view(
                                    data.num_nodes, data.num_nodes, -1).sum(1)
        outfluxes = outfluxes + model.selfflows
        outfluxes = torch.stack([outfluxes[node] for node in range(data.num_nodes) if not boundaries[node]])
        target_fluxes = torch.ones(outfluxes.shape)
        target_fluxes = target_fluxes.to(device)
        constraints = torch.mean((outfluxes - target_fluxes)**2)
        if daymask:
            mask = data.local_night & ~data.missing
        else:
            mask = ~data.missing
        if hasattr(model, 't_context'):
            gt = gt[:, model.t_context:]
            mask = mask[:, model.t_context:]
        loss = loss_func(output, gt, mask) + conservation_constraint * constraints

        loss.backward()
        loss_all += data.num_graphs * loss
        optimizer.step()

    return loss_all

def train_fluxes(model, train_loader, optimizer, loss_func, device, conservation_constraint=0.01,
                 teacher_forcing=1.0, daymask=True):
    model.to(device)
    model.train()
    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data, teacher_forcing) #.view(-1)
        gt = data.y

        fluxes = to_dense_adj(data.edge_index, edge_attr=model.local_fluxes).view(
                                    data.num_nodes, data.num_nodes, -1).sum(1)
        reverse_fluxes = fluxes.permute(1, 0, 2)
        deltas = fluxes + reverse_fluxes
        target = torch.zeros(deltas.shape).to(device)

        constraints = torch.mean((deltas - target)**2)
        if daymask:
            mask = data.local_night & ~data.missing
        else:
            mask = ~data.missing
        if hasattr(model, 't_context'):
            gt = gt[:, model.t_context:]
            mask = mask[:, model.t_context:]
        loss = loss_func(output, gt, mask) + conservation_constraint * constraints

        loss.backward()
        loss_all += data.num_graphs * loss
        optimizer.step()

    return loss_all


def train_dynamics(model, train_loader, optimizer, loss_func, device, teacher_forcing=0, daymask=True):
    model.to(device)
    model.train()
    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        output = model(data, teacher_forcing=teacher_forcing)
        gt = data.y

        if daymask:
            mask = data.local_night & ~data.missing
        else:
            mask = ~data.missing
        if hasattr(model, 't_context'):
            gt = gt[:, model.t_context:]
            mask = mask[:, model.t_context:]
        loss = loss_func(output, gt, mask)
        loss.backward()
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

        # use only data at t=0
        gt = data.y[..., 0]
        loss = loss_func(output, gt)
        loss.backward()
        loss_all += data.num_graphs * loss
        optimizer.step()

    return loss_all

def test_flows(model, test_loader, loss_func, device, get_outfluxes=True, bird_scale=1,
                fixed_boundary=[], daymask=True):
    model.to(device)
    model.eval()
    loss_all = []
    outfluxes = {}
    outfluxes_abs = {}
    for tidx, data in enumerate(test_loader):
        data = data.to(device)
        output = model(data) * bird_scale #.view(-1)
        gt = data.y * bird_scale

        if len(fixed_boundary) > 0:
            boundary_mask = np.ones(output.size(0))
            boundary_mask[fixed_boundary] = 0
            output = output[boundary_mask]
            gt = gt[boundary_mask]

        if get_outfluxes:
            outfluxes[tidx] = to_dense_adj(data.edge_index, edge_attr=model.flows).view(
                                    data.num_nodes, data.num_nodes, -1)
            outfluxes_abs[tidx] = to_dense_adj(data.edge_index, edge_attr=model.abs_flows).view(
                                    data.num_nodes, data.num_nodes, -1)# .sum(1)

            outfluxes[tidx] = outfluxes[tidx].cpu()
            outfluxes_abs[tidx] = outfluxes_abs[tidx].cpu()
            #constraints = torch.mean((outfluxes - torch.ones(data.num_nodes)) ** 2)

        if daymask:
            mask = data.local_night & ~data.missing
        else:
            mask = ~data.missing
        if hasattr(model, 't_context'):
            gt = gt[:, model.t_context:]
            mask = mask[:, model.t_context:]
        loss_all.append(torch.tensor([loss_func(output[:, t], gt[:, t], mask[:, t])
                                      for t in range(model.timesteps + 1)]))
        #loss_all.append(loss_func(output, gt))
        #constraints_all.append(constraints)

    if get_outfluxes:
        return torch.stack(loss_all), outfluxes , outfluxes_abs
    else:
        return torch.stack(loss_all)

def test_fluxes(model, test_loader, loss_func, device, get_fluxes=True, bird_scale=1,
                fixed_boundary=[], daymask=True):
    model.to(device)
    model.eval()
    loss_all = []
    fluxes = {}

    for tidx, data in enumerate(test_loader):
        data = data.to(device)
        output = model(data) * bird_scale #.view(-1)
        gt = data.y * bird_scale

        if len(fixed_boundary) > 0:
            boundary_mask = np.ones(output.size(0))
            boundary_mask[fixed_boundary] = 0
            output = output[boundary_mask]
            gt = gt[boundary_mask]

        if get_fluxes:
            fluxes[tidx] = to_dense_adj(data.edge_index, edge_attr=model.fluxes).view(
                                    data.num_nodes, data.num_nodes, -1).cpu()

        if daymask:
            mask = data.local_night & ~data.missing
        else:
            mask = ~data.missing
        if hasattr(model, 't_context'):
            gt = gt[:, model.t_context:]
            mask = mask[:, model.t_context:]
        loss_all.append(torch.tensor([loss_func(output[:, t], gt[:, t], mask[:, t])
                                      for t in range(model.timesteps + 1)]))

    if get_fluxes:
        return torch.stack(loss_all), fluxes
    else:
        return torch.stack(loss_all)

def test_dynamics(model, test_loader, loss_func, device, bird_scale=2000, daymask=True):
    model.to(device)
    model.eval()
    loss_all = []

    for nidx, data in enumerate(test_loader):
        data = data.to(device)
        output = model(data) * bird_scale #.view(-1)
        gt = data.y * bird_scale

        if daymask:
            mask = data.local_night & ~data.missing
        else:
            mask = ~data.missing
        if hasattr(model, 't_context'):
            gt = gt[:, model.t_context:]
            mask = mask[:, model.t_context:]
        loss_all.append(torch.tensor([loss_func(output[:, t], gt[:, t], mask[:, t])
                                      for t in range(model.timesteps + 1)]))

    return torch.stack(loss_all)

def predict_dynamics(model, test_loader, device, bird_scale=2000):
    model.to(device)
    model.eval()
    gt = []
    pred = []

    for nidx, data in enumerate(test_loader):
        data = data.to(device)
        gt.append(data.y * bird_scale)
        pred.append(model(data) * bird_scale)

    gt = torch.stack(gt, dim=0) # shape (nights, radars, timesteps)
    pred = torch.stack(pred, dim=0) # shape (nights, radars, timesteps)

    return gt, pred

def test_departure(model, test_loader, loss_func, cuda, bird_scale=2000):
    if cuda:
        model.cuda()
    model.eval()
    loss_all = []
    for tidx, data in enumerate(test_loader):
        if cuda: data = data.to('cuda')
        output = model(data).view(-1) * bird_scale
        gt = data.y[..., 0] * bird_scale

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



