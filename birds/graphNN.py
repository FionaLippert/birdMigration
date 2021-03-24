import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader, Dataset, InMemoryDataset
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, to_dense_adj
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
    def __init__(self, in_channels, hidden_channels, out_channels, timesteps, seed=1234):
        super(LSTM, self).__init__()

        torch.manual_seed(seed)

        self.lstm = torch.nn.LSTMCell(in_channels, hidden_channels)
        self.hidden2birds = torch.nn.Sequential(torch.nn.Linear(hidden_channels, out_channels),
                                                torch.nn.Sigmoid())
        self.timesteps = timesteps
        self.hidden_channels = hidden_channels


    def forward(self, data, teacher_forcing=0):

        x = data.x[:, 0]
        states = torch.zeros(1, self.hidden_channels)
        hidden = torch.zeros(1, self.hidden_channels)
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
            states, hidden = self.lstm(inputs, (states, hidden))

            x = self.hidden2birds(states).view(-1)

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

        self.lin_in = torch.nn.Linear(in_channels, hidden_channels)
        self.lin_hidden = [torch.nn.Linear(hidden_channels, hidden_channels) for _ in range(n_layers - 1)]
        self.lin_out = torch.nn.Linear(hidden_channels, out_channels)
        self.timesteps = timesteps
        self.dropout_p = dropout_p

    def forward(self, data):

        x = data.x[..., 0]

        y_hat = []
        for t in range(self.timesteps + 1):

            features = torch.cat([data.coords.flatten(),
                                  data.env[..., t].flatten()], dim=0)
            x = self.lin_in(features)
            x = x.sigmoid()
            x = x.relu()
            x = torch.nn.functional.dropout(x, p=self.dropout_p, training=self.training)

            for l in self.lin_hidden:
                x = l(x)
                x = x.relu()
                x = torch.nn.functional.dropout(x, p=self.dropout_p, training=self.training)

            x = self.lin_out(x)
            x = x.sigmoid()

            # for locations where it is night: set birds in the air to zero
            x = x * data.local_night[:, t]

            y_hat.append(x)

        return torch.stack(y_hat, dim=1)


class NodeMLP(MessagePassing):

    def __init__(self, node_n_in=6, n_hidden=16, n_out=1, timesteps=6, n_layers=2, dropout_p=0, seed=1234):
        super(NodeMLP, self).__init__(aggr='add', node_dim=0)

        torch.manual_seed(seed)

        self.lin_in = torch.nn.Linear(node_n_in, n_hidden)
        self.lin_hidden = [torch.nn.Linear(n_hidden, n_hidden) for _ in range(n_layers - 1)]
        self.lin_out = torch.nn.Linear(n_hidden, n_out)

        self.timesteps = timesteps
        self.dropout_p = dropout_p
        self.n_hidden = n_hidden


    def forward(self, data):

        y_hat = []

        for t in range(self.timesteps + 1):

            x = self.propagate(data.edge_index, coords=data.coords, env=data.env[..., t], edge_attr=data.edge_attr)

            # for locations where it is night: set birds in the air to zero
            x = x * data.local_night[:, t].view(-1, 1)

            y_hat.append(x)

        prediction = torch.cat(y_hat, dim=-1)
        return prediction


    def message(self, edge_attr):
        # set all messages to 0 --> no spatial dependencies
        n_edges = edge_attr.size(0)
        msg = torch.zeros(n_edges)
        return msg


    def update(self, aggr_out, coords, env):
        # use only location-specific features to predict migration intensities
        features = torch.cat([coords, env], dim=1)
        x = self.lin_in(features)
        x = x.sigmoid()
        x = x.relu()
        x = torch.nn.functional.dropout(x, p=self.dropout_p, training=self.training)

        for l in self.lin_hidden:
            x = l(x)
            x = x.relu()
            x = torch.nn.functional.dropout(x, p=self.dropout_p, training=self.training)

        x = self.lin_out(x)
        x = x.sigmoid()

        return x

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


class BirdFlowTime(MessagePassing):

    def __init__(self, num_nodes, timesteps, hidden_dim=16, embedding=0, model='linear', norm=True,
                 use_departure=False, seed=12345, fix_boundary=[], multinight=False, use_wind=True, dropout_p=0.5):
        super(BirdFlowTime, self).__init__(aggr='add', node_dim=0) # inflows from neighbouring radars are aggregated by adding

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
                                                 torch.nn.Sigmoid())
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
                                                 torch.nn.Sigmoid())

        #if use_departure:
        # self.departure = torch.nn.Sequential(torch.nn.Linear(in_channels_dep, hidden_channels_dep),
        #                                      torch.nn.ReLU(),
        #                                      torch.nn.Linear(hidden_channels_dep, out_channels_dep),
        #                                      torch.nn.Sigmoid())
            #                                      #torch.nn.Tanh())
            # self.departure = torch.nn.Sequential(torch.nn.Linear(in_channels_dep, out_channels_dep),
            #                                     torch.nn.Sigmoid())

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
        # if self.use_departure:
        #     features = torch.cat([coords, data.env[..., 0]], dim=1)
        #     x = self.departure(features)
        #
        #     if len(self.fix_boundary) > 0:
        #         x[self.fix_boundary] = data[self.fix_boundary, 0]
        y_hat.append(x)

        self.flows = []
        self.abs_flows = []
        for t in range(self.timesteps):
            r = torch.rand(1)
            if r < teacher_forcing:
                x = data.x[..., t].view(-1, 1)



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


    def update(self, aggr_out, coords, env, ground, local_dusk):
        # return aggregation (sum) of inflows computed by message()
        # add departure prediction if local_dusk flag is True

        if self.multinight:
            features = torch.cat([coords, env, ground], dim=1)
            departure = self.departure(features)
            departure = departure * local_dusk.view(-1, 1) # only use departure model if it is local dusk
            pred = aggr_out + departure
        else:
            pred = aggr_out

        return pred



class BirdFlowRecurrent(MessagePassing):

    def __init__(self, timesteps, hidden_dim=16, model='linear',
                 seed=12345, fix_boundary=[], multinight=False, use_wind=True, dropout_p=0.5):
        super(BirdFlowRecurrent, self).__init__(aggr='add', node_dim=0) # inflows from neighbouring radars are aggregated by adding

        torch.manual_seed(seed)

        edges_n_in = 10
        if not use_wind:
            edges_n_in -= 2
        n_hidden = hidden_dim #16 #2*in_channels #int(in_channels / 2)
        n_out = 1

        nodes_n_in = 8
        if not use_wind:
            nodes_n_in -= 2

        if model == 'linear':
            self.edgeflow = torch.nn.Linear(edges_n_in, n_out)
        elif model == 'linear+sigmoid':
            self.edgeflow = torch.nn.Sequential(torch.nn.Linear(edges_n_in, n_out),
                                                torch.nn.Sigmoid())

        else:
            self.edgeflow = torch.nn.Sequential(torch.nn.Linear(edges_n_in, n_hidden),
                                                torch.nn.Dropout(p=dropout_p),
                                                torch.nn.ReLU(),
                                                torch.nn.Linear(n_hidden, n_out),
                                                torch.nn.Sigmoid())

        self.departure = torch.nn.Sequential(torch.nn.Linear(n_hidden, n_out),
                                             torch.nn.Tanh())

        self.node_lstm = nn.LSTMCell(nodes_n_in, n_hidden)

        self.timesteps = timesteps
        self.fix_boundary = fix_boundary
        self.multinight = multinight
        self.use_wind = use_wind
        self.n_hidden = n_hidden


    def forward(self, data, teacher_forcing=0.0):
        # with teacher_forcing = 0.0 the model always uses previous predictions to make new predictions
        # with teacher_forcing = 1.0 the model always uses the ground truth to make new predictions

        # measurement at t=0
        x = data.x[..., 0].view(-1, 1)

        # birds on the ground at t=0
        ground = torch.zeros_like(x)

        # initialize lstm variables
        hidden = Variable(torch.zeros(data.x.size(0), self.n_hidden)).to(x.device)
        states = Variable(torch.zeros(data.x.size(0), self.n_hidden)).to(x.device)
        # if x.is_cuda:
        #     hidden = hidden.cuda()
        #     states = states.cuda()

        coords = data.coords
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        y_hat = []
        y_hat.append(x)

        self.flows = []
        self.abs_flows = []
        for t in range(self.timesteps):
            r = torch.rand(1)
            if r < teacher_forcing:
                x = data.x[..., t].view(-1, 1)


            env = data.env[..., t]
            if not self.use_wind:
                env = env[:, 2:]
            x, states, hidden = self.propagate(edge_index, x=x, coords=coords, env=env,
                                                states=states, hidden=hidden,
                                                edge_attr=edge_attr, ground=ground,
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
                x = x * data.local_night[:, t+1].view(-1, 1)

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

        self.flows.append(flow)

        abs_flow = flow * x_j
        self.abs_flows.append(abs_flow)

        return abs_flow


    def update(self, aggr_out, coords, env, ground, local_dusk, states, hidden):
        # return aggregation (sum) of inflows computed by message()
        # add departure prediction if local_dusk flag is True

        inputs = torch.cat([coords, env, ground.view(-1, 1), local_dusk.float().view(-1, 1)], dim=1)
        states, hidden = self.node_lstm(inputs, (states, hidden))
        departure = self.departure(states)
        #departure = departure * local_dusk.view(-1, 1) # only use departure model if it is local dusk
        pred = aggr_out + departure

        return pred, states, hidden


class BirdDynamics(MessagePassing):

    def __init__(self, msg_n_in=16, node_n_in=8, n_out=1, n_hidden=16, timesteps=6, model='linear',
                 seed=12345, multinight=False, use_wind=True, dropout_p=0):
        super(BirdDynamics, self).__init__(aggr='add', node_dim=0) # inflows from neighbouring radars are aggregated by adding

        torch.manual_seed(seed)

        if not use_wind:
            msg_n_in -= 2

        if not use_wind:
            node_n_in -= 2

        if model == 'linear':
            self.edgeflow = torch.nn.Linear(msg_n_in, n_out)
        elif model == 'linear+sigmoid':
            self.msg_nn = torch.nn.Sequential(torch.nn.Linear(msg_n_in, n_hidden),
                                                torch.nn.Sigmoid())
            self.node_nn = torch.nn.Sequential(torch.nn.Linear(n_hidden, n_hidden),
                                                 torch.nn.Tanh())
            self.departure = torch.nn.Sequential(torch.nn.Linear(node_n_in, n_out),
                                               torch.nn.Tanh())
        else:
            self.msg_nn = torch.nn.Sequential(torch.nn.Linear(msg_n_in, n_hidden),
                                                torch.nn.ReLU(),
                                                torch.nn.Linear(n_hidden, n_hidden),
                                                torch.nn.Sigmoid())
            self.node_nn = torch.nn.Sequential(torch.nn.Linear(n_hidden, n_hidden),
                                                 torch.nn.ReLU(),
                                                 torch.nn.Linear(n_hidden, n_hidden),
                                                 torch.nn.Tanh())
            self.departure = torch.nn.Sequential(torch.nn.Linear(node_n_in, n_hidden),
                                               torch.nn.ReLU(),
                                               torch.nn.Linear(n_hidden, n_out),
                                               torch.nn.Tanh())


        self.timesteps = timesteps
        self.multinight = multinight
        self.use_wind = use_wind


    def forward(self, data, teacher_forcing=0.0):
        # with teacher_forcing = 0.0 the model always uses previous predictions to make new predictions
        # with teacher_forcing = 1.0 the model always uses the ground truth to make new predictions

        # measurement at t=0
        x = data.x[..., 0].view(-1, 1)

        # birds on ground at t=0
        ground = torch.zeros_like(x)


        coords = data.coords
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        if self.node_embedding is not None:
            embedding = torch.cat([self.node_embedding.weight]*data.num_graphs)
        else:
            embedding = None


        y_hat = []
        y_hat.append(x)

        for t in range(self.timesteps):
            r = torch.rand(1)
            if r < teacher_forcing:
                x = data.x[..., t].view(-1, 1)

            env = data.env[..., t]
            if not self.use_wind:
                env = env[:, 2:]
            x = self.propagate(edge_index, x=x, coords=coords, env=env, ground=ground, dusk=data.local_dusk[:, t],
                               edge_attr=edge_attr, embedding=embedding)


            if self.multinight:
                # for locations where it is dawn: save birds to ground and set birds in the air to zero
                r = torch.rand(1)
                if r < teacher_forcing:
                    ground = ground + data.local_dawn[:, t+1].view(-1, 1) * data.x[..., t+1].view(-1, 1)
                else:
                    ground = ground + data.local_dawn[:, t+1].view(-1, 1) * x
                x = x * data.local_night[:, t+1].view(-1, 1)

                # TODO for radar data, birds can stay on the ground or depart later in the night, so
                #  at dusk birds on ground shouldn't be set to zero but predicted departing birds should be subtracted
                # for locations where it is dusk: set birds on ground to zero
                ground = ground * ~data.local_dusk[:, t].view(-1, 1)

            y_hat.append(x)

        prediction = torch.cat(y_hat, dim=-1)
        return prediction


    def message(self, x_i, x_j, coords_i, coords_j, env_i, env_j, edge_attr):
        # construct messages to node i for each edge (j,i)
        # can take any argument initially passed to propagate()
        # x_j are source features with shape [E, out_channels]

        features = torch.cat([x_i.view(-1, 1), x_j.view(-1, 1), coords_i, coords_j, env_i, env_j, edge_attr], dim=1)
        msg = self.msg_nn(features)

        return msg


    def update(self, aggr_out, x, coords, env, ground, dusk):

        #features = torch.cat([aggr_out, x.view(-1, 1), coords, env], dim=1)
        features = torch.cat([ground.view(-1, 1), dusk.view(-1, 1).float(), coords, env], dim=1)
        departure = self.departure(features)
        delta = self.node_nn(aggr_out)
        pred = x + delta + departure

        return pred



class BirdRecurrent1(MessagePassing):

    def __init__(self, msg_n_in=17, node_n_in=8, n_out=1, n_hidden=16, timesteps=6,
                 seed=12345, multinight=False, use_wind=True, dropout_p=0):
        super(BirdRecurrent1, self).__init__(aggr='add', node_dim=0) # inflows from neighbouring radars are aggregated by adding

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

            x, hidden = self.propagate(edge_index, x=x, coords=coords, env=env, ground=ground, hidden=hidden,
                                       dusk=data.local_dusk[:, t], edge_attr=edge_attr,
                                       local_night=data.local_night[:, t])


            if self.multinight:
                # for locations where it is dawn: save birds to ground and set birds in the air to zero
                r = torch.rand(1)
                if r < teacher_forcing:
                    ground = ground + data.local_dawn[:, t+1].view(-1, 1) * data.x[..., t+1].view(-1, 1)
                else:
                    ground = ground + data.local_dawn[:, t+1].view(-1, 1) * x
                x = x * data.local_night[:, t+1].view(-1, 1)

                # TODO for radar data, birds can stay on the ground or depart later in the night, so
                #  at dusk birds on ground shouldn't be set to zero but predicted departing birds should be subtracted
                # for locations where it is dusk: set birds on ground to zero
                ground = ground * ~data.local_dusk[:, t].view(-1, 1)

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


    def update(self, aggr_out, x, coords, env, dusk, ground, hidden):

        inputs = torch.cat([ground.view(-1, 1), dusk.view(-1, 1).float(), coords, env], dim=1)

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

        if not departure:
            # omit t=0
            gt = gt[:, 1:]
            output = output[:, 1:]

        if conservation:
            outfluxes = to_dense_adj(data.edge_index, edge_attr=torch.stack(model.flows, dim=-1)).view(data.num_nodes,
                                                                                                       data.num_nodes,
                                                                                                       -1).sum(1)
            outfluxes = torch.stack([outfluxes[node] for node in range(data.num_nodes) if not boundaries[node]])
            target_fluxes = torch.ones(outfluxes.shape)
            if cuda: target_fluxes = target_fluxes.to('cuda')
            constraints = torch.mean((outfluxes - target_fluxes)**2)
            loss = loss_func(output, gt, data.local_night) + 0.01 * constraints
        else:
            loss = loss_func(output, gt, data.local_night)
        loss.backward()
        #loss_all += data.num_graphs * loss.item()
        loss_all += data.num_graphs * loss
        optimizer.step()

    return loss_all

def train_dynamics(model, train_loader, optimizer, loss_func, cuda,
                 teacher_forcing=1.0):
    if cuda: model.cuda()
    model.train()
    loss_all = 0
    for data in train_loader:
        if cuda: data = data.to('cuda')
        optimizer.zero_grad()
        if teacher_forcing < 0:
            output = model(data)
        else:
            output = model(data, teacher_forcing)
        gt = data.y
        print(gt)

        loss = loss_func(output, gt, data.local_night)
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

def test_fluxes(model, test_loader, timesteps, loss_func, cuda, get_outfluxes=True, bird_scale=2000,
                departure=False, fix_boundary=[]):
    if cuda:
        model.cuda()
    model.eval()
    loss_all = []
    outfluxes = {}
    outfluxes_abs = {}
    for tidx, data in enumerate(test_loader):
        if cuda: data = data.to('cuda')
        output = model(data) * bird_scale #.view(-1)
        gt = data.y * bird_scale

        # if not departure:
        #     gt = gt[:, 1:]
        #     output = output[:, 1:]

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
        loss_all.append(torch.tensor([loss_func(output[:, t], gt[:, t], data.local_night[:, t]) for t in range(timesteps + 1)]))
        #loss_all.append(loss_func(output, gt))
        #constraints_all.append(constraints)

    if get_outfluxes:
        return torch.stack(loss_all), outfluxes , outfluxes_abs #, torch.stack(constraints_all)
    else:
        return torch.stack(loss_all)

def test_dynamics(model, test_loader, timesteps, loss_func, cuda, bird_scale=2000):
    if cuda:
        model.cuda()
    model.eval()
    loss_all = []

    for tidx, data in enumerate(test_loader):
        if cuda: data = data.to('cuda')
        output = model(data) * bird_scale #.view(-1)
        gt = data.y * bird_scale
        loss_all.append(torch.tensor([loss_func(output[:, t], gt[:, t], data.local_night[:, t]) for t in range(timesteps + 1)]))

    return torch.stack(loss_all)

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



