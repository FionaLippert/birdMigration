import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader, Dataset, InMemoryDataset
from torch_geometric.nn import MessagePassing, inits
from torch_geometric.utils import add_self_loops, degree, to_dense_adj, dense_to_sparse, softmax
#from torch_geometric_temporal.nn.recurrent import DCRNN
import numpy as np
import os.path as osp
import os


def init_weights(m):
    if type(m) == nn.Linear:
        #inits.glorot(m.weight)
        #nn.init.xavier_normal_(m.weight)
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        #if hasattr(m, 'bias'): inits.zeros(m.bias)
    # elif type(m) == nn.LSTMCell:
    #     for name, param in m.named_parameters():
    #         if 'bias' in name:
    #             inits.zeros(param)
    #         elif 'weight' in name:
    #             inits.glorot(param)

class LSTM(torch.nn.Module):
    """
   Standard LSTM taking all observed/predicted bird densities and environmental features as input to LSTM
   Args:
       in_channels (int): number of input features (node features x number of nodes)
       hidden_channels (int): number of units per hidden layer
       out_channels (int): number of nodes x number of outputs per node
       timesteps (int): length of forecasting horizon
   """
    def __init__(self, **kwargs):
        super(LSTM, self).__init__()

        self.horizon = kwargs.get('horizon', 40)
        self.dropout_p = kwargs.get('dropout_p', 0)
        self.n_hidden = kwargs.get('n_hidden', 16)
        self.n_in = 5 + kwargs.get('n_env', 4)
        self.n_nodes = kwargs.get('n_nodes', 22)
        self.n_layers = kwargs.get('n_layers', 1)
        self.force_zeros = kwargs.get('force_zeros', False)
        self.teacher_forcing = kwargs.get('teacher_forcing', 0)

        torch.manual_seed(kwargs.get('seed', 1234))


        self.fc_in = torch.nn.Linear(self.n_in*self.n_nodes, self.n_hidden)
        self.lstm_layers = nn.ModuleList([torch.nn.LSTMCell(self.n_hidden, self.n_hidden) for l in range(self.n_layers)])
        self.fc_out = torch.nn.Linear(self.n_hidden, self.n_nodes)


    def forward(self, data):

        x = data.x[:, 0]
        # states = torch.zeros(1, self.hidden_channels).to(x.device)
        # hidden = torch.zeros(1, self.hidden_channels).to(x.device)
        h_t = [torch.zeros(1, self.n_hidden, device=x.device) for l in range(self.n_layers)]
        c_t = [torch.zeros(1, self.n_hidden, device=x.device) for l in range(self.n_layers)]

        #hidden = None

        y_hat = [x]
        for t in range(self.horizon):
            r = torch.rand(1)
            if r < self.teacher_forcing:
                x = data.x[:, t]


            # use both bird prediction/observation and environmental features as input to LSTM
            inputs = torch.cat([data.coords.flatten(),
                                data.env[..., t+1].flatten(),
                                data.local_dusk[:, t].float().flatten(),
                                data.local_dawn[:, t+1].float().flatten(),
                                x], dim=0).view(1, -1)

            # multi-layer LSTM
            inputs = self.fc_in(inputs) #.relu()
            h_t[0], c_t[0] = self.lstm_layers[0](inputs, (h_t[0], c_t[0]))
            for l in range(1, self.n_layers):
                h_t[l], c_t[l] = self.lstm_layers[l](h_t[l-1], (h_t[l], c_t[l]))

            x = x + self.fc_out(h_t[-1]).tanh().view(-1)

            if self.force_zeros:
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
    def __init__(self, in_channels, hidden_channels, out_channels, horizon, n_layers=1, dropout_p=0.5, seed=12345):
        super(MLP, self).__init__()

        torch.manual_seed(seed)

        self.fc_in = torch.nn.Linear(in_channels, hidden_channels)
        self.fc_hidden = nn.ModuleList([torch.nn.Linear(hidden_channels, hidden_channels) for _ in range(n_layers - 1)])
        self.fc_out = torch.nn.Linear(hidden_channels, out_channels)
        self.horizon = horizon
        self.dropout_p = dropout_p

        self.reset_parameters()

    def reset_parameters(self):

        init_weights(self.fc_in)
        init_weights(self.fc_out)
        self.fc_hidden.apply(init_weights)

    def forward(self, data):

        y_hat = []
        for t in range(self.horizon + 1):

            features = torch.cat([data.coords.flatten(),
                                  data.env[..., t].flatten()], dim=0)
            x = self.fc_in(features)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_p, training=self.training)

            for l in self.fc_hidden:
                x = l(x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout_p, training=self.training)

            x = self.fc_out(x)
            x = x.sigmoid()

            # for locations where it is night: set birds in the air to zero
            x = x * data.local_night[:, t]

            y_hat.append(x)

        return torch.stack(y_hat, dim=1)


class LocalMLP(torch.nn.Module):

    def __init__(self, n_env, coord_dim=2, **kwargs):
        super(LocalMLP, self).__init__()

        self.horizon = kwargs.get('horizon', 1)
        self.dropout_p = kwargs.get('dropout_p', 0)
        self.n_hidden = kwargs.get('n_hidden', 16)
        self.use_acc = kwargs.get('use_acc_vars', False)
        self.n_layers = kwargs.get('n_fc_layers', 1)
        self.force_zeros = kwargs.get('force_zeros', False)

        self.n_in = n_env + coord_dim + self.use_acc * 2

        torch.manual_seed(kwargs.get('seed', 1234))

        self.fc_in = torch.nn.Linear(self.n_in, self.n_hidden)
        self.fc_hidden = nn.ModuleList([torch.nn.Linear(self.n_hidden, self.n_hidden)
                                        for _ in range(self.n_layers - 1)])
        self.fc_out = torch.nn.Linear(self.n_hidden, 1)

        self.reset_parameters()

    def reset_parameters(self):

        self.fc_hidden.apply(init_weights)
        init_weights(self.fc_in)
        init_weights(self.fc_out)


    def forward(self, data):

        y_hat = []

        for t in range(self.horizon):

            x = self.step(data.coords, data.env[..., t], acc=data.acc[..., t])

            if self.force_zeros:
                # for locations where it is night: set birds in the air to zero
                x = x * data.local_night[:, t].view(-1, 1)

            y_hat.append(x)

        prediction = torch.cat(y_hat, dim=-1)
        return prediction



    def step(self, coords, env, acc):
        # use only location-specific features to predict migration intensities
        if self.use_acc:
            features = torch.cat([coords, env, acc], dim=1)
        else:
            features = torch.cat([coords, env], dim=1)
        x = F.relu(self.fc_in(features))
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        for l in self.fc_hidden:
            x = F.relu(l(x))
            x = F.dropout(x, p=self.dropout_p, training=self.training)

        x = self.fc_out(x)
        x = x.sigmoid()

        return x


class EdgeFluxMLP(torch.nn.Module):

    def __init__(self, n_in, **kwargs):
        super(EdgeFluxMLP, self).__init__()

        #print(f'edge flux n_in = {n_in}')
        self.n_hidden = kwargs.get('n_hidden', 64)
        self.n_fc_layers = kwargs.get('n_fc_layers', 1)
        self.dropout_p = kwargs.get('dropout_p', 0)

        self.input2hidden = torch.nn.Linear(n_in, self.n_hidden, bias=False)
        self.fc_edge_in = torch.nn.Linear(self.n_hidden * 2, self.n_hidden)
        self.fc_edge_hidden = nn.ModuleList([torch.nn.Linear(self.n_hidden, self.n_hidden)
                                             for _ in range(self.n_fc_layers - 1)])
        self.hidden2output = torch.nn.Linear(self.n_hidden, 1)

        self.reset_parameters()

    def reset_parameters(self):

        init_weights(self.input2hidden)
        init_weights(self.fc_edge_in)
        self.fc_edge_hidden.apply(init_weights)
        init_weights(self.hidden2output)


    def forward(self, x_j, inputs, hidden_j):

        inputs = self.input2hidden(inputs)
        inputs = torch.cat([inputs, hidden_j], dim=1)

        flux = F.relu(self.fc_edge_in(inputs))
        flux = F.dropout(flux, p=self.dropout_p, training=self.training, inplace=False)

        for l in self.fc_edge_hidden:
            flux = F.relu(l(flux))
            flux = F.dropout(flux, p=self.dropout_p, training=self.training, inplace=False)

        flux = self.hidden2output(flux)
        flux = torch.sigmoid(flux)
        flux = flux * x_j

        return flux

class NodeLSTM(torch.nn.Module):

    def __init__(self, n_in, **kwargs):
        super(NodeLSTM, self).__init__()

        self.n_in = n_in
        self.n_hidden = kwargs.get('n_hidden', 64)
        self.n_lstm_layers = kwargs.get('n_lstm_layers', 2)
        self.use_encoder = kwargs.get('use_encoder', True)
        self.dropout_p = kwargs.get('dropout_p', 0)

        # node embedding
        self.input2hidden = torch.nn.Linear(self.n_in, self.n_hidden, bias=False)

        if self.use_encoder:
            # self.fc_attention = torch.nn.Linear(self.n_hidden * 2, self.n_hidden)
            # self.v_attention = torch.nn.Linear(self.n_hidden, 1, bias=False)
            self.lstm_in = torch.nn.LSTMCell(self.n_hidden * 2, self.n_hidden)
        else:
            self.lstm_in = torch.nn.LSTMCell(self.n_hidden, self.n_hidden)
        self.lstm_layers = nn.ModuleList([torch.nn.LSTMCell(self.n_hidden, self.n_hidden)
                                          for _ in range(self.n_lstm_layers - 1)])

        # self.hidden2output = torch.nn.Sequential(torch.nn.Linear(self.n_hidden, self.n_hidden),
        #                                    torch.nn.Dropout(p=self.dropout_p),
        #                                    torch.nn.LeakyReLU(),
        #                                    torch.nn.Linear(self.n_hidden, 1))

        self.hidden2in = torch.nn.Sequential(torch.nn.Linear(self.n_hidden, self.n_hidden),
                                                 torch.nn.Dropout(p=self.dropout_p),
                                                 torch.nn.ReLU(),
                                                 torch.nn.Linear(self.n_hidden, 1))

        self.hidden2out = torch.nn.Sequential(torch.nn.Linear(self.n_hidden, self.n_hidden),
                                                 torch.nn.Dropout(p=self.dropout_p),
                                                 torch.nn.ReLU(),
                                                 torch.nn.Linear(self.n_hidden, 1))

        self.reset_parameters()

    def reset_parameters(self):

        init_weights(self.input2hidden)
        # if self.use_encoder:
        #     init_weights(self.fc_attention)
        #     init_weights(self.v_attention)
        init_weights(self.lstm_in)
        self.lstm_layers.apply(init_weights)
        self.hidden2in.apply(init_weights)
        self.hidden2out.apply(init_weights)

    def setup_states(self, h, c, enc_state=None):
        self.h = h
        self.c = c
        self.alphas = []
        self.enc_state = h[-1] #enc_state

    def get_alphas(self):
        alphas = torch.stack(self.alphas)
        return alphas

    def get_hidden(self):
        return self.h[-1]

    def forward(self, inputs):

        inputs = self.input2hidden(inputs)

        if self.use_encoder:
            # use attention mechanism to extract information from encoder states
            # hidden = self.h[-1].unsqueeze(1).repeat(1, self.enc_states.size(1), 1)
            # energy = torch.tanh(self.fc_attention(torch.cat([self.enc_states, hidden], dim=2)))
            # scores = self.v_attention(energy).squeeze()
            # alpha = F.softmax(scores, dim=1)
            # context = torch.matmul(alpha.unsqueeze(1), self.enc_states).squeeze() # shape (radars x hidden)
            # inputs = torch.cat([inputs, context], dim=1)

            # if not self.training:
            #     self.alphas.append(alpha)

            inputs = torch.cat([inputs, self.enc_state], dim=1)

        # lstm layers
        self.h[0], self.c[0] = self.lstm_in(inputs, (self.h[0], self.c[0]))
        for l in range(self.n_lstm_layers - 1):
            self.h[0] = F.dropout(self.h[0], p=self.dropout_p, training=self.training, inplace=False)
            self.c[0] = F.dropout(self.c[0], p=self.dropout_p, training=self.training, inplace=False)
            self.h[l+1], self.c[l+1] = self.lstm_layers[l](self.h[l], (self.h[l+1], self.c[l+1]))

        x_in = torch.sigmoid(self.hidden2in(self.h[-1]))
        x_out = torch.sigmoid(self.hidden2out(self.h[-1]))

        # delta = torch.tanh(self.hidden2output(self.h[-1]))

        return x_in, x_out, self.h[-1]


class LocalLSTM(torch.nn.Module):

    def __init__(self, n_env, coord_dim=2, **kwargs):
        super(LocalLSTM, self).__init__()

        self.horizon = kwargs.get('horizon', 40)
        self.t_context = max(1, kwargs.get('context', 1))
        self.teacher_forcing = kwargs.get('teacher_forcing', 0)
        self.use_encoder = kwargs.get('use_encoder', True)

        # model components
        n_in = n_env + coord_dim + 1
        if self.use_encoder:
            self.encoder = RecurrentEncoder(n_in, **kwargs)
        self.node_lstm = NodeLSTM(n_in, **kwargs)

        seed = kwargs.get('seed', 1234)
        torch.manual_seed(seed)


    def forward(self, data):

        x = data.x[..., self.t_context - 1].view(-1, 1)
        y_hat = []

        if self.use_encoder:
            # push context timeseries through encoder to initialize decoder
            h_t, c_t = self.encoder(data)
            self.node_lstm.setup_states(h_t, c_t) #, enc_states)
        else:
            # start from scratch
            h_t = [torch.zeros(data.x.size(0), self.node_lstm.n_hidden, device=x.device) for
                   _ in range(self.node_lstm.n_lstm_layers)]
            c_t = [torch.zeros(data.x.size(0), self.node_lstm.n_hidden, device=x.device) for
                   _ in range(self.node_lstm.n_lstm_layers)]
            self.node_lstm.setup_states(h_t, c_t)

        forecast_horizon = range(self.t_context, self.t_context + self.horizon)

        for t in forecast_horizon:

            r = torch.rand(1)
            if r < self.teacher_forcing:
                x = data.x[..., t-1].view(-1, 1)

            inputs = torch.cat([x.view(-1, 1), data.coords, data.env[..., t]], dim=1)
            # delta, hidden = self.node_lstm(inputs)
            # x = x + delta

            x_in, x_out, hidden = self.node_lstm(inputs)
            x = x + x_in - (x_out * x)
            y_hat.append(x)

        prediction = torch.cat(y_hat, dim=-1)

        return prediction



class FluxGraphLSTM(MessagePassing):

    def __init__(self, n_env, n_edge_attr, coord_dim=2, **kwargs):
        super(FluxGraphLSTM, self).__init__(aggr='add', node_dim=0)

        # settings
        self.horizon = kwargs.get('horizon', 40)
        self.t_context = max(1, kwargs.get('context', 1))
        self.teacher_forcing = kwargs.get('teacher_forcing', 0)
        self.use_encoder = kwargs.get('use_encoder', True)
        self.use_boundary_model = kwargs.get('use_boundary_model', True)
        self.fixed_boundary = kwargs.get('fixed_boundary', False)
        self.n_graph_layers = kwargs.get('n_graph_layers', 0)

        # model components
        n_node_in = n_env + coord_dim + 1
        n_edge_in = 2 * n_env + 2 * coord_dim + n_edge_attr

        self.node_lstm = NodeLSTM(n_node_in + 1, **kwargs)
        self.edge_mlp = EdgeFluxMLP(n_edge_in, **kwargs)
        if self.use_encoder:
            self.encoder = RecurrentEncoder(n_node_in, **kwargs)
        if self.use_boundary_model:
            self.boundary_model = Extrapolation()

        self.graph_layers = nn.ModuleList([GraphLayer(**kwargs) for l in range(self.n_graph_layers)])

        seed = kwargs.get('seed', 1234)
        torch.manual_seed(seed)


    def forward(self, data):
        boundary_nodes = data.boundary.view(-1, 1)
        inner_nodes = torch.logical_not(data.boundary).view(-1, 1)

        x = data.x[..., self.t_context - 1].view(-1, 1)
        y_hat = []

        if self.use_encoder:
            # push context timeseries through encoder to initialize decoder
            h_t, c_t = self.encoder(data)
            #print(f'encoder states: {h_t[-1]}')
            #assert(torch.isfinite(h_t[-1]).all())
            self.node_lstm.setup_states(h_t, c_t)
        else:
            # start from scratch
            h_t = [torch.zeros(data.x.size(0), self.node_lstm.n_hidden, device=x.device) for
                   _ in range(self.node_lstm.n_lstm_layers)]
            c_t = [torch.zeros(data.x.size(0), self.node_lstm.n_hidden, device=x.device) for
                   _ in range(self.node_lstm.n_lstm_layers)]
            self.node_lstm.setup_states(h_t, c_t)

        # setup model components
        if self.use_boundary_model:
            self.boundary_model.edge_index = data.edge_index[:, torch.logical_not(data.boundary2boundary_edges)]
        hidden = h_t[-1]

        # relevant info for later
        self.edge_fluxes = torch.zeros((data.edge_index.size(1), 1, self.horizon), device=x.device)
        self.node_deltas = torch.zeros((data.x.size(0), 1, self.horizon), device=x.device)
        self.node_fluxes = torch.zeros((data.x.size(0), 1, self.horizon), device=x.device)

        forecast_horizon = range(self.t_context, self.t_context + self.horizon)

        for t in forecast_horizon:
            
            r = torch.rand(1)
            if r < self.teacher_forcing:
                x = data.x[..., t-1].view(-1, 1)
                #assert(torch.isfinite(x).all())

            if self.use_boundary_model:
                # boundary model
                x_boundary = self.boundary_model(x)
                h_boundary = self.boundary_model(hidden)
                x = x * inner_nodes + x_boundary * boundary_nodes
                hidden = hidden * inner_nodes + h_boundary * boundary_nodes

            # propagate hidden states through graph to combine spatial information
            hidden_sp = hidden
            for l in range(self.n_graph_layers):
                hidden_sp = self.graph_layers[l]([data.edge_index, hidden_sp])

            # message passing through graph
            x, hidden = self.propagate(data.edge_index,
                                         reverse_edges=data.reverse_edges,
                                         x=x,
                                         coords=data.coords,
                                         hidden=hidden,
                                         hidden_sp=hidden_sp,
                                         areas=data.areas,
                                         edge_attr=data.edge_attr,
                                         env=data.env[..., t],
                                         env_1=data.env[..., t-1],
                                         t=t-self.t_context)

            if self.fixed_boundary:
                # # use ground truth for boundary nodes
                x[data.boundary, 0] = data.y[data.boundary, t] * data.areas[data.boundary]

            y_hat.append(x)

        prediction = torch.cat(y_hat, dim=-1)
        return prediction


    def message(self, x_j, hidden_sp_j, coords_i, coords_j, env_i, env_1_j, edge_attr, t, reverse_edges):
        # construct messages to node i for each edge (j,i)
        # x_j are source features with shape [E, out_channels]

        inputs = [coords_i, coords_j, env_i, env_1_j, edge_attr]
        inputs = torch.cat(inputs, dim=1)
        #assert(torch.isfinite(inputs).all())

        flux = self.edge_mlp(x_j, inputs, hidden_sp_j)

        self.edge_fluxes[..., t] = flux
        flux = flux - flux[reverse_edges]
        flux = flux.view(-1, 1)

        return flux


    def update(self, aggr_out, x, coords, areas, env, t):

        inputs = torch.cat([x.view(-1, 1), coords, env, areas.view(-1, 1)], dim=1)
        #assert(torch.isfinite(inputs).all())

        #delta, hidden = self.node_lstm(inputs)
        x_in, x_out, hidden = self.node_lstm(inputs)
        delta = x_in - (x_out * x)

        self.node_deltas[..., t] = delta
        self.node_fluxes[..., t] = aggr_out

        pred = x + delta + aggr_out

        return pred, hidden



class GraphLayer(MessagePassing):

    def __init__(self, **kwargs):
        super(GraphLayer, self).__init__(aggr='add', node_dim=0)

        # model settings
        self.n_hidden = kwargs.get('n_hidden', 64)
        self.dropout_p = kwargs.get('dropout_p', 0)

        # model components
        self.fc_edge = torch.nn.Linear(self.n_hidden, self.n_hidden)
        self.fc_node = torch.nn.Linear(self.n_hidden, self.n_hidden)

        seed = kwargs.get('seed', 1234)
        torch.manual_seed(seed)

        self.reset_parameters()

    def reset_parameters(self):

        init_weights(self.fc_edge)
        init_weights(self.fc_node)


    def forward(self, data):

        edge_index, inputs = data

        # message passing through graph
        out = self.propagate(edge_index, inputs=inputs)

        return out


    def message(self, inputs_i):
        # construct messages to node i for each edge (j,i)
        out = self.fc_edge(inputs_i)
        out = F.relu(out)

        return out


    def update(self, aggr_out):

        out = self.fc_node(aggr_out)

        return out


class Extrapolation(MessagePassing):

    def __init__(self, **kwargs):
        super(Extrapolation, self).__init__(aggr='mean', node_dim=0)

        self.edge_index = kwargs.get('edge_index', None)

    def forward(self, var):
        var = self.propagate(self.edge_index, var=var)
        return var

    def message(self, var_j):
        return var_j


class RecurrentEncoder(torch.nn.Module):
    def __init__(self, n_in, **kwargs):
        super(RecurrentEncoder, self).__init__()

        self.t_context = kwargs.get('context', 24)
        self.n_hidden = kwargs.get('n_hidden', 64)
        self.n_lstm_layers = kwargs.get('n_lstm_layers', 1)
        self.dropout_p = kwargs.get('dropout_p', 0)
        self.use_uv = kwargs.get('use_uv', False)
        if self.use_uv:
            n_in = n_in + 2

        torch.manual_seed(kwargs.get('seed', 1234))

        self.input2hidden = torch.nn.Linear(n_in, self.n_hidden, bias=False)
        self.lstm_layers = nn.ModuleList([nn.LSTMCell(self.n_hidden, self.n_hidden)
                                          for _ in range(self.n_lstm_layers)])

        self.reset_parameters()


    def reset_parameters(self):

        self.lstm_layers.apply(init_weights)
        init_weights(self.input2hidden)


    def forward(self, data):
        # initialize lstm variables
        h_t = [torch.zeros(data.x.size(0), self.n_hidden, device=data.x.device) for _ in range(self.n_lstm_layers)]
        c_t = [torch.zeros(data.x.size(0), self.n_hidden, device=data.x.device) for _ in range(self.n_lstm_layers)]

        #states = []

        for t in range(self.t_context):
            x = data.x[:, t]
            if self.use_uv:
                h_t, c_t = self.update(x, data.coords, data.env[..., t], h_t, c_t, data.bird_uv[..., t])
            else:
                h_t, c_t = self.update(x, data.coords, data.env[..., t], h_t, c_t)
            #states.append(h_t[-1])

        #states = torch.stack(states, dim=1)
        return h_t, c_t


    def update(self, x, coords, env, h_t, c_t, bird_uv=None):

        if self.use_uv:
            inputs = torch.cat([x.view(-1, 1), coords, env, bird_uv], dim=1)
        else:
            inputs = torch.cat([x.view(-1, 1), coords, env], dim=1)

        #print(f'encoder inputs: {inputs}')
        #assert(torch.isfinite(inputs).all())

        inputs = self.input2hidden(inputs)
        h_t[0], c_t[0] = self.lstm_layers[0](inputs, (h_t[0], c_t[0]))
        for l in range(1, self.n_lstm_layers):
            h_t[l-1] = F.dropout(h_t[l-1], p=self.dropout_p, training=self.training, inplace=False)
            c_t[l-1] = F.dropout(c_t[l-1], p=self.dropout_p, training=self.training, inplace=False)
            h_t[l], c_t[l] = self.lstm_layers[l](h_t[l - 1], (h_t[l], c_t[l]))

        return h_t, c_t


#
#
# class AttentionGraphLSTM(MessagePassing):
#
#     def __init__(self, **kwargs):
#         super(AttentionGraphLSTM, self).__init__(aggr='add', node_dim=0)
#
#         self.horizon = kwargs.get('horizon', 40)
#         self.dropout_p = kwargs.get('dropout_p', 0)
#         self.n_hidden = kwargs.get('n_hidden', 16)
#         self.n_env = kwargs.get('n_env', 4)
#         self.n_node_in = 6 + self.n_env
#         self.n_edge_in = 3 + self.n_env
#         self.n_fc_layers = kwargs.get('n_fc_layers', 1)
#         self.n_lstm_layers = kwargs.get('n_lstm_layers', 1)
#         self.fixed_boundary = kwargs.get('fixed_boundary', [])
#         self.force_zeros = kwargs.get('force_zeros', True)
#         self.teacher_forcing = kwargs.get('teacher_forcing', 0)
#
#         self.use_encoder = kwargs.get('use_encoder', False)
#         self.encoder_type = kwargs.get('encoder_type', 'temporal')
#         self.t_context = kwargs.get('context', 0)
#         self.predict_delta = kwargs.get('predict_delta', True)
#
#         seed = kwargs.get('seed', 1234)
#         torch.manual_seed(seed)
#
#
#         # self.edge_env2hidden = torch.nn.Linear(self.n_env_in, self.n_hidden)
#         # self.edge_states2hidden = torch.nn.Linear(self.n_states_in, self.n_hidden)
#         self.edge2hidden = torch.nn.Linear(self.n_edge_in, self.n_hidden)
#         self.context_embedding = torch.nn.Linear(self.n_hidden, self.n_hidden)
#         self.attention_s = torch.nn.Parameter(torch.Tensor(self.n_hidden, 1))
#
#
#         self.node2hidden = torch.nn.Linear(self.n_node_in, self.n_hidden)
#
#         if self.use_encoder:
#             self.lstm_in = nn.LSTMCell(3 * self.n_hidden, self.n_hidden)
#         else:
#             self.lstm_in = nn.LSTMCell(2 * self.n_hidden, self.n_hidden)
#         self.lstm_layers = nn.ModuleList([nn.LSTMCell(self.n_hidden, self.n_hidden) for _
#                                           in range(self.n_lstm_layers - 1)])
#
#         self.hidden2delta = torch.nn.Sequential(torch.nn.Linear(self.n_hidden, self.n_hidden),
#                                                 torch.nn.Dropout(p=self.dropout_p),
#                                                 torch.nn.LeakyReLU(),
#                                                 torch.nn.Linear(self.n_hidden, 1))
#
#         if self.use_encoder:
#             if self.encoder_type == 'temporal':
#                 self.encoder = RecurrentEncoder(timesteps=self.t_context, n_env=self.n_env, n_hidden=self.n_hidden,
#                                             n_lstm_layers=self.n_lstm_layers, seed=seed, dropout_p=self.dropout_p)
#             else:
#                 self.encoder = RecurrentEncoderSpatial(timesteps=self.t_context, n_env=self.n_env, n_hidden=self.n_hidden,
#                                                 n_lstm_layers=self.n_lstm_layers, seed=seed, dropout_p=self.dropout_p)
#             self.fc_encoder = torch.nn.Linear(self.n_hidden, self.n_hidden)
#             self.fc_hidden = torch.nn.Linear(self.n_hidden, self.n_hidden)
#             self.attention_t = torch.nn.Parameter(torch.Tensor(self.n_hidden, 1))
#
#         self.reset_parameters()
#
#
#     def reset_parameters(self):
#         inits.glorot(self.edge2hidden.weight)
#         inits.glorot(self.context_embedding.weight)
#         inits.glorot(self.attention_s)
#         inits.glorot(self.node2hidden.weight)
#
#         if self.use_encoder:
#             inits.glorot(self.fc_encoder.weight)
#             inits.glorot(self.fc_hidden.weight)
#             inits.glorot(self.attention_t)
#
#         def init_weights(m):
#             if type(m) == nn.Linear:
#                 inits.glorot(m.weight)
#                 inits.zeros(m.bias)
#             elif type(m) == nn.LSTMCell:
#                 for name, param in m.named_parameters():
#                     if 'bias' in name:
#                         inits.zeros(param)
#                     elif 'weight' in name:
#                         inits.glorot(param)
#
#         self.hidden2delta.apply(init_weights)
#         self.lstm_layers.apply(init_weights)
#         init_weights(self.lstm_in)
#
#
#
#     def forward(self, data):
#         # with teacher_forcing = 0.0 the model always uses previous predictions to make new predictions
#         # with teacher_forcing = 1.0 the model always uses the ground truth to make new predictions
#
#         self.edges = data.edge_index
#         y_hat = []
#         enc_states = None
#
#         x = data.x[..., self.t_context].view(-1, 1)
#         y_hat.append(x)
#
#         # initialize lstm variables
#         if self.use_encoder:
#             # push context timeseries through encoder to initialize decoder
#             enc_states, h_t, c_t = self.encoder(data)
#             #x = torch.zeros(data.x.size(0)).to(data.x.device) # TODO eventually use this!?
#
#         else:
#             # start from scratch
#             h_t = [torch.zeros(data.x.size(0), self.n_hidden, device=x.device) for l in range(self.n_lstm_layers)]
#             c_t = [torch.zeros(data.x.size(0), self.n_hidden, device=x.device) for l in range(self.n_lstm_layers)]
#
#         coords = data.coords
#         edge_index = data.edge_index
#         edge_attr = data.edge_attr
#
#
#         if self.use_encoder:
#             forecast_horizon = range(self.t_context + 1, self.t_context + self.horizon + 1)
#         else:
#             forecast_horizon = range(1, self.horizon + 1)
#
#         self.alphas_s = torch.zeros((edge_index.size(1), 1, self.horizon + 1), device=x.device)
#         if self.use_encoder:
#             self.alphas_t = torch.zeros((x.size(0), self.t_context, self.horizon + 1), device=x.device)
#
#         for t in forecast_horizon:
#
#             r = torch.rand(1)
#             if r < self.teacher_forcing:
#                 # if data is available use ground truth, otherwise use model prediction
#                 # x = data.missing[..., t-1].view(-1, 1) * x + \
#                 #     ~data.missing[..., t-1].view(-1, 1) * data.x[..., t-1].view(-1, 1)
#                 x = data.x[..., t-1].view(-1, 1)
#
#             x, h_t, c_t = self.propagate(edge_index, x=x, coords=coords,
#                                                 h_t=h_t, c_t=c_t,
#                                                 h=h_t[-1],
#                                                 areas=data.areas,
#                                                 edge_attr=edge_attr,
#                                                 dusk=data.local_dusk[:, t-1],
#                                                 dawn=data.local_dawn[:, t],
#                                                 env=data.env[..., t],
#                                                 env_previous=data.env[..., t-1],
#                                                 t=t-self.t_context,
#                                                 boundary=data.boundary,
#                                                 night=data.local_night[:, t],
#                                                 night_previous=data.local_night[:, t-1],
#                                                 enc_states=enc_states)
#
#             if self.fixed_boundary:
#                 # use ground truth for boundary nodes
#                 x[data.boundary, 0] = data.y[data.boundary, t]
#
#             if self.force_zeros:
#                 x = x * data.local_night[:, t].view(-1, 1)
#
#             y_hat.append(x)
#
#         prediction = torch.cat(y_hat, dim=-1)
#         return prediction
#
#
#     def message(self, x_j, h_i, h_j, coords_i, coords_j, env_i, env_previous_j, edge_attr, t,
#                 night_i, night_previous_j, index):
#         # construct messages to node i for each edge (j,i)
#         # can take any argument initially passed to propagate()
#         # x_j are source features with shape [E, out_channels]
#
#         features = torch.cat([env_previous_j, night_previous_j.float().view(-1, 1),
#                               edge_attr], dim=1)
#
#         features = self.edge2hidden(features)
#         context_j = self.context_embedding(h_j)
#         context_i = self.context_embedding(h_i)
#
#         alpha = torch.tanh(features + context_i + context_j).mm(self.attention_s)
#         alpha = softmax(alpha, index)
#         self.alphas_s[..., t] = alpha
#         alpha = F.dropout(alpha, p=self.dropout_p, training=self.training, inplace=False)
#
#         msg = context_j * alpha
#         return msg
#
#
#     def update(self, aggr_out, x, coords, env, dusk, dawn, h_t, c_t, night, enc_states, t):
#
#
#         inputs = torch.cat([x.view(-1, 1), coords, env, dawn.float().view(-1, 1),
#                                 dusk.float().view(-1, 1), night.float().view(-1, 1)], dim=1)
#         inputs = self.node2hidden(inputs)
#
#         if self.use_encoder:
#             # temporal attention based on encoder states
#             enc_states = self.fc_encoder(enc_states) # shape (radars x timesteps x hidden)
#             hidden = self.fc_hidden(h_t[-1]).unsqueeze(1) # shape (radars x 1 x hidden)
#             scores = torch.matmul(torch.tanh(enc_states + hidden), self.attention_t).squeeze() # shape (radars x timesteps)
#             alpha = F.softmax(scores, dim=1)
#             self.alphas_t[..., t] = alpha
#             context = torch.matmul(alpha.unsqueeze(1), enc_states).squeeze() # shape (radars x hidden)
#
#             inputs = torch.cat([aggr_out, inputs, context], dim=1)
#         else:
#             inputs = torch.cat([aggr_out, inputs], dim=1)
#
#
#         h_t[0], c_t[0] = self.lstm_in(inputs, (h_t[0], c_t[0]))
#         h_t[0] = F.dropout(h_t[0], p=self.dropout_p, training=self.training, inplace=False)
#         c_t[0] = F.dropout(c_t[0], p=self.dropout_p, training=self.training, inplace=False)
#         for l in range(self.n_lstm_layers-1):
#             h_t[l+1], c_t[l+1] = self.lstm_layers[l](h_t[l], (h_t[l+1], c_t[l+1]))
#
#         if self.predict_delta:
#             delta = torch.tanh(self.hidden2delta(h_t[-1]))
#             pred = x + delta
#         else:
#             pred = torch.sigmoid(self.hidden2delta(h_t[-1]))
#
#         return pred, h_t, c_t
#
#
#
# class Attention2GraphLSTM(MessagePassing):
#
#     def __init__(self, **kwargs):
#         super(Attention2GraphLSTM, self).__init__(aggr='add', node_dim=0)
#
#         self.horizon = kwargs.get('horizon', 40)
#         self.dropout_p = kwargs.get('dropout_p', 0)
#         self.n_hidden = kwargs.get('n_hidden', 16)
#         self.n_env = kwargs.get('n_env', 4)
#         self.n_node_in = 6 + self.n_env
#         self.n_edge_in = 7 + self.n_env + 2*self.n_hidden
#         self.n_env_in = 3 + self.n_env
#         self.n_states_in = 3 + 2*self.n_hidden
#         self.n_fc_layers = kwargs.get('n_fc_layers', 1)
#         self.n_lstm_layers = kwargs.get('n_lstm_layers', 1)
#         self.fixed_boundary = kwargs.get('fixed_boundary', [])
#         self.force_zeros = kwargs.get('force_zeros', True)
#         self.teacher_forcing = kwargs.get('teacher_forcing', 0)
#
#         self.use_encoder = kwargs.get('use_encoder', False)
#         self.t_context = kwargs.get('context', 0)
#
#         seed = kwargs.get('seed', 1234)
#         torch.manual_seed(seed)
#
#
#         self.edge_env2hidden = torch.nn.Linear(self.n_env_in, self.n_hidden)
#         self.edge_states2hidden = torch.nn.Linear(self.n_states_in, self.n_hidden)
#         # self.edge2hidden = torch.nn.Linear(self.n_edge_in, self.n_hidden)
#         self.context_embedding1 = torch.nn.Linear(2*self.n_hidden, self.n_hidden)
#         self.context_embedding2 = torch.nn.Linear(2 * self.n_hidden, self.n_hidden)
#         self.attention1 = torch.nn.Parameter(torch.Tensor(self.n_hidden, 1))
#         self.attention2 = torch.nn.Parameter(torch.Tensor(self.n_hidden, 1))
#
#
#         self.node2hidden = torch.nn.Linear(self.n_node_in, self.n_hidden)
#
#         self.lstm_layers = nn.ModuleList([nn.LSTMCell(2*self.n_hidden, 2*self.n_hidden) for _ in range(self.n_lstm_layers)])
#
#         self.hidden2delta = torch.nn.Sequential(torch.nn.Linear(2*self.n_hidden, self.n_hidden),
#                                                 torch.nn.Dropout(p=self.dropout_p),
#                                                 torch.nn.LeakyReLU(),
#                                                 torch.nn.Linear(self.n_hidden, 1))
#
#         if self.use_encoder:
#             self.encoder = RecurrentEncoder(timesteps=self.t_context, n_env=self.n_env, n_hidden=self.n_hidden,
#                                             n_lstm_layers=self.n_lstm_layers, seed=seed, dropout_p=self.dropout_p)
#
#         self.reset_parameters()
#
#
#     def reset_parameters(self):
#         inits.glorot(self.edge2hidden.weight)
#         inits.glorot(self.context_embedding.weight)
#         inits.glorot(self.attention1)
#         inits.glorot(self.attention2)
#         inits.glorot(self.node2hidden.weight)
#
#         def init_weights(m):
#             if type(m) == nn.Linear:
#                 inits.glorot(m.weight)
#                 inits.zeros(m.bias)
#             elif type(m) == nn.LSTMCell:
#                 for name, param in m.named_parameters():
#                     if 'bias' in name:
#                         inits.zeros(param)
#                     elif 'weight' in name:
#                         inits.glorot(param)
#
#         self.hidden2delta.apply(init_weights)
#         self.lstm_layers.apply(init_weights)
#
#
#     def forward(self, data):
#         # with teacher_forcing = 0.0 the model always uses previous predictions to make new predictions
#         # with teacher_forcing = 1.0 the model always uses the ground truth to make new predictions
#
#         self.edges = data.edge_index
#         y_hat = []
#
#         # initialize lstm variables
#         if self.use_encoder:
#             # push context timeseries through encoder to initialize decoder
#             h_t, c_t = self.encoder(data)
#             #x = torch.zeros(data.x.size(0)).to(data.x.device) # TODO eventually use this!?
#             x = data.x[..., self.t_context].view(-1, 1)
#             y_hat.append(x)
#
#         else:
#             # start from scratch
#             # measurement at t=0
#             x = data.x[..., 0].view(-1, 1)
#             y_hat.append(x)
#             h_t = [torch.zeros(data.x.size(0), 2*self.n_hidden, device=x.device) for l in range(self.n_lstm_layers)]
#             c_t = [torch.zeros(data.x.size(0), 2*self.n_hidden, device=x.device) for l in range(self.n_lstm_layers)]
#
#
#         coords = data.coords
#         edge_index = data.edge_index
#         edge_attr = data.edge_attr
#
#
#         if self.use_encoder:
#             forecast_horizon = range(self.t_context + 1, self.t_context + self.horizon + 1)
#         else:
#             forecast_horizon = range(1, self.horizon + 1)
#
#         self.alphas1 = torch.zeros((edge_index.size(1), 1, self.horizon + 1), device=x.device)
#         self.alphas2 = torch.zeros((edge_index.size(1), 1, self.horizon + 1), device=x.device)
#
#         for t in forecast_horizon:
#
#             r = torch.rand(1)
#             if r < self.teacher_forcing:
#                 # if data is available use ground truth, otherwise use model prediction
#                 x = data.missing[..., t-1].view(-1, 1) * x + \
#                     ~data.missing[..., t-1].view(-1, 1) * data.x[..., t-1].view(-1, 1)
#
#             x, h_t, c_t = self.propagate(edge_index, x=x, coords=coords,
#                                                 h_t=h_t, c_t=c_t,
#                                                 h=h_t[-1],
#                                                 areas=data.areas,
#                                                 edge_attr=edge_attr,
#                                                 dusk=data.local_dusk[:, t-1],
#                                                 dawn=data.local_dawn[:, t],
#                                                 env=data.env[..., t],
#                                                 env_previous=data.env[..., t-1],
#                                                 t=t-self.t_context,
#                                                 boundary=data.boundary,
#                                                 night=data.local_night[:, t],
#                                                 night_previous=data.local_night[:, t-1])
#
#             if self.fixed_boundary:
#                 # use ground truth for boundary nodes
#                 x[data.boundary, 0] = data.y[data.boundary, t]
#
#             if self.force_zeros:
#                 x = x * data.local_night[:, t].view(-1, 1)
#
#             y_hat.append(x)
#
#         prediction = torch.cat(y_hat, dim=-1)
#         return prediction
#
#
#     def message(self, x_j, h_i, h_j, coords_i, coords_j, env_i, env_previous_j, edge_attr, t,
#                 night_i, night_previous_j, index):
#         # construct messages to node i for each edge (j,i)
#         # can take any argument initially passed to propagate()
#         # x_j are source features with shape [E, out_channels]
#
#         env_features = torch.cat([env_previous_j, night_previous_j.float().view(-1, 1),
#                               edge_attr], dim=1)
#         state_features = torch.cat([x_j, h_j, edge_attr], dim=1)
#         env_features = self.edge_env2hidden(env_features)
#         state_features = self.edge_states2hidden(state_features)
#
#         context1 = self.context_embedding1(h_i)
#         context2 = self.context_embedding2(h_i)
#
#         # alpha = (features + context).tanh().mm(self.attention)
#         # alpha = softmax(alpha, index)
#         # alpha = F.dropout(alpha, p=self.dropout_p, training=self.training)
#
#         #alpha = F.leaky_relu(self.attention.T * torch.cat([features, context], dim=1)))
#         alpha1 = (env_features + context1).tanh().mm(self.attention1)
#         alpha1 = softmax(alpha1, index)
#         alpha1 = F.dropout(alpha1, p=self.dropout_p, training=self.training)
#
#         alpha2 = (state_features + context2).tanh().mm(self.attention2)
#         alpha2 = softmax(alpha2, index)
#         alpha2 = F.dropout(alpha2, p=self.dropout_p, training=self.training)
#
#         self.alphas1[..., t] = alpha1
#         self.alphas1[..., t] = alpha2
#         msg = (env_features * alpha1) + (state_features * alpha2)
#         return msg
#
#
#     def update(self, aggr_out, x, coords, env, dusk, dawn, h_t, c_t, night):
#
#
#         inputs = torch.cat([x.view(-1, 1), coords, env, dawn.float().view(-1, 1),
#                                 dusk.float().view(-1, 1), night.float().view(-1, 1)], dim=1)
#         # TODO add attention mechanism to take past conditions into account (encoder)?
#         inputs = self.node2hidden(inputs)
#
#         inputs = torch.cat([aggr_out, inputs], dim=1)
#
#
#         h_t[0], c_t[0] = self.lstm_layers[0](inputs, (h_t[0], c_t[0]))
#         for l in range(1, self.n_lstm_layers):
#             h_t[l], c_t[l] = self.lstm_layers[l](h_t[l - 1], (h_t[l], c_t[l]))
#
#         delta = self.hidden2delta(h_t[-1]).tanh()
#         pred = x + delta
#
#         return pred, h_t, c_t
#
#

#
#
# class BirdDynamicsGraphLSTM(MessagePassing):
#
#     def __init__(self, **kwargs):
#         super(BirdDynamicsGraphLSTM, self).__init__(aggr='add', node_dim=0) # inflows from neighbouring radars are aggregated by adding
#
#         self.horizon = kwargs.get('horizon', 40)
#         self.dropout_p = kwargs.get('dropout_p', 0)
#         self.n_hidden = kwargs.get('n_hidden', 16)
#         self.n_env = kwargs.get('n_env', 4)
#         self.n_node_in = 6 + self.n_env
#         self.n_edge_in = 11 + 2 * self.n_env
#         self.n_fc_layers = kwargs.get('n_fc_layers', 1)
#         self.n_lstm_layers = kwargs.get('n_lstm_layers', 1)
#         self.predict_delta = kwargs.get('predict_delta', True)
#         self.fixed_boundary = kwargs.get('fixed_boundary', [])
#         self.force_zeros = kwargs.get('force_zeros', [])
#         self.use_encoder = kwargs.get('use_encoder', False)
#         self.t_context = kwargs.get('context', 0)
#         self.teacher_forcing = kwargs.get('teacher_forcing', 0)
#
#         self.edge_type = kwargs.get('edge_type', 'voronoi')
#         if self.edge_type == 'voronoi':
#             print('Use Voronoi tessellation')
#             self.n_edge_in += 1  # use face_length as additional feature
#             self.n_node_in += 1  # use voronoi cell area as additional feature
#
#         seed = kwargs.get('seed', 1234)
#         torch.manual_seed(seed)
#
#
#         self.fc_edge_in = torch.nn.Linear(self.n_edge_in, self.n_hidden)
#         self.fc_edge_hidden = nn.ModuleList([torch.nn.Linear(self.n_hidden, self.n_hidden)
#                                         for _ in range(self.n_fc_layers - 1)])
#         self.fc_edge_out = torch.nn.Linear(self.n_hidden, self.n_hidden)
#
#         # self.mlp_edge = torch.nn.Sequential(torch.nn.Linear(self.n_in_edge, self.n_hidden),
#         #                                     torch.nn.Dropout(p=self.dropout_p),
#         #                                     torch.nn.ReLU(),
#         #                                     torch.nn.Linear(self.n_hidden, self.n_hidden))
#
#         self.mlp_aggr = torch.nn.Sequential(torch.nn.Linear(self.n_hidden, self.n_hidden),
#                                             torch.nn.Dropout(p=self.dropout_p),
#                                             torch.nn.LeakyReLU(),
#                                             torch.nn.Linear(self.n_hidden, 1))
#
#         self.node2hidden = torch.nn.Sequential(torch.nn.Linear(self.n_node_in, self.n_hidden),
#                                                torch.nn.Dropout(p=self.dropout_p),
#                                                torch.nn.LeakyReLU(),
#                                                torch.nn.Linear(self.n_hidden, self.n_hidden))
#
#         self.lstm_layers = nn.ModuleList([nn.LSTMCell(self.n_hidden, self.n_hidden) for _ in range(self.n_lstm_layers)])
#
#         self.hidden2delta = torch.nn.Sequential(torch.nn.Linear(self.n_hidden, self.n_hidden),
#                                                torch.nn.Dropout(p=self.dropout_p),
#                                                torch.nn.LeakyReLU(),
#                                                torch.nn.Linear(self.n_hidden, 1))
#
#         if self.use_encoder:
#             self.encoder = RecurrentEncoder(timesteps=self.t_context, n_env=self.n_env, n_hidden=self.n_hidden,
#                                             n_lstm_layers=self.n_lstm_layers, seed=seed, dropout_p=self.dropout_p)
#
#
#
#     def forward(self, data):
#         # with teacher_forcing = 0.0 the model always uses previous predictions to make new predictions
#         # with teacher_forcing = 1.0 the model always uses the ground truth to make new predictions
#
#         y_hat = []
#
#         # initialize lstm variables
#         if self.use_encoder:
#             # push context timeseries through encoder to initialize decoder
#             h_t, c_t = self.encoder(data)
#             #x = torch.zeros(data.x.size(0)).to(data.x.device) # TODO eventually use this!?
#             x = data.x[..., self.t_context].view(-1, 1)
#             y_hat.append(x)
#         else:
#             # start from scratch
#             # measurement at t=0
#             x = data.x[..., 0].view(-1, 1)
#             y_hat.append(x)
#             h_t = [torch.zeros(data.x.size(0), self.n_hidden, device=x.device) for l in range(self.n_lstm_layers)]
#             c_t = [torch.zeros(data.x.size(0), self.n_hidden, device=x.device) for l in range(self.n_lstm_layers)]
#
#         coords = data.coords
#         edge_index = data.edge_index
#         edge_attr = data.edge_attr
#
#         self.fluxes = torch.zeros((data.x.size(0), 1, self.horizon + 1), device=x.device)
#         self.local_deltas = torch.zeros((data.x.size(0), 1, self.horizon + 1), device=x.device)
#
#         if self.use_encoder:
#             forecast_horizon = range(self.t_context + 1, self.t_context + self.horizon + 1)
#         else:
#             forecast_horizon = range(1, self.horizon + 1)
#
#         for t in forecast_horizon:
#
#             if True: #torch.any(data.local_night[:, t] | data.local_dusk[:, t]):
#                 # at least for one radar station it is night or dusk
#                 r = torch.rand(1)
#                 if r < self.teacher_forcing:
#                     # if data is available use ground truth, otherwise use model prediction
#                     x = data.missing[..., t-1].view(-1, 1) * x + \
#                         ~data.missing[..., t-1].view(-1, 1) * data.x[..., t-1].view(-1, 1)
#
#
#                 x, h_t, c_t = self.propagate(edge_index, x=x, coords=coords,
#                                    edge_attr=edge_attr, h_t=h_t, c_t=c_t, areas=data.areas,
#                                    dusk=data.local_dusk[:, t-1],
#                                    dawn=data.local_dawn[:, t],
#                                    env=data.env[..., t],
#                                    night=data.local_night[:, t],
#                                    t=t-self.t_context,
#                                    boundary=data.boundary)
#
#                 if self.fixed_boundary:
#                     # use ground truth for boundary cells
#                     x[data.boundary, 0] = data.y[data.boundary, t]
#
#             if self.force_zeros:
#                 # for locations where it is dawn: set birds in the air to zero
#                 x = x * data.local_night[:, t].view(-1, 1)
#
#             y_hat.append(x)
#
#         prediction = torch.cat(y_hat, dim=-1)
#         return prediction
#
#
#     def message(self, x_i, x_j, coords_i, coords_j, env_i, env_j, edge_attr, dusk_j, dawn_j, night_j):
#         # construct messages to node i for each edge (j,i)
#         # can take any argument initially passed to propagate()
#         # x_j are source features with shape [E, out_channels]
#
#         features = torch.cat([x_i.view(-1, 1), x_j.view(-1, 1), coords_i, coords_j, env_i, env_j, edge_attr,
#                               dusk_j.float().view(-1, 1), dawn_j.float().view(-1, 1), night_j.float().view(-1, 1)], dim=1)
#         #msg = self.mlp_edge(features).relu()
#
#         msg = F.leaky_relu(self.fc_edge_in(features))
#         msg = F.dropout(msg, p=self.dropout_p, training=self.training)
#
#         for l in self.fc_edge_hidden:
#             msg = F.leaky_relu(l(msg))
#             msg = F.dropout(msg, p=self.dropout_p, training=self.training)
#
#         msg = self.fc_edge_out(msg) #.relu()
#
#         return msg
#
#
#     def update(self, aggr_out, x, coords, env, areas, dusk, dawn, h_t, c_t, t, night, boundary):
#
#         # predict departure/landing
#         if self.edge_type == 'voronoi':
#             inputs = torch.cat([x.view(-1, 1), coords, env, dusk.float().view(-1, 1),
#                                 dawn.float().view(-1, 1), areas.view(-1, 1), night.float().view(-1, 1)], dim=1)
#         else:
#             inputs = torch.cat([x.view(-1, 1), coords, env, dusk.float().view(-1, 1),
#                                 dawn.float().view(-1, 1), night.float().view(-1, 1)], dim=1)
#         inputs = self.node2hidden(inputs) #.relu()
#         h_t[0], c_t[0] = self.lstm_layers[0](inputs, (h_t[0], c_t[0]))
#         for l in range(1, self.n_lstm_layers):
#             h_t[l], c_t[l] = self.lstm_layers[l](h_t[l - 1], (h_t[l], c_t[l]))
#         delta = self.hidden2delta(h_t[-1]) #.tanh()
#
#         if self.predict_delta:
#             # combine messages from neighbors into single number representing total flux
#             flux = self.mlp_aggr(aggr_out) #.tanh()
#             pred = x + ~boundary.view(-1, 1) * flux + delta
#         else:
#             # combine messages from neighbors into single number representing the new bird density
#             birds = self.mlp_aggr(aggr_out).sigmoid()
#             pred = birds + delta
#
#
#         self.fluxes[..., t] = flux
#         self.local_deltas[..., t] = delta
#
#         return pred, h_t, c_t


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
    model.train()
    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        model.teacher_forcing = teacher_forcing
        output = model(data) #.view(-1)
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
        loss_all += data.num_graphs * float(loss)
        optimizer.step()

    return loss_all

def flux_penalty(model, data, weight):

    inferred_fluxes = model.local_fluxes.squeeze()
    inferred_fluxes = inferred_fluxes - inferred_fluxes[data.reverse_edges]
    observed_fluxes = data.fluxes[..., model.t_context:].squeeze()

    diff = observed_fluxes - inferred_fluxes
    diff = torch.square(observed_fluxes) * diff  # weight timesteps with larger fluxes more

    edges = data.boundary2inner_edges + data.inner2boundary_edges + data.inner_edges
    diff = diff[edges]
    penalty = (torch.square(diff[~torch.isnan(diff)])).mean()
    penalty = weight * penalty

    return penalty

def train(model, train_loader, optimizer, loss_func, device, teacher_forcing=0, **kwargs):

    model.train()
    loss_all = 0
    flux_loss_weight = kwargs.get('flux_loss_weight', 0)
    for nidx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        if hasattr(model, 'teacher_forcing'):
            model.teacher_forcing = teacher_forcing
        output = model(data)
        gt = data.y

        if flux_loss_weight > 0:
            penalty = flux_penalty(model, data, flux_loss_weight)
        else:
            penalty = 0

        if kwargs.get('force_zeros', False):
            mask = torch.logical_and(data.local_night, torch.logical_not(data.missing))
        else:
            mask = torch.logical_not(data.missing)

        if hasattr(model, 't_context'):
            gt = gt[:, model.t_context:]
            mask = mask[:, model.t_context:]

        loss = loss_func(output, gt, mask) + penalty
        loss_all += data.num_graphs * float(loss)
        loss.backward()
        optimizer.step()

        del loss, output

    return loss_all



def train_dynamics(model, train_loader, optimizer, loss_func, device, teacher_forcing=0, daymask=True):

    model.train()
    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        model.teacher_forcing = teacher_forcing
        output = model(data)
        gt = data.y

        if daymask:
            mask = torch.logical_and(data.local_night, torch.logical_not(data.missing))
        else:
            mask = torch.logical_not(data.missing)

        if hasattr(model, 't_context'):
            gt = gt[:, model.t_context:]
            mask = mask[:, model.t_context:]

        loss = loss_func(output, gt, mask)
        loss_all += data.num_graphs * float(loss)
        loss.backward()
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
                fixed_boundary=False, daymask=True):
    model.eval()
    loss_all = []
    outfluxes = {}
    outfluxes_abs = {}
    for tidx, data in enumerate(test_loader):
        data = data.to(device)
        output = model(data) * bird_scale #.view(-1)
        gt = data.y * bird_scale

        if fixed_boundary:
            # boundary_mask = np.ones(output.size(0))
            # boundary_mask[data.boundary] = 0
            output = output[~data.boundary]
            gt = gt[~data.boundary]

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
                                      for t in range(model.horizon + 1)]))
        #loss_all.append(loss_func(output, gt))
        #constraints_all.append(constraints)

    if get_outfluxes:
        return torch.stack(loss_all), outfluxes , outfluxes_abs
    else:
        return torch.stack(loss_all)

def test(model, test_loader, loss_func, device, **kwargs):

    model.eval()
    loss_all = []

    if hasattr(model, 'teacher_forcing'):
        model.teacher_forcing = 0

    for tidx, data in enumerate(test_loader):
        data = data.to(device)
        output = model(data)
        gt = data.y

        if kwargs.get('fixed_boundary', False):
            output = output[~data.boundary]
            gt = gt[~data.boundary]

        if kwargs.get('force_zeros', False):
            mask = data.local_night & ~data.missing
        else:
            mask = ~data.missing

        if hasattr(model, 't_context'):
            gt = gt[:, model.t_context:]
            mask = mask[:, model.t_context:]

        loss_all.append(torch.tensor([loss_func(output[:, t], gt[:, t], mask[:, t]).detach()
                                      for t in range(model.horizon)]))

    return torch.stack(loss_all)


def test_dynamics(model, test_loader, loss_func, device, bird_scale=2000, daymask=True):

    model.eval()
    model.teacher_forcing = 0
    loss_all = []

    for nidx, data in enumerate(test_loader):
        data = data.to(device)

        with torch.no_grad():
            output = model(data) * bird_scale #/ data.areas.view(-1, 1)
            gt = data.y * bird_scale #/ data.areas.view(-1, 1)

            if daymask:
                mask = data.local_night & ~data.missing
            else:
                mask = ~data.missing
            if hasattr(model, 't_context'):
                gt = gt[:, model.t_context:]
                mask = mask[:, model.t_context:]

        loss_all.append(torch.tensor([loss_func(output[:, t], gt[:, t], mask[:, t])
                                      for t in range(model.horizon + 1)]).detach())
        del data, output

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

