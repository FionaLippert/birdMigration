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

    def forward(self, data):

        y_hat = []
        for t in range(self.horizon + 1):

            features = torch.cat([data.coords.flatten(),
                                  data.env[..., t].flatten()], dim=0)
            x = self.fc_in(features)
            x = F.leaky_relu(x)
            x = F.dropout(x, p=self.dropout_p, training=self.training)

            for l in self.fc_hidden:
                x = l(x)
                x = F.leaky_relu(x)
                x = F.dropout(x, p=self.dropout_p, training=self.training)

            x = self.fc_out(x)
            x = x.sigmoid()

            # for locations where it is night: set birds in the air to zero
            x = x * data.local_night[:, t]

            y_hat.append(x)

        return torch.stack(y_hat, dim=1)

class FluxMLP(torch.nn.Module):

    def __init__(self, **kwargs):
        super(FluxMLP, self).__init__()

        torch.manual_seed(kwargs.get('seed', 1234))

        self.dropout_p = kwargs.get('dropout_p', 0)
        self.n_hidden = kwargs.get('n_hidden_fluxmlp', 16)
        self.n_env = kwargs.get('n_env', 4)
        self.n_in = 10 + 2 * self.n_env
        self.n_fc_layers = kwargs.get('n_fc_layers_fluxmlp', 1)

        self.fc_in = torch.nn.Linear(self.n_in, self.n_hidden)
        self.fc_hidden = nn.ModuleList([torch.nn.Linear(self.n_hidden, self.n_hidden)
                                        for _ in range(self.n_fc_layers - 1)])
        self.fc_out = torch.nn.Linear(self.n_hidden, 1)

        self.reset_parameters()

    def reset_parameters(self):

        def init_weights(m):
            if type(m) == nn.Linear:
                inits.glorot(m.weight)
                inits.zeros(m.bias)
            elif type(m) == nn.LSTMCell:
                for name, param in m.named_parameters():
                    if 'bias' in name:
                        inits.zeros(param)
                    elif 'weight' in name:
                        inits.glorot(param)

        self.fc_hidden.apply(init_weights)
        init_weights(self.fc_in)
        init_weights(self.fc_out)


    def forward(self, env_1_j, env_i, night_1_j, night_i, coords_j, coords_i, edge_attr, day_of_year):

        features = torch.cat([env_1_j, env_i, night_1_j.float().view(-1, 1), night_i.float().view(-1, 1),
                              coords_j, coords_i, edge_attr, day_of_year.view(-1, 1)], dim=1)
        # features = torch.cat([env_1_j, env_i, night_1_j.float().view(-1, 1), night_i.float().view(-1, 1),
        #                       coords_j, coords_i, edge_attr], dim=1)

        flux = F.leaky_relu(self.fc_in(features))
        flux = F.dropout(flux, p=self.dropout_p, training=self.training)

        for l in self.fc_hidden:
            flux = F.leaky_relu(l(flux))
            flux = F.dropout(flux, p=self.dropout_p, training=self.training)

        flux = self.fc_out(flux)
        flux = flux.sigmoid()
        return flux


class FluxMLP2(torch.nn.Module):

    def __init__(self, **kwargs):
        super(FluxMLP2, self).__init__()

        torch.manual_seed(kwargs.get('seed', 1234))

        self.dropout_p = kwargs.get('dropout_p', 0)
        self.n_hidden = kwargs.get('n_hidden_fluxmlp', 16)
        self.n_env = kwargs.get('n_env', 4)
        self.n_in = 11 + 2 * self.n_env
        self.n_fc_layers = kwargs.get('n_fc_layers_fluxmlp', 1)

        self.fc_emb = torch.nn.Linear(self.n_in, self.n_hidden)
        self.fc_in = torch.nn.Linear(self.n_hidden + kwargs.get('n_hidden', 16), self.n_hidden)
        self.fc_hidden = nn.ModuleList([torch.nn.Linear(self.n_hidden, self.n_hidden)
                                        for _ in range(self.n_fc_layers - 1)])
        self.fc_out = torch.nn.Linear(self.n_hidden, 1)

        self.reset_parameters()

    def reset_parameters(self):

        def init_weights(m):
            if type(m) == nn.Linear:
                inits.glorot(m.weight)
                inits.zeros(m.bias)
            elif type(m) == nn.LSTMCell:
                for name, param in m.named_parameters():
                    if 'bias' in name:
                        inits.zeros(param)
                    elif 'weight' in name:
                        inits.glorot(param)

        self.fc_hidden.apply(init_weights)
        init_weights(self.fc_in)
        init_weights(self.fc_out)
        init_weights(self.fc_emb)


    def forward(self, h_i, x_i, env_1_j, env_i, night_1_j, night_i,
                coords_j, coords_i, edge_attr, day_of_year):

        features = torch.cat([x_i.view(-1, 1), env_1_j, env_i, night_1_j.float().view(-1, 1), night_i.float().view(-1, 1),
                              coords_j, coords_i, edge_attr, day_of_year.view(-1, 1)], dim=1)
        # features = torch.cat([env_1_j, env_i, night_1_j.float().view(-1, 1), night_i.float().view(-1, 1),
        #                       coords_j, coords_i, edge_attr], dim=1)

        features = self.fc_emb(features)
        features = torch.cat([features, h_i], dim=1)
        flux = self.fc_in(features)
        flux = F.dropout(flux, p=self.dropout_p, training=self.training)

        for l in self.fc_hidden:
            flux = F.leaky_relu(l(flux))
            flux = F.dropout(flux, p=self.dropout_p, training=self.training)

        flux = self.fc_out(flux)
        #flux = flux.relu()
        flux = flux.sigmoid()
        return flux


class FluxMLP4(torch.nn.Module):

    def __init__(self, **kwargs):
        super(FluxMLP4, self).__init__()

        torch.manual_seed(kwargs.get('seed', 1234))

        self.dropout_p = kwargs.get('dropout_p', 0)
        self.n_hidden = kwargs.get('n_hidden_fluxmlp', 16)
        self.n_env = kwargs.get('n_env', 4)
        self.n_in = 11 + 4 * self.n_env
        self.n_fc_layers = kwargs.get('n_fc_layers_fluxmlp', 1)

        self.fc_emb = torch.nn.Linear(self.n_in, self.n_hidden)
        self.fc_hidden = nn.ModuleList([torch.nn.Linear(self.n_hidden, self.n_hidden)
                                        for _ in range(self.n_fc_layers - 1)])
        self.fc_out = torch.nn.Linear(self.n_hidden, 1)

        self.reset_parameters()

    def reset_parameters(self):

        def init_weights(m):
            if type(m) == nn.Linear:
                inits.glorot(m.weight)
                inits.zeros(m.bias)
            elif type(m) == nn.LSTMCell:
                for name, param in m.named_parameters():
                    if 'bias' in name:
                        inits.zeros(param)
                    elif 'weight' in name:
                        inits.glorot(param)

        self.fc_hidden.apply(init_weights)
        init_weights(self.fc_out)
        init_weights(self.fc_emb)


    def forward(self, x_i, env_1_j, env_1_i, env_j, env_i, night_j, night_i,
                coords_j, coords_i, edge_attr, day_of_year):

        features = torch.cat([x_i.view(-1, 1), env_1_j, env_1_i, env_j, env_i,
                              night_j.float().view(-1, 1), night_i.float().view(-1, 1),
                              coords_j, coords_i, edge_attr, day_of_year.view(-1, 1)], dim=1)
        # features = torch.cat([x_i.view(-1, 1),
        #                       night_j.float().view(-1, 1), night_i.float().view(-1, 1),
        #                       coords_j, coords_i, edge_attr, day_of_year.view(-1, 1)], dim=1)
        # features = torch.cat([env_1_j, env_i, night_1_j.float().view(-1, 1), night_i.float().view(-1, 1),
        #                       coords_j, coords_i, edge_attr], dim=1)

        features = self.fc_emb(features)
        flux = F.dropout(features, p=self.dropout_p, training=self.training)

        for l in self.fc_hidden:
            flux = F.leaky_relu(l(flux))
            flux = F.dropout(flux, p=self.dropout_p, training=self.training)

        flux = self.fc_out(flux)
        #flux = flux.relu()
        #flux = flux.tanh()
        return flux


class FluxMLP3(torch.nn.Module):

    def __init__(self, **kwargs):
        super(FluxMLP3, self).__init__()

        torch.manual_seed(kwargs.get('seed', 1234))

        self.dropout_p = kwargs.get('dropout_p', 0)
        self.n_hidden = kwargs.get('n_hidden_fluxmlp', 16)
        self.n_env = kwargs.get('n_env', 4)
        self.n_in = 11 + 4 * self.n_env
        self.n_fc_layers = kwargs.get('n_fc_layers_fluxmlp', 1)

        self.fc_emb = torch.nn.Linear(self.n_in, self.n_hidden)
        self.fc_in = torch.nn.Linear(self.n_hidden + kwargs.get('n_hidden', 16), self.n_hidden)
        self.fc_hidden = nn.ModuleList([torch.nn.Linear(self.n_hidden, self.n_hidden)
                                        for _ in range(self.n_fc_layers - 1)])
        self.fc_out = torch.nn.Linear(self.n_hidden, 1)

        self.reset_parameters()

    def reset_parameters(self):

        def init_weights(m):
            if type(m) == nn.Linear:
                inits.glorot(m.weight)
                inits.zeros(m.bias)
            elif type(m) == nn.LSTMCell:
                for name, param in m.named_parameters():
                    if 'bias' in name:
                        inits.zeros(param)
                    elif 'weight' in name:
                        inits.glorot(param)

        self.fc_hidden.apply(init_weights)
        init_weights(self.fc_in)
        init_weights(self.fc_out)
        init_weights(self.fc_emb)


    def forward(self, h_i, x_i, env_1_j, env_1_i, env_j, env_i, night_j, night_i,
                coords_j, coords_i, edge_attr, day_of_year):

        features = torch.cat([x_i.view(-1, 1), env_1_j, env_1_i, env_j, env_i,
                              night_j.float().view(-1, 1), night_i.float().view(-1, 1),
                              coords_j, coords_i, edge_attr, day_of_year.view(-1, 1)], dim=1)
        # features = torch.cat([env_1_j, env_i, night_1_j.float().view(-1, 1), night_i.float().view(-1, 1),
        #                       coords_j, coords_i, edge_attr], dim=1)

        features = self.fc_emb(features)
        features = torch.cat([features, h_i], dim=1)
        flux = self.fc_in(features)
        flux = F.dropout(flux, p=self.dropout_p, training=self.training)

        for l in self.fc_hidden:
            flux = F.leaky_relu(l(flux))
            flux = F.dropout(flux, p=self.dropout_p, training=self.training)

        flux = self.fc_out(flux)
        #flux = flux.relu()
        flux = flux.tanh()
        return flux




class LocalMLP(torch.nn.Module):

    def __init__(self, **kwargs):
        super(LocalMLP, self).__init__()

        self.horizon = kwargs.get('horizon', 40)
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

        self.reset_parameters()

    def reset_parameters(self):

        def init_weights(m):
            if type(m) == nn.Linear:
                inits.glorot(m.weight)
                inits.zeros(m.bias)
            elif type(m) == nn.LSTMCell:
                for name, param in m.named_parameters():
                    if 'bias' in name:
                        inits.zeros(param)
                    elif 'weight' in name:
                        inits.glorot(param)

        self.fc_hidden.apply(init_weights)
        init_weights(self.fc_in)
        init_weights(self.fc_out)


    def forward(self, data, **kwargs):

        y_hat = []

        for t in range(self.horizon + 1):

            x = self.step(data.coords, data.env[..., t], data.areas, night=data.local_night[:, t], acc=data.acc[..., t])

            if self.force_zeros:
                print('force birds in air to be zero')
                # for locations where it is night: set birds in the air to zero
                x = x * data.local_night[:, t].view(-1, 1)

            y_hat.append(x)

        prediction = torch.cat(y_hat, dim=-1)
        return prediction



    def step(self, coords, env, areas, night, acc):
        # use only location-specific features to predict migration intensities
        if self.use_acc:
            features = torch.cat([coords, env, areas.view(-1, 1), night.float().view(-1, 1), acc], dim=1)
        else:
            features = torch.cat([coords, env, areas.view(-1, 1), night.float().view(-1, 1)], dim=1)
        x = F.leaky_relu(self.fc_in(features))
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        for l in self.fc_hidden:
            x = F.leaky_relu(l(x))
            x = F.dropout(x, p=self.dropout_p, training=self.training)

        x = self.fc_out(x)
        x = x.sigmoid()

        return x


class LocalLSTM(torch.nn.Module):

    def __init__(self, **kwargs):
        super(LocalLSTM, self).__init__()

        self.horizon = kwargs.get('horizon', 40)
        self.dropout_p = kwargs.get('dropout_p', 0)
        self.n_hidden = kwargs.get('n_hidden', 16)
        self.n_env = kwargs.get('n_env', 4)
        self.n_in = 7 + self.n_env
        self.n_lstm_layers = kwargs.get('n_lstm_layers', 1)
        self.t_context = kwargs.get('context', 0)
        self.teacher_forcing = kwargs.get('teacher_forcing', 0)
        self.predict_delta = kwargs.get('predict_delta', True)
        self.force_zeros = kwargs.get('force_zeros', True)
        self.use_encoder = kwargs.get('use_encoder', False)
        self.use_mtr_features = kwargs.get('use_mtr_features', False)
        self.return_hidden_states = kwargs.get('return_hidden_states', False)

        seed = kwargs.get('seed', 1234)
        torch.manual_seed(seed)

        self.fc_in = torch.nn.Linear(self.n_in, self.n_hidden)
        # self.mlp_in = torch.nn.Sequential(torch.nn.Linear(self.n_in, self.n_hidden),
        #                                   torch.nn.Dropout(p=self.dropout_p),
        #                                   torch.nn.ReLU(),
        #                                   torch.nn.Linear(self.n_hidden, self.n_hidden))
        if self.use_encoder:
            self.lstm_in = torch.nn.LSTMCell(self.n_hidden * 2, self.n_hidden)
        else:
            self.lstm_in = torch.nn.LSTMCell(self.n_hidden, self.n_hidden)
        self.lstm_layers = nn.ModuleList([torch.nn.LSTMCell(self.n_hidden, self.n_hidden)
                                          for _ in range(self.n_lstm_layers-1)])
        #self.fc_out = torch.nn.Linear(self.n_hidden, self.n_out)
        self.mlp_out = torch.nn.Sequential(torch.nn.Linear(self.n_hidden, self.n_hidden),
                                          torch.nn.Dropout(p=self.dropout_p),
                                          torch.nn.LeakyReLU(),
                                          torch.nn.Linear(self.n_hidden, 1))

        if self.use_encoder:
            self.attention_t = torch.nn.Parameter(torch.Tensor(self.n_hidden, 1))
            if self.use_mtr_features:
                self.encoder = RecurrentEncoder(timesteps=self.t_context, n_env=self.n_env, n_hidden=self.n_hidden,
                                                n_lstm_layers=self.n_lstm_layers, seed=seed, dropout_p=self.dropout_p)
            else:
                self.encoder = RecurrentEncoder2(timesteps=self.t_context, n_env=self.n_env, n_hidden=self.n_hidden,
                                                 n_lstm_layers=self.n_lstm_layers, seed=seed, dropout_p=self.dropout_p)
            self.fc_encoder = torch.nn.Linear(self.n_hidden, self.n_hidden)
            self.fc_hidden = torch.nn.Linear(self.n_hidden, self.n_hidden)

        self.reset_parameters()

    def reset_parameters(self):

        def init_weights(m):
            if type(m) == nn.Linear:
                inits.glorot(m.weight)
                inits.zeros(m.bias)
            elif type(m) == nn.LSTMCell:
                for name, param in m.named_parameters():
                    if 'bias' in name:
                        inits.zeros(param)
                    elif 'weight' in name:
                        inits.glorot(param)

        self.mlp_out.apply(init_weights)
        self.lstm_layers.apply(init_weights)
        init_weights(self.lstm_in)

        init_weights(self.fc_in)

        if self.use_encoder:
            init_weights(self.fc_encoder)
            init_weights(self.fc_hidden)
            init_weights(self.attention_t)


    def forward(self, data):

        x = data.x[..., self.t_context].view(-1, 1)
        y_hat = [x]

        #teacher_forcing = kwargs.get('teacher_forcing', 0)
        enc_states = None
        if self.use_encoder:
            # push context timeseries through encoder to initialize decoder
            enc_states, h_t, c_t = self.encoder(data)
            #x = torch.zeros(data.x.size(0)).to(data.x.device) # TODO eventually use this!?

        else:
            h_t = [torch.zeros(data.x.size(0), self.n_hidden, device=x.device) for l in range(self.n_lstm_layers)]
            c_t = [torch.zeros(data.x.size(0), self.n_hidden, device=x.device) for l in range(self.n_lstm_layers)]



        coords = data.coords
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        if self.use_encoder:
            forecast_horizon = range(self.t_context + 1, self.t_context + self.horizon + 1)
        else:
            forecast_horizon = range(1, self.horizon + 1)

        if self.use_encoder and not self.training:
            self.alphas_t = torch.zeros((x.size(0), self.t_context, self.horizon + 1), device=x.device)

        all_h = torch.zeros((data.x.size(0), self.n_hidden, self.horizon), device=x.device)

        for t in forecast_horizon:
            all_h[..., t-1-self.t_context] = h_t[-1]

            r = torch.rand(1)
            if r < self.teacher_forcing:
                # if data is available use ground truth, otherwise use model prediction
                x = data.missing[..., t-1].view(-1, 1) * x + \
                    ~data.missing[..., t-1].view(-1, 1) * data.x[..., t-1].view(-1, 1)

            x, h_t, c_t = self.step(x, data.coords, data.areas, data.local_dusk[:, t-1], data.local_dawn[:, t],
                                    data.env[..., t], data.local_night[:, t], h_t, c_t, enc_states, self.t_context)


            if self.force_zeros:
                # for locations where it is night: set birds in the air to zero
                x = x * data.local_night[:, t].view(-1, 1)

            y_hat.append(x)

        prediction = torch.cat(y_hat, dim=-1)

        if self.return_hidden_states:
            return prediction, all_h
        else:
            return prediction


    def step(self, x, coords, areas, dusk, dawn, env, night, h_t, c_t, enc_states, t):

        inputs = torch.cat([x.view(-1, 1), coords, env, dusk.float().view(-1, 1),
                            dawn.float().view(-1, 1), areas.view(-1, 1), night.float().view(-1, 1)], dim=1)
        inputs = self.fc_in(inputs)

        if self.use_encoder:
            # temporal attention based on encoder states
            enc_states = self.fc_encoder(enc_states) # shape (radars x timesteps x hidden)
            hidden = self.fc_hidden(h_t[-1]).unsqueeze(1) # shape (radars x 1 x hidden)
            scores = torch.tanh(enc_states + hidden).matmul(self.attention_t).squeeze() # shape (radars x timesteps)
            alpha = F.softmax(scores, dim=1)
            if not self.training:
                self.alphas_t[..., t] = alpha
            context = alpha.unsqueeze(1).matmul(enc_states).squeeze() # shape (radars x hidden)

            inputs = torch.cat([inputs, context], dim=1)


        h_t[0], c_t[0] = self.lstm_in(inputs, (h_t[0], c_t[0]))
        for l in range(self.n_lstm_layers - 1):
            h_t[0] = F.dropout(h_t[0], p=self.dropout_p, training=self.training, inplace=False)
            c_t[0] = F.dropout(c_t[0], p=self.dropout_p, training=self.training, inplace=False)
            h_t[l+1], c_t[l+1] = self.lstm_layers[l](h_t[l], (h_t[l+1], c_t[l+1]))

        if self.predict_delta:
            delta = self.mlp_out(h_t[-1]).tanh()
            x = x + delta
        else:
            x = self.mlp_out(h_t[-1]).sigmoid()

        return x, h_t, c_t


# class RecurrentGCN(torch.nn.Module):
#     def __init__(self, timesteps, node_features, n_hidden=32, n_out=1, K=1):
#         # doesn't take external features into account
#         super(RecurrentGCN, self).__init__()
#         self.recurrent = DCRNN(7, n_hidden, K, bias=True)
#         self.linear = torch.nn.Linear(n_hidden, n_out)
#         self.timesteps = timesteps
#
#     def forward(self, data, teacher_forcing=0):
#         x = data.x[:, 0].view(-1, 1)
#         predictions = [x]
#         for t in range(self.timesteps):
#             # TODO try concatenating input features and prection x to also use weather info etc
#             r = torch.rand(1)
#             if r < teacher_forcing:
#                 x = data.x[:, t].view(-1, 1)
#
#             input = torch.cat([x, data.env[..., t], data.coords], dim=1)
#             x = self.recurrent(input, data.edge_index, data.edge_weight.float())
#             x = F.relu(x)
#             x = self.linear(x)
#
#             # for locations where it is night: set birds in the air to zero
#             x = x * data.local_night[:, t+1].view(-1, 1)
#
#             predictions.append(x)
#
#         predictions = torch.cat(predictions, dim=-1)
#         return predictions
#

class BirdFlowGNN(MessagePassing):

    def __init__(self, num_nodes, timesteps, hidden_dim=16, embedding=0, model='linear', norm=True,
                 use_departure=False, seed=12345, fix_boundary=[], multinight=False, use_wind=True, dropout_p=0.5, **kwargs):
        super(BirdFlowGNN, self).__init__(aggr='add', node_dim=0) # inflows from neighbouring radars are aggregated by adding

        self.teacher_forcing = kwargs.get('teacher_forcing', 0)
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
                                                torch.nn.LeakyReLU(),
                                                torch.nn.Linear(hidden_channels, out_channels),
                                                torch.nn.Sigmoid())
            self.departure = torch.nn.Sequential(torch.nn.Linear(in_channels_dep, hidden_channels_dep),
                                                 torch.nn.Dropout(p=dropout_p),
                                                 torch.nn.LeakyReLU(),
                                                 torch.nn.Linear(hidden_channels_dep, out_channels_dep),
                                                 torch.nn.Tanh())


        self.node_embedding = torch.nn.Embedding(num_nodes, embedding) if embedding > 0 else None
        self.timesteps = timesteps
        self.norm = norm
        self.use_departure = use_departure
        self.fix_boundary = fix_boundary
        self.multinight = multinight
        self.use_wind = use_wind


    def forward(self, data):
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
            if r < self.teacher_forcing:
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
                if r < self.teacher_forcing:
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

        self.horizon = kwargs.get('horizon', 40)
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
        self.t_context = kwargs.get('context', 0)
        self.teacher_forcing = kwargs.get('teacher_forcing', 0)

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
                                               torch.nn.LeakyReLU(),
                                               torch.nn.Linear(self.n_hidden, self.n_hidden))

        self.lstm_layers = nn.ModuleList([nn.LSTMCell(self.n_hidden, self.n_hidden) for _ in range(self.n_lstm_layers)])

        self.hidden2delta = torch.nn.Sequential(torch.nn.Linear(self.n_hidden, self.n_hidden),
                                                torch.nn.Dropout(p=self.dropout_p),
                                                torch.nn.LeakyReLU(),
                                                torch.nn.Linear(self.n_hidden, 1))

        if self.use_encoder:
            self.encoder = RecurrentEncoder(timesteps=self.t_context, n_env=self.n_env, n_hidden=self.n_hidden,
                                            n_lstm_layers=self.n_lstm_layers, seed=seed, dropout_p=self.dropout_p)



    def forward(self, data):
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
            h_t = [torch.zeros(data.x.size(0), self.n_hidden, device=x.device) for l in range(self.n_lstm_layers)]
            c_t = [torch.zeros(data.x.size(0), self.n_hidden, device=x.device) for l in range(self.n_lstm_layers)]

        else:
            x = data.x[..., 0].view(-1, 1)
            y_hat.append(x)
            h_t = []
            c_t = []

        coords = data.coords
        edge_index = data.edge_index
        edge_attr = data.edge_attr


        self.flows = torch.zeros((edge_index.size(1), 1, self.horizon+1), device=x.device)
        self.abs_flows = torch.zeros((edge_index.size(1), 1, self.horizon+1), device=x.device)
        self.selfflows = torch.zeros((data.x.size(0), 1, self.horizon+1), device=x.device)
        self.abs_selfflows = torch.zeros((data.x.size(0), 1, self.horizon+1), device=x.device)
        self.deltas = torch.zeros((data.x.size(0), 1, self.horizon+1), device=x.device)
        self.inflows = torch.zeros((data.x.size(0), 1, self.horizon + 1), device=x.device)

        if self.use_encoder:
            forecast_horizon = range(self.t_context + 1, self.t_context + self.horizon + 1)
        else:
            forecast_horizon = range(1, self.horizon + 1)

        for t in forecast_horizon:

            if True: #torch.any(data.local_night[:, t+1] | data.local_dusk[:, t+1]):
                # at least for one radar station it is night or dusk

                r = torch.rand(1)
                if r < self.teacher_forcing:
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

                if self.fixed_boundary:
                    # use ground truth for boundary nodes
                    x[data.boundary, 0] = data.y[data.boundary, t]


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
            flow = F.leaky_relu(self.fc_edge_in(features))
            flow = F.dropout(flow, p=self.dropout_p, training=self.training)

            for l in self.fc_edge_hidden:
                flow = F.leaky_relu(l(flow))
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
            inputs = self.node2hidden(inputs) #.relu()

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
            selfflow = F.leaky_relu(self.fc_self_in(features))
            selfflow = F.dropout(selfflow, p=self.dropout_p, training=self.training)

            for l in self.fc_self_hidden:
                selfflow = F.leaky_relu(l(selfflow))
                selfflow = F.dropout(selfflow, p=self.dropout_p, training=self.training)

            selfflow = self.fc_edge_out(selfflow).sigmoid()

        #self.selfflows.append(selfflow)
        self.selfflows[..., t] = selfflow
        selfflow = x * selfflow
        #self.abs_selfflows.append(selfflow)
        self.abs_selfflows[..., t] = selfflow
        self.inflows[..., t] = aggr_out

        #departure = departure * local_dusk.view(-1, 1) # only use departure model if it is local dusk
        pred = selfflow + aggr_out + delta

        return pred, h_t, c_t


class Extrapolation(MessagePassing):

    def __init__(self, **kwargs):
        super(Extrapolation, self).__init__(aggr='mean', node_dim=0)

        #self.weighted = kwargs.get('weighted', 1)
        self.edge_index = kwargs.get('edge_index', None)

    def forward(self, var):
        var = self.propagate(self.edge_index, var=var)
        return var

    def message(self, var_j):
        # construct messages to node i for each edge (j,i)
        # can take any argument initially passed to propagate()
        # x_j are source features with shape [E, out_channels]

        return var_j


class BirdFluxGraphLSTM(MessagePassing):

    def __init__(self, **kwargs):
        super(BirdFluxGraphLSTM, self).__init__(aggr='add', node_dim=0)

        self.horizon = kwargs.get('horizon', 40)
        self.dropout_p = kwargs.get('dropout_p', 0)
        self.n_hidden = kwargs.get('n_hidden', 16)
        self.n_env = kwargs.get('n_env', 4)
        self.n_node_in = 6 + self.n_env
        self.n_edge_in = 10 + 2*self.n_env
        self.n_fc_layers = kwargs.get('n_fc_layers', 1)
        self.n_lstm_layers = kwargs.get('n_lstm_layers', 1)
        self.fixed_boundary = kwargs.get('fixed_boundary', False)
        self.force_zeros = kwargs.get('force_zeros', True)

        self.use_encoder = kwargs.get('use_encoder', False)
        self.t_context = kwargs.get('context', 0)
        self.enforce_conservation = kwargs.get('enforce_conservation', False)

        self.perturbation_std = kwargs.get('perturbation_std', 0)
        self.perturbation_mean = kwargs.get('perturbation_mean', 0)

        self.boundary_model = kwargs.get('boundary_model', None)
        self.fix_boundary_fluxes = kwargs.get('fix_boundary_fluxes', False)
        self.teacher_forcing = kwargs.get('teacher_forcing', 0)

        self.edge_type = kwargs.get('edge_type', 'voronoi')
        if self.edge_type == 'voronoi':
            self.n_edge_in += 1 # use face_length as additional feature
            self.n_node_in += 1 # use voronoi cell area as additional feature

        seed = kwargs.get('seed', 1234)
        torch.manual_seed(seed)


        self.fc_edge_embedding = torch.nn.Linear(self.n_edge_in, self.n_hidden)
        self.fc_edge_in = torch.nn.Linear(self.n_hidden * 2, self.n_hidden)
        # self.fc_edge_in = torch.nn.Linear(self.n_hidden, self.n_hidden)
        self.fc_edge_hidden = nn.ModuleList([torch.nn.Linear(self.n_hidden, self.n_hidden)
                                             for _ in range(self.n_fc_layers - 1)])
        self.fc_edge_out = torch.nn.Linear(self.n_hidden, 1)


        self.node2hidden = torch.nn.Sequential(torch.nn.Linear(self.n_node_in, self.n_hidden),
                                               torch.nn.Dropout(p=self.dropout_p),
                                               torch.nn.LeakyReLU(),
                                               torch.nn.Linear(self.n_hidden, self.n_hidden))

        if self.use_encoder:
            self.lstm_in = nn.LSTMCell(self.n_hidden * 2, self.n_hidden)
            self.encoder = RecurrentEncoder(timesteps=self.t_context, n_env=self.n_env, n_hidden=self.n_hidden,
                                            n_lstm_layers=self.n_lstm_layers, seed=seed, dropout_p=self.dropout_p)
            self.fc_encoder = torch.nn.Linear(self.n_hidden, self.n_hidden)
            self.fc_hidden = torch.nn.Linear(self.n_hidden, self.n_hidden)
            self.attention_t = torch.nn.Parameter(torch.Tensor(self.n_hidden, 1))
        else:
            self.lstm_in = nn.LSTMCell(self.n_hidden, self.n_hidden)
        self.lstm_layers = nn.ModuleList([nn.LSTMCell(self.n_hidden, self.n_hidden) for _ in range(self.n_lstm_layers)])

        self.hidden2delta = torch.nn.Sequential(torch.nn.Linear(self.n_hidden, self.n_hidden),
                                                torch.nn.Dropout(p=self.dropout_p),
                                                torch.nn.LeakyReLU(),
                                                torch.nn.Linear(self.n_hidden, 1))

        if self.boundary_model == 'LocalLSTM':
            kwargs['fixed_boundary'] = []
            self.boundary_lstm = LocalLSTM(**kwargs, return_hidden_states=True, use_mtr_features=True)
        elif self.boundary_model == 'FluxMLP':
            if self.use_encoder:
                self.flux_mlp = FluxMLP2(**kwargs)
            else:
                self.flux_mlp = FluxMLP(**kwargs)
        elif self.boundary_model == 'FluxMLPtanh':
            self.flux_mlp = FluxMLP3(**kwargs)
        elif self.boundary_model == 'Extrapolation':
            self.extrapolation = Extrapolation()


        self.reset_parameters()


    def reset_parameters(self):

        def init_weights(m):
            if type(m) == nn.Linear:
                inits.glorot(m.weight)
                inits.zeros(m.bias)
            elif type(m) == nn.LSTMCell:
                for name, param in m.named_parameters():
                    if 'bias' in name:
                        inits.zeros(param)
                    elif 'weight' in name:
                        inits.glorot(param)

        self.fc_edge_hidden.apply(init_weights)
        self.node2hidden.apply(init_weights)
        self.lstm_layers.apply(init_weights)
        self.hidden2delta.apply(init_weights)
        init_weights(self.lstm_in)

        init_weights(self.fc_edge_in)
        init_weights(self.fc_edge_embedding)
        init_weights(self.fc_edge_out)

        if self.use_encoder:
            init_weights(self.fc_encoder)
            init_weights(self.fc_hidden)
            init_weights(self.attention_t)



    def forward(self, data):
        # with teacher_forcing = 0.0 the model always uses previous predictions to make new predictions
        # with teacher_forcing = 1.0 the model always uses the ground truth to make new predictions

        # self.edges = data.edge_index
        # n_edges = self.edges.size(1)
        # self.boundary_edge_index = torch.tensor([idx for idx in range(n_edges)
        #                                          if data.boundary[self.edges[0, idx]]])
        # self.boundary_edges = self.edges[:, self.boundary_edge_index]
        # self.boundary = data.boundary
        #
        # self.reverse_edge_index = torch.zeros(n_edges, dtype=torch.long)
        # for idx in range(n_edges):
        #     for jdx in range(n_edges):
        #         if (self.edges[:, idx] == torch.flip(self.edges[:, jdx], dims=[0])).all():
        #             self.reverse_edge_index[idx] = jdx

        self.edges = data.edge_index
        self.boundary2inner_edges = data.boundary2inner_edges
        self.inner2boundary_edges = data.inner2boundary_edges
        # self.boundary2boundary_edges = data.boundary2boundary_edges
        self.inner_edges = data.inner_edges
        self.reverse_edges = data.reverse_edges
        self.boundary = data.boundary.bool()


        y_hat = []
        enc_states = None

        x = data.x[..., self.t_context].view(-1, 1)
        y_hat.append(x)

        # initialize lstm variables
        if self.use_encoder:
            # push context timeseries through encoder to initialize decoder
            enc_states, h_t, c_t = self.encoder(data)
            # x = torch.zeros(data.x.size(0)).to(data.x.device) # TODO eventually use this!?

        else:
            # start from scratch
            # measurement at t=0
            h_t = [torch.zeros(data.x.size(0), self.n_hidden, device=x.device) for l in range(self.n_lstm_layers)]
            c_t = [torch.zeros(data.x.size(0), self.n_hidden, device=x.device) for l in range(self.n_lstm_layers)]

        coords = data.coords
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        self.local_fluxes = torch.zeros((edge_index.size(1), 1, self.horizon + 1), device=x.device)
        if not self.training:
            self.local_deltas = torch.zeros((data.x.size(0), 1, self.horizon+1), device=x.device)
            if self.use_encoder:
                self.alphas_t = torch.zeros((x.size(0), self.t_context, self.horizon + 1), device=x.device)

        forecast_horizon = range(self.t_context + 1, self.t_context + self.horizon + 1)

        if self.boundary_model == 'LocalLSTM':
            self.boundary_lstm.teacher_forcing = self.teacher_forcing
            boundary_pred, boundary_h = self.boundary_lstm(data)
            x[data.boundary, 0] = boundary_pred[data.boundary, 0]
        elif self.boundary_model == 'Extrapolation':
            self.boundary2boundary_edges = data.boundary2boundary_edges
            self.extrapolation.edge_index = self.edges[:, torch.logical_not(self.boundary2boundary_edges)]

        for t in forecast_horizon:

            r = torch.rand(1)
            if r < self.teacher_forcing:
                print('use teacher forcing')
                # if data is available use ground truth, otherwise use model prediction
                x = data.missing[..., t-1].bool().view(-1, 1) * x + \
                    ~data.missing[..., t-1].bool().view(-1, 1) * data.x[..., t-1].view(-1, 1)

            if self.boundary_model == 'LocalLSTM':
                h_t[-1] = h_t[-1] * torch.logical_not(data.boundary.view(-1, 1)) + \
                          boundary_h[..., t-self.t_context-1] * data.boundary.view(-1, 1)
            elif self.boundary_model == 'Extrapolation':
                x_extrapolated = self.extrapolation(x)
                h_extrapolated = self.extrapolation(h_t[-1])

                x = x * torch.logical_not(data.boundary.view(-1, 1)) + \
                    x_extrapolated * data.boundary.view(-1, 1)
                h_t[-1] = h_t[-1] * torch.logical_not(data.boundary.view(-1, 1)) + \
                    h_extrapolated * data.boundary.view(-1, 1)

            x, h_t, c_t = self.propagate(edge_index, x=x, coords=coords,
                                                h_t=h_t, c_t=c_t,
                                                h=h_t[-1], c=c_t[-1],
                                                areas=data.areas,
                                                edge_attr=edge_attr,
                                                dusk=data.local_dusk[:, t-1],
                                                dawn=data.local_dawn[:, t],
                                                dawn_1=data.local_dawn[:, t-1],
                                                env=data.env[..., t],
                                                env_1=data.env[..., t-1],
                                                t=t-self.t_context,
                                                boundary=data.boundary,
                                                night=data.local_night[:, t],
                                                night_1=data.local_night[:, t-1],
                                                day_of_year=data.day_of_year[t],
                                                enc_states=enc_states)#,
                                                #radar_fluxes=data.fluxes[:, t])

            if self.fixed_boundary:
                # # use ground truth for boundary nodes
                perturbation = torch.randn(data.boundary.sum()).to(x.device) * self.perturbation_std + self.perturbation_mean
                if self.boundary_model == 'LocalLSTM':
                    x[data.boundary, 0] = boundary_pred[data.boundary, t-self.t_context] + perturbation
                else:
                    x[data.boundary, 0] = data.y[data.boundary, t] + perturbation

            if self.force_zeros:
                x = x * data.local_night[:, t].view(-1, 1)

            y_hat.append(x)

        prediction = torch.cat(y_hat, dim=-1)
        return prediction


    def message(self, x_i, x_j, h_i, h_j, coords_i, coords_j, env_i, env_j, env_1_i, env_1_j, edge_attr, t,
                night_i, night_j, night_1_j, boundary, day_of_year, dawn_i, dawn_1_j): #, radar_fluxes):
        # construct messages to node i for each edge (j,i)
        # can take any argument initially passed to propagate()
        # x_j are source features with shape [E, out_channels]


        # inputs = [x_j.view(-1, 1), coords_i, coords_j, env_i, env_1_j, edge_attr,
        #                       night_i.float().view(-1, 1), night_1_j.float().view(-1, 1),
        #                     dusk_i.float().view(-1, 1), dawn_i.float().view(-1, 1)]

        inputs = [coords_i, coords_j, env_i, env_1_j, edge_attr,
                  night_i.float().view(-1, 1), night_1_j.float().view(-1, 1),
                  dawn_i.float().view(-1, 1), dawn_1_j.float().view(-1, 1)]

        # inputs = [coords_i, coords_j, env_i, env_1_j, edge_attr,
        #           night_i.float().view(-1, 1), night_1_j.float().view(-1, 1)]

        inputs = torch.cat(inputs, dim=1)


        inputs = self.fc_edge_embedding(inputs)
        inputs = torch.cat([inputs, h_j], dim=1)

        flux = F.leaky_relu(self.fc_edge_in(inputs))
        flux = F.dropout(flux, p=self.dropout_p, training=self.training)

        for l in self.fc_edge_hidden:
            flux = F.leaky_relu(l(flux))
            flux = F.dropout(flux, p=self.dropout_p, training=self.training)

        flux = self.fc_edge_out(flux) #.tanh()

        # enforce fluxes to be symmetric along edges
        flux = flux.sigmoid() # bird density flying from node j to node i should be positive
        flux = flux * x_j

        # self.local_fluxes_A[self.edges[0], self.edges[1]] = flux.squeeze()

        if self.boundary_model == 'FluxMLP':
            #edge_fluxes = self.flux_mlp(env_1_j, env_i, night_1_j, night_i, coords_j, coords_i, edge_attr)
            # TODO use attention weighted encoder sequence as additional input?
            if self.use_encoder:
                # boundary_fluxes = self.flux_mlp(h_i[self.boundary_edges], env_1_j[self.boundary_edges], env_i[self.boundary_edges],
                #                             night_1_j[self.boundary_edges], night_i[self.boundary_edges],
                #                             coords_j[self.boundary_edges], coords_i[self.boundary_edges],
                #                             edge_attr[self.boundary_edges],
                #                             day_of_year.repeat(self.boundary_edges.size()))
                boundary_fluxes = self.flux_mlp(h_i, x_i, env_1_j, env_i, night_1_j, night_i,
                                                coords_j, coords_i,
                                                edge_attr, day_of_year.repeat(self.edges.size(1)))
            else:
                # boundary_fluxes = self.flux_mlp(env_1_j[self.boundary_edges], env_i[self.boundary_edges],
                #                             night_1_j[self.boundary_edges], night_i[self.boundary_edges],
                #                             coords_j[self.boundary_edges], coords_i[self.boundary_edges],
                #                             edge_attr[self.boundary_edges], day_of_year.repeat(self.boundary_edges.size()))
                boundary_fluxes = self.flux_mlp(env_1_j, env_i, night_1_j, night_i,
                                                coords_j, coords_i,
                                                edge_attr, day_of_year.repeat(self.edges.size(1)))

            flux = (self.inner_edges.view(-1, 1) + self.inner2boundary_edges.view(-1, 1)) * flux + \
                   self.boundary2inner_edges.view(-1, 1) * boundary_fluxes
            # print(boundary_fluxes[self.boundary_edges])
            #A_influx[self.fixed_boundary, :] = to_dense_adj(self.edges, edge_attr=edge_fluxes).squeeze()[self.fixed_boundary, :]

            # self.boundary_fluxes_A[self.boundary_edges[0], self.boundary_edges[1]] = edge_fluxes.squeeze()
            # self.local_fluxes_A[self.boundary, :] = self.boundary_fluxes_A[self.boundary, :]

        self.local_fluxes[..., t] = flux
        flux = flux - flux[self.reverse_edges]

        if self.boundary_model == 'FluxMLPtanh':
            boundary_fluxes = self.flux_mlp(h_i, x_i, env_1_j, env_1_i, env_j, env_i, night_j, night_i,
                coords_j, coords_i, edge_attr, day_of_year.repeat(self.edges.size(1)))

            flux = self.inner_edges.view(-1, 1) * flux + \
                    self.boundary2inner_edges.view(-1, 1) * boundary_fluxes - \
                     self.inner2boundary_edges.view(-1, 1) * boundary_fluxes[self.reverse_edges]

            self.local_fluxes[..., t] = flux

        # if self.fix_boundary_fluxes:
        #     flux = torch.logical_not(self.boundary_edges.view(-1, 1)) * flux + self.boundary_edges.view(-1, 1) * radar_fluxes

        # self.local_fluxes_A = self.local_fluxes_A - self.local_fluxes_A.T
        # flux = self.local_fluxes_A[self.edges[0], self.edges[1]]
        print(f'flux = {flux}')
        flux = flux.view(-1, 1)

        return flux


    def update(self, aggr_out, x, coords, env, dusk, dawn, areas, h_t, c_t, t, night, boundary, enc_states):

        if self.edge_type == 'voronoi':
            inputs = torch.cat([x.view(-1, 1), coords, env, dawn.float().view(-1, 1), #ground.view(-1, 1),
                                dusk.float().view(-1, 1), areas.view(-1, 1), night.float().view(-1, 1)], dim=1)
        else:
            inputs = torch.cat([x.view(-1, 1), coords, env, dawn.float().view(-1, 1),  # ground.view(-1, 1),
                                dusk.float().view(-1, 1), night.float().view()], dim=1)
        inputs = self.node2hidden(inputs)

        if self.use_encoder:
            # temporal attention based on encoder states
            enc_states = self.fc_encoder(enc_states) # shape (radars x timesteps x hidden)
            hidden = self.fc_hidden(h_t[-1]).unsqueeze(1) # shape (radars x 1 x hidden)
            scores = torch.tanh(enc_states + hidden).matmul(self.attention_t).squeeze() # shape (radars x timesteps)
            alpha = F.softmax(scores, dim=1)
            if not self.training:
                self.alphas_t[..., t] = alpha
            context = alpha.unsqueeze(1).matmul(enc_states).squeeze() # shape (radars x hidden)

            inputs = torch.cat([inputs, context], dim=1)

        h_t[0], c_t[0] = self.lstm_in(inputs, (h_t[0], c_t[0]))
        for l in range(self.n_lstm_layers - 1):
            h_t[0] = F.dropout(h_t[0], p=self.dropout_p, training=self.training, inplace=True)
            c_t[0] = F.dropout(c_t[0], p=self.dropout_p, training=self.training, inplace=True)
            h_t[l+1], c_t[l+1] = self.lstm_layers[l](h_t[l], (h_t[l+1], c_t[l+1]))

        delta = self.hidden2delta(h_t[-1]).tanh()
        if not self.training:
            self.local_deltas[..., t] = delta

        print(f'delta = {delta}')
        print(f'aggr out = {aggr_out}')
        pred = x + delta + aggr_out # take messages into account for inner cells only
        print(f'pred = {pred}')
        #pred = pred.relu() # enforce positive bird densities

        return pred, h_t, c_t

class testFluxMLP(MessagePassing):

    def __init__(self, **kwargs):
        super(testFluxMLP, self).__init__(aggr='add', node_dim=0)

        self.horizon = kwargs.get('horizon', 40)
        self.dropout_p = kwargs.get('dropout_p', 0)
        self.n_hidden = kwargs.get('n_hidden', 16)
        self.n_env = kwargs.get('n_env', 4)
        self.n_node_in = 6 + self.n_env
        self.n_edge_in = 8 + 2*self.n_env
        self.n_fc_layers = kwargs.get('n_fc_layers', 1)
        self.n_lstm_layers = kwargs.get('n_lstm_layers', 1)
        self.fixed_boundary = kwargs.get('fixed_boundary', False)
        self.force_zeros = kwargs.get('force_zeros', True)

        self.use_encoder = kwargs.get('use_encoder', False)
        self.t_context = kwargs.get('context', 0)
        self.enforce_conservation = kwargs.get('enforce_conservation', False)

        self.perturbation_std = kwargs.get('perturbation_std', 0)
        self.perturbation_mean = kwargs.get('perturbation_mean', 0)

        self.boundary_model = kwargs.get('boundary_model', None)
        self.fix_boundary_fluxes = kwargs.get('fix_boundary_fluxes', False)

        self.edge_type = kwargs.get('edge_type', 'voronoi')
        if self.edge_type == 'voronoi':
            self.n_edge_in += 1 # use face_length as additional feature
            self.n_node_in += 1 # use voronoi cell area as additional feature

        seed = kwargs.get('seed', 1234)
        torch.manual_seed(seed)


        self.flux_mlp = FluxMLP4(**kwargs)


    def forward(self, data):
        # with teacher_forcing = 0.0 the model always uses previous predictions to make new predictions
        # with teacher_forcing = 1.0 the model always uses the ground truth to make new predictions

        # self.edges = data.edge_index
        # n_edges = self.edges.size(1)
        # self.boundary_edge_index = torch.tensor([idx for idx in range(n_edges)
        #                                          if data.boundary[self.edges[0, idx]]])
        # self.boundary_edges = self.edges[:, self.boundary_edge_index]
        # self.boundary = data.boundary
        #
        # self.reverse_edge_index = torch.zeros(n_edges, dtype=torch.long)
        # for idx in range(n_edges):
        #     for jdx in range(n_edges):
        #         if (self.edges[:, idx] == torch.flip(self.edges[:, jdx], dims=[0])).all():
        #             self.reverse_edge_index[idx] = jdx

        self.edges = data.edge_index
        self.boundary2inner_edges = data.boundary2inner_edges
        self.inner2boundary_edges = data.inner2boundary_edges
        self.inner_edges = data.inner_edges
        self.reverse_edges = data.reverse_edges
        self.boundary = data.boundary.bool()


        y_hat = []
        enc_states = None

        x = data.x[..., self.t_context].view(-1, 1)
        y_hat.append(x)


        coords = data.coords
        edge_index = data.edge_index
        edge_attr = data.edge_attr


        self.local_fluxes = torch.zeros((edge_index.size(1), 1, self.horizon+1), device=x.device)
        # self.local_fluxes_A = torch.zeros((data.x.size(0), data.x.size(0))).to(x.device)
        # self.boundary_fluxes_A = torch.zeros((data.x.size(0), data.x.size(0))).to(x.device)
        # self.fluxes = torch.zeros((data.x.size(0), 1, self.timesteps + 1)).to(x.device)
        self.local_deltas = torch.zeros((data.x.size(0), 1, self.horizon+1), device=x.device)

        forecast_horizon = range(self.t_context + 1, self.t_context + self.horizon + 1)


        for t in forecast_horizon:

            r = torch.rand(1)

            x = data.missing[..., t-1].bool().view(-1, 1) * x + \
                ~data.missing[..., t-1].bool().view(-1, 1) * data.x[..., t-1].view(-1, 1)


            _ = self.propagate(edge_index, x=x, coords=coords,
                                                areas=data.areas,
                                                edge_attr=edge_attr,
                                                dusk=data.local_dusk[:, t-1],
                                                dawn=data.local_dawn[:, t],
                                                dawn_1=data.local_dawn[:, t-1],
                                                env=data.env[..., t],
                                                env_1=data.env[..., t-1],
                                                t=t-self.t_context,
                                                boundary=data.boundary,
                                                night=data.local_night[:, t],
                                                night_1=data.local_night[:, t-1],
                                                day_of_year=data.day_of_year[t],
                                                enc_states=enc_states)#,
                                                #radar_fluxes=data.fluxes[:, t])


            y_hat.append(x)

        prediction = torch.cat(y_hat, dim=-1)
        return prediction


    def message(self, x_i, coords_i, coords_j, env_i, env_j, env_1_i, env_1_j, edge_attr, t,
                night_i, night_j, day_of_year):
        # construct messages to node i for each edge (j,i)
        # can take any argument initially passed to propagate()
        # x_j are source features with shape [E, out_channels]



        boundary_fluxes = self.flux_mlp(x_i, env_1_j, env_1_i, env_j, env_i, night_j, night_i,
                                        coords_j, coords_i, edge_attr, day_of_year.repeat(self.edges.size(1)))

        flux = self.boundary2inner_edges.view(-1, 1) * boundary_fluxes #- \
               #self.inner2boundary_edges.view(-1, 1) * boundary_fluxes[self.reverse_edges]
        self.local_fluxes[..., t] = flux


        flux = flux.view(-1, 1)

        return flux


    def update(self, aggr_out):

        return aggr_out


class BirdFluxGraphLSTM2(MessagePassing):

    def __init__(self, **kwargs):
        super(BirdFluxGraphLSTM2, self).__init__(aggr='add', node_dim=0)

        self.horizon = kwargs.get('horizon', 40)
        self.dropout_p = kwargs.get('dropout_p', 0)
        self.n_hidden = kwargs.get('n_hidden', 16)
        self.n_env = kwargs.get('n_env', 4)
        self.n_node_in = 6 + self.n_env
        self.n_edge_in = 8 + 2*self.n_env
        self.n_fc_layers = kwargs.get('n_fc_layers', 1)
        self.n_lstm_layers = kwargs.get('n_lstm_layers', 1)
        self.fixed_boundary = kwargs.get('fixed_boundary', False)
        self.force_zeros = kwargs.get('force_zeros', True)

        self.use_encoder = kwargs.get('use_encoder', False)
        self.t_context = kwargs.get('context', 0)
        self.enforce_conservation = kwargs.get('enforce_conservation', False)
        self.teacher_forcing = kwargs.get('teacher_forcing', 0)

        self.perturbation_std = kwargs.get('perturbation_std', 0)
        self.perturbation_mean = kwargs.get('perturbation_mean', 0)

        self.boundary_model = kwargs.get('boundary_model', None)
        self.fix_boundary_fluxes = kwargs.get('fix_boundary_fluxes', False)

        self.edge_type = kwargs.get('edge_type', 'voronoi')
        if self.edge_type == 'voronoi':
            self.n_edge_in += 1 # use face_length as additional feature
            self.n_node_in += 1 # use voronoi cell area as additional feature

        seed = kwargs.get('seed', 1234)
        torch.manual_seed(seed)


        self.fc_edge_embedding = torch.nn.Linear(self.n_edge_in, self.n_hidden)
        self.fc_edge_in = torch.nn.Linear(self.n_hidden * 2, self.n_hidden)
        # self.fc_edge_in = torch.nn.Linear(self.n_hidden, self.n_hidden)
        self.fc_edge_hidden = nn.ModuleList([torch.nn.Linear(self.n_hidden, self.n_hidden)
                                             for _ in range(self.n_fc_layers - 1)])
        self.fc_edge_out = torch.nn.Linear(self.n_hidden, 1)


        self.node2hidden = torch.nn.Sequential(torch.nn.Linear(self.n_node_in, self.n_hidden),
                                               torch.nn.Dropout(p=self.dropout_p),
                                               torch.nn.LeakyReLU(),
                                               torch.nn.Linear(self.n_hidden, self.n_hidden))

        if self.use_encoder:
            self.lstm_in = nn.LSTMCell(self.n_hidden * 2, self.n_hidden)
            self.encoder = RecurrentEncoder(timesteps=self.t_context, n_env=self.n_env, n_hidden=self.n_hidden,
                                            n_lstm_layers=self.n_lstm_layers, seed=seed, dropout_p=self.dropout_p)
            self.fc_encoder = torch.nn.Linear(self.n_hidden, self.n_hidden)
            self.fc_hidden = torch.nn.Linear(self.n_hidden, self.n_hidden)
            self.attention_t = torch.nn.Parameter(torch.Tensor(self.n_hidden, 1))
        else:
            self.lstm_in = nn.LSTMCell(self.n_hidden, self.n_hidden)
        self.lstm_layers = nn.ModuleList([nn.LSTMCell(self.n_hidden, self.n_hidden) for _ in range(self.n_lstm_layers)])

        self.hidden2delta = torch.nn.Sequential(torch.nn.Linear(self.n_hidden, self.n_hidden),
                                                torch.nn.Dropout(p=self.dropout_p),
                                                torch.nn.LeakyReLU(),
                                                torch.nn.Linear(self.n_hidden, 1))

        if self.boundary_model == 'LocalLSTM':
            kwargs['fixed_boundary'] = []
            self.boundary_lstm = LocalLSTM(**kwargs, return_hidden_states=True, use_mtr_features=True)
        elif self.boundary_model == 'FluxMLP':
            if self.use_encoder:
                self.flux_mlp = FluxMLP2(**kwargs)
            else:
                self.flux_mlp = FluxMLP(**kwargs)
        elif self.boundary_model == 'FluxMLPtanh':
            self.flux_mlp = FluxMLP3(**kwargs)


        self.reset_parameters()


    def reset_parameters(self):

        def init_weights(m):
            if type(m) == nn.Linear:
                inits.glorot(m.weight)
                inits.zeros(m.bias)
            elif type(m) == nn.LSTMCell:
                for name, param in m.named_parameters():
                    if 'bias' in name:
                        inits.zeros(param)
                    elif 'weight' in name:
                        inits.glorot(param)

        self.fc_edge_hidden.apply(init_weights)
        self.node2hidden.apply(init_weights)
        self.lstm_layers.apply(init_weights)
        self.hidden2delta.apply(init_weights)
        init_weights(self.lstm_in)

        init_weights(self.fc_edge_in)
        init_weights(self.fc_edge_embedding)
        init_weights(self.fc_edge_out)

        if self.use_encoder:
            init_weights(self.fc_encoder)
            init_weights(self.fc_hidden)
            init_weights(self.attention_t)



    def forward(self, data):
        # with teacher_forcing = 0.0 the model always uses previous predictions to make new predictions
        # with teacher_forcing = 1.0 the model always uses the ground truth to make new predictions

        # self.edges = data.edge_index
        # n_edges = self.edges.size(1)
        # self.boundary_edge_index = torch.tensor([idx for idx in range(n_edges)
        #                                          if data.boundary[self.edges[0, idx]]])
        # self.boundary_edges = self.edges[:, self.boundary_edge_index]
        # self.boundary = data.boundary
        #
        # self.reverse_edge_index = torch.zeros(n_edges, dtype=torch.long)
        # for idx in range(n_edges):
        #     for jdx in range(n_edges):
        #         if (self.edges[:, idx] == torch.flip(self.edges[:, jdx], dims=[0])).all():
        #             self.reverse_edge_index[idx] = jdx

        self.edges = data.edge_index
        self.boundary2inner_edges = data.boundary2inner_edges
        self.inner2boundary_edges = data.inner2boundary_edges
        self.inner_edges = data.inner_edges
        self.reverse_edges = data.reverse_edges
        self.boundary = data.boundary.bool()


        y_hat = []
        enc_states = None

        x = data.x[..., self.t_context].view(-1, 1)
        y_hat.append(x)

        # initialize lstm variables
        if self.use_encoder:
            # push context timeseries through encoder to initialize decoder
            enc_states, h_t, c_t = self.encoder(data)
            # x = torch.zeros(data.x.size(0)).to(data.x.device) # TODO eventually use this!?

        else:
            # start from scratch
            # measurement at t=0
            h_t = [torch.zeros(data.x.size(0), self.n_hidden, device=x.device) for l in range(self.n_lstm_layers)]
            c_t = [torch.zeros(data.x.size(0), self.n_hidden, device=x.device) for l in range(self.n_lstm_layers)]

        coords = data.coords
        edge_index = data.edge_index
        edge_attr = data.edge_attr


        self.local_fluxes = torch.zeros((edge_index.size(1), 1, self.horizon+1), device=x.device)
        # self.local_fluxes_A = torch.zeros((data.x.size(0), data.x.size(0))).to(x.device)
        # self.boundary_fluxes_A = torch.zeros((data.x.size(0), data.x.size(0))).to(x.device)
        # self.fluxes = torch.zeros((data.x.size(0), 1, self.timesteps + 1)).to(x.device)
        self.local_deltas = torch.zeros((data.x.size(0), 1, self.horizon+1), device=x.device)

        forecast_horizon = range(self.t_context + 1, self.t_context + self.horizon + 1)

        if self.use_encoder:
            self.alphas_t = torch.zeros((x.size(0), self.t_context, self.horizon + 1)).to(x.device)

        if self.boundary_model == 'LocalLSTM':
            self.boundary_model.teacher_forcing = self.teacher_forcing
            boundary_pred, boundary_h = self.boundary_lstm(data)
            x[data.boundary, 0] = boundary_pred[data.boundary, 0]

        for t in forecast_horizon:

            r = torch.rand(1)
            if r < self.teacher_forcing:
                # if data is available use ground truth, otherwise use model prediction
                x = data.missing[..., t-1].bool().view(-1, 1) * x + \
                    ~data.missing[..., t-1].bool().view(-1, 1) * data.x[..., t-1].view(-1, 1)

            if self.boundary_model == 'LocalLSTM':
                h_t[-1] = h_t[-1] * torch.logical_not(data.boundary.view(-1, 1)) + \
                          boundary_h[..., t-self.t_context-1] * data.boundary.view(-1, 1)

            x, h_t, c_t = self.propagate(edge_index, x=x, coords=coords,
                                                h_t=h_t, c_t=c_t,
                                                h=h_t[-1], c=c_t[-1],
                                                areas=data.areas,
                                                edge_attr=edge_attr,
                                                dusk=data.local_dusk[:, t-1],
                                                dawn=data.local_dawn[:, t],
                                                dawn_1=data.local_dawn[:, t-1],
                                                env=data.env[..., t],
                                                env_1=data.env[..., t-1],
                                                t=t-self.t_context,
                                                boundary=data.boundary,
                                                night=data.local_night[:, t],
                                                night_1=data.local_night[:, t-1],
                                                day_of_year=data.day_of_year[t],
                                                enc_states=enc_states)#,
                                                #radar_fluxes=data.fluxes[:, t])

            if self.fixed_boundary:
                # # use ground truth for boundary nodes
                perturbation = torch.randn(data.boundary.sum()).to(x.device) * self.perturbation_std + self.perturbation_mean
                if self.boundary_model == 'LocalLSTM':
                    x[data.boundary, 0] = boundary_pred[data.boundary, t-self.t_context] + perturbation
                else:
                    x[data.boundary, 0] = data.y[data.boundary, t] + perturbation

            if self.force_zeros:
                x = x * data.local_night[:, t].view(-1, 1)

            y_hat.append(x)

        prediction = torch.cat(y_hat, dim=-1)
        return prediction


    def message(self, x_i, x_j, h_i, h_j, coords_i, coords_j, env_i, env_j, env_1_i, env_1_j, edge_attr, t,
                night_i, night_j, night_1_j, boundary, day_of_year, dawn_i, dawn_1_j): #, radar_fluxes):
        # construct messages to node i for each edge (j,i)
        # can take any argument initially passed to propagate()
        # x_j are source features with shape [E, out_channels]


        # inputs = [x_j.view(-1, 1), coords_i, coords_j, env_i, env_1_j, edge_attr,
        #                       night_i.float().view(-1, 1), night_1_j.float().view(-1, 1),
        #                     dusk_i.float().view(-1, 1), dawn_i.float().view(-1, 1)]
        inputs = [coords_i, coords_j, env_i, env_1_j, edge_attr,
                  night_i.float().view(-1, 1), night_1_j.float().view(-1, 1)]
        # features = [coords_i, coords_j, env_i, env_1_j, edge_attr,
        #             night_i.float().view(-1, 1), night_1_j.float().view(-1, 1)]
        inputs = torch.cat(inputs, dim=1)


        inputs = self.fc_edge_embedding(inputs)
        inputs = torch.cat([inputs, h_j], dim=1)

        flux = F.leaky_relu(self.fc_edge_in(inputs))
        flux = F.dropout(flux, p=self.dropout_p, training=self.training)

        for l in self.fc_edge_hidden:
            flux = F.leaky_relu(l(flux))
            flux = F.dropout(flux, p=self.dropout_p, training=self.training)

        flux = self.fc_edge_out(flux) #.tanh()

        # enforce fluxes to be symmetric along edges
        flux = flux.sigmoid() # bird density flying from node j to node i should be positive
        #flux = flux * x_j
        # self.local_fluxes_A[self.edges[0], self.edges[1]] = flux.squeeze()

        if self.boundary_model == 'FluxMLP':
            #edge_fluxes = self.flux_mlp(env_1_j, env_i, night_1_j, night_i, coords_j, coords_i, edge_attr)
            # TODO use attention weighted encoder sequence as additional input?
            if self.use_encoder:
                # boundary_fluxes = self.flux_mlp(h_i[self.boundary_edges], env_1_j[self.boundary_edges], env_i[self.boundary_edges],
                #                             night_1_j[self.boundary_edges], night_i[self.boundary_edges],
                #                             coords_j[self.boundary_edges], coords_i[self.boundary_edges],
                #                             edge_attr[self.boundary_edges],
                #                             day_of_year.repeat(self.boundary_edges.size()))
                boundary_fluxes = self.flux_mlp(h_i, x_i, env_1_j, env_i, night_1_j, night_i,
                                                coords_j, coords_i,
                                                edge_attr, day_of_year.repeat(self.edges.size(1)))
            else:
                # boundary_fluxes = self.flux_mlp(env_1_j[self.boundary_edges], env_i[self.boundary_edges],
                #                             night_1_j[self.boundary_edges], night_i[self.boundary_edges],
                #                             coords_j[self.boundary_edges], coords_i[self.boundary_edges],
                #                             edge_attr[self.boundary_edges], day_of_year.repeat(self.boundary_edges.size()))
                boundary_fluxes = self.flux_mlp(env_1_j, env_i, night_1_j, night_i,
                                                coords_j, coords_i,
                                                edge_attr, day_of_year.repeat(self.edges.size(1)))


            flux = (self.inner_edges.view(-1, 1) + self.inner2boundary_edges.view(-1, 1)) * flux + \
                   self.boundary2inner_edges.view(-1, 1) * boundary_fluxes
            # print(boundary_fluxes[self.boundary_edges])
            #A_influx[self.fixed_boundary, :] = to_dense_adj(self.edges, edge_attr=edge_fluxes).squeeze()[self.fixed_boundary, :]

            # self.boundary_fluxes_A[self.boundary_edges[0], self.boundary_edges[1]] = edge_fluxes.squeeze()
            # self.local_fluxes_A[self.boundary, :] = self.boundary_fluxes_A[self.boundary, :]


        self.local_fluxes[..., t] = flux
        flux = flux - flux[self.reverse_edges]

        if self.boundary_model == 'FluxMLPtanh':
            boundary_fluxes = self.flux_mlp(h_i, x_i, env_1_j, env_1_i, env_j, env_i, night_j, night_i,
                coords_j, coords_i, edge_attr, day_of_year.repeat(self.edges.size(1)))

            flux = self.inner_edges.view(-1, 1) * flux + \
                    self.boundary2inner_edges.view(-1, 1) * boundary_fluxes - \
                     self.inner2boundary_edges.view(-1, 1) * boundary_fluxes[self.reverse_edges]
            self.local_fluxes[..., t] = flux

        # if self.fix_boundary_fluxes:
        #     flux = torch.logical_not(self.boundary_edges.view(-1, 1)) * flux + self.boundary_edges.view(-1, 1) * radar_fluxes

        # self.local_fluxes_A = self.local_fluxes_A - self.local_fluxes_A.T
        # flux = self.local_fluxes_A[self.edges[0], self.edges[1]]

        flux = flux.view(-1, 1)

        return flux


    def update(self, aggr_out, x, coords, env, dusk, dawn, areas, h_t, c_t, t, night, boundary, enc_states):

        if self.edge_type == 'voronoi':
            inputs = torch.cat([x.view(-1, 1), coords, env, dawn.float().view(-1, 1), #ground.view(-1, 1),
                                dusk.float().view(-1, 1), areas.view(-1, 1), night.float().view(-1, 1)], dim=1)
        else:
            inputs = torch.cat([x.view(-1, 1), coords, env, dawn.float().view(-1, 1),  # ground.view(-1, 1),
                                dusk.float().view(-1, 1), night.float().view()], dim=1)
        inputs = self.node2hidden(inputs)

        if self.use_encoder:
            # temporal attention based on encoder states
            enc_states = self.fc_encoder(enc_states) # shape (radars x timesteps x hidden)
            hidden = self.fc_hidden(h_t[-1]).unsqueeze(1) # shape (radars x 1 x hidden)
            scores = torch.tanh(enc_states + hidden).matmul(self.attention_t).squeeze() # shape (radars x timesteps)
            alpha = F.softmax(scores, dim=1)
            self.alphas_t[..., t] = alpha
            context = alpha.unsqueeze(1).matmul(enc_states).squeeze() # shape (radars x hidden)

            inputs = torch.cat([inputs, context], dim=1)

        h_t[0], c_t[0] = self.lstm_in(inputs, (h_t[0], c_t[0]))
        h_t[0] = F.dropout(h_t[0], p=self.dropout_p, training=self.training)
        c_t[0] = F.dropout(c_t[0], p=self.dropout_p, training=self.training)
        for l in range(self.n_lstm_layers - 1):
            h_t[l+1], c_t[l+1] = self.lstm_layers[l](h_t[l], (h_t[l+1], c_t[l+1]))

        delta = self.hidden2delta(h_t[-1]).tanh()
        self.local_deltas[..., t] = delta

        pred = x + delta + aggr_out # take messages into account for inner cells only
        #pred = pred.relu() # enforce positive bird densities

        return pred, h_t, c_t


class AttentionGraphLSTM(MessagePassing):

    def __init__(self, **kwargs):
        super(AttentionGraphLSTM, self).__init__(aggr='add', node_dim=0)

        self.horizon = kwargs.get('horizon', 40)
        self.dropout_p = kwargs.get('dropout_p', 0)
        self.n_hidden = kwargs.get('n_hidden', 16)
        self.n_env = kwargs.get('n_env', 4)
        self.n_node_in = 6 + self.n_env
        self.n_edge_in = 3 + self.n_env
        self.n_fc_layers = kwargs.get('n_fc_layers', 1)
        self.n_lstm_layers = kwargs.get('n_lstm_layers', 1)
        self.fixed_boundary = kwargs.get('fixed_boundary', [])
        self.force_zeros = kwargs.get('force_zeros', True)
        self.teacher_forcing = kwargs.get('teacher_forcing', 0)

        self.use_encoder = kwargs.get('use_encoder', False)
        self.encoder_type = kwargs.get('encoder_type', 'temporal')
        self.t_context = kwargs.get('context', 0)
        self.predict_delta = kwargs.get('predict_delta', True)

        seed = kwargs.get('seed', 1234)
        torch.manual_seed(seed)


        # self.edge_env2hidden = torch.nn.Linear(self.n_env_in, self.n_hidden)
        # self.edge_states2hidden = torch.nn.Linear(self.n_states_in, self.n_hidden)
        self.edge2hidden = torch.nn.Linear(self.n_edge_in, self.n_hidden)
        self.context_embedding = torch.nn.Linear(self.n_hidden, self.n_hidden)
        self.attention_s = torch.nn.Parameter(torch.Tensor(self.n_hidden, 1))


        self.node2hidden = torch.nn.Linear(self.n_node_in, self.n_hidden)

        if self.use_encoder:
            self.lstm_in = nn.LSTMCell(3 * self.n_hidden, self.n_hidden)
        else:
            self.lstm_in = nn.LSTMCell(2 * self.n_hidden, self.n_hidden)
        self.lstm_layers = nn.ModuleList([nn.LSTMCell(self.n_hidden, self.n_hidden) for _
                                          in range(self.n_lstm_layers - 1)])

        self.hidden2delta = torch.nn.Sequential(torch.nn.Linear(self.n_hidden, self.n_hidden),
                                                torch.nn.Dropout(p=self.dropout_p),
                                                torch.nn.LeakyReLU(),
                                                torch.nn.Linear(self.n_hidden, 1))

        if self.use_encoder:
            if self.encoder_type == 'temporal':
                self.encoder = RecurrentEncoder(timesteps=self.t_context, n_env=self.n_env, n_hidden=self.n_hidden,
                                            n_lstm_layers=self.n_lstm_layers, seed=seed, dropout_p=self.dropout_p)
            else:
                self.encoder = RecurrentEncoderSpatial(timesteps=self.t_context, n_env=self.n_env, n_hidden=self.n_hidden,
                                                n_lstm_layers=self.n_lstm_layers, seed=seed, dropout_p=self.dropout_p)
            self.fc_encoder = torch.nn.Linear(self.n_hidden, self.n_hidden)
            self.fc_hidden = torch.nn.Linear(self.n_hidden, self.n_hidden)
            self.attention_t = torch.nn.Parameter(torch.Tensor(self.n_hidden, 1))

        self.reset_parameters()


    def reset_parameters(self):
        inits.glorot(self.edge2hidden.weight)
        inits.glorot(self.context_embedding.weight)
        inits.glorot(self.attention_s)
        inits.glorot(self.node2hidden.weight)

        if self.use_encoder:
            inits.glorot(self.fc_encoder.weight)
            inits.glorot(self.fc_hidden.weight)
            inits.glorot(self.attention_t)

        def init_weights(m):
            if type(m) == nn.Linear:
                inits.glorot(m.weight)
                inits.zeros(m.bias)
            elif type(m) == nn.LSTMCell:
                for name, param in m.named_parameters():
                    if 'bias' in name:
                        inits.zeros(param)
                    elif 'weight' in name:
                        inits.glorot(param)

        self.hidden2delta.apply(init_weights)
        self.lstm_layers.apply(init_weights)
        init_weights(self.lstm_in)



    def forward(self, data):
        # with teacher_forcing = 0.0 the model always uses previous predictions to make new predictions
        # with teacher_forcing = 1.0 the model always uses the ground truth to make new predictions

        self.edges = data.edge_index
        y_hat = []
        enc_states = None

        x = data.x[..., self.t_context].view(-1, 1)
        y_hat.append(x)

        # initialize lstm variables
        if self.use_encoder:
            # push context timeseries through encoder to initialize decoder
            enc_states, h_t, c_t = self.encoder(data)
            #x = torch.zeros(data.x.size(0)).to(data.x.device) # TODO eventually use this!?

        else:
            # start from scratch
            h_t = [torch.zeros(data.x.size(0), self.n_hidden, device=x.device) for l in range(self.n_lstm_layers)]
            c_t = [torch.zeros(data.x.size(0), self.n_hidden, device=x.device) for l in range(self.n_lstm_layers)]

        coords = data.coords
        edge_index = data.edge_index
        edge_attr = data.edge_attr


        if self.use_encoder:
            forecast_horizon = range(self.t_context + 1, self.t_context + self.horizon + 1)
        else:
            forecast_horizon = range(1, self.horizon + 1)

        self.alphas_s = torch.zeros((edge_index.size(1), 1, self.horizon + 1), device=x.device)
        if self.use_encoder:
            self.alphas_t = torch.zeros((x.size(0), self.t_context, self.horizon + 1), device=x.device)

        for t in forecast_horizon:

            r = torch.rand(1)
            if r < self.teacher_forcing:
                # if data is available use ground truth, otherwise use model prediction
                x = data.missing[..., t-1].view(-1, 1) * x + \
                    ~data.missing[..., t-1].view(-1, 1) * data.x[..., t-1].view(-1, 1)

            x, h_t, c_t = self.propagate(edge_index, x=x, coords=coords,
                                                h_t=h_t, c_t=c_t,
                                                h=h_t[-1],
                                                areas=data.areas,
                                                edge_attr=edge_attr,
                                                dusk=data.local_dusk[:, t-1],
                                                dawn=data.local_dawn[:, t],
                                                env=data.env[..., t],
                                                env_previous=data.env[..., t-1],
                                                t=t-self.t_context,
                                                boundary=data.boundary,
                                                night=data.local_night[:, t],
                                                night_previous=data.local_night[:, t-1],
                                                enc_states=enc_states)

            if self.fixed_boundary:
                # use ground truth for boundary nodes
                x[data.boundary, 0] = data.y[data.boundary, t]

            if self.force_zeros:
                x = x * data.local_night[:, t].view(-1, 1)

            y_hat.append(x)

        prediction = torch.cat(y_hat, dim=-1)
        return prediction


    def message(self, x_j, h_i, h_j, coords_i, coords_j, env_i, env_previous_j, edge_attr, t,
                night_i, night_previous_j, index):
        # construct messages to node i for each edge (j,i)
        # can take any argument initially passed to propagate()
        # x_j are source features with shape [E, out_channels]

        features = torch.cat([env_previous_j, night_previous_j.float().view(-1, 1),
                              edge_attr], dim=1)

        features = self.edge2hidden(features)
        context_j = self.context_embedding(h_j)
        context_i = self.context_embedding(h_i)

        alpha = (features + context_i + context_j).tanh().mm(self.attention_s)
        alpha = softmax(alpha, index)
        self.alphas_s[..., t] = alpha
        alpha = F.dropout(alpha, p=self.dropout_p, training=self.training)

        msg = context_j * alpha
        return msg


    def update(self, aggr_out, x, coords, env, dusk, dawn, h_t, c_t, night, enc_states, t):


        inputs = torch.cat([x.view(-1, 1), coords, env, dawn.float().view(-1, 1),
                                dusk.float().view(-1, 1), night.float().view(-1, 1)], dim=1)
        inputs = self.node2hidden(inputs)

        if self.use_encoder:
            # temporal attention based on encoder states
            enc_states = self.fc_encoder(enc_states) # shape (radars x timesteps x hidden)
            hidden = self.fc_hidden(h_t[-1]).unsqueeze(1) # shape (radars x 1 x hidden)
            scores = torch.tanh(enc_states + hidden).matmul(self.attention_t).squeeze() # shape (radars x timesteps)
            alpha = F.softmax(scores, dim=1)
            self.alphas_t[..., t] = alpha
            context = alpha.unsqueeze(1).matmul(enc_states).squeeze() # shape (radars x hidden)

            inputs = torch.cat([aggr_out, inputs, context], dim=1)
        else:
            inputs = torch.cat([aggr_out, inputs], dim=1)


        h_t[0], c_t[0] = self.lstm_in(inputs, (h_t[0], c_t[0]))
        h_t[0] = F.dropout(h_t[0], p=self.dropout_p, training=self.training)
        c_t[0] = F.dropout(c_t[0], p=self.dropout_p, training=self.training)
        for l in range(self.n_lstm_layers-1):
            h_t[l+1], c_t[l+1] = self.lstm_layers[l](h_t[l], (h_t[l+1], c_t[l+1]))

        if self.predict_delta:
            delta = self.hidden2delta(h_t[-1]).tanh()
            pred = x + delta
        else:
            pred = self.hidden2delta(h_t[-1]).sigmoid()

        return pred, h_t, c_t



class Attention2GraphLSTM(MessagePassing):

    def __init__(self, **kwargs):
        super(Attention2GraphLSTM, self).__init__(aggr='add', node_dim=0)

        self.horizon = kwargs.get('horizon', 40)
        self.dropout_p = kwargs.get('dropout_p', 0)
        self.n_hidden = kwargs.get('n_hidden', 16)
        self.n_env = kwargs.get('n_env', 4)
        self.n_node_in = 6 + self.n_env
        self.n_edge_in = 7 + self.n_env + 2*self.n_hidden
        self.n_env_in = 3 + self.n_env
        self.n_states_in = 3 + 2*self.n_hidden
        self.n_fc_layers = kwargs.get('n_fc_layers', 1)
        self.n_lstm_layers = kwargs.get('n_lstm_layers', 1)
        self.fixed_boundary = kwargs.get('fixed_boundary', [])
        self.force_zeros = kwargs.get('force_zeros', True)
        self.teacher_forcing = kwargs.get('teacher_forcing', 0)

        self.use_encoder = kwargs.get('use_encoder', False)
        self.t_context = kwargs.get('context', 0)

        seed = kwargs.get('seed', 1234)
        torch.manual_seed(seed)


        self.edge_env2hidden = torch.nn.Linear(self.n_env_in, self.n_hidden)
        self.edge_states2hidden = torch.nn.Linear(self.n_states_in, self.n_hidden)
        # self.edge2hidden = torch.nn.Linear(self.n_edge_in, self.n_hidden)
        self.context_embedding1 = torch.nn.Linear(2*self.n_hidden, self.n_hidden)
        self.context_embedding2 = torch.nn.Linear(2 * self.n_hidden, self.n_hidden)
        self.attention1 = torch.nn.Parameter(torch.Tensor(self.n_hidden, 1))
        self.attention2 = torch.nn.Parameter(torch.Tensor(self.n_hidden, 1))


        self.node2hidden = torch.nn.Linear(self.n_node_in, self.n_hidden)

        self.lstm_layers = nn.ModuleList([nn.LSTMCell(2*self.n_hidden, 2*self.n_hidden) for _ in range(self.n_lstm_layers)])

        self.hidden2delta = torch.nn.Sequential(torch.nn.Linear(2*self.n_hidden, self.n_hidden),
                                                torch.nn.Dropout(p=self.dropout_p),
                                                torch.nn.LeakyReLU(),
                                                torch.nn.Linear(self.n_hidden, 1))

        if self.use_encoder:
            self.encoder = RecurrentEncoder(timesteps=self.t_context, n_env=self.n_env, n_hidden=self.n_hidden,
                                            n_lstm_layers=self.n_lstm_layers, seed=seed, dropout_p=self.dropout_p)

        self.reset_parameters()


    def reset_parameters(self):
        inits.glorot(self.edge2hidden.weight)
        inits.glorot(self.context_embedding.weight)
        inits.glorot(self.attention1)
        inits.glorot(self.attention2)
        inits.glorot(self.node2hidden.weight)

        def init_weights(m):
            if type(m) == nn.Linear:
                inits.glorot(m.weight)
                inits.zeros(m.bias)
            elif type(m) == nn.LSTMCell:
                for name, param in m.named_parameters():
                    if 'bias' in name:
                        inits.zeros(param)
                    elif 'weight' in name:
                        inits.glorot(param)

        self.hidden2delta.apply(init_weights)
        self.lstm_layers.apply(init_weights)


    def forward(self, data):
        # with teacher_forcing = 0.0 the model always uses previous predictions to make new predictions
        # with teacher_forcing = 1.0 the model always uses the ground truth to make new predictions

        self.edges = data.edge_index
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
            h_t = [torch.zeros(data.x.size(0), 2*self.n_hidden, device=x.device) for l in range(self.n_lstm_layers)]
            c_t = [torch.zeros(data.x.size(0), 2*self.n_hidden, device=x.device) for l in range(self.n_lstm_layers)]


        coords = data.coords
        edge_index = data.edge_index
        edge_attr = data.edge_attr


        if self.use_encoder:
            forecast_horizon = range(self.t_context + 1, self.t_context + self.horizon + 1)
        else:
            forecast_horizon = range(1, self.horizon + 1)

        self.alphas1 = torch.zeros((edge_index.size(1), 1, self.horizon + 1), device=x.device)
        self.alphas2 = torch.zeros((edge_index.size(1), 1, self.horizon + 1), device=x.device)

        for t in forecast_horizon:

            r = torch.rand(1)
            if r < self.teacher_forcing:
                # if data is available use ground truth, otherwise use model prediction
                x = data.missing[..., t-1].view(-1, 1) * x + \
                    ~data.missing[..., t-1].view(-1, 1) * data.x[..., t-1].view(-1, 1)

            x, h_t, c_t = self.propagate(edge_index, x=x, coords=coords,
                                                h_t=h_t, c_t=c_t,
                                                h=h_t[-1],
                                                areas=data.areas,
                                                edge_attr=edge_attr,
                                                dusk=data.local_dusk[:, t-1],
                                                dawn=data.local_dawn[:, t],
                                                env=data.env[..., t],
                                                env_previous=data.env[..., t-1],
                                                t=t-self.t_context,
                                                boundary=data.boundary,
                                                night=data.local_night[:, t],
                                                night_previous=data.local_night[:, t-1])

            if self.fixed_boundary:
                # use ground truth for boundary nodes
                x[data.boundary, 0] = data.y[data.boundary, t]

            if self.force_zeros:
                x = x * data.local_night[:, t].view(-1, 1)

            y_hat.append(x)

        prediction = torch.cat(y_hat, dim=-1)
        return prediction


    def message(self, x_j, h_i, h_j, coords_i, coords_j, env_i, env_previous_j, edge_attr, t,
                night_i, night_previous_j, index):
        # construct messages to node i for each edge (j,i)
        # can take any argument initially passed to propagate()
        # x_j are source features with shape [E, out_channels]

        env_features = torch.cat([env_previous_j, night_previous_j.float().view(-1, 1),
                              edge_attr], dim=1)
        state_features = torch.cat([x_j, h_j, edge_attr], dim=1)
        env_features = self.edge_env2hidden(env_features)
        state_features = self.edge_states2hidden(state_features)

        context1 = self.context_embedding1(h_i)
        context2 = self.context_embedding2(h_i)

        # alpha = (features + context).tanh().mm(self.attention)
        # alpha = softmax(alpha, index)
        # alpha = F.dropout(alpha, p=self.dropout_p, training=self.training)

        #alpha = F.leaky_relu(self.attention.T * torch.cat([features, context], dim=1)))
        alpha1 = (env_features + context1).tanh().mm(self.attention1)
        alpha1 = softmax(alpha1, index)
        alpha1 = F.dropout(alpha1, p=self.dropout_p, training=self.training)

        alpha2 = (state_features + context2).tanh().mm(self.attention2)
        alpha2 = softmax(alpha2, index)
        alpha2 = F.dropout(alpha2, p=self.dropout_p, training=self.training)

        self.alphas1[..., t] = alpha1
        self.alphas1[..., t] = alpha2
        msg = (env_features * alpha1) + (state_features * alpha2)
        return msg


    def update(self, aggr_out, x, coords, env, dusk, dawn, h_t, c_t, night):


        inputs = torch.cat([x.view(-1, 1), coords, env, dawn.float().view(-1, 1),
                                dusk.float().view(-1, 1), night.float().view(-1, 1)], dim=1)
        # TODO add attention mechanism to take past conditions into account (encoder)?
        inputs = self.node2hidden(inputs)

        inputs = torch.cat([aggr_out, inputs], dim=1)


        h_t[0], c_t[0] = self.lstm_layers[0](inputs, (h_t[0], c_t[0]))
        for l in range(1, self.n_lstm_layers):
            h_t[l], c_t[l] = self.lstm_layers[l](h_t[l - 1], (h_t[l], c_t[l]))

        delta = self.hidden2delta(h_t[-1]).tanh()
        pred = x + delta

        return pred, h_t, c_t


class RecurrentEncoderSpatial(MessagePassing):
    def __init__(self, **kwargs):
        super(RecurrentEncoderSpatial, self).__init__()

        self.timesteps = kwargs.get('timesteps', 12)
        self.n_in = 4 + kwargs.get('n_env', 4)
        self.n_edge_in = 3 + kwargs.get('n_env', 4)
        self.n_hidden = kwargs.get('n_hidden', 16)
        self.n_lstm_layers = kwargs.get('n_layers_lstm', 1)
        self.dropout_p = kwargs.get('dropout_p', 0)

        torch.manual_seed(kwargs.get('seed', 1234))

        # self.node2hidden = torch.nn.Sequential(torch.nn.Linear(self.n_in, self.n_hidden),
        #                                        torch.nn.Dropout(p=self.dropout_p),
        #                                        torch.nn.ReLU(),
        #                                        torch.nn.Linear(self.n_hidden, self.n_hidden))
        self.node2hidden = torch.nn.Linear(self.n_in, self.n_hidden)

        self.lstm_in = nn.LSTMCell(self.n_hidden * 2, self.n_hidden)
        self.lstm_layers = nn.ModuleList([nn.LSTMCell(self.n_hidden, self.n_hidden) for _ in range(self.n_lstm_layers - 1)])

        self.edge2hidden = torch.nn.Linear(self.n_edge_in, self.n_hidden)
        self.context_embedding = torch.nn.Linear(self.n_hidden, self.n_hidden)
        self.attention_s = torch.nn.Parameter(torch.Tensor(self.n_hidden, 1))

        self.reset_parameters()


    def reset_parameters(self):

        def init_weights(m):
            if type(m) == nn.Linear:
                inits.glorot(m.weight)
                inits.zeros(m.bias)
            elif type(m) == nn.LSTMCell:
                for name, param in m.named_parameters():
                    if 'bias' in name:
                        inits.zeros(param)
                    elif 'weight' in name:
                        inits.glorot(param)

        inits.glorot(self.node2hidden.weight)
        self.lstm_layers.apply(init_weights)
        init_weights(self.lstm_in)


    def forward(self, data):
        # initialize lstm variables
        h_t = [torch.zeros(data.x.size(0), self.n_hidden, device=data.x.device) for l in range(self.n_lstm_layers)]
        c_t = [torch.zeros(data.x.size(0), self.n_hidden, device=data.x.device) for l in range(self.n_lstm_layers)]

        self.alphas = torch.zeros((data.edge_index.size(1), 1, self.timesteps), device=data.x.device)
        states = []


        coords = data.coords
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        for t in range(self.timesteps):

            h_t, c_t = self.propagate(edge_index, x=data.x[:, t].view(-1,1), coords=coords,
                           h_t=h_t, c_t=c_t,
                           h=h_t[-1],
                           edge_attr=edge_attr,
                           dusk=data.local_dusk[:, t],
                           dawn=data.local_dawn[:, t],
                           env=data.env[..., t],
                           t=t,
                           night=data.local_night[:, t])
            states.append(h_t[-1])
        states = torch.stack(states, dim=1) # shape (radars x timesteps x hidden features)
        return states, h_t, c_t



    def message(self, env_j, night_j, edge_attr, h_i, h_j, t, index):

        features = torch.cat([env_j, night_j.float().view(-1, 1),
                              edge_attr], dim=1)

        features = self.edge2hidden(features)
        context_j = self.context_embedding(h_j)
        context_i = self.context_embedding(h_i)

        alpha = (features + context_i + context_j).tanh().mm(self.attention_s)
        alpha = softmax(alpha, index)
        alpha = F.dropout(alpha, p=self.dropout_p, training=self.training)

        self.alphas[..., t] = alpha

        msg = context_j * alpha
        print(msg.shape)
        return msg

    def update(self, aggr_out, env, coords, x, local_night, h_t, c_t):
        inputs = torch.cat([env, coords, x.view(-1, 1),
                            local_night.float().view(-1, 1)], dim=1)
        inputs = self.node2hidden(inputs)
        inputs = torch.cat([inputs, aggr_out], dim=1)

        h_t[0], c_t[0] = self.lstm_in(inputs, (h_t[0], c_t[0]))
        h_t[0] = F.dropout(h_t[0], p=self.dropout_p, training=self.training)
        c_t[0] = F.dropout(c_t[0], p=self.dropout_p, training=self.training)
        for l in range(self.n_lstm_layers - 1):
            h_t[l+1], c_t[l+1] = self.lstm_layers[l](h_t[l], (h_t[l+1], c_t[l+1]))

        return h_t, c_t

class RecurrentEncoder(torch.nn.Module):
    def __init__(self, **kwargs):
        super(RecurrentEncoder, self).__init__()

        self.timesteps = kwargs.get('timesteps', 12)
        self.n_in = 10 + kwargs.get('n_env', 4)
        self.n_hidden = kwargs.get('n_hidden', 16)
        self.n_lstm_layers = kwargs.get('n_layers_lstm', 1)
        self.dropout_p = kwargs.get('dropout_p', 0)

        torch.manual_seed(kwargs.get('seed', 1234))

        self.node2hidden = torch.nn.Linear(self.n_in, self.n_hidden)

        self.lstm_layers = nn.ModuleList([nn.LSTMCell(self.n_hidden, self.n_hidden) for _ in range(self.n_lstm_layers)])

        self.reset_parameters()


    def reset_parameters(self):

        def init_weights(m):
            if type(m) == nn.Linear:
                inits.glorot(m.weight)
                inits.zeros(m.bias)
            elif type(m) == nn.LSTMCell:
                for name, param in m.named_parameters():
                    if 'bias' in name:
                        inits.zeros(param)
                    elif 'weight' in name:
                        inits.glorot(param)

        self.lstm_layers.apply(init_weights)
        init_weights(self.node2hidden)


    def forward(self, data):
        # initialize lstm variables
        h_t = [torch.zeros(data.x.size(0), self.n_hidden, device=data.x.device) for l in range(self.n_lstm_layers)]
        c_t = [torch.zeros(data.x.size(0), self.n_hidden, device=data.x.device) for l in range(self.n_lstm_layers)]

        states = []

        for t in range(self.timesteps):
            # h_t, c_t = self.update(data.env[..., t], data.coords, data.x[:, t], data.local_night[:, t],
            #                        data.local_dawn[:, t], data.local_dusk[:, t], data.directions[:, t],
            #                        data.speeds[:, t], h_t, c_t)
            h_t, c_t = self.update(data.env[..., t], data.coords, data.x[:, t], data.local_night[:, t],
                                   data.local_dawn[:, t], data.local_dusk[:, t], data.bird_uv[..., t],
                                   data.directions[:, t], data.speeds[:, t], h_t, c_t)

            states.append(h_t[-1])
        states = torch.stack(states, dim=1) # shape (radars x timesteps x hidden features)
        return states, h_t, c_t



    def update(self, env, coords, x, local_night, local_dawn, local_dusk, bird_uv, directions, speeds, h_t, c_t):

        #print(env.shape, coords.shape, bird_uv.shape)

        inputs = torch.cat([env, coords, x.view(-1, 1), local_dawn.float().view(-1, 1),
                            local_dusk.float().view(-1, 1), local_night.float().view(-1, 1), bird_uv,
                            directions.view(-1, 1), speeds.view(-1, 1)], dim=1)

        # inputs = torch.cat([env, coords, x.view(-1, 1), local_dawn.float().view(-1, 1),
        #                     local_dusk.float().view(-1, 1), local_night.float().view(-1, 1),
        #                     directions.view(-1, 1), speeds.view(-1, 1)], dim=1)

        inputs = self.node2hidden(inputs)
        h_t[0], c_t[0] = self.lstm_layers[0](inputs, (h_t[0], c_t[0]))
        h_t[0] = F.dropout(h_t[0], p=self.dropout_p, training=self.training, inplace=False)
        c_t[0] = F.dropout(c_t[0], p=self.dropout_p, training=self.training, inplace=False)
        for l in range(1, self.n_lstm_layers):
            h_t[l], c_t[l] = self.lstm_layers[l](h_t[l - 1], (h_t[l], c_t[l]))

        return h_t, c_t


class RecurrentEncoder2(torch.nn.Module):
    def __init__(self, **kwargs):
        super(RecurrentEncoder2, self).__init__()

        self.timesteps = kwargs.get('timesteps', 12)
        self.n_in = 6 + kwargs.get('n_env', 4)
        self.n_hidden = kwargs.get('n_hidden', 16)
        self.n_lstm_layers = kwargs.get('n_layers_lstm', 1)
        self.dropout_p = kwargs.get('dropout_p', 0)

        torch.manual_seed(kwargs.get('seed', 1234))

        self.node2hidden = torch.nn.Linear(self.n_in, self.n_hidden)

        self.lstm_layers = nn.ModuleList([nn.LSTMCell(self.n_hidden, self.n_hidden) for _ in range(self.n_lstm_layers)])

        self.reset_parameters()


    def reset_parameters(self):

        def init_weights(m):
            if type(m) == nn.Linear:
                inits.glorot(m.weight)
                inits.zeros(m.bias)
            elif type(m) == nn.LSTMCell:
                for name, param in m.named_parameters():
                    if 'bias' in name:
                        inits.zeros(param)
                    elif 'weight' in name:
                        inits.glorot(param)

        inits.glorot(self.node2hidden.weight)
        self.lstm_layers.apply(init_weights)


    def forward(self, data):
        # initialize lstm variables
        h_t = [torch.zeros(data.x.size(0), self.n_hidden, device=data.x.device) for l in range(self.n_lstm_layers)]
        c_t = [torch.zeros(data.x.size(0), self.n_hidden, device=data.x.device) for l in range(self.n_lstm_layers)]

        states = []

        for t in range(self.timesteps):
            h_t, c_t = self.update(data.env[..., t], data.coords, data.x[:, t], data.local_night[:, t],
                                   data.local_dawn[:, t], data.local_dusk[:, t], h_t, c_t)

            states.append(h_t[-1])
        states = torch.stack(states, dim=1) # shape (radars x timesteps x hidden features)
        return states, h_t, c_t



    def update(self, env, coords, x, local_night, local_dawn, local_dusk, h_t, c_t):
        inputs = torch.cat([env, coords, x.view(-1, 1), local_dawn.float().view(-1, 1),
                            local_dusk.float().view(-1, 1), local_night.float().view(-1, 1)], dim=1)
        inputs = self.node2hidden(inputs)
        h_t[0], c_t[0] = self.lstm_layers[0](inputs, (h_t[0], c_t[0]))
        for l in range(1, self.n_lstm_layers):
            h_t[0] = F.dropout(h_t[0], p=self.dropout_p, training=self.training, inplace=False)
            c_t[0] = F.dropout(c_t[0], p=self.dropout_p, training=self.training, inplace=False)
            h_t[l], c_t[l] = self.lstm_layers[l](h_t[l - 1], (h_t[l], c_t[l]))


        return h_t, c_t



class BirdDynamicsGraphLSTM(MessagePassing):

    def __init__(self, **kwargs):
        super(BirdDynamicsGraphLSTM, self).__init__(aggr='add', node_dim=0) # inflows from neighbouring radars are aggregated by adding

        self.horizon = kwargs.get('horizon', 40)
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
        self.t_context = kwargs.get('context', 0)
        self.teacher_forcing = kwargs.get('teacher_forcing', 0)

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
                                            torch.nn.LeakyReLU(),
                                            torch.nn.Linear(self.n_hidden, 1))

        self.node2hidden = torch.nn.Sequential(torch.nn.Linear(self.n_node_in, self.n_hidden),
                                               torch.nn.Dropout(p=self.dropout_p),
                                               torch.nn.LeakyReLU(),
                                               torch.nn.Linear(self.n_hidden, self.n_hidden))

        self.lstm_layers = nn.ModuleList([nn.LSTMCell(self.n_hidden, self.n_hidden) for _ in range(self.n_lstm_layers)])

        self.hidden2delta = torch.nn.Sequential(torch.nn.Linear(self.n_hidden, self.n_hidden),
                                               torch.nn.Dropout(p=self.dropout_p),
                                               torch.nn.LeakyReLU(),
                                               torch.nn.Linear(self.n_hidden, 1))

        if self.use_encoder:
            self.encoder = RecurrentEncoder(timesteps=self.t_context, n_env=self.n_env, n_hidden=self.n_hidden,
                                            n_lstm_layers=self.n_lstm_layers, seed=seed, dropout_p=self.dropout_p)



    def forward(self, data):
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
            h_t = [torch.zeros(data.x.size(0), self.n_hidden, device=x.device) for l in range(self.n_lstm_layers)]
            c_t = [torch.zeros(data.x.size(0), self.n_hidden, device=x.device) for l in range(self.n_lstm_layers)]

        coords = data.coords
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        self.fluxes = torch.zeros((data.x.size(0), 1, self.horizon + 1), device=x.device)
        self.local_deltas = torch.zeros((data.x.size(0), 1, self.horizon + 1), device=x.device)

        if self.use_encoder:
            forecast_horizon = range(self.t_context + 1, self.t_context + self.horizon + 1)
        else:
            forecast_horizon = range(1, self.horizon + 1)

        for t in forecast_horizon:

            if True: #torch.any(data.local_night[:, t] | data.local_dusk[:, t]):
                # at least for one radar station it is night or dusk
                r = torch.rand(1)
                if r < self.teacher_forcing:
                    # if data is available use ground truth, otherwise use model prediction
                    x = data.missing[..., t-1].view(-1, 1) * x + \
                        ~data.missing[..., t-1].view(-1, 1) * data.x[..., t-1].view(-1, 1)


                x, h_t, c_t = self.propagate(edge_index, x=x, coords=coords,
                                   edge_attr=edge_attr, h_t=h_t, c_t=c_t, areas=data.areas,
                                   dusk=data.local_dusk[:, t-1],
                                   dawn=data.local_dawn[:, t],
                                   env=data.env[..., t],
                                   night=data.local_night[:, t],
                                   t=t-self.t_context,
                                   boundary=data.boundary)

                if self.fixed_boundary:
                    # use ground truth for boundary cells
                    x[data.boundary, 0] = data.y[data.boundary, t]

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

        msg = F.leaky_relu(self.fc_edge_in(features))
        msg = F.dropout(msg, p=self.dropout_p, training=self.training)

        for l in self.fc_edge_hidden:
            msg = F.leaky_relu(l(msg))
            msg = F.dropout(msg, p=self.dropout_p, training=self.training)

        msg = self.fc_edge_out(msg) #.relu()

        return msg


    def update(self, aggr_out, x, coords, env, areas, dusk, dawn, h_t, c_t, t, night, boundary):

        # predict departure/landing
        if self.edge_type == 'voronoi':
            inputs = torch.cat([x.view(-1, 1), coords, env, dusk.float().view(-1, 1),
                                dawn.float().view(-1, 1), areas.view(-1, 1), night.float().view(-1, 1)], dim=1)
        else:
            inputs = torch.cat([x.view(-1, 1), coords, env, dusk.float().view(-1, 1),
                                dawn.float().view(-1, 1), night.float().view(-1, 1)], dim=1)
        inputs = self.node2hidden(inputs) #.relu()
        h_t[0], c_t[0] = self.lstm_layers[0](inputs, (h_t[0], c_t[0]))
        for l in range(1, self.n_lstm_layers):
            h_t[l], c_t[l] = self.lstm_layers[l](h_t[l - 1], (h_t[l], c_t[l]))
        delta = self.hidden2delta(h_t[-1]) #.tanh()

        if self.predict_delta:
            # combine messages from neighbors into single number representing total flux
            flux = self.mlp_aggr(aggr_out) #.tanh()
            pred = x + ~boundary.view(-1, 1) * flux + delta
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

        self.horizon = kwargs.get('horizon', 40)
        self.dropout_p = kwargs.get('dropout_p', 0)
        self.n_hidden = kwargs.get('n_hidden', 16)
        self.n_node_in = 4 + kwargs.get('n_env', 4)
        self.n_node_features_in = 4 + kwargs.get('n_env', 4)
        self.n_edge_in = 6 + 2*kwargs.get('n_env', 4)
        self.n_fc_layers = kwargs.get('n_fc_layers', 1)
        self.n_lstm_layers = kwargs.get('n_lstm_layers', 1)
        self.fixed_boundary = kwargs.get('fixed_boundary', [])
        self.forced_zeros = kwargs.get('forced_zeros', [])
        self.teacher_forcing = kwargs.get('teacher_forcing', 0)

        self.edge_type = kwargs.get('edge_type', 'voronoi')
        if self.edge_type == 'voronoi':
            self.n_node_features_in += 1
            self.n_edge_in += 1

        torch.manual_seed(kwargs.get('seed', 1234))



        self.edge_input2hidden = torch.nn.Linear(self.n_edge_in, self.n_hidden)
        self.aggr2hidden = torch.nn.Linear(self.n_hidden*2, self.n_hidden)
        self.fc_edge_in = torch.nn.Linear(3 * self.n_hidden, self.n_hidden)
        self.fc_edge_hidden = nn.ModuleList([torch.nn.Linear(self.n_hidden, self.n_hidden)
                                        for _ in range(self.n_fc_layers - 1)])
        self.fc_edge_out = torch.nn.Linear(self.n_hidden, self.n_hidden)

        self.node2hidden = torch.nn.Sequential(torch.nn.Linear(self.n_node_in, self.n_hidden),
                                               torch.nn.Dropout(p=self.dropout_p),
                                               torch.nn.LeakyReLU(),
                                               torch.nn.Linear(self.n_hidden, self.n_hidden))

        self.input2hidden = torch.nn.Sequential(torch.nn.Linear(self.n_node_features_in, self.n_hidden),
                                               torch.nn.Dropout(p=self.dropout_p),
                                               torch.nn.LeakyReLU(),
                                               torch.nn.Linear(self.n_hidden, self.n_hidden))

        self.lstm_layers = nn.ModuleList([nn.LSTMCell(self.n_hidden*2, self.n_hidden*2) for _ in range(self.n_lstm_layers)])

        self.hidden2birds = torch.nn.Sequential(torch.nn.Linear(2*self.n_hidden, self.n_hidden),
                                               torch.nn.Dropout(p=self.dropout_p),
                                               torch.nn.LeakyReLU(),
                                               torch.nn.Linear(self.n_hidden, 1))



    def forward(self, data):
        # with teacher_forcing = 0.0 the model always uses previous predictions to make new predictions
        # with teacher_forcing = 1.0 the model always uses the ground truth to make new predictions

        # measurement at t=0
        x = data.x[..., 0].view(-1, 1)
        y_hat = []
        y_hat.append(x)

        coords = data.coords
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        # initialize lstm variables
        h_t = [torch.zeros(data.x.size(0), self.n_hidden, device=x.device) for l in range(self.n_lstm_layers)]
        c_t = [torch.zeros(data.x.size(0), self.n_hidden, device=x.device) for l in range(self.n_lstm_layers)]

        node_states = torch.cat([x, data.env[..., 0],
                                 data.local_night[..., 0].float().view(-1, 1),
                                 data.local_dusk[..., 0].float().view(-1, 1),
                                 data.local_dawn[..., 0].float().view(-1, 1)], dim=1)
        h_t[0] = self.node2hidden(node_states)
        c_t[0] = self.node2hidden(node_states)


        for t in range(self.horizon):

            if True: #torch.any(data.local_night[:, t+1] | data.local_dusk[:, t+1]):
                # at least for one radar station it is night or dusk
                r = torch.rand(1)
                if r < self.teacher_forcing:
                    # if data is available use ground truth, otherwise use model prediction
                    x = data.missing[..., t].view(-1, 1) * x + \
                        ~data.missing[..., t].view(-1, 1) * data.x[..., t].view(-1, 1)

                x, h_t, c_t = self.propagate(edge_index, coords=coords,
                                             edge_attr=edge_attr, h_t=h_t, c_t=c_t, areas=data.areas,
                                             dusk=data.local_dusk[:, t],
                                             dawn=data.local_dawn[:, t+1],
                                             env=data.env[..., t+1])

                if self.fixed_boundary:
                    # use ground truth for boundary cells
                    x[data.boundary, 0] = data.y[data.boundary, t+1]

            if self.forced_zeros:
                # for locations where it is dawn: set birds in the air to zero
                x = x * data.local_night[:, t+1].view(-1, 1)

            y_hat.append(x)

        prediction = torch.cat(y_hat, dim=-1)
        return prediction


    def message(self, h_t_i, h_t_j, coords_i, coords_j, env_i, env_j, edge_attr):
        # construct messages to node i for each edge (j,i)
        # can take any argument initially passed to propagate()
        # x_j are source features with shape [E, out_channels]

        features = torch.cat([coords_i, coords_j, env_i, env_j, edge_attr], dim=1)
        #msg = self.mlp_edge(features).relu()

        features = self.edge_input2hidden(features) #.relu()
        msg = self.fc_edge_in(torch.cat([h_t_i, h_t_j, features], dim=1))
        msg = F.dropout(msg, p=self.dropout_p, training=self.training)

        for l in self.fc_edge_hidden:
            msg = F.leaky_relu(l(msg))
            msg = F.dropout(msg, p=self.dropout_p, training=self.training)

        msg = F.leaky_relu(self.fc_edge_out(msg))

        return msg


    def update(self, aggr_out, coords, env, areas, dusk, dawn, h_t, c_t):

        # recurrent component
        if self.edge_type == 'voronoi':
            inputs = torch.cat([coords, env, dusk.float().view(-1, 1),
                                dawn.float().view(-1, 1), areas.view(-1, 1)], dim=1)
        else:
            inputs = torch.cat([coords, env, dusk.float().view(-1, 1),
                                dawn.float().view(-1, 1)], dim=1)
        inputs = self.intput2hidden(inputs) #.relu()
        inputs = self.aggr2hidden(torch.cat([inputs, aggr_out]))
        h_t[0], c_t[0] = self.lstm_layers[0](inputs, (h_t[0], c_t[0]))
        for l in range(1, self.n_lstm_layers):
            h_t[l], c_t[l] = self.lstm_layers[l](h_t[l - 1], (h_t[l], c_t[l]))

        # combine messages from neighbors and recurrent module into single number representing the new bird density
        pred = self.hidden2birds(h_t[-1]).sigmoid()

        return pred, h_t, c_t



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

def train_fluxes(model, train_loader, optimizer, loss_func, device, conservation_constraint=0.01,
                 teacher_forcing=1.0, daymask=True, boundary_constraint_only=False):
    model.train()
    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        model.teacher_forcing = teacher_forcing
        output = model(data) #.view(-1)

        # if n_devices > 1:
        #     gt = torch.cat([d.y for d in data])
        # else:
        #     gt = data.y

        gt = data.y

        if conservation_constraint > 0:

            inferred_fluxes = model.local_fluxes[..., 1:].squeeze()
            #print('inferred fluxes shape', inferred_fluxes.shape)
            inferred_fluxes = inferred_fluxes - inferred_fluxes[data.reverse_edges]

            observed_fluxes = data.fluxes[..., model.t_context:-1].squeeze()


            diff = observed_fluxes - inferred_fluxes
            diff = observed_fluxes**2 * diff # weight timesteps with larger fluxes more
            if boundary_constraint_only:
                edges = data.boundary2inner_edges + data.inner2boundary_edges
            else:
                edges = data.boundary2inner_edges + data.inner2boundary_edges + data.inner_edges
            diff = diff[edges]
            constraints = (diff[~torch.isnan(diff)]**2).mean()
        else:
            constraints = 0

        if daymask:
            mask = data.local_night & ~data.missing
        else:
            mask = ~data.missing
        if hasattr(model, 't_context'):
            gt = gt[:, model.t_context:]
            mask = mask[:, model.t_context:]
        #print(diff.size(), loss_func(output, gt, mask).detach(), constraints.detach())
        constraints = conservation_constraint * constraints
        loss = loss_func(output, gt, mask)
        #print(loss, constraints)
        loss = loss + constraints
        loss_all += data.num_graphs * float(loss)
        loss.backward()

        optimizer.step()

        del loss, output

    return loss_all



def train_testFluxMLP(model, train_loader, optimizer, loss_func, device):

    model.train()
    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data) #.view(-1)
        gt = data.y

        observed_fluxes = data.fluxes[..., model.t_context:-1].squeeze()

        inferred_fluxes = model.local_fluxes[..., 1:].squeeze()
        # print('inferred fluxes', inferred_fluxes)
        diff = observed_fluxes - inferred_fluxes
        edges = data.boundary2inner_edges

        diff = diff[edges]
        loss = (diff[~torch.isnan(diff)]**2).mean()

        loss_all += data.num_graphs * float(loss)
        loss.backward()
        optimizer.step()

    return loss_all


def train_dynamics(model, train_loader, optimizer, loss_func, device, teacher_forcing=0, daymask=True):

    model.train()
    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        model.teacher_forcing = teacher_forcing
        output = model(data)

        # if n_devices > 1:
        #     gt = torch.cat([d.y for d in data])
        #     local_night = torch.cat([d.local_night for d in data])
        #     missing = torch.cat([d.missing for d in data])

        gt = data.y

        if daymask:
            mask = data.local_night & ~data.missing
        else:
            mask = ~data.missing

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

def test_fluxes(model, test_loader, loss_func, device, get_fluxes=True, bird_scale=1,
                fixed_boundary=False, daymask=True):
    model.eval()
    loss_all = []
    fluxes = {}

    for tidx, data in enumerate(test_loader):
        data = data.to(device)
        output = model(data) * bird_scale #.view(-1)
        gt = data.y * bird_scale

        if fixed_boundary:
            # boundary_mask = np.ones(output.size(0))
            # boundary_mask[fixed_boundary] = 0
            output = output[~data.boundary]
            gt = gt[~data.boundary]

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
        loss_all.append(torch.tensor([loss_func(output[:, t], gt[:, t], mask[:, t]).detach()
                                      for t in range(model.horizon + 1)]))

    if get_fluxes:
        return torch.stack(loss_all), fluxes
    else:
        return torch.stack(loss_all)

def test_dynamics(model, test_loader, loss_func, device, bird_scale=2000, daymask=True):

    model.eval()
    loss_all = []

    for nidx, data in enumerate(test_loader):
        data = data.to(device)

        with torch.no_grad():
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
