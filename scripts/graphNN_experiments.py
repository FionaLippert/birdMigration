# good results with: node embedding, degree normalization, multiple timesteps, outflow reg only for center nodes
# constraints weight = 0.01
# edge function: linear and sigmoid

from birds.graphNN import *
import argparse
from datetime import datetime
from matplotlib import pyplot as plt
import itertools as it
import os
import os.path as osp
import pickle5 as pickle
import pandas as pd
import torch
from torch_geometric.data import DataLoader
from torch.optim import lr_scheduler
from birds import GBT, datasets


parser = argparse.ArgumentParser(description='GraphNN experiments')
parser.add_argument('action', type=str, help='train or test')
parser.add_argument('--root', type=str, default='/home/fiona/birdMigration', help='entry point to required data')
parser.add_argument('--experiment', type=str, default='test', help='directory name for model performance output')
parser.add_argument('--data_source', type=str, default='radar', help='data source for training/testing')
parser.add_argument('--cpu', action='store_true', default=False, help='cpu or gpu')
parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_decay', type=float, default=50, help='steps with which learning rate decays')
parser.add_argument('--lr_gamma', type=float, default=0.5, help='decay rate of learning rate')
parser.add_argument('--teacher_forcing', type=float, default=1.0, help='probability with which ground truth is used to predict next state')
parser.add_argument('--teacher_forcing_gamma', type=float, default=0.9, help='decay rate of teacher forcing probability')
parser.add_argument('--repeats', type=int, default=5, help='number of models to be trained with different random seeds')
parser.add_argument('--hidden_dim', type=int, default=16, help='number of hidden nodes in mlp')
parser.add_argument('--dropout_p', type=float, default=0, help='dropout probability')
parser.add_argument('--use_dropout', action='store_true', default=False, help='use dropout in MLPs')
parser.add_argument('--ts_train', type=int, default=6, help='length of training sequences')
parser.add_argument('--ts_test', type=int, default=6, help='length of testing sequences')
parser.add_argument('--save_predictions', action='store_true', default=False, help='save predictions for each radar separately')
parser.add_argument('--plot_predictions', action='store_true', default=False, help='plot predictions for each radar separately')
parser.add_argument('--fix_boundary', action='store_true', default=False, help='fix boundary cells to ground truth')
#parser.add_argument('--use_env_cells', action='store_true', default=False, help='use entire cells to interpolate environment variables')
parser.add_argument('--use_buffers', action='store_true', default=False, help='use radar buffers for training instead of entire cells')
parser.add_argument('--conservation', action='store_true', default=False, help='use mass conservation constraints')
parser.add_argument('--multinight', action='store_true', default=False, help='use departure NN to bridge nights')
parser.add_argument('--weighted_loss', action='store_true', default=False, help='weight squared errors according '
                                'to bird densities to promote better fits for high migration events')
parser.add_argument('--no_wind', action='store_true', default=False, help='do not use wind features in models')
parser.add_argument('--use_black_box', action='store_true', default=False, help='use black box NN without interpretation of messages')
parser.add_argument('--use_black_box_rec', action='store_true', default=False, help='use recurrent black box NN without interpretation of messages')
parser.add_argument('--use_dcrnn', action='store_true', default=False, help='use DCRNN')
parser.add_argument('--recurrent', action='store_true', default=False, help='use recurrent bird flow model')
parser.add_argument('--n_layers', type=int, default=1, help='number of MLP layers')
args = parser.parse_args()

args.cuda = (not args.cpu and torch.cuda.is_available())
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root = osp.join(args.root, 'data')
model_dir = osp.join(args.root, 'models', args.experiment)
os.makedirs(model_dir, exist_ok=True)
gam_csv = osp.join(root, 'seasonal_trends', f'gam_summary_{args.data_source}.csv')

season = 'fall'
if args.data_source == 'radar':
    train_years = ['2016', '2017']
    test_year = '2015'
    val_year = test_year
    bird_scale = 1e7
else:
    train_years = ['2016', '2017', '2018']
    val_year = '2019'
    test_year = '2015'
    bird_scale = 2000

test_val_split = 0.8

def persistence(last_ob, timesteps):
    # always return last observed value
	return [last_ob] * timesteps

def MSE_weighted(output, gt, local_nights, p=0.75):
    errors = (output - gt)**2 * (1 + gt**p)
    mse = torch.mean(errors)
    return mse

def MSE_multinight(output, gt, local_nights):
    # ignore daytime data points
    errors = (output - gt)**2
    mse = torch.mean(errors[local_nights])
    #mse = torch.sum(errors) / torch.sum(local_nights)
    #mse = mse.detach().numpy()
    #mse[np.isinf(mse)] = np.nan
    return mse


def run_training(timesteps, model_type, conservation=True, recurrent=True, embedding=0, norm=False, epochs=100,
                 repeats=1, data_source='radar', output_dir=model_dir, bird_scale=2000, departure=False, dropout_p=0):

    train_data = [datasets.RadarData(root, 'train', year, season, timesteps,
                                     data_source=data_source, use_buffers=args.use_buffers,
                                     bird_scale=bird_scale, multinight=args.multinight) for year in train_years]

    boundaries = train_data[0].info['boundaries']
    if args.fix_boundary:
        fix_boundary = [ridx for ridx, b in boundaries.items() if b]
    else:
        fix_boundary = []

    train_data = torch.utils.data.ConcatDataset(train_data)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)

    val_data = datasets.RadarData(root, 'test', val_year, season, timesteps, data_source=data_source,
                                  bird_scale=bird_scale,
                                  use_buffers=args.use_buffers, multinight=args.multinight)
    val_loader = DataLoader(val_data, batch_size=1)
    if args.data_source == 'radar':
        val_start = int(len(val_loader) * test_val_split)
    else:
        val_start = 0
    val_stop = len(val_loader)

    val_loader = list(val_loader)[val_start:val_stop]



    for r in range(repeats):
        if model_type == 'standard_mlp':
            # model = MLP(6*train_data[0].num_nodes, args.hidden_dim, train_data[0].num_nodes,
            #             timesteps, recurrent, seed=r)
            model = NodeMLP(n_hidden=args.hidden_dim, timesteps=timesteps, seed=r, n_layers=args.n_layers)
        elif model_type == 'standard_lstm':
            model = NodeLSTM(n_hidden=args.hidden_dim, timesteps=timesteps, seed=r)
        elif args.use_black_box:
            model = BirdDynamics(train_data[0].num_nodes, timesteps, args.hidden_dim, embedding, model_type,
                                 seed=r, use_wind=(not args.no_wind), dropout_p=dropout_p, multinight=args.multinight)
        elif args.use_black_box_rec:
            model = BirdRecurrent1(n_hidden=args.hidden_dim, timesteps=timesteps,
                                seed=r, multinight=args.multinight, use_wind=(not args.no_wind), dropout_p=dropout_p)
        elif args.use_dcrnn:
            model = RecurrentGCN(timesteps, node_features=10)
        elif args.recurrent:
            model = BirdFlowRecurrent(timesteps, hidden_dim=args.hidden_dim, model=model_type,
                                    seed=r, fix_boundary=fix_boundary, multinight=args.multinight,
                                    use_wind=(not args.no_wind), dropout_p=dropout_p)
        else:
            model = BirdFlowTime(train_data[0].num_nodes, timesteps, args.hidden_dim, embedding, model_type, norm,
                                 use_departure=departure, seed=r, fix_boundary=fix_boundary, multinight=args.multinight,
                                 use_wind=(not args.no_wind), dropout_p=dropout_p)

        if repeats == 1:
            name = make_name(timesteps, model_type, conservation, recurrent, embedding, norm,
                             epochs, dropout=dropout_p)
        else:
            name = make_name_repeat(timesteps, model_type, conservation, recurrent, embedding, norm,
                                    epochs, repeat=r, dropout=dropout_p)

        params = model.parameters()
        optimizer = torch.optim.Adam(params, lr=args.lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay)#, gamma=args.gamma)

        if args.weighted_loss:
            loss_func = MSE_weighted
        else:
            loss_func = MSE_multinight #torch.nn.MSELoss()

        training_curve = np.zeros(epochs)
        val_curve = np.zeros(epochs)
        best_loss = np.inf
        if not recurrent:
            tf = 1.0
        elif model_type == 'standard_mlp' or model_type == 'standard_lstm':
            tf = -1 # no teacher forcing
        else:
            tf = args.teacher_forcing

        for epoch in range(epochs):
            if args.use_black_box or args.use_black_box_rec or args.use_dcrnn or 'standard' in model_type:
                loss = train_dynamics(model, train_loader, optimizer, loss_func, args.cuda, teacher_forcing=tf)
            else:
                loss = train_fluxes(model, train_loader, optimizer, boundaries, loss_func, args.cuda,
                         conservation, departure=False, teacher_forcing=tf)
            print(f'epoch {epoch + 1}: loss = {loss / len(train_data)}')
            if departure:
                loss = train_fluxes(model, train_loader, optimizer, boundaries, loss_func, args.cuda,
                             conservation, departure=departure, teacher_forcing=tf)
                print(f'epoch {epoch + 1}: loss with departure = {loss / len(train_data)}')
            training_curve[epoch] = loss / len(train_data)

            model.eval()
            if args.use_black_box or args.use_black_box_rec or args.use_dcrnn:
                val_loss = test_dynamics(model, val_loader, timesteps, loss_func, args.cuda, bird_scale=1) #,
                                         #start_idx=val_start, stop_idx=val_stop)
            else:
                val_loss = test_fluxes(model, val_loader, timesteps, loss_func, args.cuda,
                                   get_outfluxes=False, bird_scale=1) #, start_idx=val_start, stop_idx=val_stop)
            val_loss = val_loss[torch.isfinite(val_loss)].mean()
            val_curve[epoch] = val_loss
            print(f'epoch {epoch + 1}: val loss = {val_loss}')

            if val_loss < best_loss:
                # save best model so far
                torch.save(model.cpu(), osp.join(output_dir, name))
                best_loss = val_loss

            scheduler.step()
            if recurrent:
                tf = tf * args.teacher_forcing_gamma

        #torch.save(model, osp.join(output_dir, name))

        fig, ax = plt.subplots()
        ax.plot(range(1, epochs+1), training_curve, label='training')
        ax.plot(range(1, epochs + 1), val_curve, label='validation')
        ax.set(xlabel='epoch', ylabel='Loss', title=f'best model in epoch {np.argmin(val_curve)+1} with MSE={best_loss}',
               yscale='log', xscale='log')

        plt.legend()
        fig.savefig(osp.join(output_dir, f'training_loss_{name}.png'), bbox_inches='tight')



def make_name_repeat(timesteps, model_type, conservation=True, recurrent=True, embedding=0, norm=False, epochs=100, repeat=1, dropout=0):
    if dropout == 0:
        name = f'GNN_{model_type}_ts={timesteps}_embedding={embedding}_conservation={conservation}_' \
           f'epochs={epochs}_recurrent={recurrent}_norm={norm}_repeat={repeat}.pt'
    else:
        name = f'GNN_{model_type}_ts={timesteps}_embedding={embedding}_conservation={conservation}_' \
               f'epochs={epochs}_recurrent={recurrent}_norm={norm}_repeat={repeat}_dropout={dropout}.pt'
    return name

def make_name(timesteps, model_type, conservation=True, recurrent=True, embedding=0, norm=False, epochs=100, dropout=0):
    if dropout == 0:
        name = f'GNN_{model_type}_ts={timesteps}_embedding={embedding}_conservation={conservation}_' \
           f'epochs={epochs}_recurrent={recurrent}_norm={norm}.pt'
    else:
        name = f'GNN_{model_type}_ts={timesteps}_embedding={embedding}_conservation={conservation}_' \
               f'epochs={epochs}_recurrent={recurrent}_norm={norm}_dropout={dropout}.pt'
    return name

def load_model(name):
    model = torch.load(osp.join(model_dir, name))
    #model.recurrent = True
    return model

def load_gam_predictions(csv_file, test_loader, nights, time, radars, timesteps, mask, loss_func):
    df_gam = pd.read_csv(csv_file)
    df_gam.datetime = pd.DatetimeIndex(df_gam.datetime) #, tz='UTC')
    dti = pd.DatetimeIndex(time) #, tz='UTC')

    loss = np.zeros((len(radars), len(nights), timesteps + 1))
    pred_gam = np.zeros((len(radars), len(time)))
    for idx, radar in enumerate(radars):
        df_gam_idx = df_gam[df_gam.radar.str.contains(radar)]
        for nidx, data in enumerate(test_loader):
            y_gam = df_gam_idx[df_gam_idx.datetime.isin(dti[nights[nidx]])].gam_pred.to_numpy()
            pred_gam[idx, nights[nidx]] = y_gam

            start_idx = nights[nidx][0]
            dti_night = dti[start_idx:]
            dti_night = dti_night[mask[start_idx:]]
            y_gam = df_gam_idx[df_gam_idx.datetime.isin(dti_night[:timesteps+1])].gam_pred.to_numpy()
            loss[idx, nidx, :] = [np.square(y_gam[t] - data.y[idx, t] * bird_scale) for t in range(timesteps + 1)]
            #loss[idx, nidx, :] = [loss_func(torch.tensor(y_gam[t+1]), data.y[idx, t]) for t in range(timesteps-1)]

    mse = []
    for nidx, data in enumerate(test_loader):
        Z = data.local_night.sum(0)
        mse.append(loss[:, nidx, :].sum(0) / Z)
    mse = np.stack(mse, axis=1)
    mse = np.nan_to_num(mse, posinf=0).sum(1) / np.isfinite(mse).sum(1)
    mse = np.nan_to_num(mse, posinf=0)
    gam_losses = np.sqrt(mse)

    return gam_losses, pred_gam

def gbt_rmse(bird_scale, multinight, mask, seed=1234):
    gbt = GBT.fit_GBT(root, train_years, season, args.ts_train, args.data_source, bird_scale, multinight, seed)
    X, y = GBT.prepare_data_nights_and_radars('test', root, test_year, season, args.ts_test, args.data_source,
                                   bird_scale, multinight)
    # X has shape (nights, timesteps, radars, features)
    # y has shape (nights, timesteps, radars)
    # mask has shape (radars, timesteps, nights) --> change to (nights, timesteps, radars)
    mask = np.swapaxes(mask, 0, -1)
    rmse = []
    for t in range(args.ts_test + 1):
        gt = y[:, t, :] * bird_scale
        gt = np.concatenate([gt[nidx] for nidx in range(X.shape[0])])
        y_hat = gbt.predict(np.concatenate([X[nidx, t] for nidx in range(X.shape[0])])) * bird_scale
        mask_t = np.concatenate([mask[nidx, t, :] for nidx in range(X.shape[0])])
        y_hat = y_hat * mask_t
        loss = np.square(gt - y_hat)
        mse = np.nan_to_num(np.sum(loss) / np.sum(mask_t), posinf=0)
        rmse.append(np.sqrt(mse))
    return rmse



def plot_test_errors(timesteps, model_names, short_names, model_types, output_path,
                     data_source='radar', bird_scale=2000, departure=False):

    #output_dir = osp.join(root, 'model_performance', f'experiment_{datetime.now()}')
    #os.makedirs(output_dir, exist_ok=True)

    #name = make_name(timesteps, embedding, model_type, recurrent, conservation, norm, epochs)

    test_data = datasets.RadarData(root, 'test', test_year, season, timesteps, data_source=data_source, bird_scale=bird_scale,
                          use_buffers=args.use_buffers, multinight=args.multinight)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    if args.data_source == 'radar':
        split = int(len(test_loader) * test_val_split)
        test_loader = list(test_loader)[:split]

    nights = test_data.info['nights']
    local_nights = test_data.info['local_nights']
    tidx = test_data.info['tidx']
    radar_index = {idx: name for idx, name in enumerate(test_data.info['radars'])}

    with open(osp.join(osp.dirname(output_path), 'nights.pickle'), 'wb') as f:
        pickle.dump(nights, f)
    with open(osp.join(osp.dirname(output_path), 'seq_tidx.pickle'), 'wb') as f:
        pickle.dump(tidx, f)
    with open(osp.join(osp.dirname(output_path), 'local_nights.pickle'), 'wb') as f:
        pickle.dump(local_nights, f)
    with open(osp.join(osp.dirname(output_path), f'radar_index.pickle'), 'wb') as f:
        pickle.dump(radar_index, f, pickle.HIGHEST_PROTOCOL)

    models = [load_model(name) for name in model_names]
    for i, n in enumerate(short_names):
        print(f'model: {n}, num params: {sum(p.numel() for p in models[i].parameters() if p.requires_grad)}')

    loss_func = MSE_multinight #torch.nn.MSELoss()

    fig, ax = plt.subplots(figsize=(10, 4))
    loss_all = {m : [] for m in model_types}
    for midx, model in enumerate(models):
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data)

        model.timesteps = timesteps

        if 'standard' in short_names[midx]:
            loss_all[short_names[midx]].append(
                np.nanmean(test_dynamics(model, test_loader, timesteps, loss_func,
                                         args.cuda, bird_scale=bird_scale).detach().numpy(), axis=0)
            )
        elif args.use_black_box or args.use_black_box_rec or args.use_dcrnn:
            loss_all[short_names[midx]].append(
                np.nanmean(test_dynamics(model, test_loader, timesteps, loss_func,
                                         args.cuda, bird_scale).detach().numpy(), axis=0)
            )
        else:
            l, outfluxes, outfluxes_abs = test_fluxes(model, test_loader, timesteps, loss_func, args.cuda,
                                bird_scale=bird_scale, departure=departure)
            loss_all[short_names[midx]].append(np.nanmean(l.detach().numpy(), axis=0))
            with open(osp.join(osp.dirname(output_path), f'outfluxes_{short_names[midx]}.pickle'), 'wb') as f:
                pickle.dump(outfluxes, f, pickle.HIGHEST_PROTOCOL)
            with open(osp.join(osp.dirname(output_path), f'outfluxes_abs_{short_names[midx]}.pickle'), 'wb') as f:
                pickle.dump(outfluxes_abs, f, pickle.HIGHEST_PROTOCOL)



    for type in model_types:
        losses = np.sqrt(np.stack(loss_all[type]))
        mean_loss = np.nan_to_num(np.nanmean(losses, axis=0))
        std_loss = np.nan_to_num(np.nanstd(losses, axis=0))
        line = ax.plot(range(timesteps+1), mean_loss, label=f'{type}')
        ax.fill_between(range(timesteps+1), mean_loss - std_loss, mean_loss + std_loss, alpha=0.2,
                        color=line[0].get_color())

    #gam_losses, _ = load_gam_predictions(gam_csv, test_loader, test_data.info['nights'], test_data.info['timepoints'],
    #                                   test_data.info['radars'], timesteps, test_data.info['time_mask'], loss_func)
    #ax.plot(range(timesteps+1), gam_losses, label=f'GAM')

    gbt_losses = []
    mask = np.stack([local_nights[:, tidx[:, nidx]] for nidx in range(tidx.shape[1])], axis=-1)
    for r in range(args.repeats):
        gbt_losses.append(gbt_rmse(bird_scale, args.multinight, mask, seed=r))
    gbt_mean_loss = np.stack(gbt_losses, axis=0).mean(0)
    gbt_std_loss = np.stack(gbt_losses, axis=0).std(0)
    line = ax.plot(range(timesteps+1), gbt_mean_loss, label=f'GBT')
    ax.fill_between(range(timesteps+1), gbt_mean_loss - gbt_std_loss, gbt_mean_loss + gbt_std_loss, alpha=0.2,
                    color=line[0].get_color())

    naive_losses = []
    def naive(t, nidx):
        daytime_mask = local_nights[:, tidx[t, nidx]]
        return data.x[:, 0].cpu() * daytime_mask * bird_scale

    for nidx, data in enumerate(test_loader):
        naive_losses.append(torch.tensor(
            [loss_func(naive(t, nidx), data.y[:, t].cpu() * bird_scale, data.local_night[:, t].cpu()) for t in range(timesteps + 1)]))
    naive_losses = torch.stack(naive_losses, dim=0).detach().numpy()
    naive_losses = np.nan_to_num(naive_losses, posinf=0).sum(0) / np.isfinite(naive_losses).sum(0)
    naive_losses = np.sqrt(np.nan_to_num(naive_losses, posinf=0))
    ax.plot(range(timesteps+1), naive_losses, label=f'constant night')


    ax.set_xlabel('timestep')
    ax.set_ylabel('RMSE')
    ax.set_xticks(range(timesteps+1))
    ax.legend()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


def plot_predictions(timesteps, model_names, short_names, model_types, output_dir, tidx=None,
                     data_source='radar', repeats=1, bird_scale=2000, departure=False):

    dataset = datasets.RadarData(root, 'test', test_year, season, timesteps, data_source=data_source, bird_scale=bird_scale,
                        use_buffers=args.use_buffers, multinight=args.multinight)
    nights = dataset.info['nights']
    seq_tidx = dataset.info['tidx']
    time = np.array(dataset.info['timepoints'])
    with open(osp.join(output_dir, 'nights.pickle'), 'wb') as f:
        pickle.dump(nights, f)
    df_gam = pd.read_csv(gam_csv)
    df_gam.datetime = pd.DatetimeIndex(df_gam.datetime)
    dti = pd.DatetimeIndex(time) #, tz='UTC')

    gbt_models = []
    for r in range(args.repeats):
        gbt_models.append(GBT.fit_GBT(root, train_years, season, args.ts_train, args.data_source, bird_scale,
                                      args.multinight, seed=r))
    X_gbt, y_gbt = GBT.prepare_data_nights_and_radars('test', root, test_year, season, args.ts_test,
                                                      args.data_source, bird_scale, args.multinight)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    if args.data_source == 'radar':
        split = int(len(dataloader) * test_val_split)
        dataloader = list(dataloader)[:split]

    models = [load_model(name) for name in model_names]

    if tidx is None:
        tidx = range(time.size)

    for idx, radar in enumerate(dataset.info['radars']):
        gt = np.zeros(len(time))
        pred = []
        for _ in models:
            pred.append(np.zeros(len(time))) # * np.nan)
        pred = np.stack(pred, axis=0)
        pred_gam = np.zeros(len(time))
        pred_gbt = np.zeros((args.repeats, len(time)))
        df_gam_idx = df_gam[df_gam.radar.str.contains(radar)]

        for nidx, data in enumerate(dataloader):
            #gt[nights[nidx][:timesteps+1]] = data.y[idx]
            gt[seq_tidx[:, nidx]] = data.y[idx]

            if args.cuda: data = data.to('cuda')
            for midx, model in enumerate(models):
                model.timesteps = timesteps
                if args.cuda: model.cuda()
                y = model(data).cpu().detach().numpy()[idx]
                #pred[midx][nights[nidx][:timesteps+1]] = y
                pred[midx][seq_tidx[:, nidx]] = y

            pred_gam[nights[nidx][:timesteps+1]] = df_gam_idx[df_gam_idx.datetime.isin(dti[nights[nidx][:timesteps+1]])].gam_pred.to_numpy()
            for r in range(args.repeats):
                pred_gbt[r, seq_tidx[:, nidx]] = gbt_models[r].predict(X_gbt[nidx, :, idx, :]) * bird_scale

        fig, ax = plt.subplots(figsize=(20, 4))
        for midx, model_type in enumerate(model_types):
            all_pred = pred[midx*repeats:(midx+1)*repeats, tidx] * bird_scale
            mean_pred = all_pred.mean(0)
            std_pred = all_pred.std(0)

            line = ax.plot(time[tidx], pred[midx][tidx], ls='--', alpha=0.3)
            line = ax.errorbar(time[tidx], mean_pred, std_pred, ls='--', alpha=0.4, capsize=3)
            ax.scatter(time[tidx], mean_pred, s=30, facecolors='none', edgecolor=line[0].get_color(),
                       label=f'{model_type}', alpha=0.4)

        line = ax.plot(time[tidx], pred_gam[tidx], ls='--', alpha=0.4)
        ax.scatter(time[tidx], pred_gam[tidx], s=30, facecolors='none', edgecolor=line[0].get_color(),
                   label=f'GAM', alpha=0.4)

        line = ax.errorbar(time[tidx], pred_gbt[:, tidx].mean(0), pred_gbt[:, tidx].std(0), ls='--', alpha=0.4)
        ax.scatter(time[tidx], pred_gbt[:, tidx].mean(0), s=30, facecolors='none', edgecolor=line[0].get_color(),
                   label=f'GBT', alpha=0.4)

        ax.plot(time[tidx], gt[tidx] * bird_scale, label='ground truth', c='gray', alpha=0.8)

        ax.set_title(radar)
        #ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel('bird density')
        fig.legend(loc='upper right', bbox_to_anchor=(0.77, 0.85))
        fig.savefig(os.path.join(output_dir, f'{radar}.png'), bbox_inches='tight')
        plt.close(fig)


def plot_predictions_1seq(timesteps, model_names, short_names, model_types, output_dir, nidx=0,
                     data_source='radar', repeats=1, bird_scale=2000, departure=False):

    dataset = datasets.RadarData(root, 'test', test_year, season, timesteps, data_source=data_source, bird_scale=bird_scale,
                        use_buffers=args.use_buffers, multinight=args.multinight)
    nights = dataset.info['nights']
    local_nights = dataset.info['local_nights']
    seq_tidx = dataset.info['tidx']
    time = np.array(dataset.info['timepoints'])
    with open(osp.join(output_dir, 'nights.pickle'), 'wb') as f:
        pickle.dump(nights, f)
    #df_gam = pd.read_csv(gam_csv)
    #df_gam.datetime = pd.DatetimeIndex(df_gam.datetime)
    dti = pd.DatetimeIndex(time) #, tz='UTC')

    gbt_models = []
    for r in range(args.repeats):
        gbt_models.append(GBT.fit_GBT(root, train_years, season, args.ts_train, args.data_source, bird_scale,
                                      args.multinight, seed=r))
    X_gbt, y_gbt = GBT.prepare_data_nights_and_radars('test', root, test_year, season, args.ts_test,
                                                      args.data_source, bird_scale, args.multinight)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    if args.data_source == 'radar':
        split = int(len(dataloader) * test_val_split)
        dataloader = list(dataloader)[:split]

    models = [load_model(name) for name in model_names]

    for idx, radar in enumerate(dataset.info['radars']):
        pred = []
        for _ in models:
            pred.append(np.zeros(len(seq_tidx[:, nidx]))) # * np.nan)
        pred = np.stack(pred, axis=0)
        #pred_gam = np.zeros(len(seq_tidx[:, nidx]))
        pred_gbt = np.zeros((args.repeats, len(seq_tidx[:, nidx])))
        #df_gam_idx = df_gam[df_gam.radar.str.contains(radar)]

        data = list(dataloader)[nidx]

        gt = data.y[idx]

        if args.cuda: data = data.to('cuda')
        for midx, model in enumerate(models):
            model.timesteps = timesteps
            if args.cuda: model.cuda()
            y = model(data).cpu().detach().numpy()[idx]
            #pred[midx][nights[nidx][:timesteps+1]] = y
            pred[midx] = y

        #pred_gam = df_gam_idx[df_gam_idx.datetime.isin(dti[seq_tidx[:, nidx]])].gam_pred.to_numpy()
        for r in range(args.repeats):
            pred_gbt[r] = gbt_models[r].predict(X_gbt[nidx, :, idx, :]) * bird_scale

        tidx = seq_tidx[:, nidx]
        fig, ax = plt.subplots(figsize=(20, 4))
        for midx, model_type in enumerate(model_types):
            all_pred = pred[midx*repeats:(midx+1)*repeats] * bird_scale
            mean_pred = all_pred.mean(0)
            std_pred = all_pred.std(0)

            line = ax.plot(dti[tidx], pred[midx], ls='--', alpha=0.3)
            line = ax.errorbar(dti[tidx], mean_pred, std_pred, ls='--', alpha=0.4, capsize=3)
            ax.scatter(dti[tidx], mean_pred, s=30, facecolors='none', edgecolor=line[0].get_color(),
                       label=f'{model_type}', alpha=0.4)

        #line = ax.plot(dti[tidx], pred_gam, ls='--', alpha=0.4)
        #ax.scatter(dti[tidx], pred_gam, s=30, facecolors='none', edgecolor=line[0].get_color(),
        #           label=f'GAM', alpha=0.4)

        line = ax.errorbar(dti[tidx], pred_gbt.mean(0), pred_gbt.std(0), ls='--', alpha=0.4)
        ax.scatter(dti[tidx], pred_gbt.mean(0), s=30, facecolors='none', edgecolor=line[0].get_color(),
                   label=f'GBT', alpha=0.4)

        ax.plot(dti[tidx], gt.cpu() * bird_scale, label='ground truth', c='gray', alpha=0.8)

        #plt.xticks(range(tidx.size), time[tidx], rotation=45, ha='right');

        ax.set_title(radar)
        #ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel('bird density')
        fig.legend(loc='upper right', bbox_to_anchor=(0.77, 0.85))
        fig.savefig(os.path.join(output_dir, f'{radar}.png'), bbox_inches='tight')
        plt.close(fig)


def predictions(timesteps, model_names, model_types, output_dir,
                     data_source='radar', repeats=1, bird_scale=2000, departure=False):

    dataset = datasets.RadarData(root, 'test', test_year, season, timesteps, data_source=data_source, bird_scale=bird_scale,
                        use_buffers=args.use_buffers, multinight=args.multinight)
    nights = dataset.info['nights']
    time = dataset.info['timepoints']
    with open(osp.join(output_dir, 'nights.pickle'), 'wb') as f:
        pickle.dump(nights, f)

    dataloader = DataLoader(dataset, batch_size=1)
    if args.data_source == 'radar':
        split = int(len(dataloader) * test_val_split)
        dataloader = list(dataloader)[:split]

    models = [load_model(name) for name in model_names]
    dfs = []
    for idx, radar in enumerate(dataset.info['radars']):
        gt = np.zeros(len(time))
        pred = []
        for _ in models:
            pred.append(np.zeros(len(time)))
        pred = np.stack(pred, axis=0)

        for nidx, data in enumerate(dataloader):
            gt[nights[nidx][:timesteps+1]] = data.y[idx]

            if args.cuda: data = data.to('cuda')
            for midx, model in enumerate(models):
                model.timesteps = timesteps
                if args.cuda: model.cuda()
                y = model(data).cpu().detach().numpy()[idx]
                pred[midx][nights[nidx][:timesteps]] = y

        df = pd.DataFrame({'radar': [radar] * time.size,
                           'datetime': time,
                           'vid': gt})

        for midx, model_type in enumerate(model_types):
            all_pred = pred[midx*repeats:(midx+1)*repeats] * bird_scale
            mean_pred = all_pred.mean(0)
            std_pred = all_pred.std(0)
            df[model_type] = mean_pred
            df[f'{model_type}_std'] = std_pred

        dfs.append(df)
    df = pd.concat(dfs)
    df.to_csv(osp.join(output_dir, 'model_predictions.csv'))




epochs = args.epochs
norm = False
repeats = args.repeats #1
departure = False #True #True #True


if args.use_black_box_rec:
    model_types = ['RGNN']
    model_labels = model_types
elif args.use_dcrnn:
    model_types = ['DCRNN']
    model_labels = model_types
else:
    model_types = ['standard_mlp', 'standard_lstm'] #['linear+sigmoid', 'mlp']  # , 'mlp']#'linear+sigmoid', 'mlp']#, 'standard_mlp']
    model_labels = model_types #['G_linear+sigmoid', 'G_mlp']  # , 'G_mlp'] #'G_linear+sigmoid', 'G_mlp']#, 'standard_mlp']

if args.action =='train':

    cons_settings = [False] # [True, False]
    if args.use_dropout:
        dropout_settings = [0, .25, .5]
    else:
        dropout_settings = [0]
    rec_settings = [True] #, False]
    emb_settings = [0]

    all_settings = it.product(model_types, cons_settings, rec_settings, emb_settings, dropout_settings)

    for type, cons, rec, emb, dropout in all_settings:
        run_training(args.ts_train, type, cons, rec, emb, epochs=epochs, data_source=args.data_source, repeats=repeats,
                     bird_scale=bird_scale, departure=departure, dropout_p=dropout)

if args.action == 'test':

    cons = args.conservation
    rec = True
    emb = 0

    if repeats > 1:
        #all_settings = it.product(model_types, cons_settings, rec_settings, emb_settings)
        model_names = [make_name_repeat(args.ts_train, type, cons, rec, emb, epochs=epochs, repeat=r, dropout=args.dropout_p)
                       for type, r in it.product(model_types, range(repeats))]
    else:
        model_names = [make_name(args.ts_train, type, cons, rec, emb, epochs=epochs, dropout=args.dropout_p)
                       for type in model_types]
    #short_names = model_types
    short_names = [type for type, r in it.product(model_labels, range(repeats))]

    output_path = osp.join(root, 'model_performance', args.experiment,
                           f'conservation={cons}_recurrent={rec}_embedding={emb}_timesteps={args.ts_test}',
                            'test_errors.png')
    os.makedirs(osp.dirname(output_path), exist_ok=True)


    plot_test_errors(args.ts_test, model_names, short_names, model_labels, output_path, data_source=args.data_source,
                     bird_scale=bird_scale, departure=departure)
    if args.save_predictions:
        predictions(args.ts_train, model_names, model_types, osp.dirname(output_path),
                data_source=args.data_source, repeats=repeats, bird_scale=bird_scale, departure=departure)

    if args.plot_predictions:
        # plot_predictions(args.ts_test, model_names, short_names, model_labels, osp.dirname(output_path),
        #              data_source=args.data_source, repeats=repeats, tidx=range(4*24, 11*24),
        #                  departure=departure, bird_scale=bird_scale) #, tidx=range(18*24, 32*24))

        plot_predictions_1seq(args.ts_test, model_names, short_names, model_labels, osp.dirname(output_path),
                         data_source=args.data_source, repeats=repeats, nidx=6,
                         departure=departure, bird_scale=bird_scale)  # , tidx=range(18*24, 32*24))
