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
parser.add_argument('--ts_train', type=int, default=6, help='length of training sequences')
parser.add_argument('--ts_test', type=int, default=6, help='length of testing sequences')
parser.add_argument('--save_predictions', action='store_true', default=False, help='save predictions for each radar separately')
parser.add_argument('--plot_predictions', action='store_true', default=False, help='plot predictions for each radar separately')
parser.add_argument('--fix_boundary', action='store_true', default=False, help='fix boundary cells to ground truth')
args = parser.parse_args()

args.cuda = (not args.cpu and torch.cuda.is_available())

root = osp.join(args.root, 'data')
model_dir = osp.join(args.root, 'models', args.experiment)
os.makedirs(model_dir, exist_ok=True)
gam_csv = osp.join(root, 'seasonal_trends', f'gam_summary_{args.data_source}.csv')

season = 'fall'
train_years = ['2016', '2017', '2018']
val_year = '2019'
test_year = '2015'

def persistence(last_ob, timesteps):
    # always return last observed value
	return [last_ob] * timesteps


def run_training(timesteps, model_type, conservation=True, recurrent=True, embedding=0, norm=False, epochs=100,
                 repeats=1, data_source='radar', output_dir=model_dir, bird_scale=2000, departure=False):

    train_data = [RadarData(root, year, season, timesteps, data_source=data_source,
                            bird_scale=bird_scale) for year in train_years]
    boundaries = train_data[0].info['boundaries']
    if args.fix_boundary:
        fix_boundary = [ridx for ridx, b in boundaries.items() if b]
    else:
        fix_boundary = []
    train_data = torch.utils.data.ConcatDataset(train_data)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)

    val_data = RadarData(root, val_year, season, timesteps, data_source=data_source, bird_scale=bird_scale)
    val_loader = DataLoader(val_data, batch_size=1)

    for r in range(repeats):
        if model_type == 'standard_mlp':
            model = MLP(6*22, 2*22, 22, timesteps, recurrent, seed=r)
            use_conservation = False
        else:
            model = BirdFlowTime(train_data[0].num_nodes, timesteps, args.hidden_dim, embedding, model_type, norm,
                                 use_departure=departure, seed=r, fix_boundary=fix_boundary)
            use_conservation = conservation

        if repeats == 1:
            name = make_name(timesteps, model_type, conservation, recurrent, embedding, norm, epochs)
        else:
            name = make_name_repeat(timesteps, model_type, conservation, recurrent, embedding, norm, epochs, repeat=r)

        params = model.parameters()
        optimizer = torch.optim.Adam(params, lr=args.lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay)#, gamma=args.gamma)
        loss_func = torch.nn.MSELoss()

        training_curve = np.zeros(epochs)
        val_curve = np.zeros(epochs)
        best_loss = np.inf
        if not recurrent:
            tf = 1.0
        else:
            tf = args.teacher_forcing
        for epoch in range(epochs):
            loss = train_fluxes(model, train_loader, optimizer, boundaries, loss_func, args.cuda,
                         use_conservation, departure=False, teacher_forcing=tf)
            print(f'epoch {epoch + 1}: loss = {loss / len(train_data)}')
            if departure:
                loss = train_fluxes(model, train_loader, optimizer, boundaries, loss_func, args.cuda,
                             use_conservation, departure=departure, teacher_forcing=tf)
                print(f'epoch {epoch + 1}: loss with departure = {loss / len(train_data)}')
            training_curve[epoch] = loss / len(train_data)

            model.eval()
            val_loss = test_fluxes(model, val_loader, timesteps, loss_func, args.cuda,
                                   get_outfluxes=False, bird_scale=1).mean()
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



def make_name_repeat(timesteps, model_type, conservation=True, recurrent=True, embedding=0, norm=False, epochs=100, repeat=1):
    name = f'GNN_{model_type}_ts={timesteps}_embedding={embedding}_conservation={conservation}_' \
           f'epochs={epochs}_recurrent={recurrent}_norm={norm}_repeat={repeat}.pt'
    return name

def make_name(timesteps, model_type, conservation=True, recurrent=True, embedding=0, norm=False, epochs=100):
    name = f'GNN_{model_type}_ts={timesteps}_embedding={embedding}_conservation={conservation}_' \
           f'epochs={epochs}_recurrent={recurrent}_norm={norm}.pt'
    return name

def load_model(name):
    model = torch.load(osp.join(model_dir, name))
    model.recurrent = True
    return model

def load_gam_predictions(csv_file, test_loader, nights, time, radars, timesteps, loss_func):
    df_gam = pd.read_csv(csv_file)
    df_gam.datetime = pd.DatetimeIndex(df_gam.datetime) #, tz='UTC')
    dti = pd.DatetimeIndex(time, tz='UTC')

    loss = np.zeros((len(radars), len(nights), timesteps-1))
    pred_gam = np.zeros((len(radars), len(time)))
    for idx, radar in enumerate(radars):
        df_gam_idx = df_gam[df_gam.radar == radar]
        for nidx, data in enumerate(test_loader):
            y_gam = df_gam_idx[df_gam_idx.datetime.isin(dti[nights[nidx]])].gam_prediction.to_numpy()
            pred_gam[idx, nights[nidx]] = y_gam
            loss[idx, nidx, :] = loss_func(torch.tensor(y_gam[1:timesteps]), data.y)
    return loss, pred_gam


def plot_test_errors(timesteps, model_names, short_names, model_types, output_path,
                     data_source='radar', bird_scale=2000, departure=False):

    #output_dir = osp.join(root, 'model_performance', f'experiment_{datetime.now()}')
    #os.makedirs(output_dir, exist_ok=True)

    #name = make_name(timesteps, embedding, model_type, recurrent, conservation, norm, epochs)

    test_data = RadarData(root, test_year, season, timesteps, data_source=data_source, bird_scale=bird_scale)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    radar_index = {idx: name for idx, name in enumerate(test_data.info['radars'])}
    with open(osp.join(osp.dirname(output_path), f'radar_index.pickle'), 'wb') as f:
        pickle.dump(radar_index, f, pickle.HIGHEST_PROTOCOL)

    models = [load_model(name) for name in model_names]
    for i, n in enumerate(short_names):
        print(f'model: {n}, num params: {sum(p.numel() for p in models[i].parameters() if p.requires_grad)}')

    loss_func = torch.nn.MSELoss()

    fig, ax = plt.subplots()
    loss_all = {m : [] for m in model_types}
    for midx, model in enumerate(models):
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.data)

        model.timesteps = timesteps

        if short_names[midx] == 'standard_mlp':
            loss_all[short_names[midx]].append(
                test_fluxes(model, test_loader, timesteps, loss_func, args.cuda,
                     get_outfluxes=False, bird_scale=bird_scale).mean(0)
            )
        else:
            l, outfluxes, outfluxes_abs = test_fluxes(model, test_loader, timesteps, loss_func, args.cuda,
                                bird_scale=bird_scale, departure=departure)
            loss_all[short_names[midx]].append(l.mean(0))
            with open(osp.join(osp.dirname(output_path), f'outfluxes_{short_names[midx]}.pickle'), 'wb') as f:
                pickle.dump(outfluxes, f, pickle.HIGHEST_PROTOCOL)
            with open(osp.join(osp.dirname(output_path), f'outfluxes_abs_{short_names[midx]}.pickle'), 'wb') as f:
                pickle.dump(outfluxes_abs, f, pickle.HIGHEST_PROTOCOL)



    for type in model_types:
        losses = torch.stack(loss_all[type]).sqrt()
        mean_loss = losses.mean(0).detach().numpy()
        std_loss = losses.std(0).detach().numpy()
        line = ax.plot(range(1, timesteps), mean_loss, label=f'{type}')
        ax.fill_between(range(1, timesteps), mean_loss - std_loss, mean_loss + std_loss, alpha=0.2,
                        color=line[0].get_color())

    naive_losses = []
    for data in test_loader:
        naive_losses.append(torch.tensor([loss_func(data.x[:, 0]*bird_scale, data.y[:, t]*bird_scale) for t in range(timesteps-1)]))
    naive_losses = torch.stack(naive_losses, dim=0).mean(0).sqrt()
    ax.plot(range(1, timesteps), naive_losses, label=f'naive "persistence" model')

    gam_losses, _ = load_gam_predictions(gam_csv, test_loader, test_data.info['nights'], test_data.info['timepoints'],
                                       test_data.info['radars'], timesteps, loss_func)
    ax.plot(range(1, timesteps), gam_losses, label=f'GAM')


    ax.set_xlabel('timestep')
    ax.set_ylabel('RMSE')
    #ax.set_ylim(-0.005, 0.035)
    ax.set_xticks(range(1, timesteps))
    ax.legend()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


def plot_predictions(timesteps, model_names, short_names, model_types, output_dir, tidx=None,
                     data_source='radar', repeats=1, bird_scale=2000, departure=False):

    dataset = RadarData(root, test_year, season, timesteps, data_source=data_source, bird_scale=bird_scale)
    nights = dataset.info['nights']
    time = np.array(dataset.info['timepoints'])
    with open(osp.join(output_dir, 'nights.pickle'), 'wb') as f:
        pickle.dump(nights, f)
    df_gam = pd.read_csv(gam_csv)
    df_gam.datetime = pd.DatetimeIndex(df_gam.datetime, tz='UTC')
    dti = pd.DatetimeIndex(time, tz='UTC')

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

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
        df_gam_idx = df_gam[df_gam.radar==radar]

        for nidx, data in enumerate(dataloader):
            gt[nights[nidx][0]] = data.x[idx, 0]
            gt[nights[nidx][1:timesteps]] = data.y[idx]

            if args.cuda: data = data.to('cuda')
            for midx, model in enumerate(models):
                model.timesteps = timesteps
                if args.cuda: model.cuda()
                y = model(data).cpu().detach().numpy()[idx]
                if departure:
                    pred[midx][nights[nidx][:timesteps]] = y
                else:
                    pred[midx][nights[nidx][1:timesteps]] = y[1:]
                    pred[midx][nights[nidx][0]] = data.x[idx, 0]

            pred_gam[nights[nidx]] = df_gam_idx[df_gam_idx.datetime.isin(dti[nights[nidx]])].gam_prediction

        fig, ax = plt.subplots(figsize=(20, 4))
        for midx, model_type in enumerate(model_types):
            all_pred = pred[midx*repeats:(midx+1)*repeats, tidx] * bird_scale
            mean_pred = all_pred.mean(0)
            std_pred = all_pred.std(0)

            # line = ax.plot(time[tidx], pred[midx][tidx], ls='--', alpha=0.3)
            line = ax.errorbar(time[tidx], mean_pred, std_pred, ls='--', alpha=0.4, capsize=3)
            ax.scatter(time[tidx], mean_pred, s=30, facecolors='none', edgecolor=line[0].get_color(),
                       label=f'prediction ({model_type})', alpha=0.4)

        line = ax.plot(time[tidx], pred_gam[tidx], ls='--', alpha=0.4)
        ax.scatter(time[tidx], pred_gam[tidx], s=30, facecolors='none', edgecolor=line[0].get_color(),
                   label=f'prediction (GAM)', alpha=0.4)

        ax.plot(time[tidx], gt[tidx] * bird_scale, label='ground truth', c='gray', alpha=0.8)

        ax.set_title(radar)
        #ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel('bird density')
        fig.legend(loc='upper right', bbox_to_anchor=(0.77, 0.85))
        fig.savefig(os.path.join(output_dir, f'{radar}.png'), bbox_inches='tight')
        plt.close(fig)


def predictions(timesteps, model_names, model_types, output_dir,
                     data_source='radar', repeats=1, bird_scale=2000, departure=False):

    dataset = RadarData(root, test_year, season, timesteps, data_source=data_source, bird_scale=bird_scale)
    nights = dataset.info['nights']
    time = dataset.info['timepoints']
    with open(osp.join(output_dir, 'nights.pickle'), 'wb') as f:
        pickle.dump(nights, f)

    dataloader = DataLoader(dataset, batch_size=1)
    models = [load_model(name) for name in model_names]
    dfs = []
    for idx, radar in enumerate(dataset.info['radars']):
        gt = np.zeros(len(time))
        pred = []
        for _ in models:
            pred.append(np.zeros(len(time)))
        pred = np.stack(pred, axis=0)

        for nidx, data in enumerate(dataloader):
            gt[nights[nidx][0]] = data.x[idx, 0]
            gt[nights[nidx][1:timesteps]] = data.y[idx]

            if args.cuda: data = data.to('cuda')
            for midx, model in enumerate(models):
                model.timesteps = timesteps
                if args.cuda: model.cuda()
                y = model(data).cpu().detach().numpy()[idx]
                if departure:
                    pred[midx][nights[nidx][:timesteps]] = y
                else:
                    pred[midx][nights[nidx][1:timesteps]] = y[1:]
                    pred[midx][nights[nidx][0]] = data.x[idx, 0]

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




epochs = args.epochs #2 #10 #500
norm = False
repeats = args.repeats #1
departure = False #True #True #True

bird_scale = 2000

if args.action =='train':

    model_types = ['linear+sigmoid', 'mlp'] #, 'standard_mlp']
    cons_settings = [True, False]
    rec_settings = [True] #, False]
    emb_settings = [0]

    all_settings = it.product(model_types, cons_settings, rec_settings, emb_settings)

    for type, cons, rec, emb in all_settings:
        run_training(args.ts_train, type, cons, rec, emb, epochs=epochs, data_source=args.data_source, repeats=repeats,
                     bird_scale=bird_scale, departure=departure)

if args.action == 'test':

    model_types = ['linear+sigmoid', 'mlp']#, 'mlp']#'linear+sigmoid', 'mlp']#, 'standard_mlp']
    model_labels = ['G_linear+sigmoid', 'G_mlp']#, 'G_mlp'] #'G_linear+sigmoid', 'G_mlp']#, 'standard_mlp']
    cons = True
    rec = True
    emb = 0

    if repeats > 1:
        #all_settings = it.product(model_types, cons_settings, rec_settings, emb_settings)
        model_names = [make_name_repeat(args.ts_train, type, cons, rec, emb, epochs=epochs, repeat=r)
                       for type, r in it.product(model_types, range(repeats))]
    else:
        model_names = [make_name(args.ts_train, type, cons, rec, emb, epochs=epochs)
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
        plot_predictions(args.ts_test, model_names, short_names, model_labels, osp.dirname(output_path),
                     data_source=args.data_source, repeats=repeats, tidx=range(4*24, 11*24), departure=departure) #, tidx=range(18*24, 32*24))
