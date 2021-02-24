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
import torch
from torch_geometric.data import DataLoader

parser = argparse.ArgumentParser(description='GraphNN experiments')
parser.add_argument('action', type=str, help='train or test')
parser.add_argument('--root', type=str, default='/home/fiona/birdMigration', help='entry point to required data')
parser.add_argument('--experiment', type=str, default='test', help='directory name for model performance output')
parser.add_argument('--data_source', type=str, default='radar', help='data source for training/testing')
args = parser.parse_args()

root = osp.join(args.root, 'data')
model_dir = osp.join(args.root, 'models', args.experiment)
os.makedirs(model_dir, exist_ok=True)

season = 'fall'
train_years = ['2015']
test_year = '2015'


def run_training(timesteps, model_type, conservation=True, recurrent=True, embedding=0, norm=False, epochs=100,
                 repeats=1, data_source='radar', output_dir=model_dir):

    train_data = [RadarData(root, year, season, timesteps, data_source=data_source) for year in train_years]
    boundaries = train_data[0].info['boundaries']
    train_data = torch.utils.data.ConcatDataset(train_data)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)

    for r in range(repeats):
        if model_type == 'standard_mlp':
            model = MLP(6*22, 2*22, 22, timesteps, recurrent)
            use_conservation = False
        else:
            model = BirdFlowTime(train_data[0].num_nodes, timesteps, embedding, model_type, recurrent, norm)
            use_conservation = conservation

        params = model.parameters()

        optimizer = torch.optim.Adam(params, lr=0.01)
        loss_func = torch.nn.MSELoss()

        for epoch in range(epochs):
            loss = train(model, train_loader, optimizer, boundaries, loss_func, 'cpu', use_conservation)
            print(f'epoch {epoch + 1}: loss = {loss / len(train_data)}')

        if repeats == 1:
            name = make_name(timesteps, model_type, conservation, recurrent, embedding, norm, epochs)
        else:
            name = make_name_repeat(timesteps, model_type, conservation, recurrent, embedding, norm, epochs, repeat=r)
        torch.save(model, osp.join(output_dir, name))



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

def plot_test_errors(timesteps, model_names, short_names, model_types, output_path, data_source='radar'):

    #output_dir = osp.join(root, 'model_performance', f'experiment_{datetime.now()}')
    #os.makedirs(output_dir, exist_ok=True)

    #name = make_name(timesteps, embedding, model_type, recurrent, conservation, norm, epochs)

    test_data = RadarData(root, test_year, season, timesteps, data_source=data_source)
    test_loader = DataLoader(test_data, batch_size=1)

    models = [load_model(name) for name in model_names]
    for i, n in enumerate(short_names):
        print(f'model: {n}, num params: {sum(p.numel() for p in models[i].parameters() if p.requires_grad)}')

    loss_func = torch.nn.MSELoss()

    fig, ax = plt.subplots()
    loss_all = {m : [] for m in model_types}
    for midx, model in enumerate(models):
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data)

        model.timesteps = timesteps

        if short_names[midx] == 'standard_mlp':
            loss_all[short_names[midx]].append(
                test(model, test_loader, timesteps, loss_func, 'cpu', get_outfluxes=False)
            )
        else:
            l, outfluxes = test(model, test_loader, timesteps, loss_func, 'cpu')
            loss_all[short_names[midx]].append(l)
            with open(osp.join(osp.dirname(output_path), f'outfluxes_{short_names[midx]}.pickle'), 'wb') as f:
                pickle.dump(outfluxes, f, pickle.HIGHEST_PROTOCOL)


    for type in model_types:
        losses = torch.cat(loss_all[type])
        mean_loss = losses.mean(0)
        std_loss = losses.std(0)
        line = ax.plot(range(1, timesteps), mean_loss, label=f'{type}')
        ax.fill_between(range(1, timesteps), mean_loss - std_loss, mean_loss + std_loss, alpha=0.2,
                        color=line[0].get_color())


    ax.set_xlabel('timestep')
    ax.set_ylabel('MSE')
    ax.set_ylim(-0.005, 0.055)
    ax.set_xticks(range(1, timesteps))
    ax.legend()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


def plot_predictions(timesteps, model_names, short_names, model_types, output_dir, tidx=None, data_source='radar', repeats=1):

    dataset = RadarData(root, test_year, season, timesteps, data_source=data_source)
    nights = dataset.info['nights']
    with open(osp.join(output_dir, 'nights.pickle'), 'wb') as f:
        pickle.dump(nights, f)

    dataloader = DataLoader(dataset, batch_size=1)

    models = [load_model(name) for name in model_names]
    print(len(models))

    time = np.array(dataset.info['timepoints'])
    nights = dataset.info['nights']

    if tidx is None:
        tidx = range(time.size)


    for idx, radar in enumerate(dataset.info['radars']):
        gt = np.zeros(len(time))
        pred = []
        for _ in models:
            pred.append(np.ones(len(time)) * np.nan)
        pred = np.stack(pred, axis=0)
        print(pred.shape)

        for nidx, data in enumerate(dataloader):
            gt[nights[nidx][1]] = data.x[idx, 0]
            gt[nights[nidx][2:timesteps + 1]] = data.y[idx]
            for midx, model in enumerate(models):
                y = model(data).detach().numpy()[idx]
                pred[midx][nights[nidx][2:timesteps + 1]] = y
                pred[midx][nights[nidx][1]] = data.x[idx, 0]

        fig, ax = plt.subplots(figsize=(20, 4))
        for midx, model_type in enumerate(model_types):
            print(midx*repeats)
            all_pred = pred[midx*repeats:(midx+1)*repeats, tidx]
            mean_pred = all_pred.mean(0)
            std_pred = all_pred.std(0)
            # line = ax.plot(time[tidx], pred[midx][tidx], ls='--', alpha=0.3)
            line = ax.errorbar(time[tidx], mean_pred, std_pred, ls='--', alpha=0.4, capsize=3)
            ax.scatter(time[tidx], mean_pred, s=30, facecolors='none', edgecolor=line[0].get_color(),
                       label=f'prediction ({model_type})', alpha=0.4)

        ax.plot(time[tidx], gt[tidx], label='ground truth', c='gray', alpha=0.8)

        ax.set_title(radar)
        #ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel('normalized bird density')
        fig.legend(loc='upper right', bbox_to_anchor=(0.77, 0.85))
        fig.savefig(os.path.join(output_dir, f'{radar.split("/")[0]}_{radar.split("/")[1]}.png'), bbox_inches='tight')
        plt.close(fig)



timesteps = 6
epochs = 100 #10 #500
norm = False
repeats = 5

if args.action =='train':

    model_types = ['linear', 'linear+sigmoid', 'mlp', 'standard_mlp']
    cons_settings = [True, False]
    rec_settings = [True, False]
    emb_settings = [0]

    all_settings = it.product(model_types, cons_settings, rec_settings, emb_settings)

    for type, cons, rec, emb in all_settings:
        run_training(timesteps, type, cons, rec, emb, epochs=epochs, data_source=args.data_source, repeats=repeats)

if args.action == 'test':

    model_types = ['linear', 'linear+sigmoid', 'mlp', 'standard_mlp']
    model_labels = ['G_linear', 'G_linear+sigmoid', 'G_mlp', 'standard_mlp']
    cons = True
    rec = True
    emb = 0

    if repeats > 1:
        #all_settings = it.product(model_types, cons_settings, rec_settings, emb_settings)
        model_names = [make_name_repeat(timesteps, type, cons, rec, emb, epochs=epochs, repeat=r)
                       for type, r in it.product(model_types, range(repeats))]
    else:
        model_names = [make_name(timesteps, type, cons, rec, emb, epochs=epochs)
                       for type in model_types]
    #short_names = model_types
    short_names = [type for type, r in it.product(model_labels, range(repeats))]

    timesteps = 6
    output_path = osp.join(root, 'model_performance', args.experiment,
                           f'conservation={cons}_recurrent={rec}_embedding={emb}_timesteps={timesteps}',
                            'test_errors.png')
    os.makedirs(osp.dirname(output_path), exist_ok=True)


    #plot_test_errors(timesteps, model_names, short_names, model_labels, output_path, data_source=args.data_source)
    #
    plot_predictions(timesteps, model_names, short_names, model_labels, osp.dirname(output_path),
                     data_source=args.data_source, repeats=repeats, tidx=range(18*24, 32*24))
