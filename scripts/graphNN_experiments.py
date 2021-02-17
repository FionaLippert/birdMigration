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
test_years = ['2015']


def run_training(timesteps, model_type, conservation=True, recurrent=True, embedding=0, norm=False, epochs=100,
                 data_source='radar', output_dir=model_dir):
    train_data = RadarData(root, 'train', train_years, season, timesteps, data_source=data_source)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)

    model = BirdFlowTime(train_data[0].num_nodes, timesteps, embedding, model_type, recurrent, norm)
    params = model.parameters()

    optimizer = torch.optim.Adam(params, lr=0.01)
    loss_func = torch.nn.MSELoss()

    boundaries = train_data.info['all_boundaries'][0]
    for epoch in range(epochs):
        loss = train(model, train_loader, optimizer, boundaries, loss_func, 'cpu', conservation)
        print(f'epoch {epoch + 1}: loss = {loss / len(train_data)}')

    name = make_name(timesteps, model_type, conservation, recurrent, embedding, norm, epochs)
    torch.save(model, osp.join(output_dir, name))



def make_name(timesteps, model_type, conservation=True, recurrent=True, embedding=0, norm=False, epochs=100):
    name = f'GNN_{model_type}_ts={timesteps}_embedding={embedding}_conservation={conservation}_' \
           f'epochs={epochs}_recurrent={recurrent}_norm={norm}.pt'
    return name

def load_model(name):
    model = torch.load(osp.join(model_dir, name))
    model.recurrent = True
    return model

def plot_test_errors(timesteps, model_names, short_names, output_path, data_source='radar'):

    #output_dir = osp.join(root, 'model_performance', f'experiment_{datetime.now()}')
    #os.makedirs(output_dir, exist_ok=True)

    #name = make_name(timesteps, embedding, model_type, recurrent, conservation, norm, epochs)

    test_data = RadarData(root, 'test', test_years, season, timesteps, data_source=data_source)
    test_loader = DataLoader(test_data, batch_size=1)

    models = [load_model(name) for name in model_names]
    loss_func = torch.nn.MSELoss()

    fig, ax = plt.subplots()
    for midx, model in enumerate(models):
        model.timesteps = timesteps
        loss_all = test(model, test_loader, timesteps, loss_func, 'cpu')
        mean_loss = loss_all.mean(0)
        std_loss = loss_all.std(0)
        line = ax.plot(range(1, timesteps), mean_loss, label=f'{short_names[midx]}')
        ax.fill_between(range(1, timesteps), mean_loss - std_loss, mean_loss + std_loss, alpha=0.2,
                        color=line[0].get_color())
    ax.set_xlabel('timestep')
    ax.set_ylabel('MSE')
    ax.set_ylim(-0.005, 0.055)
    ax.set_xticks(range(1, timesteps))
    ax.legend()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


def plot_predictions(timesteps, model_names, output_dir, split='test', data_source='radar'):

    dataset = RadarData(root, split, test_years, season, timesteps, data_source=data_source)
    dataloader = DataLoader(dataset, batch_size=1)

    models = [load_model(name) for name in model_names]

    time = dataset.info['timepoints']
    nights = dataset.info['nights'][0]


    for idx, radar in enumerate(dataset.info['radars']):
        gt = np.zeros(len(time))
        pred = []
        for _ in models:
            pred.append(np.ones(len(time)) * np.nan)

        for nidx, data in enumerate(dataloader):
            gt[nights[nidx][1]] = data.x[idx, 0]
            gt[nights[nidx][2:timesteps + 1]] = data.y[idx]
            for midx, model in enumerate(models):
                y = model(data).detach().numpy()[idx]
                pred[midx][nights[nidx][2:timesteps + 1]] = y
                pred[midx][nights[nidx][1]] = data.x[idx, 0]

        fig, ax = plt.subplots(figsize=(20, 4))
        ax.plot(time, gt, label='ground truth', c='gray', alpha=0.5)
        for midx, model_type in enumerate(model_names):
            line = ax.plot(time, pred[midx], ls='--', alpha=0.3)
            ax.scatter(time, pred[midx], s=30, facecolors='none', edgecolor=line[0].get_color(),
                       label=f'prediction ({model_type})')

            # outfluxes = to_dense_adj(data.edge_index, edge_attr=torch.stack(models[midx].flows, dim=-1)).view(
            #     data.num_nodes,
            #     data.num_nodes,
            #     -1)
            # print(idx, radar, model_type)
            # for jdx, radar_j in enumerate(dataset.info['radars']):
            #     print(radar, radar_j, outfluxes[idx][jdx])

        ax.set_title(radar)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel('normalized bird density')
        fig.legend(loc='upper right', bbox_to_anchor=(0.77, 0.85))
        fig.savefig(os.path.join(output_dir, f'{radar.split("/")[0]}_{radar.split("/")[1]}.png'), bbox_inches='tight')
        plt.close(fig)



timesteps = 6
epochs = 10 #500
norm = False

if args.action =='train':

    model_types = ['linear', 'linear+sigmoid', 'mlp']
    cons_settings = [True, False]
    rec_settings = [True, False]
    emb_settings = [0]

    all_settings = it.product(model_types, cons_settings, rec_settings, emb_settings)

    for type, cons, rec, emb in all_settings:
        print(type, cons, rec, emb)
        run_training(timesteps, type, cons, rec, emb, epochs=epochs, data_source=args.data_source)

if args.action == 'test':

    model_types = ['linear', 'linear+sigmoid', 'mlp']
    cons = True
    rec = True
    emb = 0

    #all_settings = it.product(model_types, cons_settings, rec_settings, emb_settings)
    model_names = [make_name(timesteps, type, cons, rec, emb, epochs=epochs) for type in model_types]
    short_names = model_types

    timesteps = 12
    output_path = osp.join(root, 'model_performance', args.experiment,
                           f'conservation={cons}_recurrent={rec}_embedding={emb}_timesteps={timesteps}.png')
    os.makedirs(osp.dirname(output_path), exist_ok=True)

    plot_test_errors(timesteps, model_names, short_names, output_path, data_source=args.data_source)
