from birds.graphNN import *
import argparse
from matplotlib import pyplot as plt
import itertools as it
import os
import os.path as osp
import pickle5 as pickle
import torch
from torch_geometric.data import DataLoader


parser = argparse.ArgumentParser(description='departure model experiments')
parser.add_argument('action', type=str, help='train or test')
parser.add_argument('--root', type=str, default='/home/fiona/birdMigration', help='entry point to required data')
parser.add_argument('--experiment', type=str, default='test', help='directory name for model performance output')
parser.add_argument('--data_source', type=str, default='radar', help='data source for training/testing')
parser.add_argument('--cpu', action='store_true', default=False, help='cpu or gpu')
parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--repeats', type=int, default=5, help='number of models to be trained with different random seeds')
args = parser.parse_args()

args.cuda = (not args.cpu and torch.cuda.is_available())

root = osp.join(args.root, 'data')
model_dir = osp.join(args.root, 'departure_models', args.experiment)
os.makedirs(model_dir, exist_ok=True)

season = 'fall'
train_years = ['2016', '2017', '2018', '2019']
test_year = '2015'
bird_scale = 2000
loss_func = torch.nn.MSELoss()

def run_training(model_type, hidden_channels, output_dir=model_dir):
    train_data = [RadarData(root, year, season, data_source=args.data_source,
                            bird_scale=bird_scale, timesteps=2) for year in train_years]
    train_data = torch.utils.data.ConcatDataset(train_data)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)

    for r in range(args.repeats):

        model = Departure(7, hidden_channels, 1, model=model_type, seed=r)

        if args.repeats == 1:
            name = make_name(model_type, args.epochs, hidden_channels)
        else:
            name = make_name_repeat(model_type, args.epochs, hidden_channels, repeat=r)

        params = model.parameters()
        optimizer = torch.optim.Adam(params, lr=args.lr)

        training_curve = np.zeros(args.epochs)
        best_loss = np.inf
        for epoch in range(args.epochs):
            loss = train_departure(model, train_loader, optimizer, loss_func, args.cuda)
            print(f'epoch {epoch + 1}: loss = {loss / len(train_data)}')

            training_curve[epoch] = loss
            if loss < best_loss:
                # save best model so far
                torch.save(model, osp.join(output_dir, name))

        fig, ax = plt.subplots()
        ax.plot(range(1, epochs+1), training_curve)
        ax.set(xlabel='epoch', ylabel='Loss', title=f'best model in epoch {np.argmin(training_curve)+1}')
        fig.savefig(osp.join(output_dir, f'training_loss_{name}.png'), bbox_inches='tight')

def make_name_repeat(model_type, epochs, hidden_channels, repeat):
    name = f'GNN_{model_type}_epochs={epochs}_hiddendim={hidden_channels}_repeat={repeat}.pt'
    return name

def make_name(model_type, epochs, hidden_channels):
    name = f'GNN_{model_type}_epochs={epochs}_hiddendim={hidden_channels}.pt'
    return name

def load_model(name):
    model = torch.load(osp.join(model_dir, name))
    model.recurrent = True
    return model

def plot_predictions(model, test_loader, output_path):
    if args.cuda:
        model.cuda()
    model.eval()
    loss_all = []
    predictions = []
    gt = []
    for tidx, data in enumerate(test_loader):
        if args.cuda: data = data.to('cuda')
        y_hat = model(data) * bird_scale
        y = data.x[..., 0] * bird_scale

        loss_all.append(loss_func(y_hat, y))
        predictions.append(y_hat.detach())
        gt.append(y.detach())

    rmse = torch.stack(loss_all).mean().sqrt()
    predictions = torch.cat(predictions)
    gt = torch.cat(gt)

    fig, ax = plt.subplots()
    ax.scatter(gt, predictions)
    ax.set(xlabel='ground truth', ylabel='prediction', title=f'RMSE={rmse}')
    fig.savefig(output_path, bbox_inches='tight')



if args.action == 'train':

    model_types = ['linear', 'mlp']
    hidden_channels = range(2, 8)

    all_settings = it.product(model_types, hidden_channels)

    for type, hdim in all_settings:
        run_training(type, hdim)


if args.action == 'test':

    if args.repeats > 1:
        model_names = [make_name_repeat(type, args.epochs, repeat=r)
                       for type, r in it.product(model_types, range(args.repeats))]
    else:
        model_names = [make_name(type, args.epochs) for type in model_types]

    short_names = [type for type, r in it.product(model_types, range(args.repeats))]

    output_dir = osp.join(root, 'model_performance', 'departure_models', args.experiment)
    os.makedirs(output_dir, exist_ok=True)

    test_data = RadarData(root, test_year, season, timesteps=2, data_source=args.data_source, bird_scale=bird_scale)
    test_loader = DataLoader(test_data, batch_size=1)

    for idx, name in enumerate(model_names):
        model = load_model(name)
        output_path = osp.join(output_dir, f'predictions_{short_names[idx]}hiddendim={hdim}.png')
        plot_predictions(model, test_loader, output_path)