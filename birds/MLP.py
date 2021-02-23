import torch
from torch.utils.data import Dataset

from birds import graphNN


class RadarDataWrapper(Dataset):

    def __init__(self, root, split, year, season='fall', timesteps=1, data_source='radar'):
        super(RadarDataWrapper, self).__init__()

        dataset = graphNN.RadarData(root, split, year, season, timesteps, data_source)

        data_list = []
        for data in dataset:
            # construct array of shape (radars, features, timesteps)
            # with first features being bird density
            birds = torch.cat([data.x, data.y[..., -1].view(-1, 1)], dim=1).view(1, -1, timesteps)
            coords = torch.stack([data.coords] * timesteps, dim=-1).permute(1, 0, 2)
            print(coords.shape)
            data = torch.cat([birds, coords, data.env.permute(1, 0, 2)], dim=0)

            # flatten to 2D array
            #data = data.view(-1, timesteps)
            data_list.append(data)

        self.data = torch.stack(data_list, dim=0)   # shape (nights, features, radars, timesteps)
        self.n_nights, self.n_features, self.n_radars, self.timesteps = self.data.shape

        #self.data = self.data.reshape(self.n_nights, self.n_features*self.n_radars, self.timesteps)

    def __len__(self):
        return self.n_nights

    def __getitem__(self, item):
        return self.data[item]


if __name__ == '__main__':
    test = RadarDataWrapper('/home/fiona/birdMigration/data', 'train', '2015', 'fall', 6)
