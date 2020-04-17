import numpy as np
import os
from glob import glob
from PIL import Image
import random
import torch
from torch.utils import data
import parse

#DTSTR = parse.compile('{YYYYMMDD}T{HHMM}')
DNAME = parse.compile('{start}_to_{end}')


class RadarImages(data.Dataset):
    def __init__(self, split, data_path, n_frames_in, n_frames_out,
                 transform=None, n_channels = 1, clip = 2000):

        super(RadarImages, self).__init__()

        self.split           = split
        self.n_frames_in     = n_frames_in
        self.n_frames_out    = n_frames_out
        self.seq_len         = self.n_frames_in + self.n_frames_out
        self.n_channels      = n_channels
        self.transform       = transform

        self.data_dirs = sorted([(os.path.basename(d), d) \
                            for d in glob(os.path.join(data_path, '*'))], \
                            key = lambda x: DNAME.parse(x[0]).named['start'])

    def __len__(self):
        return len(self.data_dirs)

    def __getitem__(self, idx):
        dir    = self.data_dirs[idx][1]
        frames = sorted([(os.path.basename(f), f) \
                            for f in glob(os.path.join(dir, '*.npy'))], \
                            key = lambda x: x[0])

        # load subsequence of seq_len frames
        start  = np.random.randint(0, len(frames) - self.seq_len)
        frames = np.stack([np.load(f[1]) for f \
                            in frames[start : start + self.seq_len]])

        frames[frames > self.clip] = clip

        #if self.n_channels == 3:
            # convert into RGB image
        #   frames = ...
        #   frames /= 255.
        #else:
        frames /= np.nanmax(frames)

        frames = np.array(frames).astype(np.float32)

        # transform
        if self.transform is not None:
            frames = self.transform(frames)

        if self.n_channels == 3:
            # pytorch expects dimensions [seq_len, n_channels, height, width]
            frames = frames.transpose((0, 3, 1, 2))

        return (frames, item_idx)

if __name__ == '__main__':
    test_loader = RadarImages('test', 'preprocessing/data/numpy', 10, 10)
    for batch_idx, train_batch in enumerate(test_loader):
        print(batch_idx)
