import numpy as np
import os
from glob import glob
from PIL import Image
import random
import torch
from torch.utils import data
import parse
import cv2
import xarray as xr

FNAME = parse.compile('{start}_to_{end}.nc')

# use to start new training with weights learned before:
# model.load_state_dict(torch.load(‘file_with_model’))


class RadarImages(data.Dataset):
    def __init__(self, split, data_path, n_frames,
                 transform=None, n_channels=1, clip=2000, img_size=64):

        super(RadarImages, self).__init__()

        self.split           = split
        self.n_frames        = n_frames
        self.n_channels      = n_channels
        self.transform       = transform
        self.clip            = clip
        self.img_size        = img_size

        # self.data_dirs = sorted([(os.path.basename(d), d) \
        #                 for d in glob(os.path.join(data_path, split, '*'))], \
        #                 key = lambda x: DNAME.parse(x[0]).named['start'])

        self.files = sorted([(os.path.basename(d), d) \
                        for d in glob(os.path.join(data_path, split, '*'))], \
                        key = lambda x: FNAME.parse(x[0]).named['start'])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx][1]
        #frames = sorted([(os.path.basename(f), f) \
        #                    for f in glob(os.path.join(dir, '*.npy'))], \
        #                    key = lambda x: x[0])

        ds = xr.open_dataset(file)


        # load subsequence of seq_len frames
        if self.split == 'train' and ds.time.size > self.n_frames:
            start = np.random.randint(0, ds.time.size - self.n_frames)
        else:
            start = 0
        frames = np.stack([self._load_frame(f[1]) for f \
                            in frames[start : start + self.n_frames]])


        frames = np.nan_to_num(frames) # set nan's to zero
        frames[frames <= 0.01] = 0.01 # to prevent log(0)
        frames[frames > self.clip] = self.clip

        frames = np.log(frames)
        frames /= np.log(self.clip)

        frames = np.array(frames).astype(np.float32)

        # transform
        if self.transform is not None:
            frames = self.transform(frames)

        if self.n_channels == 1:
            frames = np.expand_dims(frames, axis=-1)

        # pytorch expects dimensions [seq_len, n_channels, height, width]
        frames = frames.transpose((0, 3, 1, 2))

        return (frames, idx)

    def _load_frames(self, ds, xmin=300, xmax=700, ymin=300, ymax=700):
        frames = np.array(ds.VID)
        frames = np.stack([cv2.resize(frames[t, xmin:xmax, ymin:ymax], \
                            (self.img_size, self.img_size)) for t \
                            in range(start : start + self.n_frames)])

        return frames

    def _load_frame(self, path):
        frame = np.load(path)
        frame = frame[300:700, 300:700] # TODO: adjust grid extent in R preprocessing
        frame = cv2.resize(frame, (self.img_size, self.img_size))

        return frame

if __name__ == '__main__':
    test_loader = RadarImages('test', 'preprocessing/data/numpy', 20)
    for batch_idx, train_batch in enumerate(test_loader):
        print(batch_idx)
