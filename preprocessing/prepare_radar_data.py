import numpy as np
import os
from glob import glob
import parse
import yaml
import wradlib as wrl
import argparse
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split


FNAME = parse.compile('{info}_{datetime}.h5')
DATETIME_STR = '%Y%m%dT%H%M'
DELTA_T = timedelta(minutes=15)
H5_DATA_KEY = 'VID_data/data'


parser = argparse.ArgumentParser()
parser.add_argument('--in_dir', required=True, type=str,
                    help='input directory containing data as .h5 files')
parser.add_argument('--out_dir', default='./output', type=str,
                    help='output directory for processed .npy files')
parser.add_argument('--seq_len', default=50, type=int,
                    help='length of individual sequences')
parser.add_argument('--test_size', default=0.2, type=float,
                    help='percentage of sequences to use for validation')
args = parser.parse_args()


def prepare_data(input_path, output_path, seq_len, test_size):

    #print(glob(os.path.join(input_path, '*', '*.h5')))
    files = sorted([(FNAME.parse(os.path.basename(d)).named['datetime'], d) \
                        for d in glob(os.path.join(input_path, '*', '*.h5'))], \
                        key = lambda x: x[0])

    print(f'Found {len(files)} files to pe processed')

    # assert that data is available for all time steps
    if not np.all([(datetime.strptime(files[i+1][0], DATETIME_STR) \
                    - datetime.strptime(files[i][0], DATETIME_STR) \
                    == DELTA_T) for i in range(len(files)-1)]):
        for i in range(len(files)-1):
            if (datetime.strptime(files[i+1][0], DATETIME_STR) \
                            - datetime.strptime(files[i][0], DATETIME_STR)) != DELTA_T:
                print(files[i][0]m files[i+1][0])
        assert 0

    idx_list = range(0, len(files), seq_len)
    if len(files)%seq_len > 0:
        # last index has too little data
        idx_list = idx_list[:-1]
    idx_train, idx_test = train_test_split(idx_list, test_size=test_size)

    for i in idx_list:

        end    = min(len(files), i+seq_len)-1
        if i in idx_train:
            dataset = 'train'
        else:
            dataset = 'test'
        subdir = os.path.join(output_path, dataset, \
                        f'{files[i][0]}_to_{files[end][0]}')
        os.makedirs(subdir, exist_ok = True)

        all_bounds = [h5_to_numpy(f[1], os.path.join(subdir, f[0]))[1] \
                            for f in files[i:end+1]]


def h5_to_numpy(input_path, output_path=None):
    f = wrl.util.get_wradlib_data_file(os.path.abspath(input_path))
    content = wrl.io.read_opera_hdf5(f)
    frame = content[H5_DATA_KEY]

    bounds = np.array([content['how']['lon_min'],
                       content['how']['lat_min'],
                       content['how']['lon_max'],
                       content['how']['lat_max']]).reshape(2,2)
    radars   = content['what']['source'].astype(str)
    quantity = content['what']['quantity'].astype(str)[0]
    proj     = content['where']['projdef'].astype(str)[0]

    meta = {'bounds': bounds,
            'radars': radars,
            'quantity': quantity,
            'proj4str': projection}

    if output_path is not None:
        np.save(output_path, frame)
        with open(os.path.join(output_path, 'meta.yml'), 'w+') as f:
            yaml.dump(meta, f)

    return frame, bounds


if __name__ == '__main__':
    prepare_data(args.in_dir, args.out_dir, args.seq_len, args.test_size)
