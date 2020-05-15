import numpy as np
import os
from glob import glob
import parse
import yaml
import wradlib as wrl
import argparse
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split


FNAME = parse.compile('{info}_{datetime}.nc')
DATETIME_STR = '%Y%m%dT%H%M'
DELTA_T = timedelta(minutes=15)
H5_DATA_KEY = 'VID_data/data'


parser = argparse.ArgumentParser()
parser.add_argument('--in_dir', required=True, type=str,
                    help='input directory containing data as .nc files')
parser.add_argument('--out_dir', default='./output', type=str,
                    help='output directory for processed .nc files')
parser.add_argument('--seq_len', default=50, type=int,
                    help='length of individual sequences')
parser.add_argument('--test_size', default=0.2, type=float,
                    help='percentage of sequences to use for validation')
args = parser.parse_args()


def prepare_data(input_path, output_path, seq_len, test_size, n_subdirs=0):

    #print(glob(os.path.join(input_path, '*', '*.h5')))
    files = sorted([(FNAME.parse(os.path.basename(d)).named['datetime'], d) \
                        for d in glob(os.path.join(input_path, f'{n_subdirs*"*/"}*.nc'))], \
                        key = lambda x: x[0])

    print(f'Found {len(files)} files to pe processed')

    idx_list = range(0, len(files), seq_len)
    if len(files)%seq_len > 0:
        # last index has too little data
        idx_list = idx_list[:-1]

    if test_size == 0:
        idx_train = idx_list
    elif test_size == 1:
        idx_train == []
    else:
        idx_train, idx_test = train_test_split(idx_list, test_size=test_size)

    for k, idx in enumerate(idx_list):

        end = min(len(files), idx+seq_len)-1

        check_delta = [(datetime.strptime(files[j+1][0], DATETIME_STR) \
                        - datetime.strptime(files[j][0], DATETIME_STR) \
                        == DELTA_T) for j in range(idx, end)]
        if np.all(check_delta):
            if idx in idx_train:
                dataset = 'train'
            else:
                dataset = 'test'

            output_file = os.path.join(output_path, dataset, \
                            f'{files[idx][0]}_to_{files[end][0]}.nc')
            #os.makedirs(subdir, exist_ok = True)
            input_files = [f[1] for f in files[idx:end+1]]
            combine_nc(input_files, output_file)

            #all_bounds = [h5_to_numpy(f[1], os.path.join(subdir, f[0]))[1] \
            #                    for f in files[idx:end+1]]
        else:
            print(f'discarding sequence {k} due to missing data')

def combine_nc(input_files, output_file):
    new_nc = xr.open_mfdataset(input_files, combine='by_coords')
    new_nc.to_netcdf('output_file', mode='w', format='NETCDF3_64BIT')

def h5_to_numpy(input_path, output_path=None):
    f = wrl.util.get_wradlib_data_file(os.path.abspath(input_path))
    content = wrl.io.read_opera_hdf5(f)
    frame = content[H5_DATA_KEY]

    bounds = np.array([content['how']['lon_min'],
                       content['how']['lat_min'],
                       content['how']['lon_max'],
                       content['how']['lat_max']]).flatten()
    radars   = content['what']['source'].astype(str)
    quantity = content['what']['quantity'].astype(str)[0]
    proj     = content['where']['projdef'].astype(str)[0]

    meta = {'bounds': bounds.tolist(),
            'radars': radars.tolist(),
            'quantity': str(quantity),
            'proj4str': str(proj)}

    if output_path is not None:
        np.save(output_path, frame)
        with open(f'{output_path}_meta.yml', 'w+') as f:
            yaml.dump(meta, f)

    return frame, bounds


if __name__ == '__main__':
    prepare_data(args.in_dir, args.out_dir, args.seq_len, args.test_size)
