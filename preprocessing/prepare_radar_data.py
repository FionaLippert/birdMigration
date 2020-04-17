import numpy as np
import os
from glob import glob
import parse
import wradlib as wrl
import argparse


FNAME = parse.compile('{info}_{datetime}.h5')
H5_DATA_KEY = 'VID_data/data'


parser = argparse.ArgumentParser()
parser.add_argument('--in_dir', required=True, type=str,
                    help='input directory containing data as .h5 files')
parser.add_argument('--out_dir', default='./output', type=str,
                    help='output directory for processed .npy files')
parser.add_argument('--seq_len', default=50, type=str,
                    help='length of individual sequences')
args = parser.parse_args()


def prepare_data(input_path, output_path, seq_len):

    files = sorted([(FNAME.parse(os.path.basename(d)).named('datetime'), d) \
                        for d in glob(os.path.join(input_path, '*.h5'))], \
                        key = lambda x: x[0])

    for i in range(0, len(files), seq_len):

        start  = files[i][0]
        end    = files[i+seq_len-1][0]

        subdir = os.path.join(output_path, f'{start}_to_{end}')
        os.makedirs(subdir, exist_ok = True)

        for j in range(seq_len):
            datetime, fpath = files[i+j]
            h5_to_numpy(fpath, os.path.join(subdir, datetime))


def h5_to_numpy(input_path, output_path=None):
    f = wrl.util.get_wradlib_data_file(input_path)
    content = wrl.io.read_opera_hdf5(f)
    frame = content[H5_DATA_KEY]

    if output_path is not None:
        np.save(output_path, frame)

    return frame


if __name__ == '__main__':
    prepare_data(args.in_dir, args.out_dir, args.seq_len)
