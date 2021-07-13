import os
import os.path as osp
import pandas as pd
import ruamel.yaml
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('hp_tuning_dir', type=str, help='directory with sub-directories that contain the output '
                                                    'of runs with different hyperparameter settings')
parser.add_argument('output_file', type=str, help='file to which best hyperparameter settings will be written',
                    default='best_hyperparameters.txt')
args = parser.parse_args()


def determine_best_hp():
    job_dirs = [f.path for f in os.scandir(args.hp_tuning_dir) if f.is_dir()]
    losses = []
    cfgs = []
    for dir in job_dirs:
        # load cv summary
        df = pd.read_csv(osp.join(dir, 'summary.csv'))
        losses.append(df.val_loss.mean())

        # load config
        yaml = ruamel.yaml.YAML()
        with open(osp.join(dir, 'config.yaml'), 'r') as f:
            cfgs.append(yaml.load(f))

    # determine config with best average validation loss
    best_idx = np.argmax(losses)
    best_cfg = cfgs[best_idx]
    hp_str = " ".join([f'{name}={val}' for name, val in best_cfg.model.items()])

    with open(args.output_file, 'w') as f:
        f.write(hp_str)



if __name__ == "__main__":
    determine_best_hp()