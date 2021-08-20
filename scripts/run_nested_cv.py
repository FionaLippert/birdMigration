from omegaconf import DictConfig, OmegaConf
import hydra
import itertools as it
import os.path as osp
import os
from subprocess import Popen, PIPE
from datetime import datetime
import numpy as np
import pandas as pd
from shutil import copy


@hydra.main(config_path="conf", config_name="config")
def run(cfg: DictConfig):

    if cfg.verbose: print(f'hydra working directory: {os.getcwd()}')

    target_dir = osp.join(cfg.device.root, cfg.output_dir, f'nested_cv_{cfg.model.name}')
    os.makedirs(target_dir, exist_ok=True)

    if cfg.task.name == 'innerCV':
        run_inner_cv(cfg, target_dir)
    elif cfg.task.name == 'outerCV':
        run_outer_cv(cfg, target_dir)

def run_inner_cv(cfg: DictConfig, target_dir):

    hp_file, n_comb = generate_hp_file(cfg, target_dir)

    for year in cfg.datasource.years:
        # run inner cv for all hyperparameter settings
        output_dir = osp.join(target_dir, f'test_{year}', 'hp_grid_search')
        hp_grid_search(cfg, year, n_comb, hp_file, output_dir)


def run_outer_cv(cfg: DictConfig, target_dir):

    for year in cfg.datasource.years:
        # determine best hyperparameter setting
        input_dir = osp.join(target_dir, f'test_{year}', 'hp_grid_search')
        determine_best_hp(input_dir)

        # use this setting and train on all data except for one year
        output_dir = osp.join(target_dir, f'test_{year}', 'final_evaluation')
        final_train_eval(cfg, year, output_dir)


def final_train_eval(cfg: DictConfig, test_year: int, output_dir: str, timeout=10):

    if cfg.verbose: print(f"Start train/eval for year {test_year}")

    config_path = osp.dirname(output_dir)
    repeats = cfg.task.repeats

    if cfg.device.slurm:
        job_file = osp.join(cfg.device.root, cfg.task.slurm_job)
        proc = Popen(['sbatch', f'--array=1-{repeats}', job_file, cfg.device.root, output_dir, config_path,
                      str(test_year)], stdout=PIPE, stderr=PIPE)
    else:
        job_file = osp.join(cfg.device.root, cfg.task.local_job)
        proc = Popen([f'./{job_file}', cfg.device.root, output_dir, config_path,
                      str(test_year), repeats], stdout=PIPE, stderr=PIPE)
    stdout, stderr = proc.communicate()
    start_time = datetime.now()

    while True:
        if stderr:
            print(stderr.decode("utf-8"))
            return
        if stdout:
            print(stdout.decode("utf-8"))
            return
        if (datetime.now() - start_time).seconds > timeout:
            print(f'timeout after {timeout} seconds')
            return


def determine_best_hp(input_dir: str):
    job_dirs = [f.path for f in os.scandir(input_dir) if f.is_dir()]
    best_loss = np.inf
    for dir in job_dirs:
        # load cv summary
        df = pd.read_csv(osp.join(dir, 'summary.csv'))
        loss = df.final_val_loss.mean()

        if loss < best_loss:
            # copy config file to parent directory
            copy(osp.join(dir, 'config.yaml'), osp.dirname(input_dir))
            best_loss = loss



def hp_grid_search(cfg: DictConfig, test_year: int, n_comb: int, hp_file: str, output_dir: str, timeout=10):

    if cfg.verbose: print(f"Start grid search for year {test_year}")

    # directory created by hydra, containing current config
    # including settings overwritten from command line
    config_path = osp.join(os.getcwd(), '.hydra')

    # run inner cross-validation loop for all different hyperparameter settings
    if cfg.device.slurm:
        job_file = osp.join(cfg.device.root, cfg.task.slurm_job)
        proc = Popen(['sbatch', f'--array=1-{n_comb}', job_file, cfg.device.root, output_dir, config_path,
                      hp_file, str(test_year)], stdout=PIPE, stderr=PIPE)
    else:
        job_file = osp.join(cfg.device.root, cfg.task.local_job)
        proc = Popen([job_file, cfg.device.root, output_dir, config_path,
                      hp_file, str(test_year), str(n_comb)], stdout=PIPE, stderr=PIPE)

    stdout, stderr = proc.communicate()
    start_time = datetime.now()

    # wait until job has been submitted (at most 10s)
    while True:
        if stderr: 
            # something went wrong
            print(stderr.decode("utf-8"))
        if stdout: 
            # successful job submission
            print(stdout.decode("utf-8"))
            return
        if (datetime.now() - start_time).seconds > timeout:
            print(f'timeout after {timeout} seconds')
            return


def generate_hp_file(cfg: DictConfig, target_dir):
    search_space = {k: v for k, v in cfg.hp_search_space.items() if k in cfg.model.keys()}
    hp_file = osp.join(target_dir, 'hyperparameters.txt')

    names, values = zip(*search_space.items())
    all_combinations = [dict(zip(names, v)) for v in it.product(*values)]

    with open(hp_file, 'w') as f:
        for combi in all_combinations:
            hp_str = " ".join([f'model.{name}={val}' for name, val in combi.items()]) + "\n"
            f.write(hp_str)

    if cfg.verbose:
        print("successfully generated hyperparameter settings file")
        print(f"File path: {hp_file}")
        print(f"Number of combinations: {len(all_combinations)} \n")

    return hp_file, len(all_combinations)


if __name__ == '__main__':

    run()


