from omegaconf import DictConfig, OmegaConf
import hydra
import itertools as it
import os.path as osp
import os
from subprocess import Popen, PIPE
from datetime import datetime


@hydra.main(config_path="conf2", config_name="config")
def run(cfg: DictConfig):
    if cfg.task.name == 'innerCV':
        run_inner_cv(cfg)
    elif cfg.task.name == 'outerCV':
        run_outer_cv(cfg)

def run_outer_cv(cfg: DictConfig):

    for year in cfg.datasource.years:
        # train on all data except for one year
        final_train_eval(cfg, year)


def run_inner_cv(cfg: DictConfig):

    hp_file, n_comb = generate_hp_file(cfg)

    for year in cfg.datasource.years:
        # run inner cv to determine best hyperparameters
        hp_grid_search(cfg, year, n_comb, hp_file)


def final_train_eval(cfg: DictConfig, test_year: int, timeout=10):

    print(f"Start train/eval for year {test_year}")
    repeats = cfg.task.repeats
    job_file = osp.join(cfg.root, cfg.task.job_file)
    Popen(['sbatch', f'--array=1-{repeats}', job_file, cfg.model.name, str(test_year)], stdout=PIPE, stderr=PIPE)
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



def hp_grid_search(cfg: DictConfig, test_year: int, n_comb: int, hp_file: str):

    if cfg.verbose: print(f"Start grid search for year {test_year}")

    job_file = osp.join(cfg.root, cfg.task.job_file)

    proc = Popen(['sbatch', f'--array=1-{n_comb}', job_file,
                     hp_file, cfg.model.name, str(test_year)], stdout=PIPE, stderr=PIPE)

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


def generate_hp_file(cfg: DictConfig):
    search_space = {k: v for k, v in cfg.hp_search_space.items() if k in cfg.model.keys()}
    hp_file = osp.join(cfg.root, 'hyperparameters.txt')

    names, values = zip(*search_space.items())
    all_combinations = [dict(zip(names, v)) for v in it.product(*values)]

    with open(hp_file, 'w') as f:
        for combi in all_combinations:
            hp_str = " ".join([f'model.{name}={val}' for name, val in combi.items()]) + "\n"
            f.write(hp_str)

    if cfg.verbose:
        print("successfully generated hyperparameter settings file")
        print(f"File path: {hp_file}")
        print(f"Number of combinations: {len(all_combinations)}")

    return hp_file, len(all_combinations)


if __name__ == '__main__':

    run()


