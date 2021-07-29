from omegaconf import DictConfig, OmegaConf
import hydra
import itertools as it
import os.path as osp
import os
from subprocess import Popen, PIPE
import datetime


@hydra.main(config_path="conf2", config_name="config")
def run(cfg: DictConfig):
    assert cfg.action.name == 'cv'

    hp_file, n_comb = generate_hp_file(cfg)

    for year in cfg.datasource.years:
        # run inner cv to determine best hyperparameters
        hp_grid_search(cfg, year, n_comb, hp_file)


def hp_grid_search(cfg: DictConfig, test_year: int, n_comb: int, hp_file: str):

    if cfg.verbose: print(f"Run grid search for year {test_year}")

    process = Popen(['sbatch', f'--array=1-{n_comb}', cfg.action.task.job_file,
                     hp_file, cfg.model.name, str(test_year)], stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    start_time = datetime.datetime.now()

    # wait until job has been submitted (at most 10s)
    while True:
        if stdout: print(stdout.decode("utf-8"))
        if 'Submitted batch job' in stdout.decode("utf-8") or (datetime.datetime.now() - start_time).seconds > 10:
            return


def generate_hp_file(cfg: DictConfig):
    search_space = {k: v for k, v in cfg.action.hp_search_space.items() if k in cfg.model.keys()}
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


