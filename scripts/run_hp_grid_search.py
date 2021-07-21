from omegaconf import DictConfig, OmegaConf
import hydra
import itertools as it
import os.path as osp
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--job_file', type=str, help='slurm array job file')


@hydra.main(config_path="conf2", config_name="config")
def hp_grid_search(cfg: DictConfig, job_file: str):

    hp_file, n_comb = generate_hp_file(cfg)

    print("Start cross-validation for all hyperparameter settings")
    subprocess.Popen(['sbatch', f'--array=1-{n_comb}', job_file, hp_file])


def generate_hp_file(cfg: DictConfig):
    search_space = {k: v for k, v in cfg.hp_search_space.items() if k in cfg.model.keys()}
    hp_file = osp.join(cfg.root, 'hyperparameters.txt')

    names, values = zip(*search_space.items())
    all_combinations = [dict(zip(names, v)) for v in it.product(*values)]

    with open(hp_file, 'w') as f:
        for combi in all_combinations:
            hp_str = " ".join([f'model.{name}={val}' for name, val in combi.items()]) + "\n"
            f.write(hp_str)
    print("successfully generated hyperparameter settings file")
    print(f"File path: {hp_file}")
    print(f"Number of combinations: {len(all_combinations)}")

    return hp_file, len(all_combinations)


if __name__ == '__main__':

    # to run nested cross-validation run the following in the terminal:
    # python run_hp_grid_search.py --job_file cv_job_file +datasource.test_year=range(2015, 2018)

    args = parser.parse_args()
    hp_grid_search(job_file=args.job_file)
