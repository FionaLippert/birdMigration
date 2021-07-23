from omegaconf import DictConfig, OmegaConf
import hydra
import itertools as it
import os.path as osp
import os
import subprocess


@hydra.main(config_path="conf2", config_name="config")
def run(cfg: DictConfig):

    task = cfg.cv_task

    if task == 'inner':
        job_file = osp.join(cfg.root, 'scripts', 'run_inner_cv.job')
        inner_cv(cfg, job_file)
    elif task == 'outer':
        job_file = osp.join(cfg.root, 'scripts', 'run_outer_cv.job')
        outer_cv(cfg, job_file)



def outer_cv(cfg: DictConfig, job_file: str):

    # outer cv loop
    for year in cfg.datasource.years:
        # train on all data except for one year
        final_train_eval(cfg, job_file, year)


def inner_cv(cfg: DictConfig, job_file: str):

    hp_file, n_comb = generate_hp_file(cfg)

    for year in cfg.datasource.years:
        # run inner cv to determine best hyperparameters
        hp_grid_search(cfg, job_file, year, n_comb, hp_file)


def final_train_eval(cfg: DictConfig, job_file: str, test_year: int):

    print(f"Run train/eval for year {test_year}")
    repeats = cfg.cv_repeats
    subprocess.Popen(['sbatch', f'--array=1-{repeats}', job_file, cfg.model.name, str(test_year)])


def hp_grid_search(cfg: DictConfig, job_file: str, test_year: int, n_comb: int, hp_file: str):

    print(f"Run grid search for year {test_year}")
    subprocess.Popen(['sbatch', f'--array=1-{n_comb}', job_file, hp_file, cfg.model.name, str(test_year)])



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
    # python run_nested_cv.py inner --job_file run_inner_cv.job
    # and once all jobs are done and 'best_hp_settings.txt' exists:
    # # python run_nested_cv.py outer --job_file run_outer_cv.job


    run()


