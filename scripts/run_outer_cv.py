from omegaconf import DictConfig, OmegaConf
import hydra
import itertools as it
import os.path as osp
import os
from subprocess import Popen, PIPE
import datetime


@hydra.main(config_path="conf2", config_name="config")
def run(cfg: DictConfig):
    assert cfg.task == 'outerCV'

    # outer cv loop
    for year in cfg.datasource.years:
        # train on all data except for one year
        final_train_eval(cfg, year)


def final_train_eval(cfg: DictConfig, test_year: int):

    print(f"Run train/eval for year {test_year}")
    repeats = cfg.action.repeats
    Popen(['sbatch', f'--array=1-{repeats}', cfg.task.job_file, cfg.model.name, str(test_year)])



if __name__ == '__main__':

    run()


