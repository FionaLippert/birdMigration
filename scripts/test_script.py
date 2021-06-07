import os.path as osp
import os
import traceback

import logging
import time

import hydra
import submitit
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


@hydra.main(config_path=".", config_name="config")
def run(cfg: DictConfig):
    env = submitit.JobEnvironment()
    log.info(f"Process ID {os.getpid()} executing task {cfg.task}, with {env}")

    # output_dir = osp.join(cfg.root, 'results', cfg.datasource.name, cfg.action.name,
    #                       cfg.model.name, cfg.experiment)
    # os.makedirs(output_dir, exist_ok=True)

    time.sleep(1)


if __name__ == "__main__":
    run()