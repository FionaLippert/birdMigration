from omegaconf import DictConfig, OmegaConf
import hydra
import itertools as it
import os.path as osp
import os
import traceback

@hydra.main(config_path="conf2", config_name="config")
def generate(cfg: DictConfig):
    search_space = {k : v for k, v in cfg.hp_search_space.items() if k in cfg.model.keys()}
    hp_file = osp.join(cfg.root, 'hyperparameters.txt')

    names, values = zip(*search_space.items())
    all_combinations = [dict(zip(names, v)) for v in it.product(*values)]

    with open(hp_file, 'w') as f:
        for combi in all_combinations:
            hp_str = " ".join([f'model.{name}={val}' for name, val in combi.items()]) + "\n"
            f.write(hp_str)
    print("successfully generated hyperparameter settings file")

if __name__ == '__main__':
    generate()
