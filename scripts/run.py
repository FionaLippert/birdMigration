from omegaconf import DictConfig, OmegaConf
import hydra
import os.path as osp
import os
import traceback
import run_GBT, run_NNs, run_FlowNNs, run_GAM, run_HA, run_preprocessing

@hydra.main(config_path="conf", config_name="config")
def run(cfg: DictConfig):

    if cfg.action == 'preprocessing':
        # run preprocessing
        run_preprocessing.run(cfg)
    else:
        # run training/testing
        # directory to which outputs will be written
        output_dir = osp.join(cfg.root, 'results', cfg.datasource.name, cfg.action.name,
                              cfg.model.name, cfg.experiment)
        os.makedirs(output_dir, exist_ok=True)

        log_file = os.path.join(output_dir, 'log.txt')
        log = open(log_file, 'w')

        try:
            if cfg.model.name == 'GBT':
                run_GBT.run(cfg, output_dir, log)
            elif cfg.model.name == 'GAM':
                run_GAM.run(cfg, output_dir, log)
            elif cfg.model.name in ['BirdFlowGraphLSTM']:
                run_FlowNNs.run(cfg, output_dir, log)
            elif cfg.model.name == 'HA':
                run_HA.run(cfg, output_dir, log)
            else:
                run_NNs.run(cfg, output_dir, log)

        except Exception:
            print(traceback.format_exc(), file=log)

        log.flush()
        log.close()

if __name__ == "__main__":
    run()