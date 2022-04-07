import subprocess
import os
import os.path as osp
import yaml
import argparse
from datetime import datetime
import multiprocessing as mp


parser = argparse.ArgumentParser(description='VPTS processing pipeline')
parser.add_argument('output_dir', help='path to directory where data will be written to')
args = parser.parse_args()

with open('config.yml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
with open('sdvvp_config.yml') as f:
    sdvvp_config = yaml.load(f, Loader=yaml.FullLoader)

ts_str = config['ts'].strftime("%Y%m%dT%H%M")
te_str = config['te'].strftime("%Y%m%dT%H%M")
# subdir = config['data']['vpi_local'] if args.test_local else config['data']['vpi']
# subdir = os.path.join(subdir, f'{ts_str}_to_{te_str}')

os.makedirs(args.output_dir, exist_ok = True)

with open(os.path.join(args.output_dir, 'config.yml'), 'w+') as f:
    yaml.dump(config, f)
with open(os.path.join(args.output_dir, 'sdvvp_config.yml'), 'w+') as f:
    yaml.dump(sdvvp_config, f)
logfile = os.path.join(args.output_dir, 'log.txt')

start_time = datetime.now()


processes = set()
max_processes = mp.cpu_count() - 1

for r in config['radars']:
    processes.add(subprocess.Popen(['Rscript', 'generate_vpts.R', args.output_dir, r],
                            stdout=open(logfile, 'a+'),
                            stderr=open(logfile, 'a+')))
    if len(processes) >= max_processes:
        os.wait()
        processes.difference_update(
            [p for p in processes if p.poll() is not None])

#Check if all the child processes were closed
for p in processes:
    if p.poll() is None:
        p.wait()


time_elapsed = datetime.now() - start_time
with open(logfile, 'a+') as f:
    f.write('\n')
    f.write(f'Time elapsed (hh:mm:ss.ms) {time_elapsed} \n')
