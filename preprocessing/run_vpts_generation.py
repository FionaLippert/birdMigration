import subprocess
import os
import yaml
import argparse
from datetime import datetime
import multiprocessing as mp

os.system("conda activate birdmigration")

parser = argparse.ArgumentParser(description='VPTS processing pipeline')
parser.add_argument('--test_local', default=False, action='store_true', help='use local file paths')
args = parser.parse_args()

with open('config.yml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
with open('sdvvp_config.yml') as f:
    sdvvp_config = yaml.load(f, Loader=yaml.FullLoader)

ts_str = config['ts'].strftime("%Y%m%dT%H%M")
te_str = config['te'].strftime("%Y%m%dT%H%M")
subdir = config['data']['vpi_local'] if args.test_local else config['data']['vpi']
subdir = os.path.join(subdir, f'{ts_str}_to_{te_str}')

os.makedirs(subdir, exist_ok = True)
print(subdir)
with open(os.path.join(subdir, 'config.yml'), 'w+') as f:
    yaml.dump(config, f)
with open(os.path.join(subdir, 'sdvvp_config.yml'), 'w+') as f:
    yaml.dump(sdvvp_config, f)
logfile = os.path.join(subdir, 'log.txt')

start_time = datetime.now()


processes = set()
max_processes = mp.cpu_count() - 1

for r in config['radars']:
    #print('---------- start new process ------------')
    processes.add(subprocess.Popen(['Rscript', 'generate_vpts.R', subdir, r],
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
