import subprocess
import os
import os.path as osp
import yaml
import argparse
from datetime import datetime
import multiprocessing as mp


parser = argparse.ArgumentParser(description='VPTS processing pipeline')
parser.add_argument('data_dir', help='path to directory containing vpts data')
parser.add_argument('output_dir', help='path to directory where data will be written to')
args = parser.parse_args()

with open('nexrad_config.yml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

ts_str = config['ts'].strftime("%Y%m%dT%H%M")
te_str = config['te'].strftime("%Y%m%dT%H%M")
year = config['ts'].strftime("%Y")

output_dir = osp.join(args.output_dir, year)
os.makedirs(output_dir, exist_ok = True)

with open(os.path.join(output_dir, 'config.yml'), 'w+') as f:
    yaml.dump(config, f)
logfile = os.path.join(output_dir, 'log.txt')

start_time = datetime.now()

processes = set()
max_processes = mp.cpu_count() - 1

for r in config['radars']:
    radar_file = osp.join(args.data_dir, year, f'{r}{year}.rds')
    processes.add(subprocess.Popen(['Rscript', 'vpts2vpi.R', output_dir, radar_file],
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
