import subprocess
import os, glob
import os.path as osp
import yaml
import argparse
from datetime import datetime
import multiprocessing as mp


parser = argparse.ArgumentParser(description='Compute average vertical profile from VPTS')
parser.add_argument('data_dir', help='path to directory containing vpts data')
parser.add_argument('output_dir', help='path to directory where data will be written to')
args = parser.parse_args()

with open('nexrad_config.yml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

years = config['years']
ts_str = config['ts']
te_str = config['te']

for y in years:

    print(f'process data for year {y}')

    y = str(y)

    config['ts'] = datetime.strptime(ts_str.replace('YEAR', y), "%Y-%m-%d %H:%M:%S")
    config['te'] = datetime.strptime(te_str.replace('YEAR', y), "%Y-%m-%d %H:%M:%S")

    ts_y = config['ts'].strftime("%Y%m%dT%H%M")
    te_y = config['te'].strftime("%Y%m%dT%H%M")

    output_dir = osp.join(args.output_dir, y)
    os.makedirs(output_dir, exist_ok=True)

    print(output_dir)

    with open(os.path.join(output_dir, 'config.yml'), 'w+') as f:
        yaml.dump(config, f)
    logfile = os.path.join(output_dir, 'log.txt')

    start_time = datetime.now()

    processes = set()
    max_processes = mp.cpu_count() - 1

    radar_list = config.get('radars', [])
    for file in os.listdir(osp.join(args.data_dir, y)):
        if file.endswith(f'{y}.rds'):
            r = file.split(y)[0] # get radar name
            if len(radar_list) == 0 or r in radar_list:
                # start process for radar r
                radar_file = osp.join(args.data_dir, y, file)
                print(radar_file)
                processes.add(subprocess.Popen(['Rscript', 'vpts2vp_avg.R', output_dir, radar_file],
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
