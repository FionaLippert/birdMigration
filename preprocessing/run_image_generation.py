import subprocess
import os
import time
import yaml
import numpy as np
from datetime import datetime, timedelta
import multiprocessing as mp

with open('config.yml') as f:
      config = yaml.load(f, Loader = yaml.FullLoader)

time_delta = timedelta(minutes = config['tr'])
time_range = np.arange(start = config['ts'],
                       stop  = config['te'] + time_delta,
                       step  = time_delta,
                       dtype = datetime)

ts_str = time_range[0].strftime("%Y%m%dT%H%M")
te_str = time_range[-1].strftime("%Y%m%dT%H%M")
subdir = os.path.join(config['data']['ppi'], f'{ts_str}-{te_str}')
#subdir = os.path.join(config['data']['ppi'], f'{time_range[0]} - {time_range[-1]}')

os.makedirs(subdir, exist_ok = True)
with open(os.path.join(subdir, 'config.yml'), 'w+') as f:
    yaml.dump(config, f)
logfile = os.path.join(subdir, 'log.txt')

for r in config['radars']:
    os.makedirs(os.path.join(subdir, r), exist_ok=True)

start_time = datetime.now()

subprocess.call(['Rscript', 'setup_image_generation.R', subdir])

print("done with setup")

processes = set()
max_processes = mp.cpu_count()

for t in time_range:
    #print('---------- start new process ------------')
    processes.add(subprocess.Popen(['Rscript', 'generate_radar_images.R', subdir, str(t)],
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
