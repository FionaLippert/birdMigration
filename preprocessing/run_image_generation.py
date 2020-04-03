import subprocess
import os
import time
import yaml
import numpy as np
from datetime import datetime, timedelta

with open('config.yml') as f:
      config = yaml.load(f, Loader = yaml.FullLoader)

time_delta = timedelta(minutes = config['tr'])

time_range = np.arange(start = config['ts'],
                       stop  = config['te'] + time_delta,
                       step  = time_delta,
                       dtype = datetime)


subdir = os.path.join(config['data']['tiff'], f'{time_range[0]} - {time_range[-1]}')
os.makedirs(subdir, exist_ok = True)

subprocess.call(['Rscript', 'setup_image_generation.R', subdir])

#subprocess.Popen(['Rscript', 'generate_radar_images.R', subdir, str(time_range[0])])

processes = set()
max_processes = 5

for t in time_range:
    print('---------- start new process ------------')
    processes.add(subprocess.Popen(['Rscript', 'generate_radar_images.R', subdir, str(t)],
                            stdout=open(os.path.join(subdir, 'log.txt'), 'w'),
                            stderr=open(os.path.join(subdir, 'log.txt'), 'w')))
    if len(processes) >= max_processes:
        os.wait()
        processes.difference_update(
            [p for p in processes if p.poll() is not None])
#Check if all the child processes were closed
for p in processes:
    if p.poll() is None:
        p.wait()
