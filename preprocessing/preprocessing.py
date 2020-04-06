import yaml
import os
import json
import rasterio as rio
import numpy as np
from datetime import datetime, timedelta

def tiff_timeseries_to_numpy():

    with open('config.yml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    dirs = [f.path for f in os.scandir(config['data']['tiff']) if f.is_dir()]

    for d in dirs:
        numpy_dir = os.path.join(config['data']['numpy'], os.path.basename(d))
        if not os.path.isdir(numpy_dir):
            # if not yet processed to numpy
            for root, subdirs, _ in os.walk(d):
               for idx, sd in enumerate(sorted(subdirs)):
                    with open(os.path.join(root, sd, 'timestamps.json')) as f:
                        t = json.load(f)
                        if idx == 0:
                            timestamps = np.array(t)
                        else:
                            timestamps = np.hstack((timestamps, np.array(t)))

                    data_chunk = rio.open(os.path.join(root, sd, f'{config["quantity"]}.tif'))
                    if idx == 0:
                        bounds = np.array([[data_chunk.bounds.bottom, data_chunk.bounds.left], \
                                [data_chunk.bounds.top, data_chunk.bounds.right]])
                        data = data_chunk.read()
                    else:
                        data = np.vstack((data, data_chunk.read()))

            result = RadarTimeSeries(data, timestamps, bounds)
            result.save(numpy_dir)

def tiff_images_to_numpy():

    with open('config.yml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    dirs = [f.path for f in os.scandir(config['data']['tiff']) if f.is_dir()]

    for d in dirs:
        numpy_dir = os.path.join(config['data']['numpy'], os.path.basename(d))
        if not os.path.isdir(numpy_dir):
            # if not yet processed to numpy
            data = {}
            for _, _, files in os.walk(d):
               for idx, f in enumerate(sorted(files)):
                   if f.endswith('.tif'):
                        data_chunk = rio.open(os.path.join(d, f))
                        if idx == 0:
                            bounds = np.array([[data_chunk.bounds.bottom, data_chunk.bounds.left], \
                                    [data_chunk.bounds.top, data_chunk.bounds.right]])
                        timestamp = os.path.splitext(os.path.basename(f))[0])
                        data[timestamp] = data_chunk.read()

            result = RadarTimeSeries(data = np.array(list(data.values())), /
                                     timestamps = np.array(list(data.keys())), /
                                     bounds = bounds)
            result.save(numpy_dir)

def get_radar_sequence(dir, ts, tl, tr):

    i = 0
    data = []
    for _, _, files in os.walk(dir):
       for f in sorted(files):
           if f == (f'{datetime(ts) + timedelta(minutes=tr*i)}.tif'):
               data.append(rio.open(os.path.join(dir, f)).read())
               i += 1
           if i == tl:
               return np.array(data)
    if i < tl:
        return None


class RasterImage:

    def __init__(self, root, radar, timestamp):
        


class RadarTimeSeries:

    def __init__(self, data, timestamps, bounds):
        self.data = data
        self.timestamps = timestamps
        self.bound = bounds

    def save(self, dir):
        os.makedirs(numpy_dir, exist_ok=True)
        np.save(os.path.join(dir, 'data.npy'), self.data)
        np.save(os.path.join(dir, 'timestamps.npy'), self.timestamps)
        np.save(os.path.join(dir, 'bounds.npy'), self.bounds)

    def to_pngs(self, log=True, colormap=cm.rainbow,
                        min_var = None, max_intensity = 2000):

        img_arr = self.data.copy()
        img_arr[img_arr>max_intensity] = max_intensity #np.nan
        if log:
            img_arr[img_arr==0] = np.nan
            img_arr = np.log(img_arr)
        img_arr = (img_arr - np.nanmin(img_arr)) / (np.nanmax(img_arr) - np.nanmin(img_arr))

        if min_var is not None:
            img_arr[:, np.nanvar(img_arr, axis=0)<min_var] = np.nan

        #dirname = f'min_var={min_var}'
        #os.makedirs(os.path.join(config['data']['png'], dirname), exist_ok=True)

        x, y,z = np.where(np.isnan(img_arr))
        img_arr = colormap(img_arr, bytes = True)
        img_arr[x,y,z, -1] = 0

        images = [Image.fromarray(img) for img in img_arr]
        # for img in images:
            #img.save(f'{dirname}/{self.timestamps[i]}.png','PNG')

        return t_range




"""
def raster_to_pngs(raster_data, log=True, colormap=cm.rainbow,
                    min_var = None, max_intensity = 2000):
    t_start = config['ts']
    t_delta = timedelta(minutes=config['tr'])
    img_arr = raster_data.read()

    img_arr[img_arr>max_intensity] = max_intensity #np.nan
    if log:
        img_arr[img_arr==0] = np.nan
        img_arr = np.log(img_arr)
    img_arr = (img_arr - np.nanmin(img_arr)) / (np.nanmax(img_arr) - np.nanmin(img_arr))

    if min_var is not None:
        img_arr[:, np.nanvar(img_arr, axis=0)<min_var] = np.nan

    dirname = f'min_var={min_var}'
    os.makedirs(os.path.join(config['data']['png'], dirname), exist_ok=True)

    x, y,z = np.where(np.isnan(img_arr))
    img_arr = colormap(img_arr, bytes = True)
    img_arr[x,y,z, -1] = 0




    t_range = np.array([t_start + t_delta*i for i in range(img_arr.shape[0])], dtype=datetime)

    for i in range(img_arr.shape[0]):
        im = Image.fromarray(img_arr[i])
        im.save(f'{dirname}/{t_range[i]}.png','PNG')


    #t_range = np.arange(t_start, datetime(2015,7,1), timedelta(days=1)).astype(datetime)
    return t_range
"""
