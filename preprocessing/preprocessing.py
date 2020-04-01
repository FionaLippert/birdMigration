import yaml
import os
import json
import rasterio as rio
import numpy as np

def tiff_to_numpy():

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
