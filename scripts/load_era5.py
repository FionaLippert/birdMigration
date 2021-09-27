from birds import datasets, era5interface
import geopandas as gpd
import os.path as osp
import os


years = ['2015', '2016', '2017']
seasons = ['fall']
buffer_x, buffer_y = 4, 4

data_dir = '/home/fiona/birdMigration/data/raw'
df = gpd.read_file(osp.join(data_dir, 'abm', 'all_radars.shp'))
radars = dict(zip(zip(df.lon, df.lat), df.radar.values))
dl = era5interface.ERA5Loader(radars)

minx, miny, maxx, maxy = df.total_bounds
bounds = [maxy + buffer_y, minx - buffer_x, miny - buffer_y, maxx + buffer_x]  # North, West, South, East

for year in years:
    for season in seasons:
        output_dir = osp.join(data_dir, 'env', season, year)
        os.makedirs(output_dir, exist_ok=True)
        dl.download_season(season, year, output_dir, pl=850, bounds=bounds,
                           buffer_x=4, buffer_y=4, surface_data=True)
