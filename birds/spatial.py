import numpy as np
import pandas as pd
from geovoronoi import voronoi_regions_from_coords, plotting
from shapely import geometry
import pyproj
from geopy.distance import geodesic, lonlat
import itertools as it
import networkx as nx
import geopandas as gpd



class Spatial:
    def __init__(self, radars, seed=1234, buffer=150_000, n_dummy_radars=0): #, epsg_local='3035'):
        """
        Initialization of Spatial object
        Args:
            radars (dict): mapping from radar coordinates (lon, lat) to names
            epsg (str): coordinate reference system as epsg string
            epsg_local (str): coordinate reference system as epsg string
        """

        self.radars = radars
        self.rng = np.random.default_rng(seed)

        # setup geodesic (lonlat) coordinate system
        self.epsg_lonlat = '4326'
        self.pts_lonlat = gpd.GeoSeries([geometry.Point(xy) for xy in radars.keys()], crs=f'EPSG:{self.epsg_lonlat}')

        # setup local "aximuthal equidistant" coordinate system
        lat_0 = self.pts_lonlat.y.mean()
        lon_0 = self.pts_lonlat.x.mean()
        self.crs_local = pyproj.Proj(proj='aeqd', ellps='WGS84', datum='WGS84', lat_0=lat_0, lon_0=lon_0).srs
        self.pts_local = self.pts_lonlat.to_crs(self.crs_local)

        # add dummy radars if applicable
        self.add_dummy_radars(n_dummy_radars, buffer=buffer)
        self.N_dummy = n_dummy_radars
        self.N = len(radars) + n_dummy_radars
        self.radar_names = list(self.radars.values()) + [f'boundary_{i}' for i in range(self.N_dummy)]

    def voronoi(self, buffer=150_000, self_edges=False):
        """
        Construct Voronoi diagram based on radar coordinates
        Args:
            buffer (float): max distance around radar stations (in meters)
        Returns:
            adj (GeoDataFrame): edges between neighbouring radars, weight=distance [m]
            cells (GeoDataFrame): polygons describing the Voronoi cells
        """

        boundary = geometry.MultiPoint(self.pts_local).buffer(buffer)
        sink = boundary.buffer(buffer).difference(boundary)
        self.sink = gpd.GeoSeries(sink, crs=self.crs_local)  # .to_crs(epsg=self.epsg_equal_area)

        xy = self.pts2coords(self.pts_local)
        lonlat = self.pts2coords(self.pts_lonlat)

        polygons, pts = voronoi_regions_from_coords(np.array(xy), boundary)
        polygons = [polygons[pid] for pid, pt in sorted(pts.items(), key=lambda kv: kv[1])]

        cells = gpd.GeoDataFrame({'radar': self.radar_names,
                                  'observed' : [True] * (self.N-self.N_dummy) + [False] * self.N_dummy,
                                  'x': [c[0] for c in xy],
                                  'y': [c[1] for c in xy],
                                  'lon': [c[0] for c in lonlat],
                                  'lat': [c[1] for c in lonlat],
                                  },
                                 geometry=polygons,
                                 crs=self.crs_local)
        cells['boundary'] = cells.geometry.map(lambda x: x.intersects(sink))

        adj = np.zeros((self.N, self.N))
        G = nx.DiGraph()
        edges = []
        face_len = []
        for i, j in it.combinations(cells.index, 2):
            intersec = cells.geometry.iloc[i].intersection(cells.geometry.iloc[j])
            if type(intersec) is geometry.LineString:
                adj[i, j] = self.distance(xy[i], xy[j])
                adj[j, i] = adj[i, j]
                face = gpd.GeoSeries(intersec, crs=self.crs_local)
                p1 = face.iloc[0].coords[0]
                p2 = face.iloc[0].coords[1]

                distance = self.distance(xy[i], xy[j])
                face_length = self.distance(p1, p2)
                face_len.append(face_length)
                G.add_edge(i, j, distance=distance, face_length=face_length,
                                 angle=self.angle(lonlat[i], lonlat[j]))
                G.add_edge(j, i, distance=distance, face_length=face_length,
                           angle=self.angle(lonlat[j], lonlat[i]))
                edges.append(geometry.LineString((xy[i], xy[j])))
        if self_edges:
            [G.add_edge(i, i, distance=0, face_length=0, angle=0) for i in cells.index]


        nx.set_node_attributes(G, pd.Series(cells['radar']).to_dict(), 'radar')
        nx.set_node_attributes(G, pd.Series(cells['boundary']).to_dict(), 'boundary')
        # nx.set_node_attributes(G, 'measured', name='type')
        #
        # # add one global sink node
        # sink_id = len(G)
        # G.add_node(sink_id, type='sink')
        # G.add_edges_from([(sink_id, n) for n in G.nodes])
        # G.add_edges_from([(n, sink_id) for n in G.nodes])

        self.cells = cells
        self.G = G
        self.max_dist = np.max(adj)
        self.edges = gpd.GeoSeries(edges, crs=self.crs_local)

        return cells, G

    def sample_point(self, area):
        minx, miny, maxx, maxy = area.total_bounds
        x = self.rng.uniform(minx, maxx)
        y = self.rng.uniform(miny, maxy)
        pos = geometry.Point(x, y)
        while not area.contains(pos).any():
            x = np.random.uniform(minx, maxx)
            y = np.random.uniform(miny, maxy)
            pos = geometry.Point(x, y)
        return x, y

    def add_dummy_radars(self, n, buffer=150_000):
        if n == 0: return
        boundary = geometry.MultiPoint(self.pts_local).buffer(buffer).boundary

        distances = np.linspace(0, boundary.length, n+1)
        points = [boundary.interpolate(d) for d in distances[:-1]]

        dummy_radars = gpd.GeoSeries([p for p in points], crs=self.crs_local).to_crs(epsg=self.epsg_lonlat)

        self.pts_lonlat = self.pts_lonlat.append(dummy_radars, ignore_index=True)
        self.pts_local = self.pts_local.append(dummy_radars.to_crs(self.crs_local), ignore_index=True)


    def G_max_dist(self, max_distance):
        # create graph with edges between any two radars with distance <= max_distance [km]
        xy = self.pts2coords(self.pts_local)
        lonlat = self.pts2coords(self.pts_lonlat)
        max_distance = max_distance * 1000 # kilometers to meters
        G = nx.DiGraph()
        for i in range(self.N):
            for j in range(self.N):
                dist = self.distance(xy[i], xy[j])
                if dist <= max_distance:
                    G.add_edge(i, j, distance=dist,
                               angle=self.angle(lonlat[i], lonlat[j]))
                    G.add_edge(j, i, distance=dist,
                               angle=self.angle(lonlat[j], lonlat[i]))

        nx.set_node_attributes(G, pd.Series(self.cells['radar']).to_dict(), 'radar')
        nx.set_node_attributes(G, pd.Series(self.cells['boundary']).to_dict(), 'boundary')
        nx.set_node_attributes(G, 'measured', name='type')

        return G

    def subgraph(self, G, attr, value):
        node_gen = (n for n, data in G.nodes(data=True) if data.get(attr) == value)
        subgraph = G.subgraph(node_gen)
        return nx.DiGraph(subgraph)

    def pts2coords(self, pts, reverse_xy=False):
        """
        Convert point objects to (x, y) coordinates
        Args:
            reverse_xy (bool): if true, the order of the two coordinates is switched to (y, x)
        Returns:
            coords: list of projected coordinates
        """
        coords = [[p.xy[0][0], p.xy[1][0]] for p in pts]
        if reverse_xy:
            coords = [[p.xy[1][0], p.xy[0][0]] for p in pts]
            #coords = np.flip(coords, axis=1)
        return coords

    def distance(self, coord1, coord2):
        """
        Compute distance between two geographical locations in local crs
        Args:
            coord1 (tuple): coordinates of first location (x, y)
            coord2 (tuple): coordinates of second location (x, y)
        Returns:
            dist (float): distance in meters
        """
        dist = np.linalg.norm(np.array(coord1) - np.array(coord2))
        return dist

    def angle(self, coord1, coord2):
        # coords should be in lonlat crs
        y = coord2[0] - coord1[0]
        x = coord2[1] - coord1[1]

        rad = np.arctan2(y, x)
        deg = np.rad2deg(rad)

        # make sure angle is between 0 and 360 degree
        deg = (deg + 180) % 360 + 180

        return deg

    def flip(self, coord):
        return (coord[1], coord[0])

    def voronoi_with_sink(self):
        gdf_sink = gpd.GeoDataFrame()

        if not hasattr(self, 'cells'):
            self.voronoi()
        for c in self.cells.columns:
            gdf_sink[c] = [np.nan]
        gdf_sink['radar'] = 'sink'
        gdf_sink['geometry'] = self.sink.geometry
        voronoi_with_sink = self.cells.append(gdf_sink, ignore_index=True)
        return voronoi_with_sink


if __name__ == '__main__':

    from birds import datahandling
    import os.path as osp

    path = '/home/fiona/birdMigration/data/raw/radar/fall/2015'
    radars = datahandling.load_radars(path)

    sp = Spatial(radars, n_dummy_radars=15)
    sp.cells.to_file(osp.join(path, 'voronoi_test.shp'))
    sp.sink.to_file(osp.join(path, 'voronoi_sink_test.shp'))
    nx.write_gpickle(sp.subgraph('type', 'measured'), osp.join(path, 'delaunay_test.gpickle'), protocol=4)

    # for index, row in sp.cells.iterrows():
    #     area = row.geometry.area / 1000_000
    #     partial = row.geometry.buffer(-36_000).area / 1000_000
    #     print(row.radar, partial/area)