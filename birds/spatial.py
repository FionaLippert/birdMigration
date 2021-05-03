import numpy as np
import pandas as pd
from geovoronoi import voronoi_regions_from_coords, plotting
from shapely import geometry
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
        self.epsg_lonlat = '4326'          # epsg code for lon lat crs
        #self.epsg_equidistant = '4087'     # epsg code for "WGS 84 / World Equidistant Cylindrical"
        self.epsg_equal_area = '3035'      # epsg code for "Lambert Azimuthal Equal Area projection"
        #self.epsg_local = '32632'          # epsg code for "WGS84 / UTM zone 32N"


        # projections of radar positions
        self.pts_lonlat = gpd.GeoSeries([geometry.Point(xy) for xy in radars.keys()],
                                        crs=f'EPSG:{self.epsg_lonlat}')
        self.pts_equal_area = self.pts_lonlat.to_crs(epsg=self.epsg_equal_area)
        # equidistant projection centered around mean location of radar stations
        self.crs_local = f'+proj=aeqd +lat_0={self.pts_lonlat.y.mean():.7f} ' \
                         f'+lon_0={self.pts_lonlat.x.mean():.7f} +units=m +ellps=WGS84'
        self.pts_local = self.pts_lonlat.to_crs(self.crs_local)

        self.rng = np.random.default_rng(seed)
        self.add_dummy_radars(n_dummy_radars, buffer=buffer)

        self.N_dummy = n_dummy_radars
        self.N = len(radars) + n_dummy_radars

        self.voronoi()

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

        # compute voronoi cells
        xy_equal_area = self.pts2coords(self.pts_equal_area)
        #xy_equidistant = self.pts2coords(self.pts_equidistant)
        xy = self.pts2coords(self.pts_local)
        lonlat = self.pts2coords(self.pts_lonlat)

        polygons, pts = voronoi_regions_from_coords(np.array(xy), boundary)
        #print(polygons, pts)

        # reindex polygons to match order of radars
        #idx = np.array(poly2pt).flatten().argsort()
        #idx = np.array(list(pts.values())).argsort()

        #polygons = np.array(list(polygons.values()))[idx]
        polygons = [polygons[pid] for pid, pt in sorted(pts.items(), key=lambda kv: kv[1])]

        cells = gpd.GeoDataFrame({'radar': list(self.radars.values()) + ['boundary'] * self.N_dummy,
                                  'x': [c[0] for c in xy],
                                  'y': [c[1] for c in xy],
                                  'x_eqa': [c[0] for c in xy_equal_area],
                                  'y_eqa': [c[1] for c in xy_equal_area],
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
                adj[i, j] = self.distance(lonlat[i], lonlat[j], epsg=self.epsg_lonlat)
                adj[j, i] = adj[i, j]
                face = gpd.GeoSeries(intersec, crs=self.crs_local).to_crs(epsg=self.epsg_lonlat)
                p1 = face.iloc[0].coords[0]
                p2 = face.iloc[0].coords[1]
                face_len.append(self.distance(p1, p2, epsg=self.epsg_lonlat))

                distance = self.distance(lonlat[i], lonlat[j], epsg=self.epsg_lonlat)
                face_length = self.distance(p1, p2, epsg=self.epsg_lonlat)
                G.add_edge(i, j, distance=distance, face_length=face_length,
                                 angle=self.angle(lonlat[i], lonlat[j]))
                G.add_edge(j, i, distance=distance, face_length=face_length,
                           angle=self.angle(lonlat[j], lonlat[i]))
                edges.append(geometry.LineString((xy[i], xy[j])))
        if self_edges:
            [G.add_edge(i, i, distance=0, face_length=0, angle=0) for i in cells.index]

        # create network
        #G = nx.from_numpy_matrix(adj, create_using=nx.DiGraph())
        nx.set_node_attributes(G, pd.Series(cells['radar']).to_dict(), 'radar')
        nx.set_node_attributes(G, pd.Series(cells['boundary']).to_dict(), 'boundary')
        nx.set_node_attributes(G, 'measured', name='type')
        #nx.set_node_attributes(G, pd.Series(cells['xy'].to_dict()), 'coords')

        # add sink nodes
        # for i, row in cells[cells['boundary']].iterrows():
        #     nidx = len(G)
        #     G.add_node(nidx, type='sink', radar=row.radar)
        #     G.add_edge(nidx, i)

        # add one global sink node
        sink_id = len(G)
        G.add_node(sink_id, type='sink')
        G.add_edges_from([(sink_id, n) for n in G.nodes])
        G.add_edges_from([(n, sink_id) for n in G.nodes])

        # add self-loops to graph
        #G.add_weighted_edges_from([(n, n, 0) for n in G.nodes])

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
        # sink = boundary.buffer(buffer).difference(boundary)
        # sink = gpd.GeoSeries(sink, crs=self.crs_local).to_crs(epsg=self.epsg_lonlat)

        distances = np.linspace(0, boundary.length, n+1)
        points = [boundary.interpolate(d) for d in distances[:-1]]

        dummy_radars = gpd.GeoSeries([p for p in points], crs=self.crs_local).to_crs(epsg=self.epsg_lonlat)

        self.pts_lonlat = self.pts_lonlat.append(dummy_radars, ignore_index=True)
        self.pts_local = self.pts_local.append(dummy_radars.to_crs(self.crs_local), ignore_index=True)
        self.pts_equal_area = self.pts_equal_area.append(dummy_radars.to_crs(epsg=self.epsg_equal_area),
                                                         ignore_index=True)

    def G_max_dist(self, max_distance):
        # create graph with edges between any two radars with distance <= max_distance [km]
        xy_equal_area = self.pts2coords(self.pts_equal_area)
        lonlat = self.pts2coords(self.pts_lonlat)

        G = nx.DiGraph()
        for i in range(self.N):
            for j in range(self.N):
                dist = self.distance(xy_equal_area[i], xy_equal_area[j], epsg=self.epsg_equal_area) / 1000
                if dist <= max_distance:
                    G.add_edge(i, j, distance=dist,
                               angle=self.angle(lonlat[i], lonlat[j]))
                    G.add_edge(j, i, distance=dist,
                               angle=self.angle(lonlat[j], lonlat[i]))

        nx.set_node_attributes(G, pd.Series(self.cells['radar']).to_dict(), 'radar')
        nx.set_node_attributes(G, pd.Series(self.cells['boundary']).to_dict(), 'boundary')
        nx.set_node_attributes(G, 'measured', name='type')

        return G

    def subgraph(self, attr, value):
        node_gen = (n for n, data in self.G.nodes(data=True) if data.get(attr) == value)
        subgraph = self.G.subgraph(node_gen)
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

    def distance(self, coord1, coord2, epsg):
        """
        Compute distance between two geographical locations
        Args:
            coord1 (tuple): coordinates of first location (lon, lat) or (x, y)
            coord2 (tuple): coordinates of second location (lon, lat) or (x, y)
        Returns:
            dist (float): distance in meters
        """
        if epsg == self.epsg_lonlat:
            dist = geodesic(self.flip(coord1), self.flip(coord2)).kilometers
        else:
            dist = np.linalg.norm(np.array(coord1) - np.array(coord2))
        return dist

    def angle(self, coord1, coord2):
        # coords should be in lonlat crs
        y = coord2[0] - coord1[0]
        x = coord2[1] - coord1[1]

        rad = np.arctan2(y, x)
        deg = np.rad2deg(rad)
        deg = (deg + 360) % 360

        return deg

    def flip(self, coord):
        return (coord[1], coord[0])

    def voronoi_with_sink(self):
        gdf_sink = gpd.GeoDataFrame()
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