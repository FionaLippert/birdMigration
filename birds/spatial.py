import numpy as np
import pandas as pd
from geovoronoi import voronoi_regions_from_coords, plotting
from shapely import geometry
from geopy.distance import geodesic, lonlat
import itertools as it
import networkx as nx
import geopandas as gpd


class Spatial:
    def __init__(self, radars, epsg='4326', epsg_local='3035'):
        """
        Initialization of Spatial object
        Args:
            radars (dict): mapping from radar coordinates (lon, lat) to names
            epsg (str): coordinate reference system as epsg string
            epsg_local (str): coordinate reference system as epsg string
        """
        self.radars = radars
        self.epsg = epsg
        self.epsg_local = epsg_local
        self.N = len(radars)

        # projection to local crs
        self.pts_lonlat = gpd.GeoSeries([geometry.Point(xy) for xy in radars.keys()],
                                        crs=f'EPSG:{epsg}')
        self.pts_local = self.pts_lonlat.to_crs(epsg=epsg_local)

        self.voronoi()

    def voronoi(self, buffer=150_000, plot=False):
        """
        Construct Voronoi diagram based on radar coordinates
        Args:
            buffer (float): max distance around radar stations (in meters)
            plot (bool): if True, plot Voronoi diagram
        Returns:
            adj (GeoDataFrame): edges between neighbouring radars, weight=distance [m]
            cells (GeoDataFrame): polygons describing the Voronoi cells
        """

        boundary = geometry.MultiPoint(self.pts_local).buffer(buffer)
        sink = boundary.buffer(buffer).difference(boundary)

        # compute voronoi cells
        xy = self.pts2coords(self.pts_local)
        lonlat = self.pts2coords(self.pts_lonlat)
        polygons, pts = voronoi_regions_from_coords(np.array(xy), boundary)
        #print(polygons, pts)

        # reindex polygons to match order of radars
        #idx = np.array(poly2pt).flatten().argsort()
        #idx = np.array(list(pts.values())).argsort()

        #polygons = np.array(list(polygons.values()))[idx]
        polygons = [polygons[pid] for pid, pt in sorted(pts.items(), key=lambda kv: kv[1])]

        cells = gpd.GeoDataFrame({'radar': list(self.radars.values()),
                                  'xy': xy,
                                  'lonlat': list(self.radars.keys()),
                                  },
                                 geometry=polygons,
                                 crs=f'EPSG:{self.epsg_local}')
        cells['boundary'] = cells.geometry.map(lambda x: x.intersects(sink))

        adj = np.zeros((self.N, self.N))
        edges = []
        for i, j in it.combinations(cells.index, 2):
            intersec = cells.geometry.iloc[i].intersection(cells.geometry.iloc[j])
            if type(intersec) is geometry.LineString:
                adj[i, j] = self.distance(xy[i], xy[j],
                                          epsg=self.epsg_local)
                adj[j, i] = adj[i, j]
                edges.append(geometry.LineString((xy[i], xy[j])))

        # create network
        G = nx.from_numpy_matrix(adj, create_using=nx.DiGraph())
        nx.set_node_attributes(G, pd.Series(cells['radar']).to_dict(), 'radar')
        nx.set_node_attributes(G, pd.Series(cells['boundary']).to_dict(), 'boundary')
        nx.set_node_attributes(G, 'measured', name='type')
        nx.set_node_attributes(G, pd.Series(cells['xy'].to_dict()), 'coords')

        # add sink nodes
        for i, row in cells[cells['boundary']].iterrows():
            nidx = len(G)
            G.add_node(nidx, type='sink', radar=row.radar)
            G.add_edge(nidx, i)

        # add self-loops to graph
        G.add_weighted_edges_from([(n, n, 0) for n in G.nodes])

        self.cells = cells
        self.G = G
        self.max_dist = np.max(adj)
        self.edges = gpd.GeoSeries(edges, crs=f'EPSG:{self.epsg_local}')

        # plotting
        if plot:
            fig, ax = plotting.subplot_for_map(figsize=(7, 7))
            plotting.plot_voronoi_polys_with_points_in_area(ax,
                                                            boundary, polygons, xy, poly2pt)
            self.edges.plot(ax=ax)
            gpd.GeoSeries([sink.difference(boundary)], crs=f'EPSG:{self.epsg_local}').plot(ax=ax, color='lightgray')
            fig.show()

        return cells, G

    def subgraph(self, attr, value):
        node_gen = (n for n, data in self.G.nodes(data=True) if data.get(attr) == value)
        subgraph = self.G.subgraph(node_gen)
        return nx.Graph(subgraph)

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
        if epsg == self.epsg_local:
            dist = np.linalg.norm(np.array(coord1) - np.array(coord2))
        elif epsg == '4326':
            dist = geodesic(self.flip(coord1), self.flip(coord2)).kilometers
        else:
            dist = None  # raise error?
        return dist

    def angle(self, coord1, coord2):

        y = coord1[0] - coord2[0]
        x = coord1[1] - coord2[1]

        rad = np.arctan2(y, x)
        deg = np.rad2deg(rad)
        deg = (deg + 360) % 360

        return deg

    def flip(self, coord):
        return (coord[1], coord[0])


# if __name__ == '__main__':
#
#     import datahandling
#     path = '/home/fiona/birdMigration/data/raw/radar'
#     _, radars, _ = datahandling.load_season(path, 'fall', '2015')
#
#     sp = Spatial(radars)
#     sp.voronoi()
#     for index, row in sp.cells.iterrows():
#         area = row.geometry.area / 1000_000
#         partial = row.geometry.buffer(-36_000).area / 1000_000
#         print(row.radar, partial/area)