import numpy as np
import pandas as pd
# from geovoronoi import voronoi_regions_from_coords
from scipy.spatial import Voronoi
import shapely
from shapely import geometry
import pyproj
import itertools as it
import networkx as nx
import geopandas as gpd



class Spatial:
    """
    Spatial object holding radar information and allowing for construction of the associated Voronoi tessellation,
    Delaunay triangulation, and other static node (radar) and edge (Voronoi face) features.
    """

    def __init__(self, radars, seed=1234, buffer=150_000, n_dummy_radars=0, n_helper_points=10):
        """
        Initialization of Spatial object

        :param radars: mapping from radar coordinates (lon, lat) to names
        :param seed: random seed
        :param buffer: radius for outline of boundary cells (in meters)
        :param n_dummy_radars: number of dummy radars, i.e. unobserved boundary cells
        :param n_helper_points: number of points to use to avoid infinite cells in Voronoi tessellation
        """

        self.radars = radars
        self.rng = np.random.default_rng(seed)
        self.buffer = buffer
        self.n_helper_points = n_helper_points

        # setup geodesic (lonlat) coordinate system
        self.epsg_lonlat = '4326'
        self.pts_lonlat = gpd.GeoSeries([geometry.Point(xy) for xy in radars.keys()],
                                        crs=f'epsg:{self.epsg_lonlat}')

        # setup local "azimuthal equidistant" coordinate system
        lat_0 = self.pts_lonlat.y.mean()
        lon_0 = self.pts_lonlat.x.mean()
        self.crs_local = pyproj.Proj(proj='aeqd', ellps='WGS84', datum='WGS84', lat_0=lat_0, lon_0=lon_0).crs
        self.pts_local = self.pts_lonlat.to_crs(self.crs_local)
    
        # add dummy radars if applicable
        self.add_dummy_radars(n_dummy_radars)
        self.N_dummy = n_dummy_radars
        self.N = len(radars) + n_dummy_radars
        self.radar_names = list(self.radars.values()) + [f'boundary_{i}' for i in range(self.N_dummy)]


    def voronoi(self, self_edges=False):
        """
        Construct Voronoi diagram based on radar coordinates

        :param self_edges: add self-edges to graph (booloean)
        :return: cells (polygons describing the Voronoi cells), G (Delaunay triangulation with edge attributes)
        """

        boundary = geometry.MultiPoint(self.pts_local).buffer(self.buffer)
        sink = boundary.buffer(self.buffer).difference(boundary)

        # add helper points to handle infinite edges in Voronoi diagram
        helper_boundary = geometry.MultiPoint(self.pts_local).buffer(3 * self.buffer)
        helper_points = [helper_boundary.interpolate(d) for d in
            np.linspace(0, helper_boundary.length, self.n_helper_points + 1)[:-1]]

        pts_voronoi = pd.concat([self.pts_local, gpd.GeoSeries(helper_points, crs=self.crs_local)], ignore_index=True)

        voronoi = Voronoi(points=self.pts2coords(pts_voronoi))
        lines = [geometry.LineString(voronoi.vertices[line]) for line in voronoi.ridge_vertices if -1 not in line]
        polygons = gpd.GeoDataFrame(geometry=gpd.GeoSeries(shapely.ops.polygonize(lines)), crs=self.crs_local)
        polygons = polygons.intersection(boundary)

        # match polygons with radar locations
        sorted_polygons = [polygons[polygons.contains(point) == True].values[0] for point in self.pts_local]

        xy = self.pts2coords(self.pts_local)
        lonlat = self.pts2coords(self.pts_lonlat)

        cells = gpd.GeoDataFrame({'radar': self.radar_names,
                                  'observed' : [True] * (self.N-self.N_dummy) + [False] * self.N_dummy,
                                  'x': [c[0] for c in xy],
                                  'y': [c[1] for c in xy],
                                  'lon': [c[0] for c in lonlat],
                                  'lat': [c[1] for c in lonlat],
                                  },
                                 geometry=sorted_polygons,
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

        self.cells = cells
        self.G = G
        self.max_dist = np.max(adj)
        self.edges = gpd.GeoSeries(edges, crs=self.crs_local)

        return cells, G


    def add_dummy_radars(self, n):
        """
        Add dummy radars to list of radars.

        :param n: number of dummy radars to add
        """
        if n == 0: return
        boundary = geometry.MultiPoint(self.pts_local).buffer(self.buffer).boundary

        distances = np.linspace(0, boundary.length, n+1)
        points = [boundary.interpolate(d) for d in distances[:-1]]

        print(f'add {n} dummy radars')
        dummy_radars = gpd.GeoSeries([p for p in points], crs=self.crs_local).to_crs(f'epsg:{self.epsg_lonlat}')
        self.pts_lonlat = pd.concat([self.pts_lonlat, dummy_radars], ignore_index=True)
        self.pts_local = pd.concat([self.pts_local, dummy_radars.to_crs(self.crs_local)], ignore_index=True)
        self.pts_local = gpd.GeoSeries(self.pts_local, crs=self.crs_local)


    def G_max_dist(self, max_distance):
        """
        Create graph with edges between any two radars with distance <= max_distance [km]

        :param max_distance: maximum distance for which radars will still be connected
        :return: graph G
        """

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
        """
        Extract subgraph from larger graph G.

        :param G: larger graph
        :param attr: attribute to filter on
        :param value: only nodes where attr==value will be included in subgraph
        :return: subgraph
        """

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
        return coords

    def distance(self, coord1, coord2):
        """
        Compute distance between two geographical locations in local crs

        :param coord1: coordinates of first location (x, y)
        :param coord2: coordinates of second location (x, y)
        :return: distance in meters
        """

        dist = np.linalg.norm(np.array(coord1) - np.array(coord2))
        return dist

    def angle(self, coord1, coord2):
        """
        Compute angle between two geographical locations, i.e. the direction of the line from coord1 to coord2.

        :param coord1: coordinates of first location (x, y)
        :param coord2: coordinates of second location (x, y)
        :return: angle (in degrees north)
        """
        # coords should be in lonlat crs
        y = coord2[0] - coord1[0]
        x = coord2[1] - coord1[1]

        rad = np.arctan2(y, x)
        deg = np.rad2deg(rad)

        # make sure angle is between 0 and 360 degree
        deg = (deg + 360) % 360

        return deg

    def flip(self, coord):
        """
        Flip x and y (or lon and lat) coordinates.

        :param coord: coordinates (x, y)
        :return: flipped coordinates (y, x)
        """
        return (coord[1], coord[0])

