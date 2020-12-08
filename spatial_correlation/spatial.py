import numpy as np
import geopandas as gpd
from geovoronoi import voronoi_regions_from_coords, plotting
from shapely import geometry
from geopy.distance import geodesic, lonlat
import itertools as it

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
        coords = self.pts2coords(self.pts_local)
        boundary = geometry.MultiPoint(self.pts_local).buffer(buffer)
        polygons, pts, poly2pt = voronoi_regions_from_coords(coords, boundary)

        idx = np.array(poly2pt).flatten().argsort()
        cells = gpd.GeoSeries(np.array(polygons)[idx],
                              crs=f'EPSG:{self.epsg_local}')

        adj = np.zeros((self.N, self.N))
        edges = []
        for i, j in it.combinations(cells.index, 2):
            intersec = cells.iloc[i].intersection(cells.iloc[j])
            if type(intersec) is geometry.LineString:
                adj[i, j] = self.distance(coords[i], coords[j],
                                          epsg=self.epsg_local)
                adj[j, i] = adj[i, j]
                edges.append(geometry.LineString((coords[i], coords[j])))

        self.cells = cells
        self.adj = adj
        self.max_dist = np.max(adj)
        self.edges = gpd.GeoSeries(edges, crs=f'EPSG:{self.epsg_local}')

        if plot:
            fig, ax = plotting.subplot_for_map(figsize=(7, 7))
            plotting.plot_voronoi_polys_with_points_in_area(ax,
                                                            boundary, polygons, coords, poly2pt)
            self.edges.plot(ax=ax)
            plt.show()

        return adj, cells

    def pts2coords(self, pts, reverse_xy=False):
        """
        Convert point objects to [x, y] coordinates
        Args:
            reverse_lonlat (bool): if true, the order of the two coordinates is switched
        Returns:
            coords: np.array of projected coordinates
        """
        coords = np.array([[p.xy[0][0], p.xy[1][0]] for p in pts])
        if reverse_xy:
            coords = np.flip(coords, axis=1)
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
            dist = geodesic(self.flip(coord1), self.flip(coord2)).meters
        else:
            dist = None  # raise error?
        return dist

    def flip(self, coord):
        return (coord[1], coord[0])