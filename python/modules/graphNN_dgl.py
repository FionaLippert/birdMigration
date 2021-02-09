import dgl
from dgl.data import DGLDataset
from dgl import save_graphs, load_graphs
from dgl.data.utils import makedirs, save_info, load_info
import torch
import numpy as np
import networkx as nx
import spatial, datahandling
import os.path as osp
import os
import pandas as pd


class RadarData(DGLDataset):

    def __init__(self, root, year='2016', season='fall', timesteps=1,
                 force_reload=False,
                 verbose=False,
                 environment_vars = [],
                 vpi_path='/home/fiona/radar_data/vpi/night_only',
                 #processed_dir = '/home/fiona/birdMigration/data',
                 wind_path=None):

        self.root = root
        self.season = season
        self.year = year

        super(RadarData, self).__init__(name='radar_data',
                                        url=None,
                                        raw_dir=osp.join(self.root, 'raw'),
                                        save_dir=osp.join(self.root, 'processed'),
                                        force_reload=force_reload,
                                        verbose=verbose)


    def download(self):
        pass

    def process(self):

        data, radars, t_range = datahandling.load_season(osp.join(self.raw_dir, 'vpi'), self.year, self.season)

        check = np.isfinite(data).all(axis=0)
        dft = pd.DataFrame({'check': np.append(np.logical_and(check[:-1], check[1:]), False),
                                 'tidx': range(len(t_range))}, index=t_range)
        data = data[:, dft.check]
        print(data.shape)

        # construct graph
        space = spatial.Spatial(radars)
        space.voronoi()
        nx_g = space.subgraph('type', 'measured')  # graph without sink nodes

        #print(nx_g.nodes(data=True))
        #print(radars.values())

        coord_dict = nx.get_node_attributes(nx_g, 'coords')
        nx.set_edge_attributes(nx_g, {(u,v): {'norm': 1/len(list(nx_g.neighbors(u))),
                                          'angle': angle(coord_dict[u], coord_dict[v])}
                                      for (u,v) in list(nx_g.edges())})

        G = dgl.from_networkx(nx.DiGraph(nx_g), node_attrs=['coords'], edge_attrs=['norm', 'weight', 'angle'])
        print(G)

        #print(G.edata['angle'][0])

        self.graphs = []
        for t in range(data.shape[-1] - 1):
            G = dgl.from_networkx(nx.DiGraph(nx_g), node_attrs=['coords'], edge_attrs=['norm', 'weight', 'angle'])
            G.ndata['birds'] = torch.tensor(data[..., t], dtype=torch.long)
            G.ndata['gt'] = torch.tensor(data[..., t+1], dtype=torch.long)

            self.graphs.append(G)


    def save(self):
        path = osp.join(self.save_dir, self.year, self.season, f'dgl_graph_{idx}.bin')




def angle(coord1, coord2):
    y = coord1[0] - coord2[0]
    x = coord1[1] - coord2[1]

    rad = np.arctan2(y, x)
    deg = np.rad2deg(rad)
    deg = (deg + 360) % 360

    return deg

if __name__ == '__main__':

    dataset = RadarData('/home/fiona/birdMigration/data', '2016', 'fall', 1)
