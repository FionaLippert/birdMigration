import numpy as np
import scipy as sp
from spatial import Spatial
import itertools as it
from geopy.distance import geodesic, lonlat
from matplotlib import pyplot as plt
import seaborn as sb
import pandas as pd


def all_data(data):
    all_data = []
    for c, d in data.items():
        all_data.append(np.array(d))
    all_data = np.concatenate(all_data)
    return all_data


def angle_from_latlon(lon1, lat1, lon2, lat2):
    y = lon1 - lon2
    x = lat1 - lat2

    brng = np.arctan2(y, x)
    brng = np.rad2deg(brng)
    brng = (brng + 360) % 360

    return brng


def compute_corr(x, y, lag):
    if lag > 0:
        x = x[:x.size - lag]
        y = y[lag:]
    else:
        x = x[-lag:]
        y = y[:y.size + lag]

    x_mask = np.isfinite(x)
    y_mask = np.isfinite(y)
    xy_mask = np.logical_and(x_mask, y_mask)

    x = x[xy_mask]
    y = y[xy_mask]

    r, p = sp.stats.pearsonr(x, y)

    return r, p


def corr(data, names, lag, neighbours_only=False):
    dist_lon = []
    dist_lat = []
    dist = []
    angle = []
    pearsonr = []

    spatial = Spatial(names)
    adj, _ = spatial.voronoi()
    mapping = {ci: idx for idx, ci in enumerate(names.keys())}
    for ci, cj in it.permutations(names.keys(), 2):
        d = geodesic(lonlat(*ci), lonlat(*cj)).kilometers
        if neighbours_only:
            cond = (adj[mapping[ci], mapping[cj]] > 0)
        else:
            cond = True
        if cond:
            dist_lon.append(cj[0] - ci[0])
            dist_lat.append(cj[1] - ci[1])
            dist.append(d)
            angle.append(np.abs(angle_from_latlon(*cj, *ci)))  # from ci to cj

            # corr of earlier time point at ci with later time point at cj
            x = data[ci].values.flatten()
            y = data[cj].values.flatten()

            r, p = compute_corr(x, y, lag)

            pearsonr.append(r)

    fig, ax = plt.subplots(1, 2, figsize=(12, 3))

    # ax[0].scatter(angle, pearsonr, c=dist)
    # ax[2].scatter(dist_lat, pearsonr, c=dist)
    # ax[2].scatter(dist, pearsonr)
    # ax[0].set_xlabel('angle')
    # ax[2].set_xlabel('lat diff')
    # ax[2].set_xlabel('dist')
    # ax[2].axvline(dist_lat[np.argmax(pearsonr)], ls = '--')

    if neighbours_only:
        bins = np.linspace(-1.5, 1.5, 4)
    else:
        bins = np.linspace(-5.5, 5.5, 12)
    A = np.vstack((np.digitize(dist_lat, bins), pearsonr)).T
    means = [np.mean(A[A[:, 0] == i, 1]) for i in range(len(bins))]
    std = [np.std(A[A[:, 0] == i, 1]) for i in range(len(bins))]
    maxr = [np.max(A[A[:, 0] == i, 1]) for i in range(len(bins))]
    # ax[3].errorbar(bins-(bins[1]-bins[0])/2, means, std)
    # ax[3].plot(bins-(bins[1]-bins[0])/2, maxr)

    binsize = (bins[1] - bins[0]) / 2
    binc = bins - binsize
    binc = np.append(binc, bins[-1] + binsize)
    lats = [binc[i] for i in np.digitize(dist_lat, bins)]
    df = pd.DataFrame({'corr': pearsonr, 'lat diff': lats})
    ax[0].axvline(int(bins.size / 2), ls='--', c='lightgray')
    ax[0] = sb.boxplot(x="lat diff", y="corr", data=df, color='gray', ax=ax[0])
    ax[0].set_ylim(0, 1)
    ax[0].set_title(f'Lag = {lag}h')

    if neighbours_only:
        bins = np.linspace(0, 360, 9)[1:]
    else:
        bins = np.linspace(0, 360, 17)[1:]
    A = np.vstack((np.digitize(angle, bins), pearsonr)).T
    means = np.array([np.mean(A[A[:, 0] == i, 1]) for i in range(len(bins))])
    std = np.array([np.std(A[A[:, 0] == i, 1]) for i in range(len(bins))])
    maxr = [np.nanmax(A[A[:, 0] == i, 1]) for i in range(len(bins))]
    # ax[1].errorbar(bins-(bins[1]-bins[0])/2, means, std)
    # ax[1].plot(bins-(bins[1]-bins[0])/2, maxr)

    binsize = (bins[1] - bins[0]) / 2
    binc = bins - binsize
    binc = np.append(binc, bins[-1] + binsize)
    angles = np.array([binc[i] for i in np.digitize(angle, bins)])
    angles = angles / 360 * 2 * np.pi
    df = pd.DataFrame({'corr': pearsonr, 'angle': angles})
    # ax[1] = sb.boxplot(x="angle", y="corr", data=df, color='gray', ax=ax[1])

    ax[1].remove()
    x = (bins - (bins[1] - bins[0]) / 2) / 360 * 2 * np.pi
    axis = fig.add_subplot(1, 2, 2, projection='polar')
    bars = axis.bar(x, means, width=0.3, bottom=0, yerr=std, ecolor='gray')
    # bars = axis.errorbar(bins-(bins[1]-bins[0])/2, means, std)
    # x = (bins-(bins[1]-bins[0])/2)/360 *2*np.pi
    # axis.plot(x, means, c='blue')
    # axis.fill_between(x, means-std, means+std,
    #       alpha=0.5, edgecolor='blue', facecolor='blue')

    # Use custom colors and opacity
    for r, bar in zip(means, bars):
        bar.set_facecolor(plt.cm.jet(r))
        # bar.set_alpha(1)

    axis.set_rlim(0, 1)
    axis.set_theta_zero_location("N")
    axis.set_theta_direction(-1)


def corr_neighbours(data_list, names_list, lags, neighbours_only=True, title='', plot='angle'):
    rows = []

    if neighbours_only:
        bins_lat = np.linspace(-1.5, 1.5, 4)
    else:
        bins_lat = np.linspace(-5, 5, 6)

    bins_angle = np.linspace(0, 360, 9)[1:]

    def binc(bins):
        binsize = (bins[1] - bins[0]) / 2
        binc = np.append(bins - binsize, bins[-1] + binsize)
        return binc

    for data, names in zip(data_list, names_list):
        spatial = Spatial(names)
        adj, _ = spatial.voronoi()
        mapping = {ci: idx for idx, ci in enumerate(names.keys())}
        for ci, cj in it.permutations(names.keys(), 2):
            if neighbours_only:
                cond = (adj[mapping[ci], mapping[cj]] > 0)
            else:
                cond = True
            if cond:
                for lag in lags:
                    x = data[ci].values.flatten()
                    y = data[cj].values.flatten()

                    r, p = compute_corr(x, y, lag)
                    rows.append({'diff lat': binc(bins_lat)[np.digitize(cj[1] - ci[1], bins_lat)],
                                 'angle': binc(bins_angle)[np.digitize(np.abs(angle_from_latlon(*cj, *ci)),
                                                                       bins_angle)] / 360 * 2 * np.pi,
                                 'lag': lag,
                                 'corr': r})

    df = pd.DataFrame(rows)

    if plot == 'lat':
        fig, ax = plt.subplots(1, 1, figsize=(18, 3))
        fig.suptitle(title)
        ax = sb.boxplot(x="lag", y="corr", data=df, hue="diff lat", ax=ax)
        ax.set_ylim(0, 1.1)
        ax.legend(loc='upper right', title='difference in latitude [deg]', bbox_to_anchor=(1.0, 1.0),
                  ncol=len(lags) + 1)

    if plot == 'angle':
        fig, ax = plt.subplots(2, int(len(lags) / 2), figsize=(len(lags) * 2, 8), subplot_kw={'projection': 'polar'})
        for idx, lag in enumerate(lags):
            df_lag = df[df.lag == lag]
            df_lag = df_lag.groupby('angle')

            # x = (bins_angle-(bins_angle[1]-bins_angle[0])/2)/360 *2*np.pi
            means = df_lag.mean()['corr']
            i, j = int(lag / int(len(lags) / 2)), lag % int(len(lags) / 2)
            bars = ax[i, j].bar(df_lag.mean().index, means, width=0.3, bottom=0, yerr=df_lag.std()['corr'],
                                ecolor='gray')
            for r, bar in zip(means, bars):
                bar.set_facecolor(cm.jet(r))

            ax[i, j].set_rlim(0, 1)
            ax[i, j].set_theta_zero_location("N")
            ax[i, j].set_theta_direction(-1)
            ax[i, j].set_title(f'lag = {lag}h', pad=20)
            # ax[i].colorbar()

        cmap = cm.jet
        # ax[-1].remove()
        # axis = fig.add_subplot(1, len(lags), len(lags))
        axis = fig.add_axes([0.95, 0.25, 0.02, 0.5])
        cb = colorbar.ColorbarBase(axis, cmap=cmap,
                                   orientation='vertical')
        cb.set_label('correlation coefficient')

        plt.subplots_adjust(wspace=0.3)
        plt.subplots_adjust(hspace=0.4)
    return fig


def corr_matrix(data, names, lag=1):
    coord_list = sorted(list(names.keys()), key=lambda x: x[1], reverse=True)
    codes = [names[ci] for ci in coord_list]
    N = len(data)
    corr = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            x = data[coord_list[i]].values.flatten()
            y = data[coord_list[j]].values.flatten()

            r, p = compute_corr(x, y, lag)
            corr[i, j] = r

    fig, ax = plt.subplots(figsize=(10, 8))
    sb.heatmap(pd.DataFrame(corr), cmap='RdBu_r', ax=ax)
    ax.set(title=f'cross-correlation with lag={lag}', ylabel='time t', xlabel=f'time t+{lag}')
    ax.set_yticklabels(codes, rotation=0)
    ax.set_xticklabels(codes, rotation=90)


#     spatial = Spatial(names)
#     adj, _ = spatial.voronoi()
#     mapping = {ci : idx for idx, ci in enumerate(names.keys())}

#     for i in range(N):
#         for j in range(N):
#             dist = adj[mapping[coord_list[i]], mapping[coord_list[j]]]
#             if dist > 0:
#                 text = ax.text(j, i, '1',
#                                ha="left", va="top", color="black")

def spatio_temp_corr(rcode, data, names, lags):
    coord_list = list(names.keys())
    names_list = np.array(list(names.values()))

    spatial = Spatial(names)
    adj, _ = spatial.voronoi()
    mapping = {ci: idx for idx, ci in enumerate(names.keys())}

    ci = {v: k for k, v in names.items()}[rcode]
    di = data[ci]
    sorted_radars = dict(sorted(names.items(), key=lambda x: geodesic(lonlat(*ci), lonlat(*x[0])).kilometers))
    corr = []
    codes = []
    for cj in list(sorted_radars.keys()):
        if adj[mapping[ci], mapping[cj]] > 0:
            codes.append(f'** {names[cj]}')
        else:
            codes.append(names[cj])
        corr.append([])
        for lag in lags:
            x = data[ci].vlaues.flatten()
            y = data[cj].values.flatten()

            r, p = compute_corr(x, y, lag)
            corr[-1].append(r)

    fig, ax = plt.subplots(figsize=(10, 5))
    sb.heatmap(pd.DataFrame(corr), cmap='RdBu_r', ax=ax)
    ax.set(title=rcode, xlabel='Lag [hours]', ylabel='Radars')
    ax.set_yticklabels(codes, rotation=0)
    ax.set_xticklabels(np.array(lags))