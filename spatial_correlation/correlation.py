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


def corr(data, names, lag, neighbours_only=False):
    dist_lon = []
    dist_lat = []
    dist = []
    angle = []
    pearsonr = []

    spatial = Spatial(names)
    adj = spatial.voronoi()
    mapping = {ci: idx for idx, ci in enumerate(names.keys())}
    for ci, cj in it.permutations(names.keys(), 2):
        d = geodesic(lonlat(*ci), lonlat(*cj)).kilometers
        if neighbours_only:
            cond = (adj[mapping[ci], mapping[cj]] > 0)
        else:
            cond = True
        # if adj[mapping[ci], mapping[cj]] > 0:
        # if d < 50000:
        if cond:
            dist_lon.append(cj[0] - ci[0])
            dist_lat.append(cj[1] - ci[1])
            dist.append(d)
            angle.append(np.abs(angle_from_latlon(*cj, *ci)))  # from ci to cj
            # print(names[ci], names[cj], ci, cj, angle[-1])

            # corr of earlier time point at ci with later time point at cj
            x = np.nan_to_num(np.roll(np.array(data[ci]).flatten(), lag))
            y = np.nan_to_num(np.array(data[cj]).flatten())

            # only take non-zero measurements into account
            # xy = np.vstack([x, y])
            # xy[xy==0] = np.nan
            # print(xy.shape)
            # xy = xy[:,np.isfinite(xy).all(axis=0)]
            # print(xy.shape)
            r, p = sp.stats.pearsonr(x, y)
            pearsonr.append(r)

    fig, ax = plt.subplots(1, 4, figsize=(18, 3))

    ax[0].scatter(angle, pearsonr, c=dist)
    ax[2].scatter(dist_lat, pearsonr, c=dist)
    # ax[2].scatter(dist, pearsonr)
    ax[0].set_xlabel('angle')
    ax[2].set_xlabel('lat diff')
    # ax[2].set_xlabel('dist')

    ax[2].axvline(dist_lat[np.argmax(pearsonr)], ls='--')

    if neighbours_only:
        bins = np.linspace(-1.5, 1.5, 4)
    else:
        bins = np.linspace(-5.5, 5.5, 12)
    # bins = np.linspace(-5, 5, 6)
    # bins = np.arange(-5.5, 5.5, 1.5)[1:]
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
    ax[3] = sb.boxplot(x="lat diff", y="corr", data=df, color='gray', ax=ax[3])

    if neighbours_only:
        bins = np.linspace(0, 360, 9)[1:]
    else:
        bins = np.linspace(0, 360, 17)[1:]
    # print(len(bins), np.digitize(angle, bins))
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
    axis = fig.add_subplot(1, 4, 2, projection='polar')
    bars = axis.bar(x, means, width=0.3, bottom=0, yerr=std, ecolor='gray')
    # bars = axis.errorbar(bins-(bins[1]-bins[0])/2, means, std)
    # x = (bins-(bins[1]-bins[0])/2)/360 *2*np.pi
    # axis.plot(x, means, c='blue')
    # axis.fill_between(x, means-std, means+std,
    #       alpha=0.5, edgecolor='blue', facecolor='blue')

    # Use custom colors and opacity
    for r, bar in zip(means, bars):
        # bar.set_facecolor(plt.cm.jet(r))
        bar.set_facecolor(plt.cm.jet(r))
        # bar.set_alpha(1)

    axis.set_rlim(0, 1)
    axis.set_theta_zero_location("N")
    axis.set_theta_direction(-1)


def spatio_temp_corr(rcode, radars, names, lags):
    coord_list = list(names.keys())
    names_list = np.array(list(names.values()))

    ci = {v: k for k, v in names.items()}[rcode]
    di = radars[ci]
    sorted_radars = dict(sorted(names.items(), key=lambda x: geodesic(ci, x[0]).kilometers))
    corr = []
    for cj in list(sorted_radars.keys()):
        corr.append([])
        for lag in lags:
            x = resample_shift(radars[cj])

            y = np.roll(resample_shift(radars[ci]), lag)
            if len(x) != len(y):
                r = abs(len(x) - len(y))
                if len(x) < len(y):
                    x = np.concatenate((x, np.zeros(r)))
                else:
                    y = np.concatenate((y, np.zeros(r)))

            # r = np.corrcoef(x, y)[0, 1]
            r = sp.stats.pearsonr(np.nan_to_num(x), np.nan_to_num(y))[0]
            corr[-1].append(r)

    fig, ax = plt.subplots(figsize=(10, 5))
    sb.heatmap(pd.DataFrame(corr), cmap='RdBu_r', ax=ax)
    ax.set(title=rcode, xlabel='Lag [hours]', ylabel='Radars')
    ax.set_yticklabels(list(sorted_radars.values()), rotation=0)
    ax.set_xticklabels(np.array(lags))
