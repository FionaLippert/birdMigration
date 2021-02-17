
def construct_arrow(coords, data, time):
    lon = coords[0]
    lat = coords[1]

    s = 0.1
    u = s * float(data.sel(time=time, lon=lon, lat=lat).u)
    v = s * float(data.sel(time=time, lon=lon, lat=lat).v)

    norm = np.linalg.norm([ 2 *v, u] ) *5
    norm = np.linalg.norm([ 2 *v, u])
    u_n = u/ (norm * 5)
    v_n = v / (norm * 5)
    line = Polyline(
        locations=[
            [lat - v_n, lon - u_n],
            [lat + v_n, lon + u_n]
        ],
        color="green",
        fill=False,
        # line_cap='square',
        weight=int(1 + np.nan_to_num(norm) * 3)
    )

    s_head = 0.1
    polygon = Polygon(
        locations=[
            [lat + v_n - u * s_head, lon + u_n + 2 * v * s_head],
            [lat + v_n + v * s_head, lon + u_n + u * s_head],
            [lat + v_n + u * s_head, lon + u_n - 2 * v * s_head]
        ],
        color="green",
        fill_color="green",
        fill_opacity=1,
        weight=1
    )

    return line, polygon