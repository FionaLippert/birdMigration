require(bioRad)
require(uvaRadar)
require(lubridate)
require(raster)
require(stars)

library(parallel)
library(MASS)

num_cores <- detectCores()

# set credentials for UvA Radar Data Storage
s3_set_key(username = "flippert",
           password = "eFd5cqJqpv8hJN2D")

# time range of interest
ts <- as.POSIXct("2016-10-3 17:00", "UTC")  # POSIXct start time
tr <- 15                                     # time resolution [min]
tl <- 50                                     # total length [min]
tseq <- seq(0, tl, tr)                      # delta t sequence
te <- ts + minutes(tl)                      # POSIXct end time

radars <- c("NL/DBL", "NL/DHL", "NL/HRW", "BE/JAB", "BE/ZAV", "BE/WID")
param  <- "VID"                             # quantity of interest
res    <- 500                               # raster resolution [m]


bbox <- st_bbox(get_radars_df(radars)$geometry)
extent_x <- bbox$xmax - bbox$xmin
extent_y <- bbox$ymax - bbox$ymin

reach <- 5
grid <- raster(xmn=bbox$xmin-reach,xmx=bbox$xmax+reach,ymn=bbox$ymin-reach,ymx=bbox$ymax+reach, res=0.01)
img_size <- c(dim(grid)[[2]], dim(grid)[[1]]) # size of final images [pixels]

# for each time step compute vertically integrated radar composite
l <- list()
for(dt in tseq) {
  timestamp <- ts + minutes(dt)
  keys <- get_keys(radars, timestamp)
  message(timestamp)

  # apply vertical integration to all available radars at time t=ts+dt
  # and combine the resulting ppi's into composite raster

  # TODO: load pvol and vp prior to parallel execution of vertical integration
  if(length(keys) > 0) {
    names(keys) <- keys
    pvol_list <- sapply(keys, function(k) { retrieve_pvol(vp_key_to_pvol(k))},
                        simplify = FALSE, USE.NAMES = TRUE)

    vp_list <- sapply(keys, retrieve_vp, simplify = FALSE, USE.NAMES = TRUE)

    ppi_list <- lapply(keys, function(k) {
                    integrate_to_ppi(pvol=pvol_list[[k]],
                                     vp=vp_list[[k]],
                                     raster=grid)
                    })#, mc.cores=num_cores-1)

    #message(ppi_list)

    # TODO: adjust when new bioRad release is available (with res as input)
    composite <- composite_ppi(ppi_list, param=param, dim=img_size)
    l[[as.character(timestamp)]] <- st_as_stars(composite$data)

    #bm <- download_basemap(composite)
    #png(as.character(timestamp))
    #map(composite, bm)
  }
}
strs <- do.call(c, c(l, list(along=3)))
result <- st_set_dimensions(strs, 3, names="time",
                    values = as.POSIXct(st_get_dimension_values(strs, 3)))

write_stars(result, "2016-10-3_200min.tif", driver = "GTiff")
write_stars(result, "2016-10-3_200min.nc")

#png("composite_ts.png")

# plot first attribute (result only has one, namely "param") at time point ts
#plot(result[1,,,1])

#print("done")
#dev.off()
#browseURL("composite_ts.png")
