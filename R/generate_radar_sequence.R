require(bioRad)
require(uvaRadar)
require(lubridate)
require(raster)
require(stars)
library(jsonlite)
library(parallel)
library(MASS)

num_cores <- detectCores() - 1
num_cores <- 3

# set credentials for UvA Radar Data Storage
s3_set_key(username = "flippert",
           password = "eFd5cqJqpv8hJN2D")

dir <- "data"
if(!dir.exists(dir)){
  dir.create(file.path(getwd(), dir))
}
#filename <- "2016-10"
#path <- file.path(dir, filename)

# time range of interest
ts <- as.POSIXct("2016-10-3 17:00", "UTC")  # POSIXct start time
te <- as.POSIXct("2016-10-3 19:00", "UTC")  # POSIXct end time
tl <- difftime(te, ts, units = "mins")      # total ength in minutes
tr <- 15                                     # time resolution [min]
#tl <- 50                                     # total length [min]
tseq <- seq(0, tl, tr)                      # delta t sequence
#te <- ts + minutes(tl)                      # POSIXct end time

radars <- c("NL/DBL", "NL/DHL", "NL/HRW", "BE/JAB", "BE/ZAV", "BE/WID")
param  <- "VID"                             # quantity of interest
res    <- 500                               # raster resolution [m]


bbox <- st_bbox(get_radars_df(radars)$geometry)
extent_x <- bbox$xmax - bbox$xmin
extent_y <- bbox$ymax - bbox$ymin

reach <- 2 #5
grid <- raster(xmn=bbox$xmin-reach,xmx=bbox$xmax+reach,ymn=bbox$ymin-reach,ymx=bbox$ymax+reach, res=0.01)
img_size <- c(dim(grid)[[2]], dim(grid)[[1]]) # size of final images [pixels]

chunk_size = ceil(length(tseq)/num_cores)

composite_timeseries <- function(job_idx){
    ts_job = ts + minutes((job_idx-1) * tr)
    te_job = min(ts_job + minutes(chunk_size * tr), te)

    message(ts_job, te_job)

    tl_job <- difftime(te_job, ts_job, units = "mins")
    tseq_job <- seq(0, tl_job, tr)

    # for each time step compute vertically integrated radar composite
    l <- list()
    for(dt in tseq) {
      timestamp <- ts_job + minutes(dt)
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

        # TODO: adjust when new bioRad release is available (with res as input)
        composite <- composite_ppi(ppi_list, param=param, dim=img_size)
        l[[as.character(timestamp)]] <- st_as_stars(composite$data)
      }
    }
    strs <- do.call(c, c(l, list(along=3)))
    result <- st_set_dimensions(strs, 3, names="time",
                        values = as.POSIXct(st_get_dimension_values(strs, 3)))

    write_json(as.POSIXct(st_get_dimension_values(strs, 3)), paste0(file.path(dir, offset), ".json"))
    write_stars(result, paste0(file.path(dir, offset), ".tif"), driver = "GTiff")
}

jobs <- seq(1, num_cores)
system.time({
  results <- lapply(jobs, composite_timeseries, mc.cores = num_cores)
})
