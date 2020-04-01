require(bioRad)
require(uvaRadar)
require(lubridate)
require(raster)
require(stars)
require(jsonlite)
require(parallel)
require(MASS)
require(numform)
require(yaml)

num_cores <- detectCores() - 1

config = yaml.load_file("config.yml")

# set credentials for UvA Radar Data Storage
s3_set_key(username = config$login$username,
           password = config$login$password)

ts <- as.POSIXct(config$ts, "UTC")  # POSIXct start time
te <- as.POSIXct(config$te, "UTC")  # POSIXct end time
tl   <- as.numeric(difftime(te, ts, units = "mins"))    # total length in minutes
tseq <- seq(0, tl, config$tr)                                  # delta t sequence

bbox <- st_bbox(get_radars_df(config$radars)$geometry)
extent_x <- bbox$xmax - bbox$xmin
extent_y <- bbox$ymax - bbox$ymin

reach <- 5
grid <- raster(xmn=bbox$xmin-reach,xmx=bbox$xmax+reach,ymn=bbox$ymin-reach,ymx=bbox$ymax+reach, res=config$res)
img_size <- c(dim(grid)[[2]], dim(grid)[[1]]) # size of final images [pixels]

chunk_size <- ceiling(length(tseq)/num_cores)

if(!dir.exists(config$data$tiff)){
  dir.create(config$data$tiff, recursive=TRUE)
}
subdir <- file.path(config$data$tiff, paste(ts, "-", te))
if(!dir.exists(subdir)){
  dir.create(subdir)
}

config_file <- file(file.path(subdir, "config.yml"), open="w")
write_yaml(config, config_file)


composite_timeseries <- function(job_idx){

    tryCatch(
      {
        ts_job = ts + minutes((job_idx-1) * (chunk_size) * config$tr)
        te_job = min(ts_job + minutes((chunk_size-1) * config$tr), te)

        if(ts_job <= te){
          path <- file.path(subdir, formatC(job_idx,
                            width=floor(log10(num_cores))+1, flag="0"))
          if(!dir.exists(path)){
            dir.create(path)
          }
          log <- file(file.path(path, "log.txt"), open="w")
          sink(log, type='message', append=TRUE)

          message(paste(ts_job, te_job))

          tl_job <- as.numeric(difftime(te_job, ts_job, units = "mins"))
          tseq_job <- seq(0, tl_job, config$tr)

          # for each time step compute vertically integrated radar composite
          l <- list()
          for(dt in tseq_job) {
            timestamp <- ts_job + minutes(dt)
            keys <- get_keys(config$radars, timestamp)
            message(timestamp)

            # apply vertical integration to all available radars at time t=ts+dt
            # and combine the resulting ppi's into composite raster

            # TODO: load pvol and vp prior to parallel execution of vertical integration
            if(length(keys) > 0) {
              names(keys) <- keys
              pvol_list <- sapply(keys, function(k) { retrieve_pvol(vp_key_to_pvol(k))},
                            simplify = FALSE, USE.NAMES = TRUE)

              vp_list <- sapply(keys, retrieve_vp, simplify = FALSE, USE.NAMES = TRUE)

              # TODO; fix error arising when vp$data[['eta']] is all NAN (with BE/WID at 2016-10-01 00:15)
              # maybe fix it by using 'dens' instead of 'eta'?
              ppi_list <- lapply(keys, function(k) {
                        integrate_to_ppi(pvol=pvol_list[[k]],
                                         vp=vp_list[[k]],
                                         raster=grid)
                        })#, mc.cores=num_cores-1)

              # TODO: adjust when new bioRad release is available (with res as input)
              composite <- composite_ppi(ppi_list, param=config$quantity, dim=img_size)
              l[[as.character(timestamp)]] <- st_as_stars(composite$data)
            }
          }

          strs <- do.call(c, c(l, list(along=3)))

          fname_t <- file.path(path, "timestamps.json")
          fname_d <- file.path(path, paste0(config['quantity'], ".tif"))

          if(length(l)>1){
            #result <- st_set_dimensions(strs, 3, names="time",
            #                 values = as.POSIXct(st_get_dimension_values(strs, 3)))
            write_json(as.POSIXct(st_get_dimension_values(strs, 3)), fname_t)
            write_stars(strs, fname_d, driver = "GTiff")
          }else{
            write_json(ts_job, fname_t)
            write_stars(strs, fname_d, driver = "GTiff")
          }
        }
      },
      error = function(e) print(e),
      finally = {
        sink(NULL,type='message')
      }
    )
}

jobs <- seq(1, num_cores)
system.time({
  results <- mclapply(jobs, composite_timeseries, mc.cores = num_cores)
})
