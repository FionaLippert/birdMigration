require(bioRad)
require(uvaRadar)
require(lubridate)
require(raster)
require(stars)

library(parallel)
library(MASS)

num_cores <- detectCores()
message(num_cores)

# set credentials for UvA Radar Data Storage
s3_set_key(username = "flippert",
           password = "eFd5cqJqpv8hJN2D")

ts <- as.POSIXct("2016-10-4 21:00", "UTC")

radars <- c("NL/DBL", "NL/DHL") #, "NL/HRW", "BE/JAB", "BE/ZAV", "BE/WID")
param  <- "VID"                             # quantity of interest

system.time(
  keys <- get_keys(radars, ts)
)

names(keys) <- keys
pvol_list <- sapply(keys, function(k) { retrieve_pvol(vp_key_to_pvol(k))},
                    simplify = FALSE, USE.NAMES = TRUE)
message(names(pvol_list))

system.time(
  ppi_list <- mclapply(keys, function(k) {
                  integrate_to_ppi(pvol=pvol_list[[k]],
                                   vp=retrieve_vp(k),
                                   res=500)
                  }, mc.cores=num_cores-1)
)

system.time(
  ppi_list <- lapply(keys, function(k) {
                  integrate_to_ppi(pvol=retrieve_pvol(vp_key_to_pvol(k)),
                                   vp=retrieve_vp(k),
                                   res=500)
                  })
)

system.time(
  composite <- composite_ppi(ppi_list, param=param)
)
