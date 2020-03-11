require(bioRad)
require(uvaRadar)
require(lubridate)
require(raster)
require(stars)

# set credentials for UvA Radar Data Storage
s3_set_key(username = "flippert",
           password = "eFd5cqJqpv8hJN2D")

# time range of interest
ts <- as.POSIXct("2016-10-4 21:00", "UTC")
te <- ts+minutes(7)

radars <- c("NL/DBL", "BE/JAB")
grid <- raster(xmn=1,xmx=10,ymn=49,ymx=56, res=.01)
radarList <- list()
for(r in radars) {
  keys <- get_keys(r, ts  %--%te)
  message(r)
  message(keys)
  l <- list()
  for(k in keys) {
    res <- integrate_to_ppi(raster = grid,
              pvol = retrieve_pvol(vp_key_to_pvol(k)), vp=retrieve_vp(k))
    l[[k]] <- st_as_stars(res$data)
  }
  rr <- do.call(c, c(l, list(along=3)))
  t <- key_to_timestamp(st_get_dimension_values(rr, 3))
  rr <- st_set_dimensions(rr, 3, names="time", values=as.POSIXct(t))
  radarList[[r]] <- rr
}

png("mygraphic.png")

strs <- do.call(c,c(radarList, list(along=4)))
result <- st_set_dimensions(strs, 4, names="radar",
              values=st_get_dimension_values(strs, 4))
plot(result)

print("done")
dev.off()
browseURL("mygraphic.png")

#warnings()
