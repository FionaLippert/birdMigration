#!/bin/bash

parallel -j0 Rscript generate_radar_images.R ::: {1..10}
