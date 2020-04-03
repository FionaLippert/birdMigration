#!/bin/bash

Rscript setup_image_generation.R
parallel -j0 Rscript generate_radar_images.R ::: {1..10}
