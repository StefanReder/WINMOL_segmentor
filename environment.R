#install.packages("imager")
#install.packages("tensorflow")
#install.packages("keras")
#install.packages("tidyverse")
#install.packages("tfdatasets")
#install.packages("rsample")
#install.packages("foreach")
#install.packages("doParallel")
#install.packages("reticulate")
#install.packages("magick")
#install.packages("raster")
#install.packages("tfio")
#install.packages("rgdal")


library(tensorflow)
library(keras)
library(tidyverse)
library(tfdatasets)
library(rsample)
library(foreach)
library(doParallel)
library(reticulate)
library(magick)
library(raster)
library(tfio)

#install_tensorflow(version = "gpu")

gpu <- tf$config$experimental$get_visible_devices('GPU')[[1]]
tf$config$experimental$set_memory_growth(device = gpu, enable = TRUE)

#policy <-tf$keras$mixed_precision$Policy('mixed_float16')
#tf$keras$mixed_precision$set_global_policy(policy)

tf$xla$experimental$jit_scope()
