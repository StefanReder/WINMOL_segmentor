
cost_train <- function(model, stage) {

cl <- makePSOCKcluster(10) 

clusterEvalQ(cl, {
  
#include input_pipeline.R
 source("input_pipeline.R")
})

if (stage==1){
registerDoParallel(cl)

#Build training and validation data sets
train_data <- tibble(
  img=list.files(traindir, full.names=TRUE, pattern="\\.jpeg$"),
  mask=list.files(maskdir, full.names=TRUE, pattern="\\.gif$"))
train_data <- initial_split(train_data, prop = 0.8)

train_set <- create_dataset(training(train_data), train=TRUE)
val_set <- create_dataset(testing(train_data), train=FALSE)

####  Fit the model to the data 

model %>% fit(
  train_set,
  validation_data=val_set,
  epochs = 100, 
  verbose = 1,
  callbacks = callbacks_train1)
stopCluster(cl)
gc()
}



if (stage==2) {
registerDoParallel(cl)  
train_data <- tibble(
  img=list.files(paste(wd,"datasets/SpecDS/train/", sep=""), full.names=TRUE, pattern="\\.jpeg$"),
  mask=list.files(paste(wd,"datasets/SpecDS/mask/", sep=""), full.names=TRUE, pattern="\\.gif$"))
train_data <- initial_split(train_data, prop = 0.8)

train_set <- create_dataset(training(train_data), train=TRUE)
val_set <- create_dataset(testing(train_data), train=FALSE)

model %>% fit(
  train_set,
  validation_data=val_set,
  epochs = 100, 
  verbose = 1,
  callbacks = callbacks_train2)
stopCluster(cl)
gc()
}


gc()


}