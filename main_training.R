
source("environment.R")
source("input_pipeline.R")
source("controlling.R")
source("model_ResUNet.R") 
source("model_UNet.R") 
source("model_UNet_fast.R") 
source("training.R") 
source("evaluation.R") 


##### Set up environment #####
gc()
memory.size()
memory.limit()
dataset_name<-'GenDS10'
test_dataset_name <-'TestDS'
wd <- './'
imgdir <- paste(wd,'img/',sep="")
maskdir <- paste(wd,"datasets",dataset_name,'/mask/',sep="")
traindir <- paste(wd,"datasets",dataset_name,'/train/',sep="")
testdir <- paste(wd, "datasets",test_dataset_name,"/",sep="")
checkdir <- paste(wd,'check/',sep="")
modeldir <- paste(wd,'models/',sep="")
logdir <-paste(wd,'log/',sep="")
IMG_width=256
IMG_height=256
IMG_bit=8
n_Channels=3
num_classes=1


##Compile Model 
model <-UNet()
model_name <- "UNet"

source("callbacks.R") 
model %>% compile(
  optimizer = optimizer,
  loss = F1Score_loss,
  metrics =  list(Precision, Recall, F1Score))

###Load Model
#model<-load_model_hdf5(paste(modeldir,dataset_name,"/","model_weights.hdf5",custom_objects=TRUE))
#load_model_weights_hdf5(model, filepath="modeldir,dataset_name,"/",model_name.hdf5")

### Model summary ###
summary(model)

###Train Model
model%>%cost_train(stage=1) 
model%>%cost_train(stage=2) 

###Evaluate Model
model%>%cost_eval(stage=2)

### Save Model ###
#keras::save_model_hdf5(model, paste(modeldir,dataset_name,"/",model_name,"_",dataset_name,"_model_",format(Sys.time(),"%Y-%m-%d_%H%M%S"),".hdf5", sep=""), overwrite = TRUE, include_optimizer = TRUE)
#keras::save_model_weights_hdf5(model,paste(modeldir,dataset_name,"/",model_name,"_",dataset_name,"_weights_",format(Sys.time(),"%Y-%m-%d_%H%M%S"),".hdf5",sep=""))


