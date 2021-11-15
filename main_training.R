
source("environment.R")
source("input_pipeline.R")
source("controlling.R")
source("training.R") 
source("evaluation.R") 


##### Set up environment #####
#Please update this section according to your data structure 
#and select a training and test dataset. 
#If you use the prepared folder stucture, you can leave the settings on default.
#And you only have to update the name of the datasets.
gc()
memory.size()
memory.limit()
dataset_name<-'GenDS10'
test_dataset_name <-'TestDS'
wd <- './'
imgdir <- paste(wd,'img/',sep="")
maskdir <- paste(wd,"datasets/",dataset_name,'/mask/',sep="")
traindir <- paste(wd,"datasets/",dataset_name,'/train/',sep="")
testdir <- paste(wd, "datasets/",test_dataset_name,"/",sep="")
checkdir <- paste(wd,'check/',sep="")
modeldir <- paste(wd,'models/',sep="")
logdir <-paste(wd,'log/',sep="")
IMG_width=256
IMG_height=256
IMG_bit=8
n_Channels=3
num_classes=1


##Select a network architecture and compile the model 
#To add a costum network architecture, you have to define a function
#in a seperate R file and change the following lines, accordingly.
source("model_UNet.R") 
model <- UNet()
model_name <- "UNet"

#Compilation of the model
source("callbacks.R") 
model %>% compile(
  optimizer = optimizer,
  loss = F1Score_loss,
  metrics =  list(Precision, Recall, F1Score))

###Load Model
#The following lines are predifined to load model weight, or to load a model 
#from the predifined model folder. Just update the filename.
#model<-load_model_hdf5(paste(modeldir,dataset_name,"/","model_weights.hdf5",custom_objects=TRUE))
#load_model_weights_hdf5(model, filepath="modeldir,dataset_name,"/",model_name.hdf5")

### Model summary ###
summary(model)

###Train Model
#The model is trained in two stages, first with a generic dataset (GenDS) 
#defined in the header, second with the specific dataset (SpecDS) 
model%>%cost_train(stage=1) 
model%>%cost_train(stage=2) 

###Evaluate Model
#The classification performance is evaluated with an independent test dataset
model%>%cost_eval(stage=2)

### Save Model ###
#Use the following lines to save the model or the weight to the model folder
#keras::save_model_hdf5(model, paste(modeldir,dataset_name,"/",model_name,"_",dataset_name,"_model_",format(Sys.time(),"%Y-%m-%d_%H%M%S"),".hdf5", sep=""), overwrite = TRUE, include_optimizer = TRUE)
#keras::save_model_weights_hdf5(model,paste(modeldir,dataset_name,"/",model_name,"_",dataset_name,"_weights_",format(Sys.time(),"%Y-%m-%d_%H%M%S"),".hdf5",sep=""))

#######Now you can use main_prediction to use the trained model####### 
####for the semantic segmentation of tree stems on UAV-orthophotos####
