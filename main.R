
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
dataset_name<-'DS2only'
model_name="DS2only"
tempdir <- 'c:/temp/'
imgdir <- paste(tempdir,'img/',sep="")
maskdir <- paste(tempdir,dataset_name,'/mask/',sep="")
traindir <- paste(tempdir,dataset_name,'/train/',sep="")
checkdir <- paste(tempdir,'check/',sep="")
outputdir <- paste(tempdir,'output/',sep="")
logdir <-paste(tempdir,'log/',sep="")
IMG_width=256
IMG_height=256
IMG_bit=8
n_Channels=3
num_classes=1


##Compile Model 
model <-UNet_fast()
model_name="UNet_fast"


source("callbacks.R") 
model %>% compile(
  optimizer = optimizer,
  loss = F1Score_loss,
  metrics =  list(Precision, Recall, F1Score))

###Load Model
#model<-load_model_hdf5(paste(outputdir,dataset_name,"/","UNet_aug10_weights_2021-03-24_222154.hdf5",custom_objects=TRUE))
#load_model_weights_hdf5(model, filepath="C:/temp/output/mistel/UNet_mistel_2021-03-30_233725_weights.train1.03-0.44.hdf5")

### Model summary ###
summary(model)

###Train Model
model%>%cost_train(stage=1) 
model%>%cost_train(stage=2) 

###Evaluate Model
model%>%cost_eval(stage=2)

### Save Model ###
#keras::save_model_hdf5(model, paste(outputdir,dataset_name,"/",model_name,"_",dataset_name,"_model_",format(Sys.time(),"%Y-%m-%d_%H%M%S"),".hdf5", sep=""), overwrite = TRUE, include_optimizer = TRUE)
#keras::save_model_weights_hdf5(model,paste(outputdir,dataset_name,"/",model_name,"_",dataset_name,"_weights_",format(Sys.time(),"%Y-%m-%d_%H%M%S"),".hdf5",sep=""))

maskdir <- paste("D:/temp/Data_masters_project/aug100",'/mask/',sep="")
traindir <- paste("D:/temp/Data_masters_project/aug100",'/train/',sep="")


train_data <- tibble(
  img=list.files(traindir, full.names=TRUE, pattern="\\.jpeg$"),
  mask=list.files(maskdir, full.names=TRUE, pattern="\\.gif$"))
train_data <- initial_split(train_data, prop = 0.8)

train_set <- create_dataset(training(train_data), train=TRUE)
val_set <- create_dataset(testing(train_data), train=FALSE)



load_model_weights_hdf5(model, filepath="C:/temp/output_rep/GenDS10/UNet_fast_GenDS10_2021-08-13_110403_weights.train1.22-0.022.hdf5")
model%>%cost_eval(stage=1)

load_model_weights_hdf5(model, filepath="C:/temp/output_rep/GenDS50/UNet_fast_GenDS50_2021-08-13_122532_weights.train1.20-0.018.hdf5")
model%>%cost_eval(stage=1)

load_model_weights_hdf5(model, filepath="C:/temp/output_rep/GenDS100/UNet_fast_GenDS100_2021-08-22_072809_weights.train1.{epoch02}-{val_loss.02}.hdf5")
model%>%cost_eval(stage=1)

load_model_weights_hdf5(model, filepath="C:/temp/output_rep/Baseline/res_org/UNet_fast_DS2only_2021-10-08_165057_weights.train2.07-0.37.hdf5")
model%>%cost_eval(stage=2)
load_model_weights_hdf5(model, filepath="C:/temp/output_rep/GenDS10/res_org/UNet_fast_GenDS10_2021-08-16_103008_weights.train2.43-0.27.hdf5")
model%>%cost_eval(stage=2)
load_model_weights_hdf5(model, filepath="C:/temp/output_rep/GenDS50/res_org/UNet_fast_GenDS50_2021-08-16_104414_weights.train2.34-0.32.hdf5")
model%>%cost_eval(stage=2)
load_model_weights_hdf5(model, filepath="C:/temp/output_rep/GenDS100/res_org/UNet_fast_GenDS100_2021-08-16_105645_weights.train2.43-0.31.hdf5")
model%>%cost_eval(stage=2)
