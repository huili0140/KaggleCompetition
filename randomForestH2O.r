library(data.table)
library(h2o)
h2o.init(nthreads=-1, max_mem_size='6G')

cat("Reading in the datasets...\n")
train <- read.csv("train.csv")
test  <- read.csv("test.csv")
store <- read.csv("store.csv")

cat("Formatting the datasets...\n")
train <- train[train$Sales > 0,]  
train$Id <- 0
test$Sales <- 0
test$Customers <- 0
total <- rbind(train, test)

total <- merge(total,store,by="Store")


total$Date <- as.Date(as.character(total$Date), "%Y-%m-%d")
total$Year <- as.numeric(format(total$Date, "%Y"))
total$Month <- as.numeric(format(total$Date, "%m"))
total$Day <- as.numeric(format(total$Date, "%d"))
total$Open[is.na(total$Open)] <- 1
total[is.na(total)] <- -1
total$LogSales <- log1p(total$Sales)

train <- subset(total, Date < "2015-08-01")
test <- subset(total, Date >= "2015-08-01")

cat("Building the model...\n")
cols <- names(train)[c(1,2,7:9,11:22)]
cols


## Load data into cluster from R
trainHex<-as.h2o(train)
## Train a random forest using all default parameters
rfHex <- h2o.randomForest(x=cols,
                          y="LogSales", 
                          ntrees = 200, # 100
                          max_depth = 30, # 20
                          nbins_cats = 1115, ## allow it to fit store ID
                          training_frame=trainHex)

summary(rfHex)
cat("Predicting Sales\n")
## Load test data into cluster from R
testHex<-as.h2o(test)

## Get predictions out; predicts in H2O, as.data.frame gets them into R
predictions<-as.data.frame(h2o.predict(rfHex,testHex))
## Return the predictions to the original scale of the Sales data
pred <- expm1(predictions[,1])
summary(pred)
submission <- data.frame(Id=test$Id, Sales=pred)

cat("saving the submission file\n")
write.csv(submission, "h2o.rf.200.csv",row.names=F)
