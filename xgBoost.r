library(data.table)
library(xgboost)

cat("Reading in the datasets...\n")
train <- read.csv("../input/train.csv")
test  <- read.csv("../input/test.csv")
store <- read.csv("../input/store.csv")

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


train <- subset(total, Date < "2015-08-01")
test <- subset(total, Date >= "2015-08-01")



feature.names <- names(train)[c(1,2,7:9,11:22)]
cat("Feature Names\n")
feature.names

cat("assuming text variables are categorical & replacing them with numeric ids\n")
for (f in feature.names) {
  if (class(train[[f]])=="character") {
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
    test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
  }
}

cat("train data column names after slight feature engineering\n")
names(train)
cat("test data column names after slight feature engineering\n")
names(test)
tra<-train[,feature.names]
RMPSE<- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  elab<-exp(as.numeric(labels))-1
  epreds<-exp(as.numeric(preds))-1
  err <- sqrt(mean((epreds/elab-1)^2))
  return(list(metric = "RMPSE", value = err))
}
nrow(train)
h<-sample(nrow(train),10000)

dval<-xgb.DMatrix(data=data.matrix(tra[h,]),label=log(train$Sales+1)[h])
dtrain<-xgb.DMatrix(data=data.matrix(tra[-h,]),label=log(train$Sales+1)[-h])
watchlist<-list(val=dval,train=dtrain)
param <- list(  objective           = "reg:linear", 
                booster = "gbtree",
                eta                 = 0.02, # 0.06, #0.01,
                max_depth           = 10, #changed from default of 8
                subsample           = 0.9, # 0.7
                colsample_bytree    = 0.7 # 0.7
                #num_parallel_tree   = 2
                # alpha = 0.0001, 
                # lambda = 1
)

clf <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 2500, #300, #280, #125, #250, # changed from 300
                    verbose             = 0,
                   early.stop.round    = 100,
                    watchlist           = watchlist,
                    maximize            = FALSE,
                    feval=RMPSE
)
pred1 <- exp(predict(clf, data.matrix(test[,feature.names]))) -1
submission <- data.frame(Id=test$Id, Sales=pred1)
cat("saving the submission file\n")
write_csv(submission, "rf1.csv")
