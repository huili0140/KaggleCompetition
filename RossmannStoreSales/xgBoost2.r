library(data.table)
library(xgboost)

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

cat("Feature Names...\n")
feature.names <- names(total)[c(1,2,7:9,11:22)]
feature.names

cat("Change text variables to categorical & replacing them with numeric ids...\n")
for (f in feature.names) {
  if (class(total[[f]])=="character") {
    levels <- unique(total[[f]])
    total[[f]] <- as.integer(factor(total[[f]], levels=levels))
  }
}

train <- subset(total, Date < "2015-08-01")
test <- subset(total, Date >= "2015-08-01")


tra <- train[,feature.names]
h <- sample(nrow(train),10000)

RMPSE <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  elab <- exp(as.numeric(labels))-1
  epreds <-exp(as.numeric(preds))-1
  err <- sqrt(mean((epreds/elab-1)^2))
  return(list(metric = "RMPSE", value = err))
}

dval <- xgb.DMatrix(data = data.matrix(tra[h,]), 
					label = log1p(train$Sales[h]))
dtrain <- xgb.DMatrix(data = data.matrix(tra[-h,]),
					label = log1p(train$Sales[-h]))
watchlist <- list(val = dval, train = dtrain)

set.seed(100)
param <- list(  objective = "reg:linear", 	# specify the learning task
                booster = "gbtree",			# select booster from gbtree or gblinear
                eta = 0.015, 				# control the learning rate, 0.06, 0.02, 0.01,
                max_depth = 10, 			# maximum depth of the tree
				        min_child_weight = 5, 		# larger to be more conservative
                subsample = 0.9, 			# sub-sample ratio of the training instances
                colsample_bytree = 0.7	 	# sub-sample ratio of the columns			
)

clf <- xgb.train(	params = param, 
					data = dtrain, 			# xgb.DMatrix training data
					nrounds = 5000, 		# changed from 300
					verbose = 0,			# print information of performance
					early.stop.round = 100,	# stop if performance gets worse after the first n rounds
					watchlist = watchlist,
					maximize = FALSE,		# the lower the evaluation the better
					feval = RMPSE,			# customized evaluation function
					nthreads = 4
)

pred1 <- exp(predict(clf, data.matrix(test[,feature.names]))) -1
submission <- data.frame(Id = test$Id, Sales = pred1)
cat("saving the submission file...\n")
write.csv(submission, "xgboost.15.5000.csv", row.names = FALSE)
