library(data.table)
library(randomForest)

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

train <- subset(total, Date < "2015-08-01")
test <- subset(total, Date >= "2015-08-01")

cat("Building the model...\n")
cols <- names(train)[c(1,2,7:9,11:22)]
cols

set.seed(1014)
clf <- randomForest(train[,cols], 
                    log1p(train$Sales),
                    mtry=30,
                    ntree=10,
                    sampsize=50000,
                    do.trace=TRUE)


print(clf)
importance(clf)
plot(clf)
plot(importance(clf), lty=2, pch=16)

cat("Predicting Sales...\n")
pred <- expm1(predict(clf, test[, cols])) 
submission <- data.frame(Id=test$Id, Sales=pred)

cat("Saving the submission file...\n")
write.csv(submission, "randomForest.5.20.100k.csv", row.names = FALSE)
