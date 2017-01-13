library(caret)
library(adabag)
libary(randomForest)

set.seed(5499)
df <- read.csv('datasets/sonar.csv')
df$predict <- as.factor(df$predict)

#bootstrap sample
trainIndex <- createDataPartition(df$predict, p = .8, 
                                  list = FALSE, 
                                  times = 1)
training_set <- df[trainIndex,]
testing_set  <- df[-trainIndex,]

#Random forest model 
error <- c()
for (i in c(1:50)){
  fit <- randomForest(predict ~.,
                      data=training_set, 
                      importance=TRUE, 
                      ntree=100,
                      mtry=i)
  yhat <- predict(fit,testing_set[-61])
  
  rf_error <- 1 - sum(yhat == testing_set$predict)/nrow(testing_set)
  error[i] <- rf_error
}
cat(sprintf("random forest error: %f",min(error)))

#adaboost model
adaboost_model <- boosting(predict ~ ., data = training_set, mfinal = 10,
                           control = rpart.control(maxdepth = 1))
adaboost_prediction <- predict.boosting(adaboost_model, newdata = testing_set)

cat(sprintf("adaboost error: %f",adaboost_prediction$error))




