library(caret)
library("adabag")

set.seed(5499)
df <- read.csv('datasets/iris.csv')
df$Species <- as.factor(df$Species)

#bootstrap sample
trainIndex <- createDataPartition(iris$Species, p = .8, 
                                  list = FALSE, 
                                  times = 1)
training_set <- df[trainIndex,]
testing_set  <- df[-trainIndex,]

#Random forest model 
rf_model <- train(Species ~., data=training_set,
                  method = 'rf')
rf_prediction <- predict(rf_model,testing_set)

rf_error <- 1 - sum(rf_prediction == testing_set$Species)/nrow(testing_set)
cat(sprintf("random forest error: %f \n", rf_error))

#adaboost model
adaboost_model <- boosting(Species ~ ., data = training_set, mfinal = 10,
                           control = rpart.control(maxdepth = 1))
adaboost_prediciton <- predict.boosting(adaboost_model, newdata = testing_set)

cat(sprintf("adaboost error: %f",adaboost_prediciton$error))

