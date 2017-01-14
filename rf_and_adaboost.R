library(caret)
library(randomForest)
library(adabag)

#set.seed(5499)
df <- read.csv('datasets/sonar.csv')
df$predict <- as.factor(df$predict)

error <- c()
for (i in c(1:80)){
  
  #bootstrap sample
  trainIndex <- createDataPartition(df$predict, p = .9, 
                                    list = FALSE, 
                                    times = 1)
  training_set <- df[trainIndex,]
  testing_set  <- df[-trainIndex,]
  
  fit <- randomForest(predict ~.,
                    data=training_set, 
                    importance=TRUE, 
                    ntree=1000,
                    mtry=1)
  yhat <- predict(fit,testing_set[-61])
  
  rf_error <- 1 - sum(yhat == testing_set$predict)/nrow(testing_set)
  error[i] <- rf_error
}
x = c(1:80)
f <- data.frame(x,error)
plt <- ggplot(f,aes(x,error)) + geom_step() + ylim(0,.3) +geom_hline(aes(yintercept = mean(error))) 
print(plt)
print(mean(error))

ada_error <- c()
for(i in c(1:10)){
  trainIndex <- createDataPartition(df$predict, p = .9, 
                                    list = FALSE, 
                                    times = 1)
  training_set <- df[trainIndex,]
  testing_set  <- df[-trainIndex,]
  
#adaboost model
adaboost_model <- boosting(predict ~ ., data = training_set, mfinal = 50)#,control = rpart.control(maxdepth = 1)

adaboost_prediction <- predict.boosting(adaboost_model, newdata = testing_set)
ada_error[i] <- adaboost_prediction$error
}
print(mean(ada_error))

