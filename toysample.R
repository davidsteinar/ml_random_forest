library(tidyverse)
library(randomForest)

df <- read_csv('ShipData.csv')
df$VoyageId <- NULL

sub <- sample(nrow(df), floor(nrow(df) * 0.8))
set.seed(5434)
training <- df[sub, ]
testing  <- df[-sub,]

model <- randomForest(training$ShaftPower ~ .,data=training[-6])

testing$PredictedShaftPower <- predict(model,testing[-6])

plt <- ggplot(testing,aes(ShaftPower,PredictedShaftPower)) + geom_point() + geom_smooth(method='lm')

print(plt)
print(model)
