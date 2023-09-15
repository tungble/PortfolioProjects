# LOAD
source("C:/Users/billl/Documents/School/Babson/Fall 22 Classes/Machine Learning/in class/R files/BabsonAnalytics.R")
library(rpart)
library(rpart.plot)
library(Metrics)
library(caret)
library(lubridate)
df = read.csv("C:/Users/billl/Documents/School/Babson/Fall 22 Classes/Machine Learning/Bikeshare project/train.csv")
df$hours = NULL
View(df)
# MANAGE
df$season = as.factor(df$season)
df$weather = as.factor(df$weather)
df$workingday = as.factor(df$workingday)
df$holiday= as.factor(df$holiday)
df$Time = as.factor(df$Time)
df$Date = mdy(df$Date)
df$month = month(df$Date, label=T) #change month to factor
df$registered[df$registered == 0] = 0.1
df$casual[df$casual == 0] = 0.1 # adding 0.1 to 0
df$weekday = wday(df$Date, label=TRUE) 
# PARTITION
set.seed(1234)
N = nrow(df) 
training_size = round(N*0.6)
set.seed(1234)
training_cases = sample(N, training_size) 
training = df[training_cases, ]
test = df[-training_cases, ]
# BUILD LM_CASUAL
model_lm_casual = lm(casual ~ Time +season +holiday +workingday
                     +weather +temp +atemp +humidity +windspeed +month +weekday, 
                     data=training)
step(model_lm_casual)
model_lm_casual = step(model_lm_casual)
summary(model_lm_casual)
# BUILD RTREE_CASUAL
stopping_rules = rpart.control(minsplit=1,minbucket = 1, cp=-1) # stopping rules
model_rtree_casual = rpart(casual ~ Time +season +holiday +workingday
                           +weather +temp +atemp +humidity +windspeed +month +weekday, 
                           data=training, control=stopping_rules)
model_rtree_casual = easyPrune(model_rtree_casual)
rpart.plot(model_rtree_casual)
# BUILD KNN_CASUAL
trControl <- trainControl(method = 'repeatedcv', 
                          number = 10,   
                          repeats = 5) #cross val
model_knn_casual = knnreg(casual ~ temp +atemp +humidity +windspeed, 
                 data=training, 
                 tuneGrid = expand.grid(k=1:10), 
                 trControl = trControl, 
                 preProc = c('center', 'scale')) #standardized
# BUILD LM_RGST
model_lm_rgst = lm(registered ~ Time +season +holiday +workingday
                   +weather +temp +atemp +humidity +windspeed +month +weekday, 
                   data=training)
step(model_lm_rgst)
model_lm_rgst = step(model_lm_rgst)
summary(model_lm_rgst)
# BUILD RTREE_RGST
model_rtree_rgst = rpart(registered ~ Time +season +holiday +workingday
                         +weather +temp +atemp +humidity +windspeed +month +weekday, 
                         data=training, control=stopping_rules)
model_rtree_rgst = easyPrune(model_rtree_rgst)
rpart.plot(model_rtree_rgst)
# BUILD KNN_RGST 
model_knn_rgst = knnreg(registered ~ temp +atemp +humidity +windspeed,
                 data=training, 
                 tuneGrid = expand.grid(k=1:10), 
                 trControl = trControl,
                 preProc = c('center', 'scale')) #standardized
# BUILD CASUAL_STACK
pred_casual_lm_full = predict(model_lm_casual, df)
pred_casual_rtree_full = predict(model_rtree_casual, df)
pred_casual_knn_full = predict(model_knn_casual, df)
df_casual_stack = cbind(df, pred_casual_lm_full, 
                 pred_casual_rtree_full, 
                 pred_casual_knn_full)
# PREDICT CASUAL_STACK
train_casual_stack = df_casual_stack[training_cases, ] 
test_casual_stack = df_casual_stack[-training_cases, ] #same partition data
model_casual_stack = rpart(casual ~Time +season +holiday +workingday
                      +weather +temp +atemp +humidity +windspeed +month +weekday, 
                      data=train_casual_stack, control=stopping_rules)
model_casual_stack = easyPrune(model_casual_stack)
predictions_casual_stack = predict(model_casual_stack, test_casual_stack)
# BUILD RGST_STACK 
pred_rgst_lm_full = predict(model_lm_rgst, df)
pred_rgst_rtree_full = predict(model_rtree_rgst, df)
pred_rgst_knn_full = predict(model_knn_rgst, df)
df_rgst_stack = cbind(df, pred_rgst_lm_full, 
                 pred_rgst_rtree_full, 
                 pred_rgst_knn_full)
# PREDICT RGST_STACK 
train_rgst_stack = df_rgst_stack[training_cases, ] 
test_rgst_stack = df_rgst_stack[-training_cases, ]
model_rgst_stack = rpart(registered ~Time +season +holiday +workingday
                         +weather +temp +atemp +humidity +windspeed +month +weekday, 
                         data=train_rgst_stack, control=stopping_rules)
model_rgst_stack = easyPrune(model_rgst_stack)
predictions_rgst_stack = predict(model_rgst_stack, test_rgst_stack)
# PREDICT COUNT_STACK
predictions_count_stack = predictions_casual_stack + predictions_rgst_stack
# BENCHMARK COUNT
observations_count = test$count
predictions_count_bench = mean(training$count)
errors_count_bench = observations_count - predictions_count_bench
rmse_count_bench = sqrt(mean(errors_count_bench^2))
mape_count_bench = mean(abs(errors_count_bench/observations_count))
# EVALUATE COUNT_STACK 
errors_count_stack = observations_count - predictions_count_stack
rmse_count_stack = sqrt(mean(errors_count_stack^2))
mape_count_stack = mean(abs(errors_count_stack/observations_count))
# EVALUATE LM_COUNT
predictions_casual_lm = predict(model_lm_casual, test)
predictions_rgst_lm = predict(model_lm_rgst, test)
predictions_count_lm = predictions_casual_lm + predictions_rgst_lm
errors_count_lm = observations_count - predictions_count_lm
rmse_count_lm = sqrt(mean(errors_count_lm^2))
mape_count_lm = mean(abs(errors_count_lm/observations_count))
# EVALUATE RTREE_COUNT
predictions_casual_rtree = predict(model_rtree_casual, test)
predictions_rgst_rtree = predict(model_rtree_rgst, test)
predictions_count_rtree = predictions_casual_rtree + predictions_rgst_rtree
errors_count_rtree = observations_count - predictions_count_rtree
rmse_count_rtree = sqrt(mean(errors_count_rtree^2))
mape_count_rtree = mean(abs(errors_count_rtree/observations_count))
# EVALUATE KNN_COUNT
predictions_casual_knn = predict(model_knn_casual, test)
predictions_rgst_knn = predict(model_knn_rgst, test)
predictions_count_knn = predictions_casual_knn + predictions_rgst_knn
errors_count_knn = observations_count - predictions_count_knn
rmse_count_knn = sqrt(mean(errors_count_knn^2))
mape_count_knn = mean(abs(errors_count_knn/observations_count))








