library(tidyverse)
library(DescTools)
library(rpart.plot)
library(ggplot2)
library(dplyr)
library(cowplot)
library(mice)
library(FSelector)
library(arules)
library(tibble)
library(rattle)
library(smotefamily)
library(caret)
library(corrplot)

set.seed(0)
options("scipen"=100, "digits"=5)

new_card_data <- read.csv("C:\\Users\\Olan\\Documents\\Olan\\3rd_year\\2nd_semester\\Data_Mining\\Case_study_Algos\\new_card_data.csv")
new_card_data$loan_status <- factor(new_card_data$loan_status)
new_card_data$initial_list_status <- factor(new_card_data$initial_list_status)

library(RSBID)
#data.smotenc <- SMOTE_NC(new_card_data, 'loan_status')
data.smotenc <- read.csv("datasmotenc.csv")
data.smotenc$loan_status <- factor(data.smotenc$loan_status)
data.smotenc$initial_list_status <- factor(data.smotenc$initial_list_status)


########## SPLITTING ###########
inTrain <- createDataPartition(y = new_card_data$loan_status, p = .7, list = FALSE)
new_card_data_train <- new_card_data %>% slice(inTrain)
new_card_data_test <- new_card_data %>% slice(-inTrain)
inTrainB <- createDataPartition(y = data.smotenc$loan_status, p = .5, list = FALSE)
train.smotenc <- data.smotenc %>% slice(inTrainB)
test.smotenc <- data.smotenc %>% slice(-inTrainB)

#k is number of fold in cross validation
train_index <- createFolds(new_card_data_train$loan_status, k = 10)

#Enable parallel processing to speed up training
library(doParallel)
cl <- makePSOCKcluster(5)
registerDoParallel(cl)


##  !!!   Replace data.smotenc with new_card_data.
##  !!!    for balanced and unbalanced respectively

##### RPART Decision Tree ###########

#k determines number of folds
set.seed(0)
train_index <- createFolds(data.smotenc$loan_status, k = 50)

#fine tuning parameter. seq(start, end, interval) or explicit
#tuneGrid = data.frame(cp = seq(0, 0.08, 0.001)), 
rpartGrid = data.frame(cp = 0)

rpartFit <- data.smotenc %>%
  train(loan_status ~ .,
        data = . ,
        method = "rpart",
        tuneGrid = rpartGrid,
        tuneLength = 5,
        trControl = trainControl(method = "cv",
                                 indexOut = train_index,
                                 savePredictions = "final"))
#rpartFit
confusionMatrix(rpartFit$pred$pred, rpartFit$pred$obs, mode= "everything")

summary(data.smotenc)

#------------- Template for hold-out method ------------#
#NOTE: for different splits, edit the splitting code above

inTrainB <- createDataPartition(y = data.smotenc$loan_status, p = .6, list = FALSE)
train.smotenc <- data.smotenc %>% slice(inTrainB)
test.smotenc <- data.smotenc %>% slice(-inTrainB)

rpartFitH <- train.smotenc %>%
  train(loan_status ~ .,
        data = . ,
        method = "rpart",
        tuneGrid = rpartGrid,
        tuneLength = 5,
        trControl = trainControl(savePredictions = "final"))
pred <- predict(rpartFitH, newdata = test.smotenc)
cm<- confusionMatrix(pred, test.smotenc$loan_status, mode= "everything")

cm$byClass[5:7]
cm$overall[1:2]

########## KNN ##########
#k determines number of folds
train_index <- createFolds(data.smotenc$loan_status, k = 10)

set.seed(0)
knnFit <- data.smotenc %>% train(loan_status ~ .,
                              method = "knn",
                              data = .,
                              tuneLength = 5,
                              #tuneGrid=data.frame(k = 1:10),
                              tuneGrid=data.frame(k = 1), #neigbors parameter
                              trControl = trainControl(method = "cv",
                                                       indexOut = train_index,
                                                       savePredictions = "final"))
#knnFit
confusionMatrix(knnFit$pred$pred, knnFit$pred$obs, mode= "everything")

inTrainB <- createDataPartition(y = data.smotenc$loan_status, p = .9, list = FALSE)
train.smotenc <- data.smotenc %>% slice(inTrainB)
test.smotenc <- data.smotenc %>% slice(-inTrainB)

knnFitH <- train.smotenc %>%
  train(loan_status ~ .,
        data = . ,
        method = "knn",
        tuneLength = 5,
        tuneGrid=data.frame(k = 1),
        trControl = trainControl(savePredictions = "final"))
pred <- predict(knnFitH, newdata = test.smotenc)
cm <- confusionMatrix(pred, test.smotenc$loan_status, mode= "everything")
cm$byClass[5:7]
cm$overall[1:2]


########## PART - Rule-Based Classifier ###########
#k determines number of folds
train_index <- createFolds(data.smotenc$loan_status, k = 20)
set.seed(0)

rulesFit <- data.smotenc %>% train(loan_status ~ .,
                                method = "PART",
                                data = .,
                                tuneLength = 5,
                                trControl = trainControl(method = "cv",
                                                         indexOut = train_index,
                                                         savePredictions = "final"))
rulesFit
rulesFit$finalModel #print decision list
confusionMatrix(rulesFit$pred$pred, rulesFit$pred$obs, mode= "everything")


inTrainB <- createDataPartition(y = data.smotenc$loan_status, p = .9, list = FALSE)
train.smotenc <- data.smotenc %>% slice(inTrainB)
test.smotenc <- data.smotenc %>% slice(-inTrainB)

rulesFitH <- train.smotenc %>%
  train(loan_status ~ .,
        data = . ,
        method = "PART",
        tuneLength = 5,
        trControl = trainControl(savePredictions = "final"))
pred <- predict(rulesFitH, newdata = test.smotenc)
cm<- confusionMatrix(pred, test.smotenc$loan_status, mode= "everything")
cm$byClass[5:7]
cm$overall[1:2]

rulesFitH$finalModel

############ Linear Support-Vector-Machine SVM #########
set.seed(0)
train_index <- createFolds(data.smotenc$loan_status, k = 20)

svmFit <- data.smotenc %>% train(loan_status ~.,
                              method = "svmLinear",
                              data = .,
                              tuneLength = 5,
                              trControl = trainControl(method = "cv",
                                                       indexOut = train_index,
                                                       savePredictions = "final"))
svmFit
#svmFit$finalModel
confusionMatrix(svmFit$pred$pred, svmFit$pred$obs, mode= "everything")


inTrain <- createDataPartition(y = new_card_data$loan_status, p = .9, list = FALSE)
new_card_data_train <- new_card_data %>% slice(inTrain)
new_card_data_test <- new_card_data %>% slice(-inTrain)

svmFitH <- new_card_data_train %>%
  train(loan_status ~ .,
        data = . ,
        method = "svmLinear",
        tuneLength = 5,
        trControl = trainControl(savePredictions = "final"))
pred <- predict(svmFitH, newdata = new_card_data_test)
cm<- confusionMatrix(pred, new_card_data_test$loan_status, mode= "everything")
cm$byClass[5:7]
cm$overall[1:2]

########### Random Forest #############
set.seed(0)
train_index <- createFolds(data.smotenc$loan_status, k = 50)

#create tunegrid with 8 values from 1:15, for mtry to tuning model. Upper value depend on Tune length
#Our train function will change number of entry variable at each split according to tunegrid. 
RFgrid <- expand.grid(.mtry = (1:5))

randomForestFit <- data.smotenc %>% train(loan_status ~ .,
                                                 method = "rf",
                                                 data = .,
                                                 tuneLength = 5,
                                                 tuneGrid = RFgrid,
                                                 trControl = trainControl(method = "cv",
                                                                          indexOut = train_index,
                                                                          savePredictions = "final"))
randomForestFit
#randomForestFit$finalModel

plot(randomForestFit)
confusionMatrix(randomForestFit$pred$pred, randomForestFit$pred$obs, mode= "everything")


inTrain <- createDataPartition(y = new_card_data$loan_status, p = .9, list = FALSE)
new_card_data_train <- new_card_data %>% slice(inTrain)
new_card_data_test <- new_card_data %>% slice(-inTrain)

randomForestFitH <- new_card_data_train %>%
  train(loan_status ~ .,
        data = . ,
        method = "rf",
        tuneLength = 5,
        tuneGrid = RFgrid,
        trControl = trainControl(savePredictions = "final"))
pred <- predict(randomForestFitH, newdata = new_card_data_test)
cm<- confusionMatrix(pred, new_card_data_test$loan_status, mode= "everything")
cm$byClass[5:7]
cm$overall[1:2]
randomForestFitH
########### Naive Bayes #############

set.seed(0)
train_index <- createFolds(data.smotenc$loan_status, k = 50)

nBayesFit <- data.smotenc %>% train(loan_status ~ .,
                                     method = "nb",
                                     data = .,
                                     tuneLength = 5,
                                     #tuneGrid = NBgrid,
                                     trControl = trainControl(method = "cv",
                                                              indexOut = train_index,
                                                              savePredictions = "final"))
#nBayesFit
#nBayesFit$finalModel

confusionMatrix(nBayesFit$pred$pred, nBayesFit$pred$obs, mode= "everything")


inTrain <- createDataPartition(y = new_card_data$loan_status, p = .7, list = FALSE)
new_card_data_train <- new_card_data %>% slice(inTrain)
new_card_data_test <- new_card_data %>% slice(-inTrain)

nBayesFitH <- new_card_data_train %>%
  train(loan_status ~ .,
        data = . ,
        method = "nb",
        tuneLength = 5,
        trControl = trainControl(savePredictions = "final"))
pred <- predict(nBayesFitH, newdata = new_card_data_test)
cm<- confusionMatrix(pred, new_card_data_test$loan_status, mode= "everything")
cm$byClass[5:7]
cm$overall[1:2]

nBayesFitH

############ Artificial Neural Network #############
set.seed(0)
train_index <- createFolds(data.smotenc$loan_status, k = 10)

nnetGrid <-  expand.grid(size = seq(from = 1, to = 18, by = 3),
                         decay = seq(from = 0.1, to = 0.5, by = 0.1))

nnetFit <- data.smotenc %>% train(loan_status ~ .,
                                  method = "nnet",
                                  data = .,
                                  tuneGrid = nnetGrid,
                                  trControl = trainControl(method = "cv",
                                                            indexOut = train_index,
                                                            savePredictions = "final"),
                                  trace = FALSE)
nnetFit
plot(nnetFit)
nnetFit$finalModel

confusionMatrix(nnetFit$pred$pred, nnetFit$pred$obs, mode= "everything")

inTrain <- createDataPartition(y = new_card_data$loan_status, p = .8, list = FALSE)
new_card_data_train <- new_card_data %>% slice(inTrain)
new_card_data_test <- new_card_data %>% slice(-inTrain)

nnetGrid <-  expand.grid(size = seq(from = 1, to = 18, by = 3),
                         decay = seq(from = 0.1, to = 0.5, by = 0.1))

nnetFitH <- new_card_data_train %>%
  train(loan_status ~ .,
        data = . ,
        method = "nnet",
        tuneGrid = nnetGrid,
        trControl = trainControl(savePredictions = "final"))
pred <- predict(nnetFitH, newdata = new_card_data_test)
confusionMatrix(pred, new_card_data_test$loan_status, mode= "everything")

plot(nnetFitH)

#---------------------------    EXXXTRA    --------------------------#
####### Conditional Inference Tree ########
set.seed(0)
train_index <- createFolds(data.smotenc$loan_status, k = 20)
ctreeFit <- data.smotenc %>% train(loan_status ~ .,
                                          method = "ctree",
                                          data = .,
                                          tuneLength = 5,
                                          trControl = trainControl(method = "cv",
                                                                   indexOut = train_index,
                                                                   savePredictions = "final"))
ctreeFit
cm <- confusionMatrix(ctreeFit$pred$pred, ctreeFit$pred$obs, mode= "everything")
cm$byClass[5:7]
cm$overall[1:2]


inTrainB <- createDataPartition(y = data.smotenc$loan_status, p = .6, list = FALSE)
train.smotenc <- data.smotenc %>% slice(inTrainB)
test.smotenc <- data.smotenc %>% slice(-inTrainB)

ctreeFitH <- train.smotenc %>% train(loan_status ~ .,
                                    method = "ctree",
                                    data = .,
                                    tuneLength = 5,
                                    trControl = trainControl(savePredictions = "final"))

pred <- predict(ctreeFitH, newdata = test.smotenc)
cm<- confusionMatrix(pred, test.smotenc$loan_status, mode= "everything")
ctreeFit
cm$byClass[5:7]
cm$overall[1:2]
cm

############## C45 Fit ################
library(RWeka)
set.seed(0)
train_index <- createFolds(data.smotenc$loan_status, k = 50)
C45Fit <- data.smotenc %>% train(loan_status ~ .,
                              method = "J48",
                              data = .,
                              tuneLength = 5,
                              trControl = trainControl(method = "cv",
                                                       indexOut = train_index,
                                                       savePredictions = "final"))
C45Fit
cm <- confusionMatrix(C45Fit$pred$pred, C45Fit$pred$obs, mode= "everything")
cm$byClass[5:7]
cm$overall[1:2]




inTrainB <- createDataPartition(y = data.smotenc$loan_status, p = .9, list = FALSE)
train.smotenc <- data.smotenc %>% slice(inTrainB)
test.smotenc <- data.smotenc %>% slice(-inTrainB)

C45FitH <- train.smotenc %>% train(loan_status ~ .,
                                        method = "J48",
                                        data = .,
                                        tuneLength = 5,
                                        trControl = trainControl(savePredictions = "final"))
pred <- predict(C45FitH, newdata = test.smotenc)
cm<- confusionMatrix(pred, test.smotenc$loan_status, mode= "everything")
C45FitH
cm$byClass[5:7]
cm$overall[1:2]

########## Gradient Boosted Decision Tree ###########
library(xgboost)
set.seed(0)
train_index <- createFolds(data.smotenc$loan_status, k = 50)
xgboostFit <- data.smotenc %>% train(loan_status ~ .,
                                  method = "xgbTree",
                                  data = .,
                                  tuneLength = 5,
                                  trControl = trainControl(method = "cv",
                                                           indexOut = train_index,
                                                           savePredictions = "final"),
                                  tuneGrid = expand.grid(
                                    nrounds = 20,
                                    max_depth = 3,
                                    colsample_bytree = .6,
                                    eta = 0.1,
                                    gamma=0,
                                    min_child_weight = 1,
                                    subsample = .5
                                  ))
xgboostFit
xgboostFit$finalModel
cm <- confusionMatrix(xgboostFit$pred$pred, xgboostFit$pred$obs, mode= "everything")
cm$byClass[5:7]
cm$overall[1:2]


set.seed(0)
inTrainB <- createDataPartition(y = data.smotenc$loan_status, p = .6, list = FALSE)
train.smotenc <- data.smotenc %>% slice(inTrainB)
test.smotenc <- data.smotenc %>% slice(-inTrainB)

xgboostFitH <- train.smotenc %>% train(loan_status ~ .,
                                            method = "xgbTree",
                                            data = .,
                                            tuneLength = 5,
                                            tuneGrid = expand.grid(
                                                                  nrounds = 20,
                                                                  max_depth = 3,
                                                                  colsample_bytree = .6,
                                                                  eta = 0.1,
                                                                  gamma=0,
                                                                  min_child_weight = 1,
                                                                  subsample = .5
                                                                ))
pred <- predict(xgboostFitH, newdata = test.smotenc)
cm<- confusionMatrix(pred, test.smotenc$loan_status, mode= "everything")
xgboostFitH
cm$byClass[5:7]
cm$overall[1:2]
cm
########## MODEL PERFORMANCE COMPARISON ############
resamps <- resamples(list(
  rpart = rpartFit,
  KNN = knnFit,
  rules = rulesFit,
  SVM = svmFit,
  randomForest = randomForestFit,
  nBayes = nBayesFit,
  NeuralNet = nnetFit,
  CTree = ctreeFit,
  C45 = C45Fit,
  GradientBoosted = xgboostFit
))
resamps
summary(resamps)
bwplot(resamps, layout = c(3, 1))

difs <- diff(resamps)
difs
summary(difs)

########## Hold out Eval
resampsH <- resamples(list(
  hrpart = rpartFitH,
  HKNN = knnFitH,
  Hrules = rulesFitH,
  HSVM = svmFitH,
  HrandomForest = randomForestFitH,
  HnBayes = nBayesFitH,
  HNeuralNet = nnetFitH,
  HCTree = ctreeFitH,
  HC45 = C45FitH,
  HGradientBoosted = xgboostFitH
))
resampsH
summary(resampsH)
bwplot(resampsH, layout = c(3, 1))

difs <- diff(resampsH)
difs
summary(difs)
