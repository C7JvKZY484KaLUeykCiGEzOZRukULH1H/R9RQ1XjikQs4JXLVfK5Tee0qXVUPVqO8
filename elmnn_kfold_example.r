# https://github.com/C7JvKZY484KaLUeykCiGEzOZRukULH1H/R9RQ1XjikQs4JXLVfK5Tee0qXVUPVqO8/blob/master/elmnn_kfold_example.r
# 
# genetic algorithm is used to find the best parameters for elmnn model
# using kfold for validation to overcome overfit
# 
# this particular example is suited for regression or 2-class problem
#
# TODO: 
# - binary GA mode, representing integers as set of bits
# - MultiLogLoss prediction metric for classification problems
# - feature selection together with parameter optimisation in the same genetics fitness function
# - caching of fitness function results


# install.packages(c("elmNN", "MLmetrics", "GA"), dependencies = TRUE)
library(elmNN)
library(MLmetrics)
library(GA)

data(Melanoma)
trainTable <- Melanoma[-(round(nrow(Melanoma)*2/3):nrow(Melanoma)),]
testTable  <- Melanoma[round(nrow(Melanoma)*2/3):nrow(Melanoma),]

PREDICTOR_COLUMNS <- 1:(ncol(trainTable)-1)
TARGET_COLUMN <- ncol(trainTable)

KFOLDS <- 10
NHID_MAX <- 200
SEED_MAX <- 1000
GA_POPSIZE <- 50

ACTFUN_NAMES <- c("sig",
                  "sin",
                  "radbas",
                  "hardlim",
                  "hardlims",
                  "satlins",
                  "tansig",
                  "tribas",
                  "poslin",
                  "purelin")

elmnn_kfold <- function(x,y,nhid,actfun,seed){
  split2 <- function(x,n) split(x, cut(seq_along(x), n, labels = FALSE))
  folds <- split2(1:nrow(x), KFOLDS)
  modelList <- list()
  prediction <- rep(NA, nrow(x))
  for(i in 1:KFOLDS){
    set.seed(seed)
    modelList[[i]] <- elmtrain(x = x[-folds[[i]], , drop=FALSE], y = y[-folds[[i]]], nhid=nhid, actfun=actfun)
    prediction[folds[[i]]] <- predict(modelList[[i]], x[folds[[i]], , drop=FALSE])
  }
  resultModel <- list(modelList = modelList, prediction = prediction)
  class(resultModel) <- "elmnn_kfold"
  return(resultModel)
}

predict.elmnn_kfold <- function(model, data){
  prediction <- rep(0, nrow(data))
  for(i in 1:length(model$modelList)){
    prediction <- prediction + predict(model$modelList[[i]], data)[,1] / length(model$modelList)
  }
  return(prediction)
}

GaFitness <- function(x){
  tryCatch({
    nhid <- round(x[1])
    actfun <- ACTFUN_NAMES[round(x[2])]
    seed <- round(x[3])
    
    elmnnModels <- elmnn_kfold(x = trainTable[,PREDICTOR_COLUMNS],
                               y = trainTable[,TARGET_COLUMN],
                               nhid = nhid,
                               actfun = actfun,
                               seed = seed)
    score <- R2_Score(y_pred = elmnnModels$prediction, y_true = trainTable[,TARGET_COLUMN])
    return(score)
  },error=function(e){})
  return(-10)
}

gaResult <- ga(type="real-valued",
               fitness = GaFitness,
               min     = c(1,1,0),
               max     = c(NHID_MAX, length(ACTFUN_NAMES), SEED_MAX),
               monitor = plot,
               popSize = GA_POPSIZE)
cat("Best R^2 score:",max(gaResult@fitness),"\n")

nhidOptimised <- round(gaResult@solution[1,1])
actfunOptimised <- ACTFUN_NAMES[round(gaResult@solution[1,2])]
seedOptimised <- round(gaResult@solution[1,3])

model <- elmnn_kfold(x = trainTable[,PREDICTOR_COLUMNS],
                     y = trainTable[,TARGET_COLUMN],
                     nhid = nhidOptimised,
                     actfun = actfunOptimised,
                     seed = seedOptimised)

predictionTrain <- predict(model, trainTable[,PREDICTOR_COLUMNS])
predictionTrain[predictionTrain>=0.5] <- 1
predictionTrain[predictionTrain<0.5] <- 0

predictionTest  <- predict(model, testTable[,PREDICTOR_COLUMNS])
predictionTest[predictionTest>=0.5] <- 1
predictionTest[predictionTest<0.5] <- 0

cat("Accuracy on train data:", Accuracy(y_pred = predictionTrain, trainTable[,TARGET_COLUMN]), "\n")
cat("Accuracy on test data:", Accuracy(y_pred = predictionTest, testTable[,TARGET_COLUMN]), "\n")
