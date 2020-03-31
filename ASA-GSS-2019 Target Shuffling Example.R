#-----------------------------#
#       How Rare is Rare?     #
#   IMportance of Validation  #
#                             #
#        Dr. Aric LaBarr      #
#         Institute for       # 
#      Advanced Analytics     #
#-----------------------------#

# Load Needed Libraries #
library(ISLR) # Library for data from "Introduction to Statistical Learning with Applications in R" #
library(dplyr) # Library used for data manipulation #
library(pROC) # Library for model assessment of logistic regression #

# Set Seed - Getting random sample from the data set for modeling purposes #
set.seed(12345)

# Load Data - Default data from ISLR package #
def <- Default

# Assigning ID's to the observations since they don't already have them #
def$ID <- 1:10000

# Splitting data in training and validation #
train <- def %>% group_by(default) %>% sample_n(250)
valid <- def[which(!(def$ID %in% train$ID)),] 

# Logistic Model - Classification model predicting default based on student and income variables #
Logit.Model <- glm(default ~ factor(student) + income, data = train, family=binomial(logit))
summary(Logit.Model)

# Evaluating the logistic model based on ROC curves #
model.roc <- roc(Logit.Model$y, Logit.Model$fitted.values)
AUC.True <- auc(model.roc)
AUC.True

# Target Shuffling - Developing the shuffle of simulations #
sim <- 1000 # Setting the number of simulations #

Y.Shuffle <- matrix(0, nrow=length(train$default), ncol=sim)
for(j in 1:sim){
  Uniform <- runif(length(train$default))
  Y.Shuffle[,j] <- as.numeric(train$default[order(Uniform)]) - 1
}

Y.Shuffle <- data.frame(Y.Shuffle)
colnames(Y.Shuffle) <- paste('Y.',seq(1:sim),sep="")

train <- data.frame(train, Y.Shuffle)

# Calucating the ROC curve from logistic regressions from simulated shuffles of the data #
AUC.Y <- rep(0,sim)
for(i in 1:sim){
  AUC.Y[i] <- auc(roc(glm(train[,5+i] ~ income + factor(student), data = train, family=binomial(logit))$y, glm(train[,5+i] ~ balance + income, data = train, family=binomial(logit))$fitted.values))
}

# Plotting histogram of all ROC curve values as well as marking the true model ROC curve #
hist(c(AUC.Y,AUC.True), breaks=25, col = "blue", main='Distribution of AUC Values', xlab='Area Under the Curve')
abline(v = AUC.True, col="red", lwd=2)
mtext("True Model", at=AUC.True, col="red")

# How many ROC curves beat our ROC curve from the real data? #
sum(AUC.Y > AUC.True)

# Only 45 out of 1000 beat out our model - 4.5% of random models beat our model #
# The likelihood our model occurred by random chance is 4.5% #
# Go back and change the seed number to see what happens with a new sample! #
