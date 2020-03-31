
### Example code for Estimating Tree, Forest and Main Effects Logistic Regression Models using 
### the NHIS sample data referenced in the Article.  
### Code prepared by Trent D. Buskirk, May 30, 2017.

##################################################
##########   newrespond20 Prediction    ########## ##################################################


################################################################# 
###  Step 0:   Create the training and test sets to be used in  #              ###  the simple cross validation                                #
#################################################################

## rename the levels of the educ3 variable to facilitate printing ##of the final tree models.

levels(SPData2$educ3)<-c("HSorLess", "Some Col", "BSorMore")
levels(SPData2$newrespond20)<-c("No", "Yes")

## create simple training and testing set based on pre-defined ##folds of the dataset that were randomly assigned.

SPDtrain<-SPData2[SPData2$fold<=10,]

# fold 11 represents rougly 15% of the total data set and will #form our "test" set
SPDtest<-SPData2[SPData2$fold>10,]

# removing extra variables that won't be included in the models

SPDtrain<-SPDtrain[,c(1:5,7:9, 11:13,30)]
SPDtest<-SPDtest[,c(1:5,7:9, 11:13,30)]

SPDtrain<-data.frame(SPDtrain[2:12], SPDtrain$id)
names(SPDtrain)[12]<-c("id")
SPDtest<-data.frame(SPDtest[2:12], SPDtest$id)
names(SPDtest)[12]<-c("id")

require(rpart)
require(randomForest)

############################################################## #######       STEP 1:  Estimate the tree model          ###### 
##############################################################

set.seed(5312017)
nr2tree<-rpart(newrespond20~., data=SPDtrain[,-c(13)], control=rpart.control(cp=0.0005))
printcp(nr2tree)
plotcp(nr2tree)

# Based on the "One Standard Error" rule, the largest value of cp that has an error that is within 
#one standard error of the smallest error is .0022.  We will use the final tree with this value of CP.
# we can either prune the tree, or grow it and specify the value of cp. 

nr2tree1<-prune.rpart(nr2tree, cp=0.0022)

#############################################################
########        STEP 2: PLot the TREE MODEL          ######## #############################################################
require(rpart.plot)
rpart.plot(nr2tree1, box.palette="BuRd",
branch.lty=1, branch.lwd=3, branch.col="darkgreen", type=1,shadow.col="white", nn=TRUE, cex=.95)


prop.table(table())

require("partykit")
tree.2<- as.party(nr2tree1)
plot(tree.2, gp = gpar(fontsize = 9.5))

names(tree.2)
tree.2$fitted[1:10,]

# enhancing the plot 
# see:  http://stackoverflow.com/questions/33314187/partykit-displaying-terminal-node-
# percentile-values-above-terminal-node-boxplot 

tab<-prop.table(table(tree.2$fitted[,1], tree.2$fitted[,2]), margin=1)[,2]

tab<-round(tab,2)
mlab <- function(id, nobs) paste("Rspd%=", tab[id])
plot(tree.2, tp_args = list(mainlab = mlab), gp = gpar(fontsize = 6.5, col="darkblue", lwd=2))

##############################################################
###### STEP 3: Assess Model Performance on the test set  #####  ##############################################################

#  Since the outcome variable is categorical we will compute a 
#  confusion matrix that can be used to
#  compute relevant accuracy measures such as the Sensitivity, 
#  Specificity, Percent Correctly Classified etc.

require(caret)
require(pROC)

# now check how well the tree model performed on the test set
nr2treepreds<-predict(nr2tree1, SPDtest[,-c(12,13)],type="class")

# get predicted probabilities for use in computing AUC 
nr2treepreds2<-predict(nr2tree1, SPDtest[,-c(12,13)])[,2]
# Compute Area Under the Receiver Operating Characteristic Curve
auc(SPDtest$newrespond20,nr2treepreds2)
#under the curve: 81.93
roc(SPDtest$newrespond20,nr2treepreds2, plot=TRUE, add=TRUE,col=1, lwd=2.5, lty=5)

nr2treeconfuse<-confusionMatrix(nr2treepreds, SPDtest$newrespond20, positive=c("Yes"))
nr2treeconfuse

# cross validated estimate of the Percentage Correctly Classified # for the tree model: 78.32%
# cross validated estimate of the Sensitivity for the tree model:  # 57.49% (true positive rate)
# cross validated estimate of the Specificity for the tree model: # 92.12% (true negative rate) 
# overall balanced accuracy for the tree model (i.e. average of 
# true negative and true positive rate): 74.81%

#############################################################
########### Main Effects Logistic Regression Model for   #### ########### newrespond20                                 ####
#############################################################.

nr2LR <- glm(newrespond20 ~ age + educ3 + hispanic2 + incgrp4 + telstat+
  ratcat2 + region + sex + wborace + wrkcata, 
  family=binomial(logit), data=SPDtrain[,-c(12)])
summary(nr2LR)
nr2treepredsLR<-predict(nr2LR, SPDtest[,-c(11,12)], type="response")
nr2treepredsLRfinal<-as.numeric((nr2treepredsLR>=.5))
nr2treepredsLRfinal<-as.factor(nr2treepredsLRfinal)
levels(nr2treepredsLRfinal)<-c("No", "Yes")

# Area Under the Curve Calculations
auc(SPDtest$newrespond20,nr2treepredsLR)
#under the curve: 74.19

nr2treeconfuseLR<-confusionMatrix(nr2treepredsLRfinal, SPDtest$newrespond20, positive=c("Yes"))
nr2treeconfuseLR

# cross validated estimate of the Percentage Correctly Classified for the LR model: 69.92%
# cross validated estimate of the Sensitivity for the LR model:  61.41% (true positive rate)
# cross validated estimate of the Specificity for the LR model: 77.30% (true negative rate)
# overall balanced accuracy for the LR model (i.e. average of true negative and true positive rate): 69.36%
