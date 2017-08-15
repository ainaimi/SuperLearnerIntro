# R code to demonstrate implementation of Super Learner 
#	for binary classification (Example 2) in
# "Stacked Generalization: An Introdution to Super Learning"
#	by Ashley I. Naimi and Laura B. Balzer

# load the relevant packages
library(SuperLearner);library(data.table);library(nnls);library(mvtnorm)
library(ranger);library(xgboost);library(splines);library(Matrix)
library(ggplot2);library(xtable);library(pROC)

# set the working directory: e.g. 
setwd("~/Dropbox/Documents/Research/Papers/SuperLearnerIntro/")

# EXAMPLE 2
set.seed(123)
n=10000
sigma <- abs(matrix(runif(25,0,1), ncol=5))
sigma <- forceSymmetric(sigma)
sigma <- as.matrix(nearPD(sigma)$mat)
x <- rmvnorm(n, mean=c(0,.25,.15,0,.1), sigma=sigma)
modelMat<-model.matrix(as.formula(~ (x[,1]+x[,2]+x[,3]+x[,4]+x[,5])^3))
beta<-runif(ncol(modelMat)-1,0,1)
beta<-c(2,beta) # setting intercept
mu <- 1-plogis(modelMat%*%beta) # true underlying risk of the outcome
y<-rbinom(n,1,mu)

hist(mu);mean(y)

x<-data.frame(x)
D<-data.frame(x,y)

# Specify the number of folds for V-fold cross-validation
folds=5
## split data into 5 groups for 5-fold cross-validation 
## we do this here so that the exact same folds will be used in 
## both the SL fit with the R package, and the hand coded SL
index<-split(1:1000,1:folds)
splt<-lapply(1:folds,function(ind) D[index[[ind]],])
# view the first 6 observations in the first [[1]] and second [[2]] folds
head(splt[[1]])
head(splt[[2]])

#-------------------------------------------------------------------------------
# Fit using the SuperLearner Package
#-------------------------------------------------------------------------------
# Specify the outcome-for-prediction (y), the predictors (x), 
#		family (for a binary outcome), measure of performance (1-AUC),
#		the library (sl.lib), and number of folds
sl.lib <- c("SL.bayesglm","SL.polymars")
fitY<-SuperLearner(Y=y,X=x,family="binomial",
                   method="method.AUC",
                   SL.library=sl.lib,
                   cvControl=list(V=folds))

# Note: for rare binary outcomes, consider using the stratifyCV option to 
#		maintain roughly the same # of outcomes per fold
# View the output: 'Risk' column returns the CV estimates of (1-AUC)
#		'Coef' column gives the weights for the final SuperLearner (meta-learner)
fitY

# Obtain the predicted probability of the outcome
y_pred<-predict(fitY, onlySL=T)$pred
p <- data.frame(y=y, y_pred=y_pred)
head(p)

# Use the roc() function to obtain measures of performance for binary classification
a <- roc(p$y, p$y_pred, direction="auto")
# To plot the ROC curve, we need the sensitivity and specificity
C<-data.frame(sens=a$sensitivities,spec=a$specificities)

ggplot() + geom_step(data=C, aes(1-spec,sens),color="blue",size=.25) + 
  theme_light() + theme(panel.grid.major = element_blank(),
                        panel.grid.minor = element_blank()) + 
  labs(x = "1 - Specificity",y = "Sensitivity") +
  geom_abline(intercept=0,slope=1,col="gray")

#-------------------------------------------------------------------------------
# Hand-coding Super Learner
#-------------------------------------------------------------------------------
## 1: split data into 5 groups for 5-fold cross-validation 
head(D,10)
splt<-split(D,1:folds)
# view the first 6 observations in the first fold
head(splt[[1]])

#----------------------------
## 2: the lapply() function is an efficient way to rotate through the folds to execute:
#	(a) set the ii-th fold to be the validation set; (b) fit each algorithm on the training set; 
#  (c) obtain the predicted outcomes for observations in the validation set;
#  (d) estimate the estimated risk (1-AUC) for each fold
#
## 2b: fit each algorithm on the training set (but not the ii-th validation set)
m1<-lapply(1:folds,function(ii) bayesglm(formula=y~.,data=do.call(rbind,splt[-ii]),family="binomial"))
m2<-lapply(1:folds,function(ii) polyclass(do.call(rbind,splt[-ii])[,6],do.call(rbind,splt[-ii])[,1:5],cv=5))

## 2c: obtain the predicted probability of the outcome for observation in the ii-th validation set
p1<-lapply(1:folds,function(ii) predict(m1[[ii]],newdata=rbindlist(splt[ii]),type="response"))
p2<-lapply(1:folds,function(ii) ppolyclass(fit=m2[[ii]],cov=rbindlist(splt[ii])[,1:5])[,2])

# update dataframe 'splt' so that column1 is the observed outcome (y)
#   column2 is the CV-predicted probability of the outcome from bayesglm
#   column3 is the CV-predicted probability of the outcome from random forest
for(i in 1:folds){
		splt[[i]]<-cbind(splt[[i]][,6],p1[[i]],p2[[i]])
}
# view the first 6 observations in the first fold 

## 2d: calculate CV risk for each method for the ii-th validation set
# our loss function is the rank loss; so our risk is (1-AUC)
#		use the AUC() function with input as the predicted outcomes and 'labels' as the true outcomes
risk1<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,2], labels=splt[[ii]][,1]))    # CV-risk for bayesglm
risk2<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,3], labels=splt[[ii]][,1]))		# CV-risk for knn

#----------------------------
## 3: average the estimated 5 risks across the folds to obtain 1 measure of performance for each algorithm
a<-rbind(cbind("bayesglm",mean(do.call(rbind, risk1),na.rm=T)),
      cbind("polymars",mean(do.call(rbind, risk2),na.rm=T)))
# output a table of the CV-risk estimates
# xtable(a)
# compare with the package output
fitY;a

#----------------------------
## 4: estimate SL weights using the optim() function to minimize (1-AUC)
X<-data.frame(do.call(rbind,splt),row.names=NULL);  names(X)<-c("y","bayesglm","poly")
head(X)

bounds = c(0, Inf)
SL.r<-function(A, y, par){
  A<-as.matrix(A)
  names(par)<-c("bayesglm","poly")
  predictions <- crossprod(t(A),par)
  cvRisk <- 1 - AUC(predictions = predictions, labels = y)
}
init=(rep(1/2,2))
fit <- optim(par=init, fn=SL.r, A=X[,2:3], y=X[,1], 
             method="L-BFGS-B",lower=bounds[1],upper=bounds[2])
fit
alpha<-fit$par/sum(fit$par)
fitY; a
alpha

#---------------------
## 5a: fit all algorithms to original data
m1<-bayesglm(formula=y~.,data=D,family="binomial")
m2<-polyclass(D[,6],D[,1:5],cv=5)

## 5b: predict probabilities from each fit using all data
p1<-predict(m1,newdata=D,type="response")  # bayesglm
p2<-ppolyclass(fit=m2,cov=D[,1:5])[,2] #randomForest
predictions<-cbind(p1,p2)
head(predictions)

## 5c: for the observed data take a weighted combination of predictions using nnls coeficients as weights
y_pred <- predictions%*%alpha
p<-data.frame(y=y,y_pred=y_pred)

## #--------------------------------------------
# verify that our work predicts similar results as SL package
a<-roc(p$y, p$y_pred, direction="auto")
C2<-data.frame(sens=a$sensitivities,spec=a$specificities)
head(C2)

###--------------------------------------------
# fits from candidate algorithms
a<-roc(y, p1, direction="auto")
Cbayes<-data.frame(sens=a$sensitivities,spec=a$specificities)

a<-roc(y, p2, direction="auto") 
Cpoly<-data.frame(sens=a$sensitivities,spec=a$specificities)

pdf(file="figure2.pdf",height=4,width=5)
cols <- c("SuperLearner Package"="red","Manual SuperLearner"="blue", "Bayes GLM"="green", "PolyMARS"="black")
ggplot() +
  geom_step(data=C, aes(1-spec,sens,color="Manual SuperLearner"),size=.75) +
  geom_step(data=C2, aes(1-spec,sens,color="SuperLearner Package"),linetype=2,size=.5) +
  geom_step(data=Cbayes, aes(1-spec,sens,color="Bayes GLM"),linetype=2,size=.5) +
  geom_step(data=Cpoly, aes(1-spec,sens,color="PolyMARS"),linetype=2,size=.5) +
  #scale_colour_manual(name="",values=cols) +
  theme_light() + theme(legend.position=c(.8,.2)) +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank()) + 
  labs(x = "1 - Specificity",y = "Sensitivity") +
  geom_abline(intercept=0,slope=1,col="gray") +
  scale_colour_manual(name="",values=cols)
dev.off()