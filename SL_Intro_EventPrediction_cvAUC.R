# Learning SuperLearner 
library(SuperLearner);library(data.table);library(nnls);library(mvtnorm)
library(ranger);library(xgboost);library(splines);library(Matrix)
library(ggplot2);library(xtable);library(pROC)
expit<-function(x){1/(1+exp(-x))}

# EXAMPLE 2
set.seed(1234)
n=1000
sigma <- abs(matrix(runif(25,0,1), ncol=5))
sigma <- forceSymmetric(sigma)
sigma <- as.matrix(nearPD(sigma)$mat)
x <- rmvnorm(n, mean=c(0,.25,.15,0,.1), sigma=sigma)
modelMat<-model.matrix(as.formula(~ (x[,1]+x[,2]+x[,3]+x[,4]+x[,5])^3))
beta<-runif(ncol(modelMat)-1,0,1)
beta<-c(2,beta)
mu <- 1-expit(modelMat%*%beta)
y<-rbinom(n,1,mu)
hist(mu);mean(y)
x<-data.frame(x)
D<-data.frame(x,y)

folds=5
sl.lib <- c("SL.earth","SL.rpart")
fitY<-SuperLearner(Y=y,X=x,family="binomial",
                   method="method.AUC",
                   newX=x,SL.library=sl.lib,
                   cvControl=list(V=folds))
fitY;y_pred<-predict(fitY,onlySL=T)$pred
p<-data.frame(y=y,y_pred=y_pred)
a<-roc(p$y, p$y_pred, direction="auto")
C<-data.frame(sens=a$sensitivities,spec=a$specificities)

ggplot() + geom_step(data=C, aes(1-spec,sens),
                     color="blue",size=.25) + 
  theme_light() + 
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank()) + 
  labs(x = "1 - Specificity",y = "Sensitivity") +
  geom_abline(intercept=0,slope=1,col="gray")

# manually coded superlearner with three algorithms
## 1: split data into 5 groups
head(D,10)
splt<-split(D,1:folds)
head(splt[[1]])

## 2: fit each of the 3 algorithms on training set
m1<-lapply(1:folds,function(ii) earth(formula=y~., data=do.call(rbind,splt[-ii]), degree = 2,
                                      nk = 21, penalty = 3, pmethod = "backward", nfold = 0, 
                                      ncross = 1, minspan = 0, endspan = 0, 
                                      glm = list(family = binomial)))
m2<-lapply(1:folds,function(ii) rpart(y~.,method="class",data=do.call(rbind,splt[-ii])))
#m3<-lapply(1:folds,function(ii) mean(do.call(rbind,splt[-ii])$y))

## 3: predict from each fit in validation set
p1<-lapply(1:folds,function(ii) predict(m1[[ii]],newdata=rbindlist(splt[ii]),type="response"))
p2<-lapply(1:folds,function(ii) predict(m2[[ii]],newdata=rbindlist(splt[ii]))[,2])
#p3<-lapply(1:folds,function(ii) m3[[ii]])
#one<-function(a){1*(a>runif(length(a)))}
head(splt[[1]])
for(i in 1:folds){splt[[i]]<-cbind(splt[[i]][,6],p1[[i]],p2[[i]])}

## 4: calculate CV risk for each method
head(splt[[4]])
mse1<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,2],labels=splt[[ii]][,1]))
mse2<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,3],labels=splt[[ii]][,1]))
#mse3<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,4],labels=splt[[ii]][,1]))

a<-rbind(cbind("earth",mean(do.call(rbind,mse1),na.rm=T)),
      cbind("rpart",mean(do.call(rbind,mse2),na.rm=T)))
      #,cbind("Mean",mean(do.call(rbind,mse3),na.rm=T)))
xtable(a)

## 5: minimize 1 - AUC
X<-data.frame(do.call(rbind,splt),row.names=NULL);names(X)<-c("y","earth","rpart")
head(X)

bounds = c(0, Inf)
SL.r<-function(A,y,par){
  A<-as.matrix(A)
  names(par)<-c("earth","rpart")
  predictions <- A%*%as.matrix(par)
  cvRisk <- 1 - AUC(predictions = predictions, labels = y)
}
init=(rep(1,2))
fit <- optim(par=init, fn=SL.r, A=X[,2:3], y=X[,1], 
             method="L-BFGS-B",lower=bounds[1],upper=bounds[2])
fit
alpha<-fit$par/sum(fit$par)
alpha
# the above coefficeints seem to be reversed??

## 6: fit all algorithms to original data and generate predictions
m1<-earth(formula=y~., data=D, degree = 2,
      nk = 21, penalty = 3, pmethod = "backward", nfold = 0, 
      ncross = 1, minspan = 0, endspan = 0, 
      glm = list(family = binomial))
m2<-rpart(y~.,method="class",data=D)

## 7: predict from each fit using all data
p1<-predict(m1,newdata=D,type="response")
p2<-predict(m2,newdata=D)[,2]
predictions<-cbind(p1,p2)

## 8: take a weighted combination of predictions using nnls parameters as weights
y_pred <- predictions%*%alpha

p<-data.frame(y=y,y_pred=y_pred)
## 9: verify that our work predicts similar results as actual SL function
a<-roc(p$y, p$y_pred, direction="auto")
C2<-data.frame(sens=a$sensitivities,spec=a$specificities)
head(C2)

pdf(file="~/Dropbox/Documents/Research/Papers/SuperLearnerIntro/figure2.pdf",height=4,width=5)
cols <- c("SuperLearner"="red","Manual SuperLearner"="blue")
ggplot() +
  geom_step(data=C, aes(1-spec,sens),color="blue",size=.75) +
  geom_step(data=C2, aes(1-spec,sens),color="red",linetype=3,size=1) +
  #scale_colour_manual(name="",values=cols) +
  theme_light() + theme(legend.position=c(.8,.8)) +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank()) + 
  labs(x = "1 - Specificity",y = "Sensitivity") +
  geom_abline(intercept=0,slope=1,col="gray")
dev.off()