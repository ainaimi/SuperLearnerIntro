# Learning SuperLearner 
library(SuperLearner);library(data.table);library(nnls);library(rmutil)
library(ranger);library(xgboost);library(splines);library(Matrix)
library(ggplot2);library(xtable)
expit<-function(x){1/(1+exp(-x))}


# EXAMPLE 1
set.seed(12345)
n=1000
x = runif(n,0,8)
y = 5 + 4*sqrt(9 * x)*as.numeric(x<2) + as.numeric(x>=2)*(abs(x-6)^(2)) + rlaplace(n)
xl<-seq(0,8,.1)
yl<-5 + 4 * sqrt(9 * xl)*as.numeric(xl<2) + as.numeric(xl>=2)*(abs(xl-6)^(2))

D<-data.frame(x,y)
Dl<-data.frame(xl,yl)

source("~/SLwrappers/create.SL.gam.Wrapper.R")
source("~/SLwrappers/create.SL.nnet.R")
create.SL.gam(deg.gam = 5)
create.SL.nnet(size=5)
sl.lib <- c("SL.gam.5","SL.nnet.5")
fitY<-SuperLearner(Y=y,X=data.frame(x),newX=data.frame(x),SL.library=sl.lib,cvControl=list(V=5))
fitY
yS<-predict(fitY,newdata=data.frame(x=xl),onlySL=T)$pred

Dl1<-data.frame(xl,yS)

# manually coded superlearner with three algorithms
## 1: split data into 5 groups
folds=5
D<-data.frame(x,y)
head(D,10)
splt<-split(D,1:folds)

## 2: fit each of the 3 algorithms on training set
m1<-lapply(1:folds,function(ii) gam(y~s(x,5),family="gaussian",data=rbindlist(splt[-ii])))
m2<-lapply(1:folds,function(ii) nnet(x=x,y=y,linout=T,trace=F,maxit=500,size=5,data=rbindlist(splt[-ii])))

## 3: predict from each fit in validation set
p1<-lapply(1:folds,function(ii) predict(m1[[ii]],newdata=rbindlist(splt[ii]),type="response"))
p2<-lapply(1:folds,function(ii) predict(m2[[ii]],newdata=rbindlist(splt[ii]),type="raw"))
for(i in 1:folds){splt[[i]]<-cbind(splt[[i]],p1[[i]],p2[[i]])}

## 4: calculate CV risk for each method (MSE??)
mse1<-lapply(1:folds,function(ii) mean((splt[[ii]][,2]-splt[[ii]][,3])^2))
mse2<-lapply(1:folds,function(ii) mean((splt[[ii]][,2]-splt[[ii]][,4])^2))

a<-rbind(cbind("GAM",mean(do.call(rbind,mse1),na.rm=T)),
      cbind("nnet",mean(do.call(rbind,mse2),na.rm=T)))
xtable(a)

## 5: estimate SL weights using nnls (for convex combination) and normalize
X<-data.frame(do.call(rbind,splt))[,-1];names(X)<-c("y","gam","nnet")
head(X)
SL.r<-nnls(cbind(X[,2],X[,3]),X[,1])$x
alpha<-as.matrix(SL.r/sum(SL.r))
round(alpha,3)
fitY
## 6: fit all algorithms to original data and generate predictions
m1<-gam(y~s(x,5),family="gaussian",data=D)
m2<-nnet(x=x,y=y,linout=T,trace=F,maxit=500,size=5,data=D)

## 7: predict from each fit using all data
p1<-predict(m1,newdata=D,type="response")
p2<-predict(m2,newdata=D,type="raw")
predictions<-cbind(p1,p2)

## 8: take a weighted combination of predictions using nnls parameters as weights
y_pred <- predictions%*%alpha

## 9: verify that our work predicts similar results as actual SL function
p1<-predict(m1,newdata=data.frame(x=xl),type="response")
p2<-predict(m2,newdata=data.frame(x=xl),type="raw")
predictions<-cbind(p1,p2)
yS2 <- predictions%*%alpha

Dl2<-data.frame(xl,yS2)

pdf(file="~/Dropbox/Documents/Research/Papers/SuperLearnerIntro/figure1.pdf",height=4,width=5)
cols <- c("Truth"="gray25","SuperLearner"="red","Manual SuperLearner"="blue")
ggplot() +
  geom_point(data=D, aes(x,y),color="gray75",size=.75) + 
  geom_line(data=Dl2, aes(xl,yS2,color="Manual SuperLearner"),size=1) + 
  geom_line(data=Dl1, aes(xl,yS,color="SuperLearner"),size=1) + 
  geom_line(data=Dl, aes(xl,yl,color="Truth"),size=.5) + 
  scale_colour_manual(name="",values=cols) +
  theme_light() + theme(legend.position=c(.8,.8)) +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank()) + 
  labs(x = "Exposure",y = "Outcome")
dev.off()

lines(xl,yS2,lwd=2,col="blue")
legend("topleft",c("Real SL","Ashley's SL"),col=c("red","blue"),lty=c(1,1),lwd=c(2,2))

plot(yS,yS2)
lines(x = seq(min(yS),max(yS)), y = seq(min(yS),max(yS)),col="red")

## EXAMPLE 2

