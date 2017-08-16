Data <- read.csv("C:/Users/Oren/Desktop/machine learning class notes/Final_Data.CSV")
D <- data.frame(Data);
head(D);
dim(D);
colnames(D);
hist(D$CR)

## Transformations of the data frame. ##
## Includes histograms, comments about them, etc. ##

D$LOG_CR <- log(D$CR+1) ## CR has zero values for some, so I added 1.
hist(D$LOG_CR)
D$SC_LOG_CR <- scale(D$LOG_CR) 
hist(D$SC_LOG_CR) #Mostly GEV Distribution looking; somewhat normal?
hist(D$CR)
D$SIR <- D$SIR/10000
head(D$SIR)
D$SC_LOG_SIR <- scale(log(D$SIR))
hist(D$SC_LOG_SIR) ## Normal distribution.
colnames(D)
D$ASD <- D$ASD+1
D$ASD <- (D$ASD)/10000
hist(D$ASD)
D$SC_LOG_ASD <- scale(log(D$ASD))
hist(D$SC_LOG_ASD)
D$Lasers <- D$Lasers+1;
hist(D$Lasers)
hist(log(D$Lasers))
D$SC_LOG_Lasers <- scale(log(D$Lasers)) ##Bad distribution; don't even know.
hist(D$SC_LOG_Lasers)
D$Eff_Rate_Inc <- D$Eff_Rate_Inc +3
hist(D$Eff_Rate_Inc)
D$SC_LOG_Eff <- scale(log(D$Eff_Rate_Inc))
hist(scale(D$Eff_Rate_Inc))
hist(D$SC_LOG_Eff)
D$PPO.Factor <- D$PPO.Factor + 1;
D$SC_LOG_PPO <- scale(log(D$PPO.Factor))

hist(D$PPO.Factor)
head(D)
D$SC_LOG_EE <- scale(log(D$EE_Count))
hist(D$SC_LOG_EE)##GREAT normal distribution.
D$SC_LOG_RTM <- scale(log(D$RTM+1))
hist(D$SC_LOG_RTM)##Another great normal distribution.

D$SC_LOG_Claims <- scale(log(D$Claims+1))
hist(D$SC_LOG_Claims)#Normal on left, tall building on right.
head(D)
D <- D[-1]
head(D)
D$SC_LOG_SIREE <- scale(log(D$SIR/D$EE_Count))
hist(D$SC_LOG_SIREE) #Decent normal distribution.
head(D)
##PCA Variables ##
head(D)
dim(D)
PCA.data <- princomp(D[c(18:24, 26)])
colnames(D[c(18:24, 26)])
colnames(D)
head(PCA.data$scores)
plot(PCA.data)
D <- cbind(D, PCA.data$scores[,1:8])
head(D)
mat <- data.frame(x1=D$SC_LOG_CR, x2=D$SC_LOG_SIR, x3=D$SC_LOG_ASD, x4=D$SC_LOG_Lasers, x5=D$SC_LOG_Eff, x6=D$SC_LOG_PPO, x7=D$SC_LOG_EE, x8=D$SC_LOG_RTM,x9=D$Comp.1, x10=D$Comp.2, x11=D$Comp.3, x12=D$Comp.5, x13=D$Comp.4, x14=D$Comp.5, x15=D$Comp.6, x16=D$Comp.7, X17=D$SC_LOG_SIREE, x18=D$Comp.8)
cor(mat)
head(D)
dim(D)


##Categorical Variable MGU##
unique(levels(D$MGU))
x <- as.factor(substr(D$MGU, 1, 3))
summary(x)
MGUD <- model.matrix(D$SC_LOG_CR ~ x - 1)
head(MGUD)
dim(MGUD)
colnames(MGUD)
colnames(MGUD) <- paste("MGU_", unique(levels(x)), sep="")
head(MGUD)
D <- cbind(D[,-1], MGUD)
head(D)
dim(D)

##Categorical Variable: New_Renewal##
unique(levels(D$New_Renewal))
y <- as.factor(substr(D$New_Renewal, 1, 3))
summary(y)
New <- model.matrix(D$SC_LOG_CR ~ y - 1)
head(New)
dim(New)
colnames(New)
colnames(New) <- paste("", unique(levels(y)), sep="")
head(New)
head(D)
D <- cbind(D[,-1], New)
head(D)
dim(D)
help(substr)

head(D)
##Categorical Variable: Underwriter##

unique(levels(D$Underwriter))
z <- as.factor(substr(D$Underwriter, 1, 3))
summary(z)
UndWr <- model.matrix(D$SC_LOG_CR ~ z - 1)
head(UndWr)
dim(UndWr)
colnames(UndWr)
colnames(UndWr) <- paste("UndWr", unique(levels(z)), sep="")
unique(levels(z))
head(UndWr)
head(D)
D <- cbind(D[,-1], UndWr)
head(D)
dim(D)
help(substr)

##Categorical Variable: State.  Last one!##

unique(levels(D$State))
w <- as.factor(substr(D$State, 1, 3))
summary(w)
State <- model.matrix(D$SC_LOG_CR ~ w - 1)
head(State)
dim(State)
colnames(State)
colnames(State) <- paste("St_", unique(levels(w)), sep="")
unique(levels(w))
head(State)
head(D)
D <- cbind(D[,-10], State)
head(D)

###MODELS###
##Model 1.  I'm adding/subtracting columns in this section.##

head(D)
colnames(D)
dim(D)
D_2 <- D[-c(1:12, 21)]
colnames(D_2)
head(D_2)

##Stepwise Variable Selection & Training Data###
###This code is based off of regression.R, the midterm directions##
##and regression and logistic regression.R##

set.seed(11291988)
train.index <- sample(1:nrow(D),round(0.6*nrow(D)),replace = FALSE); length(train.index)
D.train <- D[train.index,]; dim(D.train)
D.test <- D[-train.index,]; dim(D.test)
head(D.train)
null.mod <- lm(D.train$SC_LOG_CR ~ 1, data = D.train); ## Response=SC_LOG_CR
summary(null.mod)
head(D.train)
head(D_2)

##This is Model #1.##
##Very poor model.##

colnames(D.train)
colnames(D)
dim(D.train)
xnam <- colnames(D)[c(14:20, 22:30, 31:121)]
xnam
head(xnam)
full.fmla <- as.formula(paste("SC_LOG_CR ~", paste(xnam, collapse="+")))

#full.fmla <- as.formula(SC_LOG_CR ~ ( SC_LOG_SIR + SC_LOG_ASD + SC_LOG_Lasers + SC_LOG_Eff + SC_LOG_PPO + SC_LOG_EE + 
                              # SC_LOG_RTM + SC_LOG_SIREE + Comp.1 + Comp.2 + Comp.3 + Comp.4 + Comp.5 + Comp.6 + Comp.7 + Comp.8)^4)
full.fmla
mod1 <- step(null.mod, full.fmla, trace=1000, k=2)
summary(mod1)
## Our R-squared = 0.06051.  Our residual standard error = .9759.  This is a poor model.##

y_1.hat <- predict(mod1, newdata = D.test)
hist(y_1.hat)
hist(D.test$SC_LOG_CR-y_1.hat)
summary((D.test$SC_LOG_CR-y_1.hat)^2)

## Lasso or Ridge?  Let's try it. ##

install.packages("glmnet")
library(glmnet)
help(glmnet)
cbind(colnames(D.train))
x <- as.matrix((D.train)[c(14:20, 22:30, 31:121)])  #training predictor variables
y <- D.train$SC_LOG_CR
CV.mod2 <- cv.glmnet(x,y,family=c("gaussian"), nfolds = 10)
summary(CV.mod2)
CV.mod2$lambda


mod2 <- glmnet(x, y, family=c("gaussian"), alpha = 1, lambda = CV.mod2$lambda.min)
summary(mod2);  
mod2$beta
y_2.hat <- predict(mod2, newx = as.matrix(D.test[c(14:20, 22:30, 31:121)]))
hist(D.test$SC_LOG_CR - y_2.hat)

summary(D.test$SC_LOG_CR-y_2.hat)
summary((D.test$SC_LOG_CR - y_2.hat)^2)

##The max val and mean are 18.57 and .9354, respectively.##
##Again, this model(Lasso) isn't necessarily spectacular.  It's still poor.##
##I attempted Ridge, but that's worse.  I left it out of this code.##


## I was unable to parse through Francisco's code for LIFT. ##
## I tried using owen's lift, but that didn't load either. ##
## The following lift chart is based on what the email you wrote##
## stated.  Not sure if I interpreted it correctly. ##

### LIFT PLOT/CHART ###

weights <- y_1.hat
dec <- quantile(weights, prob=seq(0,1, length=11), type=5)
## Deciles of the predicted data. ##
quantile(weights, prob=seq(0,1, length=11), type=5)
tapply(weights, findInterval(weights, dec), mean)
## Means of the deciles. ##
max(weights)
min(weights)
mean(weights)
X = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
Y = c(-0.24906663, -0.15160070, -0.08136731, -0.0234827, 0.03691150, 0.09447204, 0.17113512, 0.27693927, 0.51357474, 1.80451768)
weights2 <- D.test$SC_LOG_CR;
dec2 <- quantile(weights2, prob=seq(0,1, length=11), type=5)
quantile(weights2, prob=seq(0,1, length=11), type=5)
tapply(weights2, findInterval(weights2, dec2), mean)
Y1 <- c(-0.9440435, -0.8048147, -0.6972091, -0.4807927, -0.1886201, 0.1284564, 0.4820875, 1.0256142, 2.1545357, 4.2056728)

## Blue represents observed; red represents predicted. ##

plot(X, Y, type="b", col="red",ylim=range(c(Y, Y1)))
par(new=TRUE)
plot(X, Y1, type="b", col="blue", xlab="Deciles", ylab="Actual/Predicted", ylim=range(c(Y, Y1)), axes=FALSE)

?lines


## The Lift chart looks terrible.  That being said, the model is ##
## also absolutely abysmal.  Online, I found code for ##
## a GAINS/Lift chart. ##
## http://users.stat.umn.edu/~crolling/talks/gains_poster_jsm2013.pdf ##
## Gains chart for linear model. ##

install.packages("gains")
library(gains)
help(gains)

gains.mod<-gains(D.test$SC_LOG_CR, y_1.hat, groups=10)
plot(gains.mod)

## Gains chart for Lasso model. ##

install.packages("gains")
library(gains)
help(gains)

gains.mod1<-gains(D.test$SC_LOG_CR, y_2.hat, groups=10)
plot(gains.mod1)

##  ##

##We now try GEV.fit.  Scaled+Log+CR version seemed to look like that.##
######GEV######

library(fields)
library(evd)
library(evdbayes)
library(ismev)
library(SpatialExtremes)


fit_1 <- gev.fit(xdat = D$SC_LOG_CR)    #gev fit
summary(fit_1)					#summary of model
fit_1$conv						#should be zero if the model converged
fit_1$mle						#MLEs
sqrt(diag(fit_1$cov))				#se of MLEs
gev.diag(fit_1)					#model diagnostics
## From the PP and QP plots, it looks like a definite fit.  Perfect, really. ##
mu.hat <- fit_1$mle[1]  			#estimate of location parameter
sigma.hat <- fit_1$mle[2]				#estimate of scale parameter
xi.hat <- fit_1$mle[3]				#estimate of shape parameter
y_g.hat <- mu.hat + (sigma.hat/xi.hat)*(gamma(1-xi.hat) - 1)	#estimate of mean
y_g.hat
mean(D$SC_LOG_CR)  				#observed mean

fit_2 <- gev.fit(xdat = D$SC_LOG_CR, ydat = D.train[c(14:20, 22:30, 31:121)])
colnames(D)
colnames(D_2)
summary(fit_2)
fit_2$conv
fit_2$mle
sqrt(diag(fit_2$cov))
gev.diag(fit_2)

## The observed mean and estimate mean look absolutely terrible. ##


#########Model 2.  ROC Curve included, Weights=Total_GWP##############

cbind(colnames(D))
logit.inv <- function(x){
  exp(x)/(1+exp(x))
}
###############################
#Training & Testing Data
###############################
D$Y <- I(D$SC_LOG_CR <=1);	table(D$Y)
D$Y <- as.numeric(D$Y);	table(D$Y)	
D.train <- D[train.index,];	dim(D.train)
D.test <- D[-train.index,];	dim(D.test)

###################################
#stepwise variable selection
###################################
?glm
head(D)
head(D.train$Total_GWP)
null.mod <- glm(Y ~ 1, weights=D.train$Total_GWP/100000, family = "binomial", data = D.train);	summary(null.mod)
##Total_GWP must be positive.  Hence, I cannot scale/log it.##

summary(null.mod)
colnames(D)[c(14:20, 22:30, 31:121)]
xnam <- colnames(D)[c(14:20, 22:30, 31:121)]; xnam
colnames(xnam)
full.fmla <- as.formula(paste("Y ~ ", paste(xnam, collapse= "+")));	full.fmla
mod.step <- step(null.mod, full.fmla,trace = 1000, k = 2)
summary(mod.step)

g.hat <- logit.inv(predict(mod.step, newdata = D.test))	#predicted values of holdout data
#We need to apply the inverse of the log link function to get predicted probabilities 
summary(g.hat)
table(D.test$Y,round(g.hat))				#compare observed and predicted

install.packages("verification")
library(verification)
#help(roc.plot)								#google ROC curve. learn about sensitivity and specificity
roc.plot(D.test$Y, g.hat)

## This ROC curve gives poorer values when I add in the Total_GWP.##
##The following code gives a much better ROC curve when Total_GWP
##is not contained in the code under the weights category.

###ROC CURVE W/O WEIGHTS=Total_GWP###


cbind(colnames(D))
logit.inv <- function(x){
  exp(x)/(1+exp(x))
}
###############################
#Training & Testing Data
###############################
D$Y <- I(D$SC_LOG_CR <=1);  table(D$Y)
D$Y <- as.numeric(D$Y);	table(D$Y)
D.train <- D[train.index,];	dim(D.train)
D.test <- D[-train.index,];	dim(D.test)

###################################
#stepwise variable selection
###################################
?glm
head(D)
head(D.train$Total_GWP)
null.mod <- glm(Y ~ 1, family = "binomial", data = D.train);	summary(null.mod)
#note the change from lm() used above to glm()
#and family to from "gaussian" to "binomial"
summary(null.mod)
colnames(D)[c(14:20, 22:30, 31:121)]
xnam <- colnames(D)[c(14:20, 22:30, 31:121)]; xnam
colnames(xnam)
full.fmla <- as.formula(paste("Y ~ ", paste(xnam, collapse= "+")));	full.fmla
mod.step <- step(null.mod, full.fmla,trace = 1000, k = 2)
summary(mod.step)

g.hat <- logit.inv(predict(mod.step, newdata = D.test))	#predicted values of holdout data
#We need to apply the inverse of the log link function to get predicted probabilities 
summary(g.hat)
table(D.test$Y,round(g.hat))				#compare observed and predicted

install.packages("verification")
library(verification)
#help(roc.plot)								#google ROC curve. learn about sensitivity and specificity
roc.plot(D.test$Y, g.hat)

############ END OF TEST #################
