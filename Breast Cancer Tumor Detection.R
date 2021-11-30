##----------------Breast Cancer prediction using Oncology data----------------##

##---------------------STAGE 1: All the variables-----------------------------##
setwd("C:/Users/bhavn/OneDrive/Documents/R/Data Sets/BA with R Term Project")  
cancer<-read.csv("Cancer.csv")
cancer.df<-data.frame(cancer)
cancer.df<-cancer.df[,-c(1,33)]

#Summary Statistics
library(vtable)
st(cancer.df)

#Create a training and validation partition
numberOfRows <- nrow(cancer.df)
set.seed(1)
train.index <- sample(numberOfRows, numberOfRows*0.6)

print(train.index)
train.df <- cancer.df[train.index, ]
valid.df <- cancer.df[-train.index, ]
View(train.df)
View(valid.df)

##Decision tree-----------------------------------------------------------------

library(rpart)
library(rpart.plot)
library(caret)
library(e1071)

.ct <- rpart(diagnosis ~ ., data = train.df, method = "class", cp = 0, maxdepth = 4, minsplit = 1)

# print tree summary and plot tree. try different values for extra
printcp(.ct)
prp(.ct, type = 1, extra = 1, under = FALSE, split.font = 1, varlen = -10)


# classify records in the validation data using the classification tree.
# set argument type = "class" in predict() to generate predicted class membership.
ct.pred <- predict(.ct, valid.df, type = "class")

# generate confusion matrix for training data
confusionMatrix(ct.pred, as.factor(valid.df$diagnosis))

# build a deeper classification tree
max.ct <- rpart(diagnosis ~ ., data = train.df, method = "class", cp = 0, minsplit = 1, maxdepth = 30)

# count number of leaves
length(max.ct$frame$var[max.ct$frame$var == "<leaf>"])

# plot tree
prp(max.ct, type = 1, extra = 1, under = TRUE, split.font = 1, varlen = -10, 
    box.col=ifelse(max.ct$frame$var == "<leaf>", 'gray', 'white'))  

# classify records in the training data to show that the tree prefectly fits the training data.
# this is an example of overfitting
# set argument type = "class" in predict() to generate predicted class membership.
max.pred <- predict(max.ct, train.df, type = "class")

# generate confusion matrix for training data
confusionMatrix(max.pred, as.factor(train.df$diagnosis))


# classify records in the validation data.
# set argument type = "class" in predict() to generate predicted class membership.
max.pred <- predict(max.ct, valid.df, type = "class")

# generate confusion matrix for training data
confusionMatrix(max.pred, as.factor(valid.df$diagnosis))

##Pruned tree-------------------------------------------------------------------

# Create code to prune the tree

cv.ct <- rpart(diagnosis ~ ., data = cancer.df, method = "class", 
               control = rpart.control(cp = 0.00000005, minsplit = 5, xval = 5))

# use printcp() to print the table. 
printcp(cv.ct)
prp(cv.ct, type = 1, extra = 1, split.font = 1, varlen = -10)  

#prune the tree using the lowest value for xerror
#Note: the prune function requires cp as a parameter so we need to get cp for lowest value of xerror
pruned.ct <- prune(cv.ct, 
                   cp = cv.ct$cptable[which.min(cv.ct$cptable[,"xerror"]),"CP"])

#get count of the number of splits
cp_df <- data.frame(pruned.ct$cptable)
max(cp_df$nsplit)

#another way to get the count of the number of splits
pruned.ct$cptable[which.max(pruned.ct$cptable[,"nsplit"]),"nsplit"]

#get count of the number of nodes
length(pruned.ct$frame$var[pruned.ct$frame$var == "<leaf>"])

#plot the best fitting tree
prp(pruned.ct, type = 1, extra = 1, split.font = 1, varlen = -10,
    box.col=ifelse(pruned.ct$frame$var == "<leaf>", 'gray', 'white'))  

prune.pred <- predict(pruned.ct, valid.df, type = "class")

# generate confusion matrix for training data
confusionMatrix(prune.pred, as.factor(valid.df$diagnosis))

##Random Forest-----------------------------------------------------------------

library(randomForest)

#mtry = number of variables randomly sampled  as candidates at each split
#creates ntree number of trees by randomly sampling with replacement
#results are the average of all the trees
rf <- randomForest(as.factor(diagnosis) ~ ., data = train.df, 
                   ntree = 5000, mtry = 11, nodesize = 1, importance = TRUE, sampsize = 100) 

#plot the variables by order of importance
varImpPlot(rf, type = 1)

#create a confusion matrix
valid.df$diagnosis <- factor(valid.df$diagnosis)
rf.pred <- predict(rf, valid.df)
confusionMatrix(rf.pred, valid.df$diagnosis)

##Logistics Regression----------------------------------------------------------
library(gains)
library(caret)
library(ROCR)

View(cancer.df)

cancer.df$diagnosis<-as.factor(cancer.df$diagnosis)
levels(cancer.df$diagnosis)
cancer.df$diagnosis<-factor(cancer.df$diagnosis, levels = c("B","M"), labels = c(0,1))

#Create a training and validation partition
numberOfRows <- nrow(cancer.df)
set.seed(1)
train.index <- sample(numberOfRows, numberOfRows*0.6)

print(train.index)
train.df <- cancer.df[train.index, ]
valid.df <- cancer.df[-train.index, ]
View(train.df)
View(valid.df)


logitI.reg <- glm(diagnosis ~., data = train.df, family = "binomial") 
options(scipen=999)
summary(logitI.reg)
confusionMatrix(table(predict(logitI.reg, newdata = valid.df, 
                              type="response") >= 0.5, valid.df$diagnosis == 1))

# use predict() with type = "response" to compute predicted probabilities. 
logit.reg.pred <- predict(logitI.reg, valid.df, type = "response") 

View(logit.reg.pred)
# first 5 actual and predicted records
data.frame(actual = valid.df$diagnosis[1:10], predicted = logit.reg.pred[1:10])
confusionMatrix(table(predict(logitI.reg, newdata = valid.df, type="response") >= 0.5, valid.df$diagnosis == 1))


##____________________End of STAGE 1__________________________________________##

##----------STAGE 2:Eliminating few variables after correlation---------------##

setwd("C:/Users/bhavn/OneDrive/Documents/R/Data Sets/BA with R Term Project")  
cancer<-read.csv("Cancer.csv")
cancer.df<-data.frame(cancer)
summary(cancer.df)
View(cancer.df)
cancer.df<-cancer.df[,-c(1,33)]

#Summary Statistics
install.packages("vtable")
library(vtable)
st(cancer.df)


##Creating the correlation matrix and discarding similar variables
cancer.df.cor<-cor(cancer.df[,-c(1)], method = c("pearson", "kendall", "spearman"))
View(cancer.df.cor)
install.packages("corrplot")
library(corrplot)
corrplot(cancer.df.cor)

##Removing perimeter_mean, area_mean, radius_worst, perimeter_worst, area_worst, perimeter_se, area_se
##As these variables are highly correlated to each other

cancer.df<-cancer.df[,-c(4,5,14,15,22,24,25)]
View(cancer.df)

#Create a training and validation partition
numberOfRows <- nrow(cancer.df)
set.seed(1)
train.index <- sample(numberOfRows, numberOfRows*0.6)

print(train.index)
train.df <- cancer.df[train.index, ]
valid.df <- cancer.df[-train.index, ]
View(train.df)
View(valid.df)


##Decision tree-----------------------------------------------------------------

library(rpart)
library(rpart.plot)
library(caret)
library(e1071)

.ct <- rpart(diagnosis ~ ., data = train.df, method = "class", cp = 0, maxdepth = 4, minsplit = 1)

# print tree summary and plot tree. try different values for extra
printcp(.ct)
prp(.ct, type = 1, extra = 1, under = FALSE, split.font = 1, varlen = -10)


# classify records in the validation data using the classification tree.
# set argument type = "class" in predict() to generate predicted class membership.
ct.pred <- predict(.ct, valid.df, type = "class")

# generate confusion matrix for training data
confusionMatrix(ct.pred, as.factor(valid.df$diagnosis))

# build a deeper classification tree
max.ct <- rpart(diagnosis ~ ., data = train.df, method = "class", cp = 0, minsplit = 1, maxdepth = 30)

# count number of leaves
length(max.ct$frame$var[max.ct$frame$var == "<leaf>"])

# plot tree
prp(max.ct, type = 1, extra = 1, under = TRUE, split.font = 1, varlen = -10, 
    box.col=ifelse(max.ct$frame$var == "<leaf>", 'gray', 'white'))  

# classify records in the training data to show that the tree prefectly fits the training data.
# this is an example of overfitting
# set argument type = "class" in predict() to generate predicted class membership.
max.pred <- predict(max.ct, train.df, type = "class")

# generate confusion matrix for training data
confusionMatrix(max.pred, as.factor(train.df$diagnosis))


# classify records in the validation data.
# set argument type = "class" in predict() to generate predicted class membership.
max.pred <- predict(max.ct, valid.df, type = "class")

# generate confusion matrix for training data
confusionMatrix(max.pred, as.factor(valid.df$diagnosis))

##Pruned tree-------------------------------------------------------------------

# Create code to prune the tree

cv.ct <- rpart(diagnosis ~ ., data = cancer.df, method = "class", 
               control = rpart.control(cp = 0.00000005, minsplit = 5, xval = 5))

# use printcp() to print the table. 
printcp(cv.ct)
prp(cv.ct, type = 1, extra = 1, split.font = 1, varlen = -10)  

#prune the tree using the lowest value for xerror
#Note: the prune function requires cp as a parameter so we need to get cp for lowest value of xerror
pruned.ct <- prune(cv.ct, 
                   cp = cv.ct$cptable[which.min(cv.ct$cptable[,"xerror"]),"CP"])

#get count of the number of splits
cp_df <- data.frame(pruned.ct$cptable)
max(cp_df$nsplit)

#another way to get the count of the number of splits
pruned.ct$cptable[which.max(pruned.ct$cptable[,"nsplit"]),"nsplit"]

#get count of the number of nodes
length(pruned.ct$frame$var[pruned.ct$frame$var == "<leaf>"])

#plot the best fitting tree
prp(pruned.ct, type = 1, extra = 1, split.font = 1, varlen = -10,
    box.col=ifelse(pruned.ct$frame$var == "<leaf>", 'gray', 'white'))  

prune.pred <- predict(pruned.ct, valid.df, type = "class")

# generate confusion matrix for training data
confusionMatrix(prune.pred, as.factor(valid.df$diagnosis))


##Random Forest-----------------------------------------------------------------

library(randomForest)

#mtry = number of variables randomly sampled  as candidates at each split
#creates ntree number of trees by randomly sampling with replacement
#results are the average of all the trees
rf <- randomForest(as.factor(diagnosis) ~ ., data = train.df, 
                   ntree = 5000, mtry = 11, nodesize = 1, importance = TRUE, sampsize = 100) 

#plot the variables by order of importance
varImpPlot(rf, type = 1)

#create a confusion matrix
valid.df$diagnosis <- factor(valid.df$diagnosis)
rf.pred <- predict(rf, valid.df)
confusionMatrix(rf.pred, valid.df$diagnosis)


##Logistics Regression----------------------------------------------------------
library(gains)
library(caret)
library(ROCR)

View(cancer.df)

cancer.df$diagnosis<-as.factor(cancer.df$diagnosis)
levels(cancer.df$diagnosis)
cancer.df$diagnosis<-factor(cancer.df$diagnosis, levels = c("B","M"), labels = c(1,0))

#Create a training and validation partition
numberOfRows <- nrow(cancer.df)
set.seed(1)
train.index <- sample(numberOfRows, numberOfRows*0.6)

print(train.index)
train.df <- cancer.df[train.index, ]
valid.df <- cancer.df[-train.index, ]
View(train.df)
View(valid.df)


logitI.reg <- glm(diagnosis ~., data = train.df, family = "binomial") 
options(scipen=999)
summary(logitI.reg)
confusionMatrix(table(predict(logitI.reg, newdata = valid.df, 
                              type="response") >= 0.5, valid.df$diagnosis == 0))

# use predict() with type = "response" to compute predicted probabilities. 
logit.reg.pred <- predict(logitI.reg, valid.df, type = "response") 

View(logit.reg.pred)
# first 5 actual and predicted records
data.frame(actual = valid.df$diagnosis[1:10], predicted = logit.reg.pred[1:10])
confusionMatrix(table(predict(logitI.reg, newdata = valid.df, type="response") >= 0.5, valid.df$diagnosis == 0))


##____________________End of STAGE 2_____________________________________________##

##--STAGE 3:Eliminating few more variables after correlation and level of significance---##

setwd("C:/Users/bhavn/OneDrive/Documents/R/Data Sets/BA with R Term Project")  
cancer<-read.csv("Cancer.csv")
cancer.df<-data.frame(cancer)
cancer.df<-cancer.df[,-c(1,33)]

#Summary Statistics
install.packages("vtable")
library(vtable)
st(cancer.df)

##Creating the correlation matrix and discarding similar variables
cancer.df.cor<-cor(cancer.df[,-c(1)], method = c("pearson", "kendall", "spearman"))
View(cancer.df.cor)
install.packages("corrplot")
library(corrplot)
corrplot(cancer.df.cor)
View(cancer.df)


cancer.df<-cancer.df[,c(1,2,7,8,11,13,17,21,23,26)]


##Logistics--------------------------------------------------------------------

library(gains)
library(caret)
library(ROCR)

View(cancer.df)


cancer.df$diagnosis<-as.factor(cancer.df$diagnosis)
levels(cancer.df$diagnosis)
cancer.df$diagnosis<-factor(cancer.df$diagnosis, levels = c("B","M"), labels = c(0,1))

#Create a training and validation partition
numberOfRows <- nrow(cancer.df)
set.seed(1)
train.index <- sample(numberOfRows, numberOfRows*0.6)

print(train.index)
train.df <- cancer.df[train.index, ]
valid.df <- cancer.df[-train.index, ]
View(train.df)
View(valid.df)

logitI.reg <- glm(diagnosis ~., data = train.df, family = "binomial") 
options(scipen=999)
summary(logitI.reg)
confusionMatrix(table(predict(logitI.reg, newdata = valid.df, 
                              type="response") >= 0.5, valid.df$diagnosis == 1))


##Removing columns which don't have significant impact and running the regression
##model again
cancer.df<-cancer.df[,c(1,2,4,9,10)]

numberOfRows <- nrow(cancer.df)
set.seed(1)
train.index <- sample(numberOfRows, numberOfRows*0.6)

print(train.index)
train.df <- cancer.df[train.index, ]
valid.df <- cancer.df[-train.index, ]
View(train.df)
View(valid.df)

logitI.reg <- glm(diagnosis ~., data = train.df, family = "binomial") 
options(scipen=999)
summary(logitI.reg)
confusionMatrix(table(predict(logitI.reg, newdata = valid.df, 
                              type="response") >= 0.5, valid.df$diagnosis == 1))


##Considering these columns and building other predictive models

##Decision tree-----------------------------------------------------------------
library(rpart)
library(rpart.plot)
library(caret)
library(e1071)

levels(cancer.df$diagnosis)
cancer.df$diagnosis<-factor(cancer.df$diagnosis, levels = c("1","0"), labels = c("B","M"))

.ct <- rpart(diagnosis ~ ., data = train.df, method = "class", cp = 0, maxdepth = 4, minsplit = 1)

# print tree summary and plot tree. try different values for extra
printcp(.ct)
prp(.ct, type = 1, extra = 1, under = FALSE, split.font = 1, varlen = -10)


# classify records in the validation data using the classification tree.
# set argument type = "class" in predict() to generate predicted class membership.
ct.pred <- predict(.ct, valid.df, type = "class")

# generate confusion matrix for training data
confusionMatrix(ct.pred, as.factor(valid.df$diagnosis))

# build a deeper classification tree
max.ct <- rpart(diagnosis ~ ., data = train.df, method = "class", cp = 0, minsplit = 1, maxdepth = 30)

# count number of leaves
length(max.ct$frame$var[max.ct$frame$var == "<leaf>"])

# plot tree
prp(max.ct, type = 1, extra = 1, under = TRUE, split.font = 1, varlen = -10, 
    box.col=ifelse(max.ct$frame$var == "<leaf>", 'gray', 'white'))  

# classify records in the training data to show that the tree prefectly fits the training data.
# this is an example of overfitting
# set argument type = "class" in predict() to generate predicted class membership.
max.pred <- predict(max.ct, train.df, type = "class")

# generate confusion matrix for training data
confusionMatrix(max.pred, as.factor(train.df$diagnosis))


# classify records in the validation data.
# set argument type = "class" in predict() to generate predicted class membership.
max.pred <- predict(max.ct, valid.df, type = "class")

# generate confusion matrix for training data
confusionMatrix(max.pred, as.factor(valid.df$diagnosis))

##Pruned tree-------------------------------------------------------------------

# Create code to prune the tree

cv.ct <- rpart(diagnosis ~ ., data = cancer.df, method = "class", 
               control = rpart.control(cp = 0.00000005, minsplit = 5, xval = 5))

# use printcp() to print the table. 
printcp(cv.ct)
prp(cv.ct, type = 1, extra = 1, split.font = 1, varlen = -10)  

#prune the tree using the lowest value for xerror
#Note: the prune function requires cp as a parameter so we need to get cp for lowest value of xerror
pruned.ct <- prune(cv.ct, 
                   cp = cv.ct$cptable[which.min(cv.ct$cptable[,"xerror"]),"CP"])

#get count of the number of splits
cp_df <- data.frame(pruned.ct$cptable)
max(cp_df$nsplit)

#another way to get the count of the number of splits
pruned.ct$cptable[which.max(pruned.ct$cptable[,"nsplit"]),"nsplit"]

#get count of the number of nodes
length(pruned.ct$frame$var[pruned.ct$frame$var == "<leaf>"])

#plot the best fitting tree
prp(pruned.ct, type = 1, extra = 1, split.font = 1, varlen = -10,
    box.col=ifelse(pruned.ct$frame$var == "<leaf>", 'gray', 'white'))  

prune.pred <- predict(pruned.ct, valid.df, type = "class")

# generate confusion matrix for training data
confusionMatrix(prune.pred, as.factor(valid.df$diagnosis))


##Random Forest-----------------------------------------------------------------

library(randomForest)

#mtry = number of variables randomly sampled  as candidates at each split
#creates ntree number of trees by randomly sampling with replacement
#results are the average of all the trees
rf <- randomForest(as.factor(diagnosis) ~ ., data = train.df, 
                   ntree = 5000, mtry = 11, nodesize = 1, importance = TRUE, sampsize = 100) 

#plot the variables by order of importance
varImpPlot(rf, type = 1)

#create a confusion matrix
valid.df$diagnosis <- factor(valid.df$diagnosis)
rf.pred <- predict(rf, valid.df)
confusionMatrix(rf.pred, valid.df$diagnosis)

##_________________________End of Code________________________________________##





