############## Functions
#save.image("Phase2")
#load("Phase2")
{
  draw.text.rot <- function(t,just, i, j,rot,x=.4,y=.448) {
    library(grid)
    grid.text(t, x=x, y=y, just=just,rot=90,gp=gpar(fontsize=12))
  }
  
  draw.text.rot2 <- function(t,just, i, j,rot) {
    library(grid)
    grid.text(t, x=x[j], y=y[i], just=just,rot=90,gp=gpar(fontsize=12))
  }
  
  draw.text <- function(t,just, i, j,rot) {
    library(grid)
    grid.text(t, x=x[j], y=y[i], just=just)
  }
  
  kNNCrossVal_error <- function(form,train,norm=T){
    library(class)
    out <- tryCatch(
      {
        library(caret)
      },
      error=function(cond) {
        install.packages("caret")
        library(caret)
      })    
    
    tgtCol <- which(colnames(train) == as.character(form[[2]]))
    
    set.seed(1)
    idx <- createFolds(unlist(train[,tgtCol]), k=10)
    
    ks <- 1:12
    res <- sapply(ks, function(k) {
      ##try out each version of k from 1 to 12
      res.k <- sapply(seq_along(idx), function(i) {
        ##loop over each of the 10 cross-validation folds
        ##predict the held-out samples using k nearest neighbors
        pred <- kNN(form, train[-idx[[i]],], train[idx[[i]],], standardize=norm, k=k)
        ##the ratio of misclassified samples
        sum(unlist(train[ idx[[i]],tgtCol ]) != pred)/length(pred)
      })
      ##average over the 10 folds
      mean(res.k)
    })
    plot(res,ylab="Average Cross-Validation Error",xlab="k")
    
    return(res)
  }
  
  kNNCrossVal_error2 <- function(form,train,norm=T){
    library(class)
    out <- tryCatch(
      {
        library(caret)
      },
      error=function(cond) {
        install.packages("caret")
        library(caret)
      })    
    
    tgtCol <- which(colnames(train) == as.character(form[[2]]))
    
    set.seed(1)
    idx <- createFolds(unlist(train[,tgtCol]), k=10)
    
    ks <- (sqrt(nrow(train))-6):(sqrt(nrow(train))+6)
    res <- sapply(ks, function(k) {
      ##try out each version of k from 1 to 12
      res.k <- sapply(seq_along(idx), function(i) {
        ##loop over each of the 10 cross-validation folds
        ##predict the held-out samples using k nearest neighbors
        pred <- kNN(form, train[-idx[[i]],], train[idx[[i]],], standardize=norm, k=k)
        ##the ratio of misclassified samples
        sum(unlist(train[ idx[[i]],tgtCol ]) != pred)/length(pred)
      })
      ##average over the 10 folds
      mean(res.k)
    })
    plot(res,ylab="Average Cross-Validation Error",xlab="k")
    
    return(res)
  }
  
  kNNCrossVal_error3 <- function(form,train,norm=T){
    library(class)
    out <- tryCatch(
      {
        library(caret)
      },
      error=function(cond) {
        install.packages("caret")
        library(caret)
      })    
    
    tgtCol <- which(colnames(train) == as.character(form[[2]]))
    
    set.seed(1)
    idx <- createFolds(unlist(train[,tgtCol]), k=10)
    
    ks <- 13:24
    res <- sapply(ks, function(k) {
      ##try out each version of k from 1 to 12
      res.k <- sapply(seq_along(idx), function(i) {
        ##loop over each of the 10 cross-validation folds
        ##predict the held-out samples using k nearest neighbors
        pred <- kNN(form, train[-idx[[i]],], train[idx[[i]],], standardize=norm, k=k)
        ##the ratio of misclassified samples
        sum(unlist(train[ idx[[i]],tgtCol ]) != pred)/length(pred)
      })
      ##average over the 10 folds
      mean(res.k)
    })
    plot(res,ylab="Average Cross-Validation Error",xlab="k")
    
    return(res)
  }
  
}
##########################
library(caret)
library(rpart)
library(rpart.plot)
library(gridExtra)
library(grid)
library(gtable)
library(randomForest)
library(gbm)
library(nnet)
library(NeuralNetTools)
library(mice)
source("C:/Users/jaime/OneDrive/Escritorio/Babson/R/Data/BabsonAnalytics.R")




df_base=read.csv("C:/Users/jaime/OneDrive/Escritorio/Babson/R/TrabajoFinal/GiveMeSomeCredit/cs-training.csv")
df_base$X=NULL

#imputation Number of Dependants
idx = is.na(df_base$NumberOfDependents)
impute_sample = mice(df_base,method = "sample")
df_sample=complete(impute_sample)
df_base$NumberOfDependents=df_sample$NumberOfDependents

#imputation Monthly Income
idx = is.na(df_base$MonthlyIncome)
impute_rf = mice(df_base, method = 'rf') #rf - random forced: builds different trees and make them vote 
df_rf = complete(impute_rf)
df_base$MonthlyIncome=df_rf$MonthlyIncome


df=df_base
N = nrow(df)
training_percent=.60
trainingSize=round(training_percent*N)

set.seed(1234)
training_row=sample(1:N,trainingSize,replace = F)


##### KNN STD + KCV
df=df_base
df$SeriousDlqin2yrs=as.factor(df$SeriousDlqin2yrs)
standarizer_knn = preProcess(df,c("center","scale"))
df_knn=predict(standarizer_knn,df)
training_knn=df_knn[training_row,]
test_knn=df_knn[-training_row,]
#kk=kNNCrossVal_errorgeneral(SeriousDlqin2yrs ~ .,training_knn,knum = 40)
kk=29
model_knn_STD_CV=knn3(SeriousDlqin2yrs ~ ., data = training_knn,k=kk)
predictions_knn_STD_CV = predict(model_knn_STD_CV, test_knn,type = "class")
observation_knn_STD_CV = test_knn$SeriousDlqin2yrs
tab_STD_CV=table(predictions_knn_STD_CV, observation_knn_STD_CV)
error_rate_knn_STD_CV=sum(predictions_knn_STD_CV!=observation_knn_STD_CV)/nrow(test_knn)
errors_bench_knn_STD_CV = benchmarkErrorRate(training_knn$SeriousDlqin2yrs, test_knn$SeriousDlqin2yrs)
error_ratio_knn_STD_CV = error_rate_knn_STD_CV/errors_bench_knn_STD_CV

sensitivity_knn_STD_CV = sum(predictions_knn_STD_CV == 1 & observation_knn_STD_CV == 1)/sum(observation_knn_STD_CV == 1)
specificity_knn_STD_CV = sum(observation_knn_STD_CV == 0 & observation_knn_STD_CV == 0)/sum(observation_knn_STD_CV == 0)

#PRECISION
precision_knn_STD_CV = sum(predictions_knn_STD_CV == 1 & observation_knn_STD_CV == 1) / sum(predictions_knn_STD_CV == 1)


d <-tab_STD_CV
table <- tableGrob(d)
title <- textGrob("Observations",gp=gpar(fontsize=12))
footnote <- textGrob("0 = False, 1 = True", x=0, hjust=0,
                     gp=gpar( fontface="italic",fontsize = 8))

padding <- unit(1,"line")
table <- gtable_add_rows(table, 
                         heights = grobHeight(title) + padding,
                         pos = 0)
table <- gtable_add_rows(table, 
                         heights = grobHeight(footnote)+ padding)
table <- gtable_add_grob(table, list(title, footnote),
                         t=c(1, nrow(table)), l=c(1,2), 
                         r=ncol(table))
dev.new()
grid.newpage()
grid.draw(table)


draw.text.rot("Predictions",c("left"),2,2,x=.4,y=.448)

#########################################################
##### ##### #####      TREES ##### ##### ##### ##### #####
#########################################################

df=df_base
df$SeriousDlqin2yrs=as.factor(df$SeriousDlqin2yrs)
training=df[training_row,]
test=df[-training_row,]

model_trees = rpart(SeriousDlqin2yrs ~ ., training)

#dev.new()
#rpart.plot(model_trees)

pred_trees = predict(model_trees, test,type="class") #use if target is factor , type = "class")
observation= test$SeriousDlqin2yrs
error_trees = sum(pred_trees !=observation)/nrow(test)

stopping_tree=rpart.control(minsplit = 100,minbucket = 10,cp=0)
model_trees_overfit = rpart(SeriousDlqin2yrs ~ ., training,control = stopping_tree)
pred_trees_overfit = predict(model_trees_overfit, test,type="class")
error_trees_overfit = sum(pred_trees_overfit !=observation)/nrow(test)

#dev.new()
#rpart.plot(model_trees_overfit)


model_trees_prune=easyPrune(model_trees_overfit)
pred_trees_prune = predict(model_trees_prune, test,type="class")


error_trees_pruned = sum(pred_trees_prune !=observation)/nrow(test)

sensitivity_trees_pruned = sum(pred_trees_prune == 1 & observation == 1)/sum(observation == 1)
specificity_trees_pruned = sum(pred_trees_prune == 0 & observation == 0)/sum(observation == 0)

#PRECISION
precision_trees_pruned = sum(pred_trees_prune == 1 & observation == 1) / sum(pred_trees_prune == 1)





dev.new()
rpart.plot(model_trees_prune)

errors_bench_tree = benchmarkErrorRate(training$SeriousDlqin2yrs, test$SeriousDlqin2yrs)
error_ratio_tree = error_trees_pruned/errors_bench_tree
PercentOfTotal=round(100*model_trees_prune$variable.importance/sum(model_trees_prune$variable.importance),2)
VariableImportance=round(model_trees_prune$variable.importance,2)
varimptable=cbind(VariableImportance,PercentOfTotal)
dev.new()
grid.table(as.data.frame(varimptable))

#summary(model_trees_prune)



############# NEURAL NETS
df=df_base
df_knn$SeriousDlqin2yrs = as.factor(df$SeriousDlqin2yrs)
training=df_knn[training_row,]
test=df_knn[-training_row,]
model=nnet(SeriousDlqin2yrs ~., data = training,size=8)
#dev.new()
plotnet(model)

pred_nn= as.factor(predict(model,test,type="class"))
error_nn = sum(pred_nn!=observation)/nrow(test)

sensitivity_nn = sum(pred_nn == 1 & observation == 1)/sum(observation == 1)
specificity_nn = sum(pred_nn == 0 & observation == 0)/sum(observation == 0)

#PRECISION
precision_nn = sum(pred_nn == 1 & observation == 1) / sum(pred_nn == 1)


###STACKING###
df=df_base
df$SeriousDlqin2yrs = as.factor(df$SeriousDlqin2yrs)
pred_tree_full=predict(model_trees_prune,df,type = "class")

standarizer_knn = preProcess(df,c("center","scale"))
df_knn=predict(standarizer_knn,df)
pred_knn_full=predict(model_knn_STD_CV,df_knn,type = "class")

pred_nn_full=predict(model,df_knn,size=8)
df_stacked= cbind(df,pred_nn_full,pred_tree_full,pred_knn_full)
train_stacked=df_stacked[training_row,]
test_stacked=df_stacked[-training_row,]


trainingBoost=train_stacked
testBoost=test_stacked
trainingBoost$SeriousDlqin2yrs = as.integer(training$SeriousDlqin2yrs)-1
testBoost$SeriousDlqin2yrs = as.integer(test$SeriousDlqin2yrs)-1


######## BOOST STACKED

boost = gbm(SeriousDlqin2yrs ~., data=trainingBoost,n.trees = 5000,
            cv.folds = 4,distribution = "bernoulli")

best_size=gbm.perf(boost)

pred_boost = predict(boost, testBoost, best_size,type="response")
pred_boost = pred_boost>0.5
pred_boost=as.factor(as.integer(pred_boost))
error_boost = sum(pred_boost!=observation)/nrow(test)

sensitivity_boost = sum(pred_boost == 1 & observation == 1)/sum(observation == 1)
specificity_boost = sum(pred_boost == 0 & observation == 0)/sum(observation == 0)
precision_boost = sum(pred_boost == 1 & observation == 1) / sum(pred_boost == 1)



######################### RESULTS
results=rbind(
  
  cbind(sensitivity_knn_STD_CV ,
        specificity_knn_STD_CV ,
        precision_knn_STD_CV,
        error_rate_knn_STD_CV),
  
  cbind(sensitivity_trees_pruned ,
        specificity_trees_pruned ,
        precision_trees_pruned ,
        error_trees_pruned),
  
  cbind(sensitivity_nn ,
        specificity_nn ,
        precision_nn ,
        error_nn),
  
  cbind(sensitivity_boost, 
        specificity_boost ,
        precision_boost ,
        error_boost))


colnames(results) = c("Sensitivity","Specificity","Precision","Error Rate")
rownames(results) = c("kNN","Classification Tree","Neural Networks","Boosted (stacked)")
results=round(results,4)





###### ##### ##### #### ##### ##### ##### ##### ##### 
##### KNN Example

df=df_base
df_example=df[,c(1,3,6)]
df_example0=df_example[df_example[,1]==0,]
df_example1=df_example[df_example[,1]==1,]
x4 <- sample(1:nrow(df_example0), 10)
df_example0=df_example0[x4,]
x3 <- sample(1:nrow(df_example1), 10)
df_example1=df_example1[x3,]
df_example_plot=rbind(df_example0,df_example1,c(3,43,5900))
df_example_plot$SeriousDlqin2yrs=as.factor(df_example_plot$SeriousDlqin2yrs)
ggplot(df_example_plot,                       # Draw ggplot2 plot
       aes(x = age,
           y = MonthlyIncome,
           col = SeriousDlqin2yrs)) +
  geom_point() + 
  scale_color_manual(labels = c("False", "True","Example"), values = c("coral1", "turquoise2","purple")) +
  labs(title = "Probability of Financial Distress in the Next Two Years", x = "Age", y = "MonthlyIncome", color = "SeriousDlqin2yrs")

