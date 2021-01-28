# loading packages 
library(data.table) 
library(Amelia) 
library(corrplot) 
library(caret) 
library(glmnet) 
library(randomForest) 
library(mltools) # dummy coding 
 
# loading date set 
train_set_raw <- read.csv('C:\\Learning\\Kaggle\\House Prices Advanced Regression Techniques\\data\\house-prices-advanced-regression-techniques\\train.csv') 
test_set_raw <- read.csv('C:\\Learning\\Kaggle\\House Prices Advanced Regression Techniques\\data\\house-prices-advanced-regression-techniques\\test.csv') 
 
# drop id and response variable 
train_set <- train_set_raw[, ! names(train_set_raw) %in% c('Id','SalePrice')] 
 
# EDA 
# overview of types of features 
sapply(train_set, class) 
# number of numeric features 
sum(sapply(train_set, class) != 'character') # 36 
# number of categorical features 
sum(sapply(train_set, class) == 'character') # 43 
 
# adjust the types of features 
train_set <- within(train_set, { 
  MSSubClass <- factor(MSSubClass) 
  OverallQual <- factor(OverallQual) 
  OverallCond <- factor(OverallCond) 
  MoSold <- factor(MoSold) 
}) 
 
# missing values 
missmap(train_set, main = 'missmap of features in train set') 
sum(colSums(is.na(train_set)) != 0) # 19 
sum(rowSums(is.na(train_set)) != 0) # 1460 
missing_features <- names(train_set)[colSums(is.na(train_set)) != 0] 
missing_ratio <- colSums(is.na(train_set))/nrow(train_set) 
missing_information <- data.frame(missing_features, missing_ratio = missing_ratio[missing_features]) 
#write.csv(missing_information, 'missing_information.csv') 
 
# distribution of response variable 
hist(train_set_raw$SalePrice, main = 'Histogram of SalePrice', xlab = 'SalePrice') 
 
# correlation analysis 
cor_numeric_features <- cor(train_set[sapply(train_set, is.numeric)], use =  "complete.obs") 
corrplot(cor_numeric_features) 
 
# data cleaning  
# impute missing values 
# LotFrontage 
ls_imputation_xy_names <- c("LotFrontage","LotArea","X1stFlrSF","GrLivArea","GarageArea","PoolArea") 
ls_imputation_df <- train_set[,names(train_set) %in% ls_imputation_xy_names] 
ls_imputation_model <- lm(LotFrontage~.,data = ls_imputation_df) 
summary(ls_imputation_model) 
ls_imputation_values <- round(predict(ls_imputation_model, ls_imputation_df[is.na(ls_imputation_df$LotFrontage),]), 0) 
train_set$LotFrontage[is.na(train_set$LotFrontage)] <- ls_imputation_values 
# MasVnrType and Electrical 
Mode <- function(x) {  
  ux <- sort(unique(x)) 
  ux[which.max(tabulate(match(x, ux)))]  
} 
MasVnrType_raw <- train_set$MasVnrType 
train_set$MasVnrType <- replace(MasVnrType_raw, is.na(MasVnrType_raw), Mode(MasVnrType_raw[!is.na(MasVnrType_raw)])) 
Electrical_raw <- train_set$Electrical 
train_set$Electrical <- replace(Electrical_raw, is.na(Electrical_raw), Mode(Electrical_raw[!is.na(Electrical_raw)])) 
# MasVnrArea 
MasVnrArea_raw <- train_set$MasVnrArea 
mode_of_MasVnrType_raw <- Mode(MasVnrType_raw[!is.na(MasVnrType_raw)]) 
#MasVnrArea_raw[MasVnrType_raw == mode_of_MasVnrArea_raw] 
train_set$MasVnrArea[MasVnrType_raw == mode_of_MasVnrType_raw] <- 0 
train_set$MasVnrArea <- replace(train_set$MasVnrArea, is.na(train_set$MasVnrArea), 0) 
sum(is.na(train_set$MasVnrArea)) 
 
# replace NA with actual meaning 
miss_feature_names <- c('Alley','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1', 
                        'BsmtFinType2','FireplaceQu','GarageType','GarageYrBlt','GarageFinish', 
                        'GarageQual','GarageCond','PoolQC','Fence','MiscFeature') 
for(i in miss_feature_names){ 
  train_set[,names(train_set) == i] <- replace(train_set[,names(train_set) == i], is.na(train_set[,names(train_set) == i]), 'None') 
} 
#sum(is.na(train_set)) 
 
 
 
# YearBuilt/YrSold and SalesPrice 
cor(train_set$YrSold-train_set$YearBuilt, train_set_raw$SalePrice) 
plot(train_set$YrSold-train_set$YearBuilt, train_set_raw$SalePrice,  
     xlab = 'YrSold-YearBuilt', ylab = 'SalePrice') 
 
 
# YearBuilt/YearRemodAdd/YrSold and SalesPrice 
cor(train_set_raw$YearRemodAdd-train_set_raw$YearBuilt,train_set_raw$SalePrice) 
plot(train_set_raw$YearRemodAdd-train_set_raw$YearBuilt, train_set_raw$SalePrice,  
     xlab = 'YrSold-YearBuilt', ylab = 'SalePrice') 
 
cor(train_set_raw$YrSold-train_set_raw$YearRemodAdd,train_set_raw$SalePrice) 
plot(train_set_raw$YrSold-train_set_raw$YearRemodAdd, train_set_raw$SalePrice,  
     xlab = 'YrSold-YearRemodAdd', ylab = 'SalePrice') 
 
# build new features and remove extra features 
train_set$YearBuilt2Sold <- train_set_raw$YrSold-train_set_raw$YearBuilt 
train_set$YearRemod2Sold <- train_set_raw$YrSold-train_set_raw$YearRemodAdd 
train_set <- train_set[,! names(train_set) %in% c('YearBuilt','YearRemodAdd', 
                      'YrSold','TotalBsmtSF','LowQualFinSF','GarageYrBlt')] 
 
 
# lasso regression on continuous variables to select features 
lasso_x_df <- train_set_for_model[,names(train_set_for_model) %in% names_continuous_var] 
lambda_seq <- seq(0, 3, by = .0001) 
fit_lasso <- cv.glmnet(as.matrix(lasso_x_df),log(train_y_for_model),alpha = 1, lambda = lambda_seq,  
                       nfolds = 5) 
plot(fit_lasso) 
best_lam <- fit_lasso$lambda.min 
# Rebuilding the model with best lambda value identified 
lasso_best <- glmnet(as.matrix(lasso_x_df),log(train_y_for_model), alpha = 1, lambda = best_lam) 
coef(lasso_best) 
# remove LotArea and PoolArea 
train_set <- train_set[,! names(train_set) %in% c('LotArea','PoolArea')] 
 
 
# feature engineering on test set 
# remove features not useful for our predicting model 
test_set <- test_set_raw[,! names(test_set_raw) %in% c('Id','YearBuilt','YearRemodAdd', 
                            'YrSold','TotalBsmtSF','LowQualFinSF','GarageYrBlt')] 
# build new features 
test_set$YearBuilt2Sold <- test_set_raw$YrSold-test_set_raw$YearBuilt 
test_set$YearRemod2Sold <- test_set_raw$YrSold-test_set_raw$YearRemodAdd 
 
# adjust the types of features 
test_set <- within(test_set, { 
  MSSubClass <- factor(MSSubClass) 
  OverallQual <- factor(OverallQual) 
  OverallCond <- factor(OverallCond) 
  MoSold <- factor(MoSold) 
}) 
 
# missing values 
missmap(test_set_raw, main = 'missmap of features in test set') 
sum(colSums(is.na(test_set_raw)) != 0) # 33 
sum(rowSums(is.na(test_set_raw)) != 0) # 1459 
# names of features with missing values 
missing_features_test <- names(test_set)[colSums(is.na(test_set)) != 0] 
missing_ratio_test <- colSums(is.na(test_set))/nrow(test_set) 
missing_information_test <- data.frame(missing_features_test, missing_ratio = missing_ratio_test[missing_features_test]) 
# replace NA with actual meaning 
test_set$BsmtFinSF1 <- replace(test_set$BsmtFinSF1, is.na(test_set$BsmtFinSF1), 0) 
test_set$BsmtFinSF2 <- replace(test_set$BsmtFinSF2, is.na(test_set$BsmtFinSF2), 0) 
test_set$GarageCars <- replace(test_set$GarageCars, is.na(test_set$GarageCars), 0) 
test_set$GarageArea <- replace(test_set$GarageArea, is.na(test_set$GarageArea), 0) 
test_set$MasVnrArea <- replace(test_set$MasVnrArea, is.na(test_set$MasVnrArea), 0) 
 
miss_feature_names_test <- c('Alley','BsmtQual',"BsmtCond","BsmtExposure","BsmtFinType1", 
                             "BsmtFinType2","FireplaceQu","GarageType","GarageFinish", 
                             "GarageQual","GarageCond","PoolQC","Fence","MiscFeature",'MasVnrType') 
for(i in miss_feature_names_test){ 
  test_set[,names(test_set) == i] <- replace(test_set[,names(test_set) == i], is.na(test_set[,names(test_set) == i]), 'None') 
} 
 
# LotFrontage 
ls_imputation_xy_names_test <- c("LotFrontage","LotArea","X1stFlrSF","GrLivArea","GarageArea","TotRmsAbvGrd") 
ls_imputation_df_test <- test_set[,names(test_set) %in% ls_imputation_xy_names_test] 
ls_imputation_model_test <- lm(LotFrontage~.,data = ls_imputation_df_test) 
summary(ls_imputation_model_test) 
ls_imputation_values_test <- round(predict(ls_imputation_model_test, ls_imputation_df_test[is.na(ls_imputation_df_test$LotFrontage),]), 0) 
test_set$LotFrontage[is.na(test_set$LotFrontage)] <- ls_imputation_values_test 
# MSZoning, Utilities, Exterior1st, Exterior2nd, KitchenQual, Functional,SaleType 
name_mode_imputation_test <- c('MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd',  
                               'KitchenQual', 'Functional', 'SaleType') 
for(i in name_mode_imputation_test){ 
  ind <- which(names(test_set)==i) 
  tmp <- test_set[,ind] 
  test_set[,ind] <- replace(tmp, is.na(tmp), Mode(tmp[!is.na(tmp)])) 
} 
# BsmtUnfSF, BsmtFullBath, BsmtHalfBath 
for(i in c('BsmtUnfSF', 'BsmtFullBath', 'BsmtHalfBath')){ 
  ind <- which(names(test_set)==i) 
  tmp <- test_set[,ind] 
  test_set[,ind] <- replace(tmp, is.na(tmp), mean(tmp[!is.na(tmp)])) 
} 
#write.csv(test_set, 'predict_set.csv') 
 
 
# names of continuous features 
names_continuous_var <- names(train_set[sapply(train_set, is.numeric)]) 
# names of categorical features 
names_categorical_var <- names(train_set[!sapply(train_set, is.numeric)]) 
 
# special topic: cardinality 
# explore the cardinality of each categorical variable in train.csv 
# define function 
cardinality_num <- function(names_variable, data_set){ 
  cardinality_feature <- vector(length = length(names_variable)) 
  cardinality_categories_num <- vector(length = length(names_variable)) 
  for(i in 1:length(names_variable)){ 
    cardinality_feature[i] <- names_variable[i] 
    cardinality_categories_num[i] <- length(unique(data_set[,names(data_set)==names_variable[i]])) 
  } 
  cardinality_csv <- data.frame(cardinality_feature = cardinality_feature,  
                                cardinality_categories_num = cardinality_categories_num) 
  return(cardinality_csv) 
} 
 
cardinality_train <- cardinality_num(names_categorical_var, train_set) 
 
#write.csv(cardinality_train, 'cardinality_train.csv') 
 
# explore the cardinality of each categorical variable in test.csv 
cardinality_test <- cardinality_num(names_categorical_var, test_set) 
 
#write.csv(cardinality_test, 'cardinality_test.csv') 
 
# explore the number of categories of features which in test_set but not in train_set(key = 0) 
# explore the number of categories of features which in train_set but not in test_set(key = 1) 
cardinality_cross_set <- function(names_variable, train_set, test_set, key){ 
  cardinality_feature <- vector(length = length(names_variable)) 
  cardinality_categories_num <- vector(length = length(names_variable)) 
  cardinality_categories_unique <- vector(length = length(names_variable)) 
  for(i in 1:length(names_variable)){ 
    feature <- names_variable[i] 
    if(key == 0){ 
      num <- sum(!test_set[,names(test_set) == feature] %in%  
                   train_set[,names(train_set) == feature]) 
      unique_num <- length(unique(test_set[,names(test_set) == feature][!test_set[,names(test_set) == feature]  
                                                                   %in% train_set[,names(train_set) == feature]])) 
    }else if(key == 1){ 
      num <- sum(!train_set[,names(train_set) == feature] %in%  
                   test_set[,names(test_set) == feature]) 
      unique_num <- length(unique(train_set[,names(train_set) == feature][!train_set[,names(train_set) == feature]  
                                                                   %in% test_set[,names(test_set) == feature]])) 
    }else{ 
      print('there is a bug!') 
    } 
    cardinality_feature[i] <- feature 
    cardinality_categories_num[i] <- num 
    cardinality_categories_unique[i] <- unique_num 
  } 
  cardinality_test_set_csv <- data.frame(feature = cardinality_feature, 
                                         num = cardinality_categories_num, 
                                         unique_num = cardinality_categories_unique) 
  return(cardinality_test_set_csv) 
} 
 
cardinality_test_set <- cardinality_cross_set(names_categorical_var,  
                                              train_set, test_set, key = 0) 
 
cardinality_train_set <- cardinality_cross_set(names_categorical_var,  
                                              train_set, test_set, key = 1) 
#write.csv(cardinality_test_set, 'cardinality_test_set.csv') 
#write.csv(cardinality_train_set, 'cardinality_train_set.csv') 
 
 
 
# dealing with cardinality 
# distribution of MSSubClass in both train set and test set 
histogram(train_set$MSSubClass, xlab = 'MSSubClass', ylab = 'count', 
            main = 'the distribution of MSSubClass in train set') 
histogram(test_set$MSSubClass, xlab = 'MSSubClass', ylab = 'count', 
          main = 'the distribution of MSSubClass in test set') 
#write.csv(table(train_set$MSSubClass)/nrow(train_set), 'dist_train_MSSubClass.csv') 
#write.csv(table(test_set$MSSubClass)/nrow(test_set),'dist_test_MSSubClass.csv') 
# combine '150' and '40' to one category 
test_set$MSSubClass[test_set$MSSubClass == '150'] <- '40' 
 
 
# partition train set and test set 
set.seed(1) 
train_ind <- createDataPartition(train_set_raw$SalePrice, p=0.8, list=FALSE) 
test_ind <- setdiff(seq(1,nrow(train_set)), train_ind) 
train_set_for_model <- cbind(train_set[train_ind,],SalePrice = train_set_raw$SalePrice[train_ind]) 
test_set_for_model <- cbind(train_set[-train_ind,],SalePrice = train_set_raw$SalePrice[-train_ind]) 
 
# modeling 
# random forest 
# define tuning function 
rf_tuning_func <- function(min_ntree, max_ntree, sep_ntree, min_nodesize,  
                           max_nodesize, sep_nodesize, train_set, test_set, response_var){ 
  set.seed(1) 
  ntree_seq <- seq(min_ntree, max_ntree, by = sep_ntree) 
  nodesize_seq <- seq(min_nodesize, max_nodesize, by = sep_nodesize) 
  parameter_mt <- matrix(nrow = length(ntree_seq), ncol = length(nodesize_seq)) 
  rmse_list <- vector(length = length(ntree_seq)*length(nodesize_seq)) 
  colnames(parameter_mt) <- nodesize_seq 
  rownames(parameter_mt) <- ntree_seq 
  n = 1 
  for(i in 1:length(ntree_seq)){ 
    for(j in 1:length(nodesize_seq)){ 
      rf_model <- randomForest(SalePrice~., data = train_set, ntree = ntree_seq[i], 
                               nodesize = nodesize_seq[j]) 
      rf_pred <- predict(rf_model, test_set) 
      rmse <- RMSE(rf_pred, test_set[,names(test_set)==response_var]) 
      parameter_mt[i, j] <- rmse 
      rmse_list[n] <- rmse 
      cat('Progress: ',round(n*100/(length(ntree_seq)*length(nodesize_seq)),2), "%","\r", sep = '') 
      flush.console() 
      n <- n + 1 
    } 
  } 
  #heatmap(parameter_mt, keep.dendro = F) 
  min_rmse <- min(parameter_mt) 
  return(paste('ntree: ',which(parameter_mt==min_rmse, arr.ind=TRUE)[1],  
               '; nodesize: ',which(parameter_mt==min_rmse, arr.ind=TRUE)[2], 
               '; min_rmse: ',min_rmse, sep = '')) 
} 
# tuning parameters 
rf_tuning_func(10,200,1,10,100,1,train_set_for_model,test_set_for_model,'SalePrice') 
# "ntree: 10; nodesize: 53; min_rmse: 27305.1811509686" 
 
rf_tuning_func <- function(min_ntree, max_ntree, sep_ntree, min_nodesize,  
                           max_nodesize, sep_nodesize, train_set, test_set, response_var){ 
  set.seed(1) 
  ntree_seq <- seq(min_ntree, max_ntree, by = sep_ntree) 
  nodesize_seq <- seq(min_nodesize, max_nodesize, by = sep_nodesize) 
  ntree_result <- vector(length = length(ntree_seq)*length(nodesize_seq)) 
  nodesize_result <- vector(length = length(ntree_seq)*length(nodesize_seq)) 
  rmse_result <- vector(length = length(ntree_seq)*length(nodesize_seq)) 
  n = 1 
  for(i in ntree_seq){ 
    for(j in nodesize_seq){ 
      rf_model <- randomForest(SalePrice~., data = train_set, ntree = i, 
                               nodesize = j) 
      rf_pred <- predict(rf_model, test_set) 
      rmse <- RMSE(rf_pred, test_set[,names(test_set)==response_var]) 
      ntree_result[n] <- i 
      nodesize_result[n] <- j 
      rmse_result[n] <- rmse 
      cat('Progress: ',round(n*100/(length(ntree_seq)*length(nodesize_seq)),2), "%","\r", sep = '') 
      flush.console() 
      n <- n + 1 
    } 
  } 
  #heatmap(parameter_mt, keep.dendro = F) 
  min_ind <- which.min(rmse_result) 
  return(paste('ntree: ',ntree_result[ind],  
               '; nodesize: ',nodesize_result[ind], 
               '; min_rmse: ',rmse_result[min_ind], sep = '')) 
} 
 
# linear regression 
# frequency encoding 
#dataframe: entire data; var: names of categorical variables 
frequencyEncoding <- function(dataframe, var){ 
  pos <- match(var, names(dataframe)) 
  for(i in pos){ 
    target_col <- dataframe[,i] 
    dict <- table(target_col)/nrow(dataframe) 
    names_key <- names(dict) 
    dataframe[,i] <- dict[match(target_col, names_key)] 
  } 
  return(dataframe) 
} 
 
train_set_freq <- frequencyEncoding(train_set, names_categorical_var) 
train_set_model_freq <- cbind(train_set_freq[train_ind,],SalePrice = train_set_raw$SalePrice[train_ind]) 
test_set_model_freq <- cbind(train_set_freq[test_ind,],SalePrice = train_set_raw$SalePrice[-train_ind]) 
 
test <- lm(SalePrice~., data = train_set_model_freq) 
summary(test) 
# fit the linear model 
lm_model <- function(train_set,test_set, response_var){ 
  model <- lm(SalePrice~., data = train_set) 
  pred <- predict(model, test_set) 
  rmse <- RMSE(pred, test_set[, names(test_set)==response_var]) 
  return(rmse) 
} 
lm_model(train_set_model_freq, test_set_model_freq, 'SalePrice')
