
####################################################################################################################
#Titanic Capstone Project

#This project analyzes the Titanic data, uses different models and predictors to train and predict the survival of the Titanic passengers. 
#We use the full Titanic extended dataset (Kaggle + Wikipedia) at https://www.kaggle.com/pavlofesenko/titanic-extended?select=full.csv which has more completed data than the one in the Titanic package. 

#The Titanic extended dataset include extra age data in Age_wiki. We will remove the Age field and use Age_wiki as Age. 
#Other extra data will be used for our training are Embarked, Cabin and Lifeboat. We will remove other unused extra fields such as Name, Hometown, WikiId and Destination. 
#The full data set will be split into training and test data sets. 

#Using the skills and knowledge learning from the EDX PH125.8x: Data Science: Machine Learning course and other parts from the Data science course we will experiment different data modeling methods on the Titanic data training and prediction for the survival of the Titanic passenger.

####################################################################################################################  
  
  
####################################################################################################################
## Data Generation
####################################################################################################################
###Import the Titanic full.csv data from Kaggle.  
#install required libraries if they have not been installed
if(!require(caret)) 
  install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(tidyverse)) 
  install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(rpart)) 
  install.packages("randomForest", repos = "http://cran.us.r-project.org")
#load required libraries
library(caret, quietly = TRUE)
library(tidyverse, quietly = TRUE)
library(rpart, quietly = TRUE)
options(warn = -1)
options(pillar.subtle=FALSE)
options(pillar.sigfig = 7)

#import data from CSV
full <- read.csv("full.csv", stringsAsFactors = TRUE)


####################################################################################################################
## Data Cleaning
####################################################################################################################
###Clean unused data columns to free memory. Prepare data for Age, Embarked, Lifeboat, FamilySize and Cabin. Fill empty factor data in Embarked, Lifeboat and Cabin. Remove temporary data.

### Examine NA data and data dimension
# extract only data contained survived for training and test
full_clean <- full[!is.na(full$Survived),]
#Examine NA data and data dimension
colSums(is.na(full_clean)) 
dim(full_clean)

### Select feature predictors
#  select feature predictors
full_clean <- full_clean %>%
  select(Survived,  Sex, Pclass, Age_wiki, Fare, SibSp, Parch, Embarked, Lifeboat, Cabin)

### Process age and family size data

# process age and family size 
full_clean <- full_clean %>%
  mutate(Survived = factor(Survived),
         # NA age to median age
         Age = ifelse(is.na(Age_wiki), median(Age_wiki, na.rm = TRUE), Age_wiki), 
         # count family members
         FamilySize = SibSp + Parch + 1) %>%    
  select(Survived, Sex, Pclass, Age, Fare, SibSp, Parch, 
         FamilySize, Embarked, Lifeboat, Cabin)

### Fill empty factor data in Embarked, Lifeboat and Cabin.
# fill empty Embarked with X
levels(full_clean$Embarked)[match("",levels(full_clean$Embarked))] <- "X"

#Examine NA data again
colSums(is.na(full_clean)) 
#Examine full_clean data 
str(full_clean)

# Keep only cabin character
full_clean$CabinN <- vector("character", nrow(full_clean))
for (i in 1:nrow(full_clean)) {
  pattern <- "[0-9]*|\\s"
  full_clean$CabinN[i] <- substr(gsub(pattern, "", full_clean$Cabin[i]),1,1)
}
full_clean$Cabin <- factor(full_clean$CabinN)
full_clean$CabinN <- NULL

# fill empty Cabin with X
levels(full_clean$Cabin)[match("",levels(full_clean$Cabin))] <- "X"

# fill empty Lifeboat with X, ? with Q
levels(full_clean$Lifeboat)[match("",levels(full_clean$Lifeboat))] <- "X"
levels(full_clean$Lifeboat)[match("?",levels(full_clean$Lifeboat))] <- "Q"

### Check cabin and lifeboat categories, Examine full_clean data
#check cabin categories
levels(full_clean$Cabin)
#check lifeboat categories
levels(full_clean$Lifeboat)
#Examine full_clean data again
str(full_clean)
dim(full_clean)
#Remove temporary data
rm(full)


####################################################################################################################
## Data Preparation 
####################################################################################################################
###Generate training and test set by 80% and 20% split to make it consistent with the principle that 
#most of the data are in the training set, also align with the 80/20 Pareto principle. 
set.seed(8,sample.kind = "Rounding")

#Generate training and test set by 80% and 20% split.
test_index <- createDataPartition(full_clean$Survived, times = 1, p = 0.2, list = FALSE)
train_set <- full_clean[-test_index,]
test_set <- full_clean[test_index,]



####################################################################################################################
## Data Exploration and Visualization
####################################################################################################################
### Overall Survival rate
mean(full_clean$Survived == 1)

### Survival rate  by Sex
#We can see female survival rate is higher, and among the first two Pclass this is particularly obvious. 
full_clean %>% 
  ggplot(aes(Sex,y=..count.., fill = Survived)) +
  geom_bar(alpha = 0.3)

full_clean %>%
  ggplot(aes(Age, y = ..count.., fill = Survived)) +
  geom_density(alpha = 0.2, position = "identity") +
  facet_grid(Sex ~ Pclass)

### Survival rate  by Pclass
#Pclass 1 and 2 have higher survival rate, but for male passengers the survival rate is relatively low across all Pclass. 
full_clean %>% 
  ggplot(aes(Pclass,y=..count.., fill = Survived)) +
  geom_bar(alpha = 0.3, position = "stack")

full_clean %>%
  ggplot(aes(Pclass, y = ..count.., fill = Survived)) +
  geom_bar(alpha = 0.3, position = "stack") +
  facet_grid(Sex ~ .)

### Survival rate  by Family size
#Male passengers with low family number have lower survival rate; female passengers seem to be the opposite.
full_clean %>%
  ggplot(aes(FamilySize, y = ..count.., fill = Survived)) +
  #geom_density(alpha = 0.2, position = "stack") +
  geom_density(alpha = 0.2) +
  scale_x_continuous(trans = "log2")

full_clean %>%
  ggplot(aes(FamilySize, y = ..count.., fill = Survived)) +
  #geom_density(alpha = 0.2, position = "stack") +
  geom_density(alpha = 0.2) +
  scale_x_continuous(trans = "log2")+
  facet_grid(Sex ~ .)

### Survival rate  by Fare
#It seems lower fare indicates lower survival rates, this might be related to Pclass and Cabin, and how easy it is for them to access a Lifeboat. 
full_clean %>%
  ggplot(aes(Fare, y = ..count.., fill = Survived)) +
  geom_density(alpha = 0.2) +
  scale_x_continuous(trans = "log2")

full_clean %>%
  ggplot(aes(Fare, y = ..count.., fill = Survived)) +
  geom_density(alpha = 0.2) +
  scale_x_continuous(trans = "log2")+
  facet_grid(Sex ~ .)

full_clean %>% 
  filter(Fare > 0) %>% 
  ggplot(aes(Survived,Fare)) + geom_boxplot()+
  geom_point(position = position_jitter(width = 0.1), alpha = 0.1) +
  scale_y_continuous(trans = "log2")

### Survival rate  by Age
#Male passengers aged 20-40 and older passengers has less chance to survive, latter is particularly the case for people aged over 60. 
full_clean %>%
  ggplot(aes(Age, y = ..count.., fill = Survived)) +
  geom_density(alpha = 0.2) 

full_clean %>%
  ggplot(aes(Age, y = ..count.., fill = Survived)) +
  geom_density(alpha = 0.2) +
  facet_grid(Sex ~ .)

full_clean %>% 
  ggplot(aes(Survived,Age)) + geom_boxplot()+
  geom_point(position = position_jitter(width = 0.1), alpha = 0.1) 

### Survival rate  by Embarked
#Passenger embarked data marked with S seems to has less chance to survive, however this may be to do with the fact that more people are in this category. 

full_clean %>%
  ggplot(aes(Embarked, y = ..count.., fill = Survived)) +
  geom_bar(alpha = 0.3) 

full_clean %>%
  ggplot(aes(Embarked, y = ..count.., fill = Survived)) +
  geom_bar(alpha = 0.3) +
  facet_grid(Sex ~ .)

### Survival rate  by Lifeboat
#We can see that passengers that assigned with a lifeboat have a very high chance to survive. 
full_clean %>%
  ggplot(aes(Lifeboat, y = ..count.., fill = Survived)) +
  geom_bar(alpha = 0.3,position="fill") 

full_clean %>%
  ggplot(aes(Lifeboat, y = ..count.., fill = Survived)) +
  geom_bar(alpha = 0.3,position="fill") +
  facet_grid(Sex ~ .)

#**survival rate when Liftboat data is empty**
mean(full_clean[full_clean$Lifeboat=="X",]$Survived == 1)
#**survival rate when Liftboat data is not empty**
mean(full_clean[full_clean$Lifeboat!="X",]$Survived == 1)

### Survival rate  by Cabin
#Passengers without a cabin have much lower chance to survive. 
full_clean %>%
  ggplot(aes(Cabin, y = ..count.., fill = Survived)) +
  geom_bar(alpha = 0.3) 

full_clean %>%
  ggplot(aes(Cabin, y = ..count.., fill = Survived)) +
  geom_bar(alpha = 0.3) +
  facet_grid(Sex ~ .)


####################################################################################################################
### Modeling Approaches 
#The data has been split into training and test sets by 80% and 20%. We will use the training data set to experiment different modeling methods, and then use the test data to predict and calculate the prediction accuracy. 
#The project target is set to be 86% accuracy but the final rate is expected to be much higher than this. As the insight we get from previous data analysis and visualization section, 
#the passengers whose Lifeboat data is not empty has a survival rate over 98%, survival rate of those with empty Lifeboat data is merely 0.18%. We can see whether passenger can get on a lifeboat is the deciding factor for their survival. 
#We will remove Lifeboat data first for the training and add the Lifeboat data at the end to see if that confirm our expectation that Lifeboat data is the major deciding factor and will significantly increase the prediction accuracy. 
#It also makes sense to remove the Lifeboat data first since Lifeboat data is unknown before the tragedy happened. 

#Linear discriminant analysis (LDA) and Quadratic discriminant analysis (QDA) will be first used for the training. Different combinations of factors will be trialed and the ones with the highest prediction accuracy will be chosen. 
#Logistic regression of Generalized linear model with all factors excluding Lifeboat will also be used. They will be compared with k-nearest neighbors (kNN), Classification tree and Random forest modelings. 
#We will then choose the best prediction accuracy among the modelings for our Ensemble model. Finally we will add the Lifeboat data back to some models to see if it will significantly increase the prediction accuracy. 
####################################################################################################################



####################################################################################################################
# Modeling Results
####################################################################################################################

####################################################################################################################
## Linear discriminant analysis (LDA) model 
####################################################################################################################
###LDA model with Sex, Pclass, Fare and Age:
#After test adding other factors, LDA with Sex, Pclass, Fare and Age produces the highest accuracy among different factors using LDA. 
model_lda <- train(Survived ~ Sex + Pclass + Fare + Age, data = train_set, method = 'lda')
predict_lda <- predict(model_lda, test_set)
#Add project target to result table
result <- tibble(Method = "Project Target", Accuracy = 0.86)

#add prediction accuracy to the result, and print
result <- bind_rows(result,tibble(
  Method = "LDA model with Sex,Pclass,Fare,Age",
  Accuracy = mean(test_set$Survived == predict_lda)))
result

####################################################################################################################
## Quadratic discriminant analysis (QDA) model 
####################################################################################################################
###QDA model with Sex, Pclass, Fare, Age and FamilySize:
#After test adding other factors, QDA with with Sex, Pclass, Fare, Age and FamilySize produces the highest accuracy among different factors using QDA, however it is lower than LDA model result. 

model_qda <- train(Survived ~ Sex + Pclass + Fare + Age + FamilySize, data = train_set, method = 'qda')
predict_qda <- predict(model_qda, test_set)
#add prediction accuracy to the result, and print
result <- bind_rows(result,tibble(
  Method = "QDA model with Sex,Pclass,Fare,Age,FamilySize",
  Accuracy = mean(test_set$Survived == predict_qda)))
result

####################################################################################################################
## Logistic regression of Generalized linear model
####################################################################################################################
###Logistic regression of Generalized linear model with all data excluding Lifeboat data:
#Logistic regression model using glm is also lower than LDA results
model_log <- glm(Survived ~ . -Lifeboat, data=train_set, family="binomial")
model_log$xlevels[["Embarked"]] <- union(model_log$xlevels[["Embarked"]], 
                                         levels(test_set$Embarked))
model_log$xlevels[["Lifeboat"]] <- union(model_log$xlevels[["Lifeboat"]], 
                                         levels(test_set$Lifeboat))
#predict testset with trained model
predict_log <- ifelse(predict(model_log, test_set) >= 0, 1, 0)
#add prediction accuracy to the result, and print
result <- bind_rows(result,tibble(
  Method = "Logistic regression of glm",
  Accuracy = mean(predict_log == test_set$Survived)))
result

#estimate variable importance
varImp(model_log)


####################################################################################################################
## kNN model excluding Lifeboat data
####################################################################################################################
###Both kNN and cross-validated kNN models have lower accuracy than previous models. Unlike the training set, The prediction on test set is lower on the cross-validated kNN.
k <- seq(3,51,2)
model_knn <- train(Survived ~ . -Lifeboat, data = train_set, method = "knn", 
                   tuneGrid = data.frame(k))
#find the best k
model_knn$bestTune

#show knn plot
ggplot(model_knn)
#predict testset with trained model
predict_knn <- predict(model_knn, test_set) %>% factor(levels = levels(test_set$Survived))
cm_test <- confusionMatrix(data = predict_knn, reference = test_set$Survived)

#add prediction accuracy to the result, and print
result <- bind_rows(result,tibble(
  Method = "kNN model",
  Accuracy = cm_test$overall["Accuracy"]))
result

####################################################################################################################
## Cross-validated kNN model excluding Lifeboat data
####################################################################################################################
model_knn_cv <- train(Survived ~ . -Lifeboat, 
                      data=train_set, 
                      method = "knn",
                      tuneGrid = data.frame(k = seq(3, 51, 2)),
                      trControl = trainControl(method = "cv", number=10, p=0.9))
#show model plot
ggplot(model_knn_cv)
#predict testset with trained model
predict_knn_cv <- predict(model_knn_cv, test_set)
cm_test <- confusionMatrix(data = predict_knn_cv, reference = test_set$Survived)

#add prediction accuracy to the result, and print
result <- bind_rows(result,tibble(
  Method = "cross-validated kNN model",
  Accuracy = cm_test$overall["Accuracy"]))
result

####################################################################################################################
## Classification tree model excluding Lifeboat data
####################################################################################################################
###The classification tree model has significantly higher performance than other models
model_rpart <- train(Survived ~ . -Lifeboat, 
                     data=train_set, 
                     method = "rpart",
                     tuneGrid = data.frame(cp = seq(0, 0.05, 0.002)),
                     trControl = trainControl(method = "cv", number=10, p=0.9))
#show model plot
plot(model_rpart)
#predict testset with trained model
predict_rpart <- predict(model_rpart, test_set)
cm_test <- confusionMatrix(data = predict_rpart, reference = test_set$Survived)
#add prediction accuracy to the result, and print
result <- bind_rows(result,tibble(
  Method = "Classification tree model",
  Accuracy = cm_test$overall["Accuracy"]))
result

#print and visualize final model
model_rpart$finalModel
plot(model_rpart$finalModel, margin=0.1)
text(model_rpart$finalModel, cex = 0.75)

####################################################################################################################
## Random forest model excluding Lifeboat data
####################################################################################################################
###Random forest model accuracy on test set prediction is a bit lower than the classification tree model but it is higher than other models.
model_rf <- train(Survived ~. -Lifeboat, 
                  data = train_set,
                  method = "rf", 
                  tuneGrid = data.frame(mtry = seq(1, 30)), 
                  ntree = 100)
#best tuning value
model_rf$bestTune
#show model plot
plot(model_rf)
#predict testset with trained model
predict_rf <- predict(model_rf, test_set)

#add prediction accuracy to the result, and print
result <- bind_rows(result,tibble(
  Method = "Random forest model",
  Accuracy = mean(predict_rf == test_set$Survived)))
result

#estimate variable importance
varImp(model_rf)

####################################################################################################################
## Ensemble of different models
####################################################################################################################
###We will choose the models with accuracy higher than 80% for our ensemble: LDA, QDA, Logistic regression, Classification tree model and Random forest. We get the same result as our Random forest. 
ensemble <- cbind(lda=ifelse(predict_lda == "0", 0, 1), 
                  qda=ifelse(predict_qda == "0", 0, 1), 
                  log=ifelse(predict_log == "0", 0, 1), 
                  rpart=ifelse(predict_rpart == "0", 0, 1), 
                  rf=ifelse(predict_rf== "0", 0, 1))
#ensemble prediction
ensemble_predict <- ifelse(rowMeans(ensemble) < 0.5, 0, 1)

#add prediction accuracy to the result, and print
result <- bind_rows(result,tibble(
  Method = "Ensemble of different models",
  Accuracy = mean(ensemble_predict == test_set$Survived)))
result

####################################################################################################################
## Classification tree model including Lifeboat data
####################################################################################################################
###Finally we add Lifeboat data back to the training. We get over 92% from the classification tree model. 
model_rpart_lb <- train(Survived ~ ., 
                        data=train_set, 
                        method = "rpart",
                        tuneGrid = data.frame(cp = seq(0, 0.05, 0.002)),
                        trControl = trainControl(method = "cv", number=10, p=0.9))
#plot model
plot(model_rpart_lb)
#predict testset with trained model
predict_rpart_lb <- predict(model_rpart_lb, test_set)
cm_test <- confusionMatrix(data = predict_rpart_lb, reference = test_set$Survived)
#add prediction accuracy to the result, and print
result <- bind_rows(result,tibble(
  Method = "Classification tree model including Lifeboat",
  Accuracy = cm_test$overall["Accuracy"]))
result

#print and visualize final model
model_rpart_lb$finalModel
plot(model_rpart_lb$finalModel, margin=0.1)
text(model_rpart_lb$finalModel, cex = 0.75)

####################################################################################################################
## Random forest model including Lifeboat data
####################################################################################################################
###The random forest model get over 97% accuracy with the extra Lifeboat data, which is close to the survival rate of passengers managed to get on a lifeboat or with lifeboat data in their record. 
model_rf_lb <- train(Survived ~ ., 
                     data = train_set,
                     method = "rf", 
                     tuneGrid = data.frame(mtry = seq(1, 50)), 
                     ntree = 100)
#best tuning value
model_rf_lb$bestTune
#plot model
plot(model_rf_lb)
#predict testset with trained model
predict_rf_lb <- predict(model_rf_lb, test_set)

#add prediction accuracy to the result, and print
result <- bind_rows(result,tibble(
  Method = "Random forest model including Lifeboat",
  Accuracy = mean(predict_rf_lb == test_set$Survived)))
result


#estimate variable importance
varImp(model_rf_lb)

####################################################################################################################
# Conclusion
#In this project we test different models to predict the survival of the Titanic passengers using the full Titanic extended dataset. 
#The Classification tree, Random forest and Ensemble all achieve over 86% accuracy. 
#Adding the Lifeboat data with Classification tree and Random forest models reaches over 92% and 97% accuracy respectively, which confirms our analysis that the Lifeboat data is the major deciding factor and can significantly increase the prediction accuracy. 

####################################################################################################################

