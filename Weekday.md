Weekday
================
Zack Vaskalis
07/03/2020

  - [Introduction](#introduction)
      - [Information on Attributes for
        Analysis:](#information-on-attributes-for-analysis)
  - [Data](#data)
  - [Summarizations](#summarizations)
  - [Modeling](#modeling)

## Introduction

This project uses, for analysis, the Online News Popularity Data Set
which can be found at the University of California Irvine [UCI Machine
Learning Repository, Center for Machine Learning and Intelligent
Systems](https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity#).
The dataset being used here was compiled by several researchers in
Portugal, who collected articles from [Mashable](www.mashable.com) and
compiled associated statistics for a period of two years, with the goal
of being able to predict the number of shares in social networks (a
measure of popularity). The dataset has over 39,000 records of
information spanning the two year time period from January 07, 2013 to
December 27th, 2014. The dataset also contains 61 variables, of which 58
are predictive attributes, 2 non-predictive variables(the source url and
the time between article publication and dataset acquisition), and
finally the goal field for prediction: the number of shares.

### Information on Attributes for Analysis:

Out of the 61 Attributes available in this dataset for analysis, in
order not to overfit the model with too many dependent variables (which
could yield not enough generalization of the dataset to make
predictions, especially valid ones), I have chosen to use the following
7 variables, 6 which are predictive, and 1, the shares variable, which
is outcome variable. The weekday variables are also included, but only
for subsetting the data by the day of the week. Once we have the
“Monday” dataset, for instance, all of these `weekday_is_*`
variables will be removed from the dataset. The variables selected are
below, and the reasoning behind their selection follows.

Attribute Information:  
1\. n\_tokens\_content: Number of words in the content  
2\. n\_unique\_tokens: Rate of unique words in the content  
3\. average\_token\_length: Average length of the words in the content  
4\. global\_subjectivity: Text subjectivity  
5\. rate\_positive\_words: Rate of positive words among non-neutral
tokens  
6\. avg\_positive\_polarity: Avg. polarity of positive words  
7\. shares: Number of shares (target outcome trying to predict)  
8\. weekday\_is\_variables: Was the article published on a
\_\_\_\_\_\_\_?

  - weekday\_is\_monday: Was the article published on a Monday?  
  - weekday\_is\_tuesday: Was the article published on a Tuesday?  
  - weekday\_is\_wednesday: Was the article published on a Wednesday?  
  - weekday\_is\_thursday: Was the article published on a Thursday?  
  - weekday\_is\_friday: Was the article published on a Friday?  
  - weekday\_is\_saturday: Was the article published on a Saturday?  
  - weekday\_is\_sunday: Was the article published on a Sunday?

<!-- end list -->

``` r
#load necessary libraries
library(rmarkdown)
library(tidyverse)
library(caret)
library(lattice)
library(ggplot2)
library(dplyr)
library(haven)
library(rgl)
library(knitr)
library(tree)
library(randomForest)
library(psych)
library(DT)
library(summarytools)
```

## Data

``` r
# Read-in entire dataset
path <- "C:/Users/Zachary Vaskalis/Dropbox/ST558/OnlineNewsPopularity.csv"
weekdayDataRAW <- read_csv(path)

# Select only variables I am choosing to use for the analysis.
weekdayData1 <- select(weekdayDataRAW, n_tokens_content, n_unique_tokens,
                       average_token_length, weekday_is_monday, weekday_is_tuesday,
                       weekday_is_wednesday, weekday_is_thursday, weekday_is_friday,
                       weekday_is_saturday, weekday_is_sunday, global_subjectivity,
                       rate_positive_words, avg_positive_polarity, shares)

# Select only the specific day of the week I am interested in.
weekdayData2 <- filter(weekdayData1, weekday_is_monday == 1)

# Now we know we only have the day of the week filtered by above.
# So, we can remove all of the weekday_is_* variables now as they are no longer needed.
# Additionally, these variables are constant.  The selected weekday would contain all 1s,
# while the other weekdays are all 0s.  Thus, they are no longer useful for analysis.
weekdayData3 <- select(weekdayData2, -(weekday_is_monday:weekday_is_sunday))
```

``` r
# Get random number from the computer clock using Sys.time()
initial_seed <- Sys.time()
initial_seed <- as.integer(initial_seed)
print (initial_seed)
```

    ## [1] 1593785388

``` r
seed <- initial_seed %% 100000
#print(seed)

# -----------------RUN FROM HERE NOW THAT YOU HAVE RANDOM SEED-------------------------
# For reproducability, set seed using outcome of above process:
set.seed(42131)

# Get random sample of row numbers from large dataset to split data into train and test:
# 70% of data for training
train <- sample(1:nrow(weekdayData3), size = nrow(weekdayData3)*.70) 
# 30% of data for testing
test <- dplyr::setdiff(1:nrow(weekdayData3), train)

# Subset the Data into training and testing sets, using rows from train and test:
trainingData <- weekdayData3[train,]
testingData <- weekdayData3[test,]
```

## Summarizations

The summaries below are related to the training dataset. The table that
appears below is a summary of all of the variables from the training
dataset using the `describeBy()` function from the `psych` package.

``` r
# Summary statistics for all variables in the training dataset using the
# describeBy() function from the psych package
summaryData <- psych::describeBy(trainingData, trainingData$type)

x <- tibble(row.names(summaryData), summaryData) 
colnames(x)[1] <- "variable"
x[,4:10] <- round(x[,4:10],4)
y <- select(x,1,3,4,5,6,8,9,10)
datatable(arrange(y,desc(variable)))
```

![](Weekday_files/figure-gfm/summaryStatistics-1.png)<!-- -->

``` r
#histogram(trainingData$n_tokens_content)
#plot(trainingData$n_tokens_content,trainingData$shares)
#plot(trainingData$n_tokens_title,trainingData$shares)
#plot(trainingData$n_unique_tokens,trainingData$shares)
#plot(trainingData$global_subjectivity,trainingData$shares)


#trainingData$n_tokens_content <- log(trainingData$n_tokens_content + 1)
#trainingData$shares <- log(trainingData$shares + 1)

#trainingData <- filter(trainingData, n_tokens_content > 0)
```

## Modeling

``` r
# Multiple Regression Model chosen using all 20 predictor variables to try to predict
# shares.  This will also allow for comparison to the random forest model, and RMSE 
# values will be compared between the two models.

mlr1 <- lm(shares ~ ., data = trainingData)
summary(mlr1)
```

    ## 
    ## Call:
    ## lm(formula = shares ~ ., data = trainingData)
    ## 
    ## Residuals:
    ##    Min     1Q Median     3Q    Max 
    ##  -8224  -3031  -2053   -556 686348 
    ## 
    ## Coefficients:
    ##                         Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)            4778.3406  1406.5477   3.397 0.000686 ***
    ## n_tokens_content          0.3399     0.7685   0.442 0.658245    
    ## n_unique_tokens        3762.6855  3483.3409   1.080 0.280112    
    ## average_token_length  -1612.1299   545.6755  -2.954 0.003149 ** 
    ## global_subjectivity   10182.0603  2983.9765   3.412 0.000650 ***
    ## rate_positive_words   -1473.9988  1597.8123  -0.923 0.356310    
    ## avg_positive_polarity  1892.9030  3094.9280   0.612 0.540823    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 16850 on 4655 degrees of freedom
    ## Multiple R-squared:  0.004685,   Adjusted R-squared:  0.003402 
    ## F-statistic: 3.652 on 6 and 4655 DF,  p-value: 0.001279

``` r
trCtrl <- trainControl(method = "cv", number = 10)
mlr2 <- train(shares~., data=trainingData, method="lm",metric = "RMSE", trControl=trCtrl)
mlr2
```

    ## Linear Regression 
    ## 
    ## 4662 samples
    ##    6 predictor
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 4195, 4196, 4195, 4197, 4195, 4196, ... 
    ## Resampling results:
    ## 
    ##   RMSE      Rsquared     MAE     
    ##   14225.07  0.005721379  3951.839
    ## 
    ## Tuning parameter 'intercept' was held constant at a value of TRUE

The non-linear model chosen for this project is the random forests
model. This model is a good fit for this dataset given the number of
predictors included for analysis. In the model, a random subset of
predictors is selected, which will allow for a good predictor to not
dominate the tree fits. We will specify `mtry` to have a value of `p/3`.

``` r
# Non-linear model chosen is Random Forests Model  

# Model fit with training data
#rfFit <- randomForest(shares ~ ., data = trainingData, mtry = ncol(trainingData)/3,
#                     importance = TRUE)
# Fit Random Forest Tree using method = "rf" and tuning parameter, mtry
#trCtrl <- trainControl(method = "cv", number = 3)
#randFrst <- train(shares~., data = trainingData, method = "rf", trControl = trCtrl)
#randFrst


# Model predictions using testing data
#rfPred <- predict(rfFit, newdata = dplyr::select(testingData, -shares))

# Get the root mean squared error (RMSE) value - root of test prediction error

#rfRMSE <- sqrt(mean((rfPred-testingData$shares)^2))
#rfRMSE
```
