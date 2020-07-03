Weekday
================
Zack Vaskalis
07/03/2020

  - [Introduction](#introduction)
      - [Attribute Information:](#attribute-information)
  - [Data](#data)
  - [Summarizations](#summarizations)
  - [Modeling](#modeling)

## Introduction

This project uses the Online News Popularity Data Set which can be found
at the University of California Irvine [UCI Machine Learning Repository,
Center for Machine Learning and Intelligent
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

### Attribute Information:

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
path <- "C:/Users/Zachary Vaskalis/Dropbox/ST558/ST558Project2/ST558Project2/OnlineNewsPopularity.csv"
weekdayDataRAW <- read_csv(path)

# Select only variables I am choosing to use for the analysis.
weekdayData1 <- select(weekdayDataRAW, n_tokens_content, n_unique_tokens,
                       n_non_stop_unique_tokens, weekday_is_monday, weekday_is_tuesday,
                       weekday_is_wednesday, weekday_is_thursday, weekday_is_friday,
                       weekday_is_saturday, weekday_is_sunday, global_subjectivity,
                       global_sentiment_polarity, global_rate_positive_words,
                       global_rate_negative_words, rate_positive_words, rate_negative_words,
                       avg_positive_polarity, min_positive_polarity, max_positive_polarity,
                       avg_negative_polarity, min_negative_polarity, max_negative_polarity,
                       title_subjectivity, title_sentiment_polarity, abs_title_subjectivity,
                       abs_title_sentiment_polarity, shares)

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

    ## [1] 1593783032

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
    ## -13195  -3124  -1825   -248 684662 
    ## 
    ## Coefficients:
    ##                                Estimate Std. Error t value Pr(>|t|)   
    ## (Intercept)                   3.993e+03  1.623e+03   2.460  0.01393 * 
    ## n_tokens_content             -5.410e-02  8.948e-01  -0.060  0.95179   
    ## n_unique_tokens               6.599e+03  6.949e+03   0.950  0.34237   
    ## n_non_stop_unique_tokens     -2.536e+03  5.636e+03  -0.450  0.65273   
    ## global_subjectivity           7.690e+03  3.487e+03   2.205  0.02749 * 
    ## global_sentiment_polarity    -8.398e+03  7.020e+03  -1.196  0.23165   
    ## global_rate_positive_words   -1.024e+03  3.092e+04  -0.033  0.97357   
    ## global_rate_negative_words    3.175e+03  5.907e+04   0.054  0.95714   
    ## rate_positive_words          -7.643e+03  3.554e+03  -2.150  0.03158 * 
    ## rate_negative_words          -1.252e+04  4.618e+03  -2.711  0.00674 **
    ## avg_positive_polarity         1.044e+04  5.661e+03   1.844  0.06526 . 
    ## min_positive_polarity        -8.767e+03  4.951e+03  -1.771  0.07665 . 
    ## max_positive_polarity        -1.906e+03  1.832e+03  -1.040  0.29839   
    ## avg_negative_polarity        -1.726e+03  5.309e+03  -0.325  0.74515   
    ## min_negative_polarity        -2.711e+03  1.980e+03  -1.369  0.17101   
    ## max_negative_polarity        -4.766e+03  4.445e+03  -1.072  0.28370   
    ## title_subjectivity           -7.426e+02  1.169e+03  -0.635  0.52546   
    ## title_sentiment_polarity      4.917e+02  1.084e+03   0.454  0.65011   
    ## abs_title_subjectivity        1.753e+03  1.551e+03   1.130  0.25864   
    ## abs_title_sentiment_polarity  1.633e+03  1.689e+03   0.967  0.33374   
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 16830 on 4642 degrees of freedom
    ## Multiple R-squared:  0.009772,   Adjusted R-squared:  0.005719 
    ## F-statistic: 2.411 on 19 and 4642 DF,  p-value: 0.000552

``` r
trCtrl <- trainControl(method = "cv", number = 10)
mlr2 <- train(shares~., data=trainingData, method="lm",metric = "RMSE", trControl=trCtrl)
mlr2
```

    ## Linear Regression 
    ## 
    ## 4662 samples
    ##   19 predictor
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 4195, 4196, 4195, 4197, 4195, 4196, ... 
    ## Resampling results:
    ## 
    ##   RMSE      Rsquared     MAE     
    ##   14238.56  0.009123393  4006.427
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
