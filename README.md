ST558 Project 2
================
Zack Vaskalis
07/03/2020

  - [Introduction](#introduction)
  - [Project Objectives](#project-objectives)
  - [Data](#data)
      - [Information on Attributes for
        Analysis:](#information-on-attributes-for-analysis)
  - [Reports](#reports)

## Introduction

## Project Objectives

The main two objectives for this project are to create predictive models
and automate R Markdown documents.

## Data

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
14 predictor variables and of course the 1 outcome variable we are
trying to predict, `shares`. The weekday variables are also included,
but only for subsetting the data by the day of the week. Once we have
the “Monday” dataset, for instance, all of these `weekday_is_*`
variables will be removed from the dataset. The variables selected are
below, and the reasoning behind their selection follows.

Attribute Information:  
1\. `n_tokens_content`: Number of words in the content  
2\. `num_hrefs`: Number of links  
3\. `average_token_length`: Average length of the words in the content  
4\. `kw_avg_min`: Worst keyword (avg. shares)  
5\. `kw_avg_max`: Best keyword (avg. shares)  
6\. `kw_min_avg`: Avg. keyword (min. shares)  
7\. `kw_max_avg`: Avg. keyword (max. shares)  
8\. `kw_avg_avg`: Avg. keyword (avg. shares)  
9\. `LDA_00`: Closeness to LDA topic 0  
10\. `LDA_04`: Closeness to LDA topic 4  
11\. `global_subjectivity`: Text subjectivity  
12\. `global_sentiment_polarity`: Text sentiment polarity  
13\. `title_subjectivity`: Title subjectivity  
14\. `abs_title_subjectivity`: Absolute subjectivity level  
15\. `shares`: Number of shares (target outcome trying to predict)  
16\. `weekday_is_variables`: Was the article published on a
\_\_\_\_\_\_\_?

  - `weekday_is_monday`: Was the article published on a Monday?  
  - `weekday_is_tuesday`: Was the article published on a Tuesday?  
  - `weekday_is_wednesday`: Was the article published on a Wednesday?  
  - `weekday_is_thursday`: Was the article published on a Thursday?  
  - `weekday_is_friday`: Was the article published on a Friday?  
  - `weekday_is_saturday`: Was the article published on a Saturday?  
  - `weekday_is_sunday`: Was the article published on a Sunday?

For this dataset of nearly 60 predictor variables, a dataset for
analysis that contains approximately 25% of them seems to feel right,
since too many predictor variables can cause overfitting. This subset of
variables is a sampling across the set of variables I thought might have
an impact. For example, the number of words of content
(`n_tokens_content`), and the number of links (`num_hrefs`) seemed
reasonable that they might play a role in predicting shares. The `kw_`
variables also seem like they should be good predictors. I chose the
first and last `LDA` variables. The global ratings also seemed like they
would have an impact on predicting the shares so I chose
`global_subjectivity` and `global_sentiment_polarity`. Then finally, to
round out the dataset of predictors I chose the title subjectivity and
absolutely subjectivity level, since often a title will have the largest
impact - catchy or provoking titles are clickbait for a reason\! \#\#
Goal of Project

The goal of this project is to create 2 models (one linear and one
non-linear) for predicting the shares variable. In addition to this
goal, another primary goal is to use parameter functionality within R
Markdown to automatically generate an analysis report for each day of
the week, using each `weekday_is_` variable as a parameter.

## Reports

[General Weekday Template - pre-automation](Weekday.md)

``` r
#[The Analysis for Monday is available here](MondayAnalysis.md)
#[The Analysis for Tuesday is available here](TuesdayAnalysis.md)
#[The Analysis for Wednesday is available here](WednesdayAnalysis.md)
#[The Analysis for Thursday is available here](ThursdayAnalysis.md)
#[The Analysis for Friday is available here](FridayAnalysis.md)
#[The Analysis for Saturday is available here](SaturdayAnalysis.md)
#[The Analysis for Sunday is available here](SundayAnalysis.md)
```
