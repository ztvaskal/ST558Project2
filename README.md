ST558 Project 2
================
Zack Vaskalis
07/03/2020

  - [Introduction](#introduction)
  - [Project Objectives](#project-objectives)
  - [Data](#data)
      - [Information on Attributes for
        Analysis](#information-on-attributes-for-analysis)
  - [Goal of Project](#goal-of-project)
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

### Information on Attributes for Analysis

Out of the 61 Attributes available in this dataset for analysis, in
order not to overfit the model with too many dependent variables (which
could yield not enough generalization of the dataset to make
predictions, especially valid ones), I have chosen to use the following
7 variables, 6 which are predictive, and 1, the shares variable, which
is outcome variable. The weekday variables are also included, but only
for subsetting the data by the day of the week. Once we have the
“Monday” dataset, for instance, all of these `weekday_is_*`
variables will be removed from the dataset. The variables selected are
below.

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

## Goal of Project

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
