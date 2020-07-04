## ------------------------- eval = FALSE, warning = FALSE---------------------------------------
## load necessary libraries
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
library(packHV)
library(pacman)
library(purrr)

## Set working directory
setwd("C:/Users/Zachary Vaskalis/Dropbox/ST558/ST558Project2/ST558Project2/")

## Could not get the functions to work correctly.  Went old school to produce results
## This is adapted code from class so I hope it will suffice for this project.

params = list(dayVar = "weekday_is_monday")
rmarkdown::render("Weekday.Rmd", output_file = "MondayAnalysis.md",
                  params = list(dayVar = "weekday_is_monday"))
params = list(dayVar = "weekday_is_tuesday")
rmarkdown::render("Weekday.Rmd", output_file = "TuesdayAnalysis.md",
                  params = list(dayVar = "weekday_is_tuesday"))
params = list(dayVar = "weekday_is_wednesday")
rmarkdown::render("Weekday.Rmd", output_file = "WednesdayAnalysis.md",
                  params = list(dayVar = "weekday_is_wednesday"))
params = list(dayVar = "weekday_is_thursday")
rmarkdown::render("Weekday.Rmd", output_file = "ThursdayAnalysis.md",
                  params = list(dayVar = "weekday_is_thursday"))
params = list(dayVar = "weekday_is_friday")
rmarkdown::render("Weekday.Rmd", output_file = "FridayAnalysis.md",
                  params = list(dayVar = "weekday_is_friday"))
params = list(dayVar = "weekday_is_saturday")
rmarkdown::render("Weekday.Rmd", output_file = "SaturdayAnalysis.md",
                  params = list(dayVar = "weekday_is_saturday"))
params = list(dayVar = "weekday_is_sunday")
rmarkdown::render("Weekday.Rmd", output_file = "SundayAnalysis.md",
                  params = list(dayVar = "weekday_is_sunday"))



## Get unique day of the week variable names
#days <- unique(c("weekday_is_monday", "weekday_is_tuesday"))

## Create list of filenames for reports
#fname <- c("MondayAnalysis", "TuesdayAnalysis")
#output_file <- paste0(fname,".md")

## Create a list for each day with just the day parameter
#params <- lapply(days, FUN = function(x){list(dayVar = x)})

## Put it into a data fram
#reports <- tibble(output_file, params)

## automate the reports using apply () and render () functions
#apply(reports, MARGIN = 1,
#      FUN = function(x){
#        render(input = "Weekday.Rmd", output_file = x[[1]], params = x[[2]])
#      })

#pwalk(reports, render, input = "Weekday.Rmd")

