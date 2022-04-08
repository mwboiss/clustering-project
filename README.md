# Innis-Clustering-Project

This repo is a compilation of work done to explore possible features for improving zillow models for predicting Zestimates for single family properties sold in 2017.

## About the Project

### Project Goals

The goal of this project is to understand the drivers of error in Zillow's Zestimates for single family homes sold in 2017.

### Project Description

Here at Zillow we pride ourselves in providing acurrate data to our customer base. To keep an edge in this highly competitive online space, we must strive for consistant improvement. In this project I will explore what features we can use to improve our current models that predict Zestimate scores of Single Family Properties that had a transaction in 2017 by understanding the drivers of logerror.

### Initial Questions

1) Is Logerror affected by Location?

2) Is Logerror affected by the house attributes?

3) Is Logerror affected by the age of the property?

4) Is logerror affected by the time of year it is sold?

5) Is logerror affected by the tax values?

### Summary of Findings

- 
- 

### Project Report

https://github.com/mwboiss/clustering-project/blob/main/report.ipynb

### Data Dictionary

Variable | Meaning |
:-: | :-- |
Feature	Description
'bathrooms'	| Number of bathrooms in home including fractional bathrooms
'bedrooms'	| Number of bedrooms in home 
'area'	| Calculated total finished living area of the home 
'latitude'	| Latitude of the middle of the parcel multiplied by 10e6
'longitude'	| Longitude of the middle of the parcel multiplied by 10e6
'lot_size'	| Area of the lot in square feet
'parcelid'	| Unique identifier for parcels (lots) 
'city'	| City in which the property is located (if any)
'zip'	| Zip code in which the property is located
'yearbuilt'	| The Year the principal residence was built 
'taxable_value'	| The total tax assessed value of the parcel
'taxamount'	| The total property tax assessed for that assessment year
'month_sold' | The month the property was sold
'age' | The age of the property
'long_use' |  Type of land use the property is zoned for

### Steps to Reproduce

1. A locally stored env.py file containing hostname, username and password for the mySQL database containing the zillow dataset is needed.

2. Data Science Libraries needed: pandas, numpy, matplotlib.pyplot, seaborn, scipy.stats, sklearn

3. All files in the repo should be cloned to reproduce this project.

4. Ensuring .gitignore is setup to protect env.py file data.

## Plan of Action

### Wrangle Module

1) Create and test acquire functions

2) Add functions to wrangle.py module

3) Create and test prepare functions

4) Add functions to wrangle.py module

##### Missing Values

1) Explore data for missing values

2) Add code to prepare function to remove values

3) Test function in notebook

##### Outliers

1) Assess data for outliers

2) Remove outliers if needed

3) Create function to remove outliers

4) Add function to wrangle.py module

##### Scale Data

1) Scale data appropriately

2) Create function to scale data

3) Add function to wrangle.py module

##### Data Split

1) Write code needed to split data into train, validate and test

2) Add code to prepare function and test in notebook

##### Explore

###### Each Feature Individually

###### Pairs of Variables

###### Multiple Variables

###### Questions to Answer

1) Is Logerror affected by Location?

2) Is Logerror affected by the house attributes?

3) Is Logerror affected by the age of the property?

4) Is logerror affected by the time of year it is sold?

5) Is logerror affected by the tax values?

###### Explore through visualizations

1) Create visualizations exploring each question

###### Statistics tests

1) Run statistics test relevant to each question

###### Explore through Clustering

1) Test clusters as features, visualizations and possible samples for modeling.

###### Summary 

1) Create a summary that answers exploritory questions

#### Modeling

1) Evaluate which metrics best answer each question

2) Evaluate a basline meteric used to compare models to the most present target variable

3) Develop models to predict the Zestimates of Single Family Properties sold in 2017.

4) Fit the models to Train data

5) Evaluate on Validate data to ensure no overfitting

6) Evaluate top model on test data

#### Report

1) Create report ensuring well documented code and clear summary of findings as well as next steps to improve research