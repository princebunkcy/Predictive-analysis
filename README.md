# Regression 

In this project, machine learning algorithm is applied to the dataset houseprice_data.csv.This dataset contains information about house sales in King County, USA.
The data has 18 features, such as: number of bedrooms, bathrooms, floors etc., and a target variable: price. Using linear regression ,I developed a model to predict the price of a house. After developing the model, I analysed the results and discuss the effectiveness of the model, outlining the improvements when developing the model.

## Introduction
Regression is used to model the relationship between variables. For this project , I imported the necessary libraries and the houseprice dataset .I described the data sets and checked for null values which I couldn’t find. I checked for the relationship between the features and targeted column by carrying out a correlation analysis to know how changes in the features are associated with changes in the target variable

### Is there a way of visualising your model? (Possibly just one or two input/feature variable(s).)
I used a simple linear regression (y = βX + c + ε) in visualising the model with Price column as  dependent variable which can be found on the Y axis and Sqft_living column as the independent variable which can be found on the X axis . The table below shows the blue line which represents the linear relationship and with an R2 of 0.50 or 50% which is suitable to make prediction for the dataset . The figure below shows the Simple linear regression with two feature

![Diagram Title](/charts/prices.png)

### How will you assess the effectiveness of the model?
To access the effectiveness of the model I split the data into training and test sets and looked for the coefficient of determination denoted as R-squared which explains how well our model fits .If the coefficient of determination is close to 1 that shows it fits but if R-squared is close to 0 this shows the model doesn’t fit

### Include as many features as you can. Does the model improve?
I included 4 features ‘sqft_living‘, ‘grade’, ‘floors’ , ‘waterfront’ and caried out a multiple linear 
regression (y = β1X1 + β2X2 + · · · + βnXn + c + ε). After this, there wasn’t much improvement as the 
R2 moved slightly higher to 0.57 or 57% which is close to the previous R2 = 0.50 or 50% that I got 
while carrying out a simple linear regression with only one feature .with this fact , I can conclude that using many features can improve a model compared to using simple features. The table below shows the model with 5 features

![Diagram Title](/charts/new_test_plot.png)
