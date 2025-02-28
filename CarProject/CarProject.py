# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 12:36:36 2025

@author: lenovo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime

# Read the dataset from a CSV file
car = pd.read_csv('cardata.csv')

# Display the first 5 rows of the dataset (currently commented out)
# print(car.head())

# Display the shape of the DataFrame (number of rows and columns)
# print(car.shape)

# Display statistical summary of the dataset (currently commented out)
# print(car.describe())

# Display general information about the DataFrame (currently commented out)
# print(car.info())

# Count the number of missing values in each column (currently commented out)
# nulls = car.isnull().sum()
# print(nulls)

# Calculate the age of cars by subtracting the car's year from the current year
current_year = datetime.now().year
car['Car_Age'] = current_year - car['Year']

# Drop non-numeric columns like 'Car_Name' and 'Year' as they aren't useful for the regression model
car.drop(['Car_Name', 'Year'], axis=1, inplace=True)

# Apply one-hot encoding to convert categorical features into numerical features
car_final = pd.get_dummies(car)

# Define the features (X) and the target (Y)
x = car_final.drop(columns=["Selling_Price"])
y = car_final["Selling_Price"]

# Split the dataset into training and testing sets (30% for testing)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Initialize the linear regression model
model = LinearRegression()
# Train the model using the training set
model.fit(x_train, y_train)

# Get the importance of each feature (coefficients) and display them in a bar chart
importance = pd.Series(model.coef_, index=x.columns)
importance.sort_values(ascending=False).plot(kind="bar", figsize=(10,5), title="Feature Importance")
plt.show()

# Plot a heatmap showing the correlation between different features
plt.figure(figsize=(10,8))
sns.heatmap(car_final.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# Initialize another linear regression model for prediction
reg = LinearRegression()
# Train the model using the training set
reg.fit(x_train, y_train)

# Predict the selling prices using the test set
y_pred = reg.predict(x_test)

# Scatter plot showing the actual vs predicted prices
plt.scatter(y_test, y_pred)
plt.plot()
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.show()

# Calculate and print the mean squared error (MSE) for the predictions
mse = mean_squared_error(y_test, y_pred)
print(mse)

# Perform cross-validation to evaluate the model's performance on different subsets of the dataset
from sklearn.model_selection import cross_val_score
reg = LinearRegression()

# Cross-validation with 4 folds, using negative mean squared error as the scoring metric
cv_scores = cross_val_score(reg, x, y, scoring='neg_mean_squared_error', cv=4)
print(cv_scores)
# Print the average of the cross-validation scores
print(np.mean(cv_scores))

# Calculate and print the R² (coefficient of determination) score, which indicates the model's fit
r2 = r2_score(y_test, y_pred)
print(f"R²: {r2}")

# Scatter plot comparing actual vs predicted prices, with a red line representing perfect predictions
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='green')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='-')  # Ideal line
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()
