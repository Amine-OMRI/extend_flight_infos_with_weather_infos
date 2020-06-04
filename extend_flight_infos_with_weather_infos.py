
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 10:14:12 2020

@author: omri
"""
# Importing the libraries
import pandas as pd
import numpy as np
import seaborn as sns
import datetime as dt
from sklearn.linear_model import LinearRegression

# Importing the dataset
flights = pd.read_csv("/home/omri/PycharmProjects/extend_flight_infos_with_weather_infos/flights.csv")
weather = pd.read_csv("/home/omri/PycharmProjects/extend_flight_infos_with_weather_infos/weather.csv")
weather.head()
flights.head()
# Display the total number of non null observations present including the total number of entries
flights.info()
# Display a summary statistics of all observations
flights.describe()

weather.info()
weather.describe()

# First step
# -------------------------------------------------------------------------------------------------------
# Build a new data set in which entries extend the information in flight.csv with weather information
# -------------------------------------------------------------------------------------------------------
# Just to make sure that we have the same namber of unique dates in both of flights and wether DataFrames
len(flights['date'].unique().tolist())
len(weather['date'].unique().tolist())
# Join flights and wether DataFrames on "date" columns (potentially a many-to-many join)
extended_flights = pd.merge(flights, weather, on='date')

extended_flights.head()
extended_flights.info()
extended_flights.describe()

# Display the total number of NaN in our data
print(extended_flights.isnull().sum())

# Second step
# --------------------------------------------------------------------------------------------------------
# Remove entries with outliers in flight duration, flight occupancy or weather data
# --------------------------------------------------------------------------------------------------------

# Outliers can impact the results of our analyksis and statistical modeling in a drastic way
sns.boxplot(x=extended_flights['duration'], palette="Blues")
sns.boxplot(x=extended_flights['number_passengers'], palette="Blues")
sns.boxplot(x=extended_flights['temperature'], palette="Blues")
sns.boxplot(x=extended_flights['windSpeed'], palette="Blues")
sns.boxplot(x=extended_flights['precipitation'], palette="Blues")


# Accept a dataframe, remove outliers, return cleaned data in a new dataframe
def remove_outlier_iqr(df_in, col_name):
    quartile_1 = df_in[col_name].quantile(0.25)
    quartile_3 = df_in[col_name].quantile(0.75)
    iqr = quartile_3 - quartile_1  # Interquartile range
    lower_bound = quartile_1 - (1.5 * iqr)
    upper_bound = quartile_3 + (1.5 * iqr)
    print(quartile_1, quartile_3, upper_bound, lower_bound)
    # False that means these values are valid whereas True indicates presence of an outlier
    print((df_in[col_name] < lower_bound) | (df_in[col_name] > upper_bound))
    df_out = df_in.loc[((df_in[col_name] > lower_bound) & (df_in[col_name] < upper_bound)) | (df_in[col_name].isnull())]
    return df_out


# Remove entries with outliers in flight duration, number_passengers,temperature, windSpeed, precipitation
cleaned_df = remove_outlier_iqr(extended_flights, 'duration')
cleaned_df = remove_outlier_iqr(cleaned_df, 'number_passengers')
cleaned_df = remove_outlier_iqr(cleaned_df, 'temperature')
cleaned_df = remove_outlier_iqr(cleaned_df, 'windSpeed')
cleaned_df = remove_outlier_iqr(cleaned_df, 'precipitation')

# keep a copy of the unconverted date to use it after the prediction
dates = cleaned_df['date']
# Linear regression doesn't work on date data. Therefore we need to convert it into numerical value
cleaned_df['date'] = pd.to_datetime(cleaned_df['date'])
cleaned_df['date'] = cleaned_df['date'].map(dt.datetime.toordinal)
cleaned_df.info()

# Third step
# --------------------------------------------------------------------------------------------------------
# Predict the number of passengers travelling for the entries which have missing number_passengers column
# --------------------------------------------------------------------------------------------------------


# To predctict entries which have missing number_passengers (nan) we use number_passengers column as labels
# and we use rows without missing values to predict rows with missing values of the column number_passengers
# Our model contain the independant values of columns (duration, temperature, windSpeed, precipitation) and
# the dependant value of column number_passengers
# X : (duration, temperature, windSpeed, precipitation) y : number_passengers

# Splitting the dataset into the Training set and Test set
# Entries without missing values is used to train the Multiple Linear Regression model
flights_without_missing_values = cleaned_df[cleaned_df['number_passengers'].notnull()]
# X_train contain all independant values for rows without missing values in the column number_passengers
X_train = flights_without_missing_values.loc[:, flights_without_missing_values.columns != 'number_passengers'].values
# y_train contain lables wish are values of the column number_passengers for rows without missing values
y_train = flights_without_missing_values.iloc[:, 2].values

# Entries with missing values is used to test the Multiple Linear Regression model
flights_with_missing_values = cleaned_df[cleaned_df['number_passengers'].isnull()]
# X_test contain all independant values for rows with missing values in the column number_passengers
X_test = flights_with_missing_values.loc[:, flights_with_missing_values.columns != 'number_passengers'].values

# Fitting Multiple Linear Regression to the Training set
# We need to apply normalization because in our data we have different measurement unit
# duration (in hours), temperature (in degrees Celsius), speed (in m/s) and precipitation (in mm)
regressor = LinearRegression(normalize=True)
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
# Convert predicted values to int
y_pred = np.array([int(y) for y in y_pred])

# Reconstruct the dataset with predicted values
# Relace old values of number_passengers column with the predicted values in flights_without_missing_values
flights_with_missing_values.loc[:, 'number_passengers'] = y_pred
frames = [flights_without_missing_values, flights_with_missing_values]
# Concatenate flights_without_missing_values and flights_with_missing_values to get all flights data
new_df = pd.concat(frames)
# Replace the converted date column with the one that we kept it before the conversion
new_df.loc[:, 'date'] = dates
# sort the dataset by 'id'
new_df = new_df.sort_values(by='id')
new_df.info()

# Save the output Dataframe in the same comma-separated line format as flights.csv.
new_df.to_csv(r'/home/omri/Bureau/Data Science Challenge/extended_predicted_flights.csv', sep=',', header=True)

# Fourth step
# --------------------------------------------------------------------------------------------------------
# what fleet (number of aircrafts of each type) would you recommend to the airline servicing these flights
# --------------------------------------------------------------------------------------------------------
# Group the data on 'date' value
# Flights data will be grouped by date
groups_by_date = new_df.groupby('date')
# DataFrame that will contain the needed number of aircrafts of each type for each date (groups_by_date)
# First column Date : each unique date of our recontructed DataFrame (new_df)
# three column Type A, Type B, Type C : Number of aircrafts of each type(A, B, C) grouped by date
summary_table = pd.DataFrame()
# Iterate over each group
for group_name, same_date_groupe in groups_by_date:
    # Temporary dataframe that have the same structure of summary_table dataframe
    columns = ['date', 'Type A', 'Type B', 'Type C']
    tmp_df = pd.DataFrame([(format(group_name), 0, 0, 0)], columns=columns)
    # For each iterated group ,tmp_df will contain the date and the needed number of aircrafts of each type
    # Iterate inside each group
    # Each row of the group is a flight and all of them have the same date
    for row_index, row in same_date_groupe.iterrows():
        # Determine the number of aircraft by checking the duration and the number of passengers according
        # to table 1
        if ((row['duration'] < 2.5) and (row['number_passengers'] < 50)):
            tmp_df.iloc[:, 1] = tmp_df.iloc[:, 1] + 1
        elif ((row['duration'] < 1.5) and (row['number_passengers'] < 100)):
            tmp_df.iloc[:, 2] = tmp_df.iloc[:, 2] + 1
        elif ((row['duration'] < 6) and (row['number_passengers'] < 150)):
            tmp_df.iloc[:, 3] = tmp_df.iloc[:, 3] + 1
    summary_table = summary_table.append(tmp_df)

# Save the output summary Dataframe
summary_table.to_csv(r'/home/omri/Bureau/Data Science Challenge/needed_aircrafts_by_date.csv', sep=',', header=True)

print(summary_table.loc[:, summary_table.columns != 'date'].sum())

print(round(summary_table.loc[:, summary_table.columns != 'date'].mean()))
# from the average value it would be much more beneficial for the airline to buy 6 airecrafts type A and
# 7 type B and do not buy type C because there is not much travel with this type of aircraf
print(summary_table.loc[:, summary_table.columns != 'date'].std())

# Using the resulting data set of step 3(new_df)), and the aircraft specifications in Table 1, the recmended
# fleet (number of aircrafts of each type) to the airline servicing these flights would be the max of columns
# of the summary_table that contain the needed number of aircrafts of each type for each date to ensure
# that the airline servicing will have enough aircraft for all flights on such a date
print(summary_table.loc[:, summary_table.columns != 'date'].max())