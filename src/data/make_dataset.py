import pandas as pd

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

"""
The data used in this exercise is derived from Capital Bikeshare 
and is used in accordance with the published license agreement.
https://www.capitalbikeshare.com/system-data
"""

# Load data
bike_data = pd.read_csv("../../data/raw/daily-bike-share.csv")
bike_data.info()

# Select relevant features
relevant_features = [
    "season",
    "mnth",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
    "rentals",
]
bike_data = bike_data[relevant_features]

# Convert non-numerical columns to dtype category
categorical_features = [
    "season",
    "mnth",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
]
bike_data[categorical_features] = bike_data[categorical_features].astype("category")
bike_data.info()

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

bike_data.to_pickle("../../data/processed/bike_data_processed.pkl")
