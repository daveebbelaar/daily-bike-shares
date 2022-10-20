import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.append("..")
from utility import plot_settings
from utility.visualize import plot_predicted_vs_true, plot_residuals, regression_scatter

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

bike_data = pd.read_pickle("../../data/processed/bike_data_processed.pkl")
target = "rentals"

# --------------------------------------------------------------
# Train test split
# --------------------------------------------------------------

X = bike_data.drop(target, axis=1)
y = bike_data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

# --------------------------------------------------------------
# Train model
# --------------------------------------------------------------

# Define preprocessing for numeric columns (scale them)
numeric_features = X.select_dtypes(include=["float"]).columns
numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

# Define preprocessing for categorical features (encode them)
categorical_features = X.select_dtypes(include=["category"]).columns
categorical_transformer = Pipeline(
    steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
)

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# Build the pipeline
pipeline = Pipeline(
    steps=[("preprocessor", preprocessor), ("regressor", RandomForestRegressor())]
)

# fit the pipeline to train the model on the training set
model = pipeline.fit(X_train, y_train)

# --------------------------------------------------------------
# Evaluate the model
# --------------------------------------------------------------

# Get predictions
predictions = model.predict(X_test)

# Display metrics
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("RMSE:", rmse)
print("R2:", r2)

# Visualize results
plot_predicted_vs_true(y_test, predictions)
regression_scatter(y_test, predictions)
plot_residuals(y_test, predictions, bins=15)

# --------------------------------------------------------------
# Export model
# --------------------------------------------------------------

ref_cols = list(X.columns)

"""
In Python, you can use joblib or pickle to serialize (and deserialize) an object structure into (and from) a byte stream. 
In other words, it's the process of converting a Python object into a byte stream that can be stored in a file.

https://joblib.readthedocs.io/en/latest/generated/joblib.dump.html

"""

joblib.dump(value=[model, ref_cols, target], filename="../../models/model.pkl")
