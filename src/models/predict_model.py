import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import sys

sys.path.append("..")
from utility import plot_settings
from utility.visualize import plot_predicted_vs_true, regression_scatter, plot_residuals


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

bike_data = pd.read_csv("../../data/raw/daily-bike-share.csv")
new_sample = bike_data.sample(20)

# --------------------------------------------------------------
# Load model
# --------------------------------------------------------------

model, ref_cols, target = joblib.load("../../models/model.pkl")

# --------------------------------------------------------------
# Make predictions
# --------------------------------------------------------------

X_new = new_sample[ref_cols]
y_new = new_sample[target]
predictions = model.predict(X_new)

# --------------------------------------------------------------
# Evaluate results
# --------------------------------------------------------------

rmse = np.sqrt(mean_squared_error(y_new, predictions))
r2 = r2_score(y_new, predictions)

print("RMSE:", rmse)
print("R2:", r2)

# Visualize results
plot_predicted_vs_true(y_new, predictions)
