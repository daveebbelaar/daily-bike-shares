import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_predicted_vs_true(y_test, y_pred, sort=True):
    """
    Plot the results from a regression model in a plot to compare the prediction vs. acutal values
    
    Args:
        y_test : actual values
        y_pred : model predictions
        sort (bool, optional): Sort the values. Defaults to True.
    """
    # Create canvas
    plt.figure(figsize=(20, 5))
    
    t = pd.DataFrame({"y_pred": y_pred, "y_test": y_test})
    if sort:
        t = t.sort_values(by=["y_test"])

    plt.plot(t["y_test"].to_list(), label="True", marker="o", linestyle="none")
    plt.plot(
        t["y_pred"].to_list(),
        label="Prediction",
        marker="o",
        linestyle="none",
        color="purple",
    )
    plt.ylabel("Value")
    plt.xlabel("Observations")
    plt.title("Predict vs. True")
    plt.legend()
    plt.show()


def regression_scatter(y_test, y_pred):
    """
    Plot the results from a regression model in a scatter plot to compare the prediction vs. acutal values.
    Additionally, plots the regression line and ideal fit line.
    
    Args:
        y_test : actual values
        y_pred : model predictions
    """

    # Create canvas
    plt.figure(figsize=(20, 5))

    # Plot scatter
    plt.scatter(y_test, y_pred)

    # Plot diagonal line (perfect fit)
    z = np.polyfit(y_test, y_test, 1)
    p = np.poly1d(z)
    plt.plot(
        y_test, p(y_test), color="gray", linestyle="dotted", linewidth=3, label="Ideal"
    )

    # Overlay the regression line
    z = np.polyfit(y_test, y_pred, 1)
    p = np.poly1d(z)
    plt.plot(y_test, p(y_test), color="#4353ff", label="Predicted", alpha=0.5)

    plt.xlabel("Actual Value")
    plt.ylabel("Predicted Value")
    plt.title("Predicted vs. True")
    plt.legend()
    plt.show()


def plot_residuals(y_test, y_pred, bins=25):
    """
    Plot residuals of a regression model. A good model will have a residuals 
    distribution that peaks at zero with few residuals at the extremes.
    
    Args:
        y_test : actual values
        y_pred : model predictions
        bins (int, optional). Defaults to 25.
    """
    
    residuals = y_test - y_pred
    
    plt.figure(figsize=(20, 5))
    plt.hist(residuals, bins=bins, rwidth=0.95)
    plt.title("Residual Histogram")
    plt.show()
     