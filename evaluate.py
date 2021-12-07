import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt

# create function that plots residuals
def plot_residuals(y, yhat): 
    # creates a residual plot

    plt.figure(figsize=(8, 6))
    plt.scatter(y, yhat-y, color='dimgray')

    # set titles
    plt.title('Actual vs. Residuals', fontsize=12, color='black')
    # add axes labels
    plt.ylabel(r'$\hat{y}-y$')
    plt.xlabel('$y$')

    plt.show()


# create function that returns regression errors
def regression_errors(y, yhat): 
    # returns the following values
    return pd.Series({
    'SSE' : mean_squared_error(y, yhat)*len(y),
    'ESS' : ((yhat - y.mean())**2).sum(),
    'TSS' : ESS + SSE,
    'MSE' : mean_squared_error(y, yhat),
    'RMSE' : sqrt(mean_squared_error(y, yhat)),
    })


# create function that returns mean errors for baseline model
def baseline_mean_errors(y): 
    # computes the SSE, MSE, and RMSE for the baseline model
    yhat = y.mean()
    return pd.Series({
    'SSE' : ((yhat-y)**2).sum(),
    'MSE' : ((yhat-y)**2).sum()/len(y),
    'RMSE' : sqrt(((yhat-y)**2).sum()/len(y))
    })


# create function that checks if your model is better than the baseline
def better_than_baseline(y, yhat): 
    # returns true if your model performs better than the baseline, otherwise false
    baseline = y.mean()
    rmse_baseline = sqrt(((baseline - y)**2).sum()/len(y))
    rmse_model = sqrt(mean_squared_error(y, yhat))
    return rmse_model < rmse_baseline