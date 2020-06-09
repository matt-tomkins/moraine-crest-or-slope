import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import t
import pandas as pd
import numpy as np
import os
from numpy import zeros, ones, array, sqrt, log10, exp
from itertools import chain

'''
author: @matt-tomkins

Code to randomise SH exposure ages based on their peak age and their 1 sigma prediction bounds.
Utilises a truncated normal distribution to prevent randomised ages from being < 0 (i.e. in the future).
Converts to a format required by P-CAAT - see Dortch et al. (2020) "Probabilistic Cosmogenic Age Analysis Tool (P-CAAT), a tool for the ages"

Sources:
    - https://stackoverflow.com/questions/36894191/how-to-get-a-normal-distribution-within-a-range-in-numpy
    - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html
    - http://www.ieap.uni-kiel.de/et/people/wimmer/teaching/stats/statistics.pdf

'''

### Functions --------------------------------------

def get_truncated_normal(mean, sd, low, upp):
    return stats.truncnorm(
        a = (low - mean) / sd, b = (upp - mean) / sd, loc=mean, scale=sd)


def inverse_log_func(beta, x):
    """
    * Function to return an R value from a predicted age
    """
    # logarithmic, y = m * log(x) + c
    R = x - beta[1]
    R = R / beta[0]
    R = 10 ** R
    return R

def prediction_interval(dfdp, n, xn, yerr, signif, popt, pcov):
    """
    * Calculate Preduction Intervals
    *
    *  dfdp    : derivatives of that function (calculated using sympy.diff)
    *  n       : the number of samples in the original curve
    *  xn	   : the number of preductions being undertaken here
    *  y_err   : the maximum residual on the y axis
    *  signif  : the significance value (68., 95. and 99.7 for 1, 2 and 3 sigma respectively)
    *  popt    : the ODR parameters for the fit, calculated using scipy.odr.run()
    *  pcov    : the covariance matrix for the fit, calculated using scipy.odr.run()
    *
    * based on p.147 of Mathematical, Numerical, and Statistical Methods in Extraterrestrial Physics
    """
    # get number of fit parameters and data points
    np = len(popt)

    # convert provided value to a significance level (e.g. 95. -> 0.05), then calculate alpha
    alpha = 1. - (1 - signif / 100.0) / 2

    # students t test
    tval = t.ppf(alpha, n - np)

    # processing of covarianvce matrix and derivatives
    d = zeros(xn)
    for j in range(np):
        for k in range(np):
            d += dfdp[j] * dfdp[k] * pcov[j,k]

    # return prediction band offset for a new measurement
    return tval * sqrt(yerr**2 + d)


### Code --------------------------------------

# Sets working directory
os.chdir(r'C:\Users\Matt\Desktop\Repo\moraine-paper-2020')

# Loads data
df = pd.read_csv('data\Supplementary_Table_3_SH.csv', encoding = "ISO-8859-1")

# Extracts information for summary table
names, sub = df.loc[:, 'Sample_name'].values, df.loc[:, 'Sub-Landform'].values
lat, long = df.loc[:, 'Latitude_DD'].values, df.loc[:, 'Longitude_DD'].values

# Sets means and uncertainty
mu, sigma = df.loc[:, 'Calibrated_Age_ka'].values, df.loc[:, 'Calibrated_Uncertainty_ka'].values

# Sets model coefficients
# Available from: https://github.com/jonnyhuck/shed-earth/shedcalc/coefficients.py
coefficients = {"samples": 54, # Number of samples in the calibration curve
                "beta": array([-102.15108181,  187.82142242]), # Model slope and intercept
                "eps": 1.9924478773251222, # Maximum residual on the y axis
                "cov": array([[ 10.54929422, -17.71139805], [-17.71139805,  29.78310211]])} # Covariance matrix

# Initialises result file
result = []

# Sets seed for reproducible results
np.random.seed(214)

# 1000 iterations
for i in range(1000):

    # Generates the truncated normal, with a lower limit of 0 ka and an upper limit of 1000 ka (no effect)
    X = get_truncated_normal(mean = mu, sd = sigma, low = 0, upp = 1000)

    # Generates a simulated value for each sample (n = 635)
    simulated_age = X.rvs()

    # Converts to R value
    R = inverse_log_func(coefficients['beta'], simulated_age)

    # Calculates derivatives
    d = [log10(R), ones(len(R))]
    
    # Calculates prediction intervals
    simulated_uncertainty = prediction_interval(d, coefficients['samples'], len(R),
                                                coefficients['eps'], 68., coefficients['beta'], coefficients['cov'])

    # Converts to list
    simulated_age = simulated_age.tolist()
    simulated_uncertainty = simulated_uncertainty.tolist()
    
    # Returns simulated dataset number e.g. Tallada 1, Tallada 2, ... (+ 1 so the values go 1 > 1000, not 0 > 999)
    landform_name = sub + " " + str(i+1)

    # Combines, order required by P-CAAT (Sim. uncertainty is duplicated because P-CAAT is optimised for TCN, and requires a
    # internal and external uncertainty. 
    combined = list(zip(landform_name, simulated_age, simulated_uncertainty, simulated_uncertainty, names, lat, long))

    # Adds to result
    result.extend(combined)

### Data wrangling --------------------------------------

# Converts to pd dataframe, adds column names
result_df = pd.DataFrame(result)
result_df.columns = ['Landform', 'Age', 'Uncertainty 1', 'Uncertainty 2', 'Sample name', 'Latitude', 'Longitude']

# Assigns group (removes numeric) and sets index for speed
result_df['Group'] = result_df['Landform'].str.replace('\d+', '')
result_df['Group'] = result_df['Group'].str.strip()
result_df = result_df.set_index('Group')

# Loops through each index
for i in np.unique(result_df.index):
    # Subsets by landform
    subset = result_df.loc[i]
    # Saves to csv
    subset.to_csv(i+".csv",index=False)


