"""
Uses Monte Carlo Orthogonal Distance Regression for Schmidt Hammer exposure dating. Calculates model coefficients
(betas, residuals, covariance matrix) for different production rates/calculators
@authors: jonnyhuck, matt-tomkins

Monte Carlo ODR improves on standard ODR by explicitly incorporating XY errors, but without the complications
introduced by weighting.

References:
    https://stackoverflow.com/questions/60889680/how-to-plot-1-sigma-prediction-interval-for-scipy-odr/60927487#60927487
    https://www.astro.rug.nl/software/kapteyn/kmpfittutorial.html#confidence-and-prediction-intervals
    https://www.astro.rug.nl/software/kapteyn/EXAMPLES/kmpfit_ODRparabola_confidence.py
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.odr.Output.html
    http://www.ieap.uni-kiel.de/et/people/wimmer/teaching/stats/statistics.pdf
    
"""

# from sympy import symbols, diff
# from sympy.codegen.cfunctions import log10 as slog10
from scipy.stats import t, mode, linregress
from scipy.odr import ODR, Model, RealData, Data
from matplotlib.pyplot import subplots, savefig, rcParams
import matplotlib.pyplot as plt
from numpy import log10, array, linspace, zeros, sqrt, mean
import pandas as pd
import numpy as np
import os
import random
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as mpatches
import time

lm = LinearRegression()

# Updates font for plottinag
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']

def prediction_interval(func, dfdp, x, y, yerr, signif, popt, pcov):
    """
    * Calculate Preduction Intervals
    *  adapted from Extraterrestrial Physics pdf
    *
    *  func    : function that we are using
    *  dfdp    : derivatives of that function (calculated using sympy.diff)
    *  x       : the x values (calculated using numpy.linspace)
    *  y       : the y values (calculated by passing the ODR parameters and x to func)
    *  y_err   : the maximum residual on the y axis
    *  signif  : the significance value (68., 95. and 99.7 for 1, 2 and 3 sigma respectively)
    *  popt    : the ODR parameters for the fit, calculated using scipy.odr.run()
    *  pcov    : the covariance matrix for the fit, calculated using scipy.odr.run()
    """
    # get number of fit parameters and data points
    np = len(popt)
    n = len(x)

    # convert provided value to a significance level (e.g. 95. -> 0.05), then calculate alpha
    alpha = 1. - (1 - signif / 100.0) / 2

    # student’s t test
    tval = t.ppf(alpha, n - np)

    # ?? some sort of processing of covarianvce matrix and derivatives...
    d = zeros(n)
    for j in range(np):
        for k in range(np):
            d += dfdp[j] * dfdp[k] * pcov[j,k]

    # return prediction band offset for a new measurement
    return tval * sqrt(yerr**2 + d)


def log_func(beta, x):
    """
    * Log function for fitting using ODR
    """
    # logarithmic, m * log(x) + c
    return beta[0] * log10(x) + beta[1]


def linear_func(beta, x):
    """
    * Linear function for fitting using ODR
    """
    # logarithmic, m * log(x) + c
    return beta[0] * x + beta[1]

def monte_carlo_odr(PR, region, x_data, y_data, x_err, y_err):

    """

    1) Randomises the data (i = 1000) based on values (x, y) and associated errors (x_err, y_err).
    2) Constructs a standard logged OLS regression (used for ODR beta estimates).
    3) Detects outliers using internally studentised residuals from the OLS.  Those > 2 sigma (95%) are rejected. 
    4) Constructs an ODR and saves model coefficients (beta, covariance matrix, errors)
    5) Takes the median coefficients for final ODR model construction  
    
    """

    # Generates results files
    betas = []
    covariances = []
    eps = []

    # make function into Model instance (either log or linear)
    if (region == "British"):
        model = Model(linear_func)
    else:
        model = Model(log_func)
    
    # Sets seed for reproducible results (!)
    np.random.seed(214)

    # 1000 iterations                
    for i in range(1000):

        # Randomises the data (mean, sd)
        x = np.random.normal(x_data, x_err)
        y = np.random.normal(y_data, y_err)

        # Logs the data first (if Pyrenees), keeps the same variable names for simplicity
        if (region == "Pyrenees"):
            logX = log10(x)
        else:
            logX = x

        # Adds constant for stats model intercept
        X = sm.add_constant(logX)

        # 10 iterations (should be much less, but "just in case")
        for i in range(10):
            # runs a simple OLS model (log)
            linear_model = sm.OLS(y, X)
            results = linear_model.fit()
            
            # creates instance of influence
            influence = results.get_influence()
            # calculates internally standardized residuals
            St_Res = influence.resid_studentized_internal

            # Finds max residual
            M = np.max(abs(St_Res))

            # If any are larger than 2 standard deviations
            if M > 2:
                # Find their index
                res = [idx for idx, val in enumerate(St_Res) if val > 2 or val < -2]
                # Delete these data points
                x = np.delete(x, res)
                X = np.delete(X, res, axis = 0)
                y = np.delete(y, res)
            # If none are larger than 2 sd, continue using this dataset. Slope and intercept used for ODR fit.
            else:
                slope = results.params[1]
                intercept = results.params[0]
                continue
        
        # New data and model
        data = Data(x, y)
        # Job = 0, explicit orthogonal, slope and intercept estimates from OLS
        out = ODR(data, model, beta0=[slope, intercept], job=0).run()

        # Appends model coefficients to results file (for EPS, only the maximum residual value is recorded)
        betas.append(out.beta)
        eps.append(max(out.eps))
        covariances.append(out.cov_beta)

    # Takes the median of the model estimates, justified based on distribution of data
    Beta = np.median(betas, axis = 0)
    Eps = np.median(eps, axis = 0)
    Covariance = np.median(covariances, axis = 0)

    # Code below plots British curves with the different production rates 
    
    '''
    Code to check what the British one looks like!

    All look fine, coefficients are similar to non-MC version. Uncertainties are larger but still reasonable.
    Uncertainties also scale nicely with the production rates - larger uncertainties for the global PR, and
    smaller uncertainties for the local PRs

    
    if (region == "British"):

        # fit model using new beta, original x scale for consistency
        xn = linspace(min(x_data), max(x_data), 1000)
        yn = linear_func(Beta, xn)

        # 1 and 2 sigma prediction intervals
        pl1 = prediction_interval(linear_func, [xn, 1], xn, yn, Eps, 68., Beta, Covariance)
        pl2 = prediction_interval(linear_func, [xn, 1], xn, yn, Eps, 95., Beta, Covariance)

        # Returns minimum and maximum 1 sigma uncertainty
        print(min(pl1), max(pl1))

        # create a figure to draw on and add a subplot
        fig, ax = subplots(1)

        # plot y calculated from px against de-logged x (and 1 and 2 sigma prediction intervals)
        ax.plot(xn, yn, '#EC472F', label='Logarithmic ODR')
        ax.plot(xn, yn + pl1, '#0076D4', dashes=[9, 4.5], label='1σ Prediction limit (~68%)', linewidth=0.8)
        ax.plot(xn, yn - pl1, '#0076D4', dashes=[9, 4.5], linewidth=0.8)
        ax.plot(xn, yn + pl2, '#BFBFBF', dashes=[9, 4.5], label='2σ Prediction limit (~95%)', linewidth=0.5)
        ax.plot(xn, yn - pl2, '#BFBFBF', dashes=[9, 4.5], linewidth=0.5)

        # plot points and error bars
        ax.plot(x_data, y_data, 'k.', markerfacecolor= '#4495F3',
                 markeredgewidth=.5,  markeredgecolor = 'k',
                label='Calibration data (n = 54)', markersize=5)

        # labels, extents etc.
        ax.set_ylim(0, 30)
        ax.set_xlabel('Mean R-value')
        ax.set_ylabel('Age (ka)')
        ax.tick_params(direction = 'in')
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)

        # configure legend
        ax.legend(frameon=False,
                  fontsize=7)

        # Sets axis ratio to 1
        ratio = 1
        ax.set_aspect(1.0/ax.get_data_ratio()*ratio)

        # export the figure
        fig.set_size_inches(3.2, 3.2)
        #fnm = "%i.png" % i
        #savefig(fnm, dpi = 900, bbox_inches='tight')
        savefig(PR[0]+".png", dpi = 900, bbox_inches='tight')

    '''

    # Return final model coefficients
    return Beta, Eps, Covariance


def odr_production_rates(region, df):

    '''
    Input values:

    - Region (British/Pyrenees)
    - The dataset containing the SH and age values

    Loops through the production rate datasets and returns model coefficients

    '''

    # Generates results files
    Final_beta = []
    Final_covariances = []
    Final_eps = []

    # Sets region
    data = df.loc[(df['Region'] == region)]

    # Column names
    col = list(data.columns)

    # List of production rates and uncertainties, joins together
    Method = [s for s in col if "Age" in s]
    Uncertainty = [s for s in col if "External" in s]
    Production = list(zip(Method,Uncertainty))
    
    # Loops through the production rates
    for i in Production:
        # Sets y data and errors
        y_data = data.loc[:, i[0]].values
        y_err = data.loc[:,i[1]].values
        # Sets x data and errors (these don't change)
        x_data = data.loc[:, 'SH_Mean'].values
        x_err = data.loc[:, 'SH_SEM'].values
        # Sets production rate
        PR = i
        
        # Runs Monte Carlo ODR 
        Beta, Eps, Covariance = monte_carlo_odr(PR, region, x_data, y_data, x_err, y_err)

        # Appends results
        Final_beta.append(Beta)
        Final_eps.append(Eps)
        Final_covariances.append(Covariance)

        # Merges coefficients (Name of production rate, betas, maximum residual, covariance matrix)
        table = list(zip(Method, Final_beta, Final_eps, Final_covariances))

    return table


'''
Data loaded directly from the csv (update working directory) - Calibration_Summary.csv

'''

# Loads data from Supplementary File (update file path)
os.chdir(r'C:\Users\Matt\Desktop\Repo\moraine-paper-2020')

# Loads data
df = pd.read_csv('code\Calibration_Summary.csv', encoding = "ISO-8859-1")

# Run
start_time = time.time()
Pyrenees = odr_production_rates("Pyrenees", df)
British = odr_production_rates("British", df)
print("My program took", time.time() - start_time, "to run")








