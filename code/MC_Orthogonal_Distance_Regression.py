"""
Demonstration of Monte Carlo Orthogonal Distance Regression for Schmidt Hammer exposure dating
@authors: jonnyhuck, matt-tomkins

Improves on standard ODR by explicitly incorporating XY errors, but without the complications
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

lm = LinearRegression()

# Updates font for plottinag
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']


def prediction_interval(dfdp, n, xn, yerr, signif, popt, pcov):
    """
    * Calculate Preduction Intervals
    *
    *  dfdp    : derivatives of that function (calculated using sympy.diff)
    *  n       : the number of samples in the original curve
    *  xn      : the number of preductions being undertaken here
    *  y_err   : the maximum residual on the y axis
    *  signif  : the significance value (68., 95. and 99.7 for 1, 2 and 3 sigma respectively)
    *  popt    : the ODR parameters for the fit, calculated using scipy.odr.run()
    *  pcov    : the covariance matrix for the fit, calculated using scipy.odr.run()
    *
    * based on p.147 of Mathematical, Numerical, and Statistical Methods in Extraterrestrial Physics
    """
    
    # get number of fit parameters
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

def log_func(beta, x):
    """
    * Log function for fitting using ODR
    """
    # logarithmic, m * log(x) + c
    return beta[0] * log10(x) + beta[1]

def monte_carlo_odr(x_data, y_data, x_err, y_err, new_x_data, new_y_data, new_x_err, new_y_err):

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

    # make function into Model instance (logarithmic)
    model = Model(log_func)

    # Sets seed for reproducible results
    np.random.seed(214)

    # 1000 iterations                
    for i in range(1000):

        # Randomises the data (mean, sd)
        x = np.random.normal(x_data, x_err)
        y = np.random.normal(y_data, y_err)

        # Logs the data first
        logX = log10(x)

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
        # Job = 0, explicit orthogonal
        out = ODR(data, model, beta0=[slope, intercept], job=0).run()

        # Appends model coefficients to results file (for EPS, only the maximum value is recorded)
        betas.append(out.beta)
        eps.append(max(out.eps))
        covariances.append(out.cov_beta)

    # Takes the median of the model estimates
    Beta = np.median(betas, axis = 0)
    Eps = np.median(eps, axis = 0)
    Covariance = np.median(covariances, axis = 0)

    # fit model using new beta, original x scale for consistency
    xn = linspace(min(x_data), max(x_data), 1000)
    yn = log_func(Beta, xn)

    # 1 and 2 sigma prediction intervals
    pl1 = prediction_interval([log10(xn), 1], 54, len(xn), Eps, 68., Beta, Covariance)
    pl2 = prediction_interval([log10(xn), 1], 54, len(xn), Eps, 95., Beta, Covariance)

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
    ax.errorbar(x_data, y_data, ecolor='k', xerr=x_err, yerr=y_err, fmt=" ", linewidth=0.5, capsize=0)

    # adds new data and errors bars
    ax.plot(new_x_data, new_y_data,'k.', markerfacecolor= '#FF8130',
             markeredgewidth=.5,  markeredgecolor = 'k',
            label = 'New data (n = 15)', markersize = 5)
    ax.errorbar(new_x_data, new_y_data, ecolor='k', xerr=new_x_err, yerr=new_y_err, fmt=" ", linewidth=0.5, capsize=0)

    # labels, extents etc.
    ax.set_ylim(0, 60)
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
    savefig('Pyrenees_Monte_Carlo_ODR.png', dpi = 900, bbox_inches='tight')
    #savefig('Pyrenees_Monte_Carlo_ODR.svg')

    # Return final model coefficients
    return Beta, Eps, Covariance

def predict_odr(Beta, Eps, Covariance, predict, x_data):

    '''
    Function to predict ages using coefficients from Monte Carlo ODR and new SH values

    '''
    # Extracts values
    predict_x = predict.loc[:, 'SH_Mean'].values
    names = predict.loc[:, 'Sample_name'].values
    landform = predict.loc[:, 'Landform'].values
    sub = predict.loc[:, 'Sub-Landform'].values
    
    # Predicts age using model coefficients
    predict_y = log_func(Beta, predict_x)

    # Calculates 1 and 2 sigma prediction intervals
    pl1 = prediction_interval([log10(predict_x), 1], 54, len(predict_x), Eps, 68., Beta, Covariance)
    pl2 = prediction_interval([log10(predict_x), 1], 54, len(predict_x), Eps, 95., Beta, Covariance)

    # Returns predicted ages, and 1 sigma uncertainties
    output = pd.DataFrame(np.column_stack((names, landform, sub, predict_y, pl1)), columns=['Name','Landform', 'Sub-Landform', 'Age','1 sigma'])

    # Saves to csv
    output.to_csv('Predicted_SH_Ages.csv',index=False)

'''
Data is now loaded directly from the csv (update working directory)
10Be and 36Cl

Rows are removed which don't contain the full data (e.g. not sampled with the SH)

Then subset by group:
> Calibration dataset (n = 52) + Calibration new (n = 2)
> New moraine samples (n = 15)

'''

# Loads data from Supplementary File (update file path)
os.chdir(r'C:\Users\Matt\Desktop\Repo\moraine-paper-2020')

# Loads data
Be = pd.read_csv('data\Supplementary_Table_1_10Be.csv', encoding = "ISO-8859-1")
Cl = pd.read_csv('data\Supplementary_Table_2_36Cl.csv', encoding = "ISO-8859-1")
# Simplifies to key variables
cols = ['Group', 'Landform/Region','Publication', 'Isotope', 'Facility', 'Sample_name',
        'SH_Mean','SH_SEM','SH_STD', 'CRONUS_Age_2020_03_27','CRONUS_Internal_2020_03_27',
        'CRONUS_External_2020_03_27']
Be = Be[cols]
Cl = Cl[cols]

# Removes rows without all data
Be = Be.dropna(axis='index', how='any', subset=['SH_Mean'])
Cl = Cl.dropna(axis='index', how='any', subset=['SH_Mean'])
# Merges 10Be and 36Cl datasets
data = Be.append(Cl, ignore_index=True)

# Subsets based on group
Calibration = data.loc[(data['Group'] == 'Calibration dataset') | (data['Group'] == 'Calibration new')]
New = data.loc[data['Group'] == 'New moraine samples']

# load calibration data for curve creation
x_data = Calibration.loc[:, 'SH_Mean'].values
y_data = Calibration.loc[:, 'CRONUS_Age_2020_03_27'].values
x_err = Calibration.loc[:, 'SH_SEM'].values
y_err = Calibration.loc[:, 'CRONUS_External_2020_03_27'].values

# load new data to be plotted
new_x_data = New.loc[:, 'SH_Mean'].values
new_y_data = New.loc[:, 'CRONUS_Age_2020_03_27'].values
new_x_err = New.loc[:, 'SH_SEM'].values
new_y_err = New.loc[:, 'CRONUS_External_2020_03_27'].values

# plot the curve, Monte Carlo ODR
Beta, Eps, Covariance = monte_carlo_odr(x_data, y_data, x_err, y_err, new_x_data, new_y_data, new_x_err, new_y_err)

'''

Loads values to predict for (SH). Uses beta coefficients from Monte Carlo ODR. 

'''

# Loads data
predict = pd.read_csv('data\Supplementary_Table_3_SH.csv', encoding = "ISO-8859-1")

# Predicts ages
predict_odr(Beta, Eps, Covariance, predict, x_data)




