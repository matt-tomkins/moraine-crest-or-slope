"""
Demonstration of Orthogonal Distance Regression for Schmidt Hammer exposure dating
@author: jonnyhuck

Update working directory!

References:
    https://stackoverflow.com/questions/60889680/how-to-plot-1-sigma-prediction-interval-for-scipy-odr/60927487#60927487
    https://www.astro.rug.nl/software/kapteyn/kmpfittutorial.html#confidence-and-prediction-intervals
    https://www.astro.rug.nl/software/kapteyn/EXAMPLES/kmpfit_ODRparabola_confidence.py
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.odr.Output.html
    
"""

# from sympy import symbols, diff
# from sympy.codegen.cfunctions import log10 as slog10
from scipy.stats import t, mode, linregress, truncnorm
from scipy.odr import ODR, Model, RealData, Data
from matplotlib.pyplot import subplots, savefig, rcParams
import matplotlib.pyplot as plt
from numpy import log10, array, linspace, zeros, sqrt, mean
import pandas as pd
import numpy as np
import os
import random


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

def plot_shed_curve(x_data, y_data, x_err, y_err, new_x_data, new_y_data, new_x_err, new_y_err):
    """
    * Fit curve using ODR and plot
    """
    # make function into Model instance (either log or linear)
    model = Model(log_func)

    """
    The following line has a big(!) impact on the calculated prediction intervals.

    (1) Using:
        RealData(x_data, y_data) ... gives prediction intervals similar to OLS. 

    (2) Using: (where x_err and y_err are the raw error values). 
        RealData(x_data, y_data, sx=x_err, sy=y_err) ... gives prediction intervals ~1 ka larger (range = 5.4 ka) but this changes
        significantly if other values are used (e.g. R value SD instead of standard error of the mean). 

    
    Option (1) is valid (e.g. https://www.tutorialspoint.com/scipy/scipy_odr.htm) and is a standard orthogonal regression, but one which
    doesn't explicitly take into account the known errors. 
        
    Option (2) is preferable, because it shows we're taking errors fully into account, but I need to decide the appropriate values to use.

    One reason why Option (2) increases the error is that ODR assumes that ratio of XY errors = 1, as follows:

    "This approach assumes that although both variables are subject to measurement & equation error,
    the total error on Y is equal to the total error on X (in other words var(δX) + var(εX) = var (δY + var(εY)). Hence λ = 1.
    The two variables must obviously be measured in the same units for this to stand some chance of being true.
    The method minimizes the sum of the squared perpendicular distances of points to the line."
    https://influentialpoints.com/Training/errors-in-variables_regression-principles-properties-assumptions.htm

    Deming regression is very similar, but does *not* assume that λ = 1. Unfortunately, there's no obvious Python implementation.
    See Slide 7: http://www2.agroparistech.fr/podcast/IMG/pdf/chimiometrie_2017_bivregbls_mberger-bgfrancq_final.pdf

    Perhaps the best way forward is to normalise the errors beforehand i.e. error/value, which has two benefits:
    (i) isolates TCN colinearity (as age increases, errors increase)
    (ii) will hopefully ensure that λ ~ 1.

    """
    # make data into RealData instance
    #data = RealData(x_data, y_data, sx=x_err, sy=y_err)
    #data = Data(x_data, y_data, wd=x_err, we=y_err)                 
    data = Data(x_data, y_data)
    
    # initialise the ODR instance
    #out = ODR(data, model, beta0=[-0.89225534, 59.09509794]).run()
    out = ODR(data, model, beta0=[-0.5414, 36.08], job=0).run()
    # out.pprint()

    '''this only needed to run once as it is static
    # use sympy to calculate derivatives for each parameter (a and b)
    a, b, x = symbols('a b x')
    diffs = [diff(a * slog10(x) + b, a), diff(a * slog10(x) + b, b)]
    print (diffs)
    ## [log10(x), 1]
    '''

    # fit model using ODR params
    xn = linspace(min(x_data), max(x_data), 1000)
    yn = log_func(out.beta, xn)

    # calculate curve and confidence bands
    pl1 = prediction_interval(log_func, [log10(xn), 1], xn, yn, max(out.eps), 68., out.beta, out.cov_beta)
    pl2 = prediction_interval(log_func, [log10(xn), 1], xn, yn, max(out.eps), 95., out.beta, out.cov_beta)

    # create a figure to draw on and add a subplot
    fig, ax = subplots(1)

    # plot y calculated from px against de-logged x (and 1 and 2 sigma prediction intervals)
    ax.plot(xn, yn, '#EC472F', label='Logarithmic ODR')
    ax.plot(xn, yn + pl1, '#0076D4', dashes=[9, 4.5], label='1σ Prediction limit (~68%)', linewidth=0.8)
    ax.plot(xn, yn - pl1, '#0076D4', dashes=[9, 4.5], linewidth=0.8)
    ax.plot(xn, yn + pl2, '0.5', dashes=[9, 3], label='2σ Prediction limit (~95%)', linewidth=0.5)
    ax.plot(xn, yn - pl2, '0.5', dashes=[9, 3], linewidth=0.5)

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
    ax.set_xlabel('Mean R-Value')
    ax.set_ylabel('Age (ka)')
    ax.set_title('Orthogonal Distance Regression and Prediction Limits', pad = 11)


    # configure legend
    #ax.legend(bbox_to_anchor=(0.99, 0.99), borderaxespad=0., frameon=False,
         #     fontsize=10)
    ax.legend(frameon=False,
              fontsize=8)

    # Sets axis ratio to 1
    ratio = 1
    ax.set_aspect(1.0/ax.get_data_ratio()*ratio)

    # export the figure
    savefig('pyrenees_absolute.png', dpi = 600, bbox_inches='tight')
    #savefig('pyrenees2.svg')

    print(yn[0], pl1[0])
    print(min(pl1))
    print(yn[-1], pl1[-1])
    
    

def bootstrap_ODR(x_data, y_data, x_err, y_err):

    # Generates results files
    Result = []
    Prediction = []

    # create a figure to draw on and add a subplot
    fig, ax = subplots(1)

    # labels, extents etc.
    ax.set_xlabel('Mean R-Value')
    ax.set_ylabel('Age (ka)')
    ax.set_title('Bootstrapped ODR', pad = 11)

    # make function into Model instance (either log or linear)
    model = Model(log_func)

    # 1000 iterations                
    for i in range(1000):

        # Sets seed for reproducible results
        random.seed(214)

        # Randomises the data (mean, sd)
        x = np.random.normal(x_data, x_err)
        y = np.random.normal(y_data, y_err)

        # Truncated normal
        #x = truncnorm.rvs((x_err-x_data)/x_err, (x_err+x_data)/x_err,
        #                 loc=x_data, scale = x_err)
        #y = truncnorm.rvs((y_err-y_data)/y_err, (y_err+y_data)/y_err,
        #                  loc=y_data, scale = y_err)

              
        # Simple OLS for beta values
        slope, intercept, r_value, p_value, std_err = linregress(x[:],y[:])

        # New data and model
        data = Data(x, y)
        # Job = 0, explicit orthogonal
        out = ODR(data, model, beta0=[slope, intercept], job=0).run()

        # fit model using ODR params (from min to max of original data for consistency)
        xn = linspace(min(x_data), max(x_data), 1000)
        yn = log_func(out.beta, xn)

        # calculate curve and confidence bands
        pl1 = prediction_interval(log_func, [log10(xn), 1], xn, yn, max(out.eps), 68., out.beta, out.cov_beta)
        
        # Plots iteratively
        #ax.plot(xn, yn, '#EC472F', alpha = 0.1)
        ax.plot(xn, yn + pl1, '#0076D4', linewidth=0.8, alpha = 0.1)
        ax.plot(xn, yn - pl1, '#0076D4', linewidth=0.8, alpha = 0.1)

        # Appends results
        Result.append(yn)
        Prediction.append(pl1)

    # Returns "representative" bootstrapped values
    Central_Estimate = np.median(Result, axis = 0)
    Modal = np.median(Prediction, axis = 0)
    Minimum = np.min(Prediction, axis = 0)

    # One option using percentiles.
    #p95 = np.percentile(Result, 95, axis = 0)
    #p5 = np.percentile(Result, 5, axis = 0)

    # Returns maximum of Prediction
    #Prediction = np.round(Prediction,1)
    #Modal = mode(Prediction, axis = 0)[0]
    #Modal = Modal.flatten()

    ax.plot(xn, Central_Estimate, '#000000', alpha = 1)
    ax.plot(xn, Central_Estimate+Modal, '#000000', dashes=[9, 4.5], linewidth=0.8, alpha = 1)
    ax.plot(xn, Central_Estimate-Modal, '#000000', dashes=[9, 4.5], linewidth=0.8, alpha = 1)
    
    # Adds legend
    #ax.legend(frameon=False,
     #         fontsize=8)
    
    # plot points and error bars
    ax.plot(x_data, y_data, 'k.', markerfacecolor= '#FF8130',
             markeredgewidth=.5,  markeredgecolor = 'k',
            label='Calibration data (n = 54)', markersize=5)
    
    # Sets axis ratio to 1
    ratio = 1
    ax.set_aspect(1.0/ax.get_data_ratio()*ratio)

    savefig('pyrenees_bootstrap_prediction.png', dpi = 600, bbox_inches='tight')

    return Result, Prediction, Central_Estimate, Modal, Minimum
 
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

# check current working directory 
# cwd = os.getcwd() 
# print("Current working directory is:", cwd)

# Loads data
Be = pd.read_csv('data\Supplementary_Table_1_10Be.csv', encoding = "ISO-8859-1")
Cl = pd.read_csv('data\Supplementary_Table_2_36Cl.csv', encoding = "ISO-8859-1")
# Simplifies to key variables
cols = ['Group', 'Landform/Region','Publication', 'Isotope', 'Sample_name',
        'SH_Mean','SH_SEM','SH_STD', 'SH_Percent', 'CRONUS_Age_2020_03_27','CRONUS_Internal_2020_03_27',
        'CRONUS_External_2020_03_27', 'Age_Percent']
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

# Restricts to samples where external < 20% of age
# Calibration = Calibration[Calibration.Age_Percent < 0.2]

# load calibration data for curve creation
x_data = Calibration.loc[:, 'SH_Mean'].values
y_data = Calibration.loc[:, 'CRONUS_Age_2020_03_27'].values
x_err = Calibration.loc[:, 'SH_SEM'].values
# Using internal errors has only a minor impact on the computed prediction intervals (~0.1 ka). 
y_err = Calibration.loc[:, 'CRONUS_External_2020_03_27'].values


# load new data to be plotted
new_x_data = New.loc[:, 'SH_Mean'].values
new_y_data = New.loc[:, 'CRONUS_Age_2020_03_27'].values
new_x_err = New.loc[:, 'SH_SEM'].values
new_y_err = New.loc[:, 'CRONUS_External_2020_03_27'].values


# plot the curve
plot_shed_curve(x_data, y_data, x_err, y_err, new_x_data, new_y_data, new_x_err, new_y_err)

# Bootstrapping!
Result, Prediction, Central, Modal, Minimum = bootstrap_ODR(x_data, y_data, x_err, y_err)






