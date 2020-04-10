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
from scipy.stats import t
from scipy.odr import ODR, Model, RealData
from matplotlib.pyplot import subplots, savefig, rcParams
from numpy import log10, array, linspace, zeros, sqrt
import pandas as pd
import os

# Updates font for plotting
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

    # make data into RealData instance
    data = RealData(x_data, y_data, sx=x_err, sy=y_err)

    # initialise the ODR instance
    #out = ODR(data, model, beta0=[-0.89225534, 59.09509794]).run()
    out = ODR(data, model, beta0=[-0.5414, 36.08]).run()
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
    savefig('pyrenees.png', dpi = 600, bbox_inches='tight')
    #savefig('pyrenees.svg')


def predict_ages(x_data, y_data, x_err, y_err, new_x):
    """
    Uses prediction interval code above to predict for new values of x

    Doesn't work yet, predicted values are too big...
    
    """

    # make function into Model instance (either log or linear)
    model = Model(log_func)

    # make data into RealData instance
    data = RealData(x_data, y_data, sx=x_err, sy=y_err)

    # initialise the ODR instance
    out = ODR(data, model, beta0=[-0.89225534, 59.09509794]).run()

    # Error is around here somewhere
    # prediction values
    xn = new_x
    yn = log_func(out.beta, xn)

    # calculate curve and prediction interval (1 sigma)
    sigma = prediction_interval(log_func, [log10(xn), 1], xn, yn, max(out.eps), 68., out.beta, out.cov_beta)
    return yn, sigma




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
Be = pd.read_csv('Supplementary_Table_1_10Be.csv', encoding = "ISO-8859-1")
Cl = pd.read_csv('Supplementary_Table_2_36Cl.csv', encoding = "ISO-8859-1")
# Simplifies to key variables
cols = ['Group', 'Landform/Region','Publication', 'Isotope', 'Sample_name',
        'SH_Mean','SH_SEM', 'CRONUS_Age_2020_03_27','CRONUS_Internal_2020_03_27',
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

# British Data
# x_data = array([39.41,49.01,47.52,39.57,36.73,39.25,41.79,42.55,42.49,26.53,30.67,27.88,31,41.08,41.67,40.39,45.7,41.39,41.69,45.4,43.3,39.67,35.84,38.87,28.85,34.47,35.13,35.16,39.57,36.31,36,59.5,63,58.12,63.69,60.79,47.25,46.48,42.94,45.29,46.53,44.76,27.23,30.23,29.53,25.57,28.03,30.27,28.71,29.01,28.81,33.08,38.92,37.84,46.59,47.09,44.4,40.96,44.66,45.47,42.89,40.01,43.64,42.14,40.66])
# x_err = array([0.9,0.94,1.16,1.33,1.05,0.76,1.26,1.21,0.79,0.86,1.62,1.17,1.13,0.96,0.61,0.7,0.81,0.58,1.14,0.85,0.74,0.8,0.54,0.68,0.52,0.63,0.7,0.63,0.81,0.82,0.6,0.72,0.66,0.53,0.55,0.63,0.45,0.46,0.56,0.54,0.47,0.53,1.08,1.12,1.44,0.9,1.25,1.2,0.53,0.68,1.06,1.32,1.34,1.25,0.88,0.69,0.72,0.96,1.13,1.02,1,1.12,1.22,1.05,1.18])   
# y_data = array([15.27,12.53,12.89,13.52,13.12,14.46,12.71,13.59,12.08,21.42,18.88,19.26,21.15,14.55,14.21,14.89,11.27,12.09,11.57,12.19,12.8,16.31,19.13,15.69,20.01,15.18,13.62,15.92,16.03,14.47,15.33,2.67,1.58,5.3,0.83,2.31,10.68,10.86,12.11,11.06,10.84,11.52,20.95,20.43,20.53,22.72,20.94,21.31,21.82,21.96,20.01,17.4,15.12,15.91,10.83,8.91,11.03,13.17,12.97,12.64,15.19,14.7,14.11,12.17,13.76])
# y_err = array([1.2,1.02,1.04,1.3,1.06,1.17,1.03,1.11,0.97,1.81,2.64,2.34,2.9,2.02,1.1,1.36,1.08,1.08,1.03,1.13,1.13,1.39,1.59,1.34,1.67,1.43,1.39,1.84,1.45,1.54,1.44,0.38,0.2,0.55,0.15,0.31,0.84,0.86,0.96,0.88,0.86,0.91,1.87,1.91,2.03,2.16,2.13,2.14,1.86,1.74,1.77,1.51,1.28,1.33,0.93,0.8,0.91,1.17,1.17,1.1,1.38,1.24,1.22,1.05,1.19])

# plot the curve
plot_shed_curve(x_data, y_data, x_err, y_err, new_x_data, new_y_data, new_x_err, new_y_err)

'''
Currently uses example prediction data

'''


# prediction?
Predicted_age,Predicted_uncertainty = predict_ages(x_data, y_data, x_err, y_err, new_x_data)
