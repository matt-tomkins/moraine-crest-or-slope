"""
Demonstration of Orthogonal Distance Regression for Schmidt Hammer exposure dating
@author: jonnyhuck

Update working directory!

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

def standard_odr(x_data, y_data, x_err, y_err, new_x_data, new_y_data, new_x_err, new_y_err):
    """
    * Fit curve using ODR and plot
    """
    # make function into Model instance (either log or linear)
    model = Model(log_func)

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
    ax.plot(xn, yn + pl1, '#0076D4', dashes=[9, 4.5], label='1σ Prediction limit', linewidth=0.8)
    ax.plot(xn, yn - pl1, '#0076D4', dashes=[9, 4.5], linewidth=0.8)
    ax.plot(xn, yn + pl2, '0.5', dashes=[9, 3], label='2σ Prediction limit', linewidth=0.5)
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
    ax.errorbar(new_x_data, new_y_data, ecolor='k', xerr=new_x_err, yerr=new_y_err,
                fmt=" ", linewidth=0.5, capsize=0)

    
    # labels, extents etc.
    ax.set_ylim(0, 60)
    ax.set_xlabel('Mean R-value')
    ax.set_ylabel('Age (ka)')
    ax.set_title('Orthogonal Distance Regression', pad = 11)

    # configure legend
    ax.legend(frameon=False,
              fontsize=8)

    # Sets axis ratio to 1
    ratio = 1
    ax.set_aspect(1.0/ax.get_data_ratio()*ratio)

    # export the figure
    savefig('pyrenees_absolute.png', dpi = 600, bbox_inches='tight')
    #savefig('pyrenees2.svg')
    

def monte_carlo_odr(x_data, y_data, x_err, y_err, new_x_data, new_y_data, new_x_err, new_y_err):

    """
    Monte Carlo Orthogonal Distance Regression for Schmidt Hammer exposure dating
    @author: matt-tomkins

    Improves on standard ODR by explicitly incorporating XY errors, but without the complications
    introduced by weighting.

    1) Randomises the data (i = 1000) based on values (x, y) and associated errors (x_err, y_err).
    2) Constructs a standard logged OLS regression (used for ODR beta estimates).
    3) Detects outliers using internally standardized residuals from the OLS.  Those > 2 sigma are rejected. 
    4) Constructs an ODR and saves model coefficients (beta, covariance matrix, errors)
    5) Takes the means of these coefficients for final ODR model construction  
    
    """

    # Generates results files
    betas = []
    covariances = []
    eps = []
    counts =[]

    # make function into Model instance (either log or linear)
    model = Model(log_func)

    # Sets seed for reproducible results (!)
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
        counts.append(len(x))

    # Takes the median of the model estimates
    Beta = np.median(betas, axis = 0)
    Eps = np.median(eps, axis = 0)
    Covariance = np.median(covariances, axis = 0)

    # fit model using new beta, original x scale for consistency
    xn = linspace(min(x_data), max(x_data), 1000)
    yn = log_func(Beta, xn)

    # 1 and 2 sigma prediction intervals
    pl1 = prediction_interval(log_func, [log10(xn), 1], xn, yn, Eps, 68., Beta, Covariance)
    pl2 = prediction_interval(log_func, [log10(xn), 1], xn, yn, Eps, 95., Beta, Covariance)

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
    #ax.set_title('Monte-Carlo Orthogonal Distance Regression', pad = 11)
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
    savefig('pyrenees_Monte_Carlo.png', dpi = 900, bbox_inches='tight')

    return betas, eps, covariances, counts, pl1


def plot_coefficients(Beta, Eps, Covariance):
    
    # Converts to useful format
    beta = np.array(Beta)
    cov1 = np.array([i[0] for i in Covariance])
    cov2 = np.array([i[1] for i in Covariance])

    # Sets up axes
    fig, ax = plt.subplots(3, 2, sharey = 'row', figsize=(6,9))
    ax1, ax2, ax3, ax4, ax5, ax6 = ax.flatten()

    # plots data (%)
    n, bins, patches = ax1.hist(beta[:,0], weights=(np.ones(len(beta[:,0])) / len(beta[:,0]))*100, bins = 25, ec = "#000000", color = "#4E9DFF")
    n, bins, patches = ax2.hist(beta[:,1], weights=(np.ones(len(beta[:,1])) / len(beta[:,1]))*100, bins = 25, ec = "#000000", color = "#4E9DFF")
    n, bins, patches = ax3.hist(cov1[:,0], weights=(np.ones(len(cov1[:,0])) / len(cov1[:,0]))*100, bins = 25, ec = "#000000", color = "#4E9DFF")
    n, bins, patches = ax4.hist(cov1[:,1], weights=(np.ones(len(cov1[:,1])) / len(cov1[:,1]))*100, bins = 25, ec = "#000000", color = "#4E9DFF")
    n, bins, patches = ax5.hist(cov2[:,1], weights=(np.ones(len(cov2[:,1])) / len(cov2[:,1]))*100, bins = 25, ec = "#000000", color = "#4E9DFF")
    n, bins, patches = ax6.hist(Eps, weights=(np.ones(len(Eps)) / len(Eps))*100, bins = 25, ec = "#000000", color = "#4E9DFF")

    # Adds median values
    ax1.axvline(x=np.median(beta[:,0]), color = '#EC472F')
    ax2.axvline(x=np.median(beta[:,1]), color = '#EC472F')
    ax3.axvline(x=np.median(cov1[:,0]), color = '#EC472F')
    ax4.axvline(x=np.median(cov1[:,1]), color = '#EC472F')
    ax5.axvline(x=np.median(cov2[:,1]), color = '#EC472F')
    ax6.axvline(x=np.median(Eps), color = '#EC472F', ymax = 0.85)
    
    # Sets labels
    #ax1.set_xlabel(r'Slope $\beta$', labelpad=1)
    ax1.set_title(r'Slope $\beta$', position=(0.82, 0.85))
    ax1.set_ylabel('Frequency (%)')
    ax2.set_title(r'Intercept $\beta$', position=(0.22, 0.85))
    #ax2.set_xlabel(r'Intercept $\beta$', labelpad=1)
    ax3.set_title(r'Covariance $W$', position=(0.72, 0.85))
    #ax3.set_xlabel(r'Covariance $W$', labelpad=1.5)
    ax3.set_ylabel('Frequency (%)')
    ax4.set_title(r'Covariance $D$', position=(0.27, 0.85))
    #ax4.set_xlabel(r'Covariance $D$', labelpad=1.5)
    ax5.set_title(r'Covariance $E$', position=(0.72, 0.85))
    #ax5.set_xlabel(r'Covariance $E$', labelpad=1.5)
    ax5.set_ylabel('Frequency (%)')
    ax6.set_title('Maximum residual', position=(0.35, 0.85))
    #ax6.set_xlabel('Maximum residual', labelpad=1.5)


    fig.subplots_adjust(left=0.1)
    # saves
    savefig('pyrenees_coefficients.png', dpi = 900, bbox_inches='tight')

def tcn_errors(y_data,y_err,y_internal, facility):
    '''
    # Weighted errors (1/err^2)
    WE = 1/y_err**2

    # Sets up axes
    fig, ax = plt.subplots(1, 2, figsize=(6,3))
    ax1, ax2 = ax.flatten()
    ax3 = inset_axes(ax2, width="45%", height="45%")
    
    fig.tight_layout(pad=1.5)

    # Draws data
    ax1.plot(y_data, y_err,'k.', markerfacecolor= '#4E9DFF',
             markeredgewidth=.5,  markeredgecolor = 'k', markersize = 6)

    ax2.plot(y_data, WE,'k.', markerfacecolor= '#4E9DFF',
             markeredgewidth=.5,  markeredgecolor = 'k', markersize = 6)
    
    ax3.plot(y_data, WE,'k.', markerfacecolor= '#4E9DFF',
             markeredgewidth=.5,  markeredgecolor = 'k', markersize = 4)

    # Labels, titles
    ax2.set_yscale('log')
    ax1.set_xlabel("Age (ka)")
    ax1.set_ylabel("External age uncertainty (ka)")
    ax2.set_xlabel("Age (ka)")
    ax2.set_ylabel("Weights (log)", labelpad = 1)
    ax3.set_ylabel("Weights", fontsize=9, labelpad = 2)
    ax3.tick_params(labelsize=8)

    ax1.text(0.03, 0.90, "A", fontweight="bold", fontsize = 16, transform=ax1.transAxes)
    ax2.text(0.07, 0.90, "B", fontweight="bold", fontsize = 16, transform=ax2.transAxes)
    ax3.text(0.89, 0.88, "C", fontweight="bold", fontsize = 9, transform=ax3.transAxes)
    
    #savefig('pyrenees_collinearity.png', dpi = 900, bbox_inches='tight')
    '''

    # Merges data
    df = pd.DataFrame(dict(error=y_internal, percent=y_internal/y_data*100, facility=facility))
    df.loc[df['facility'].str.contains("AST"),'colors'] = '#3897F8'
    df.loc[df['facility'].str.contains("Tand"),'colors'] = '#F83838'

    groups = df.groupby('facility')

    # create a figure to draw on and add a subplot
    fig, ax = plt.subplots(1, 1, figsize=(3,3))

    # Plotting by category
    for name, groups in groups:
        ax.plot(groups.error, groups.percent, marker='o', linestyle='', ms=6, label=name, markeredgecolor = 'k', markeredgewidth = '.5')


    # Aesthetics
    leg = plt.legend()
    leg.get_frame().set_edgecolor('k')
    leg.get_frame().set_linewidth(.5)
    ax.set_xlabel("Internal age uncertainty (ka)")
    ax.set_ylabel("Internal uncertainty / Exposure age (%)")

    
    savefig('pyrenees_errors.png', dpi = 900, bbox_inches='tight')


# Plots TCN errors
tcn_errors(y_data,y_err, y_internal, facility)

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
cols = ['Group', 'Landform/Region','Publication', 'Isotope', 'Facility', 'Sample_name',
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

y_internal = Calibration.loc[:, 'CRONUS_Internal_2020_03_27'].values
facility = Calibration.loc[:, 'Facility'].values


# load new data to be plotted
new_x_data = New.loc[:, 'SH_Mean'].values
new_y_data = New.loc[:, 'CRONUS_Age_2020_03_27'].values
new_x_err = New.loc[:, 'SH_SEM'].values
new_y_err = New.loc[:, 'CRONUS_External_2020_03_27'].values


# plot the curve, standard ODR
standard_odr(x_data, y_data, x_err, y_err, new_x_data, new_y_data, new_x_err, new_y_err)

# plot the curve, Monte Carlo ODR
Beta, Eps, Covariance, counts, pl1 = monte_carlo_odr(x_data, y_data, x_err, y_err, new_x_data, new_y_data, new_x_err, new_y_err)

# plots model values
plot_coefficients(Beta, Eps, Covariance)





