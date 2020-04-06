"""
Demonstration of Orthogonal Distance Regression for Schmidt Hammer exposure dating
@author: jonnyhuck

References:
    https://stackoverflow.com/questions/60889680/how-to-plot-1-sigma-prediction-interval-for-scipy-odr/60927487#60927487
    https://www.astro.rug.nl/software/kapteyn/kmpfittutorial.html#confidence-and-prediction-intervals
    https://www.astro.rug.nl/software/kapteyn/EXAMPLES/kmpfit_ODRparabola_confidence.py
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.odr.Output.html
"""

# from sympy import symbols, diff
from scipy.stats import t, chisquare
from scipy.odr import ODR, Model, RealData
from matplotlib.pyplot import subplots, savefig
from sympy.codegen.cfunctions import log10 as slog10
from numpy import log10, array, linspace, zeros, sqrt

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

    # chi squared goodness of fit test (do not use out.res_var - it isn't the same)
    r_chisq, p = chisquare(y, func(popt, x), np)

    # ?? some sort of processing of covarianvce matrix and derivatives...
    d = zeros(n)
    for j in range(np):
        for k in range(np):
            d += dfdp[j] * dfdp[k] * pcov[j,k]

    # ?? the above value is then combines with that max error value
    d1 = sqrt(yerr**2 + d)

    # return prediction band offset for a new measurement
    return tval * d1


def log_func(beta, x):
    """
    * Log function for fitting using ODR
    """
    # logarithmic, m * log(x) + c
    return beta[0] * log10(x) + beta[1]


def plot_shed_curve(x_data, y_data, x_err, y_err):
    """
    * Fit curve using ODR and plot
    """
    # make function into Model instance (either log or linear)
    model = Model(log_func)

    # make data into RealData instance
    data = RealData(x_data, y_data, sx=x_err, sy=y_err)

    # initialise the ODR instance
    out = ODR(data, model, beta0=[-0.89225534, 59.09509794]).run()
    # out.pprint()

    ''' this only needed to run once as it is static '''
    # use sympy to calculate derivatives for each parameter (a and b)
    # a, b, x = symbols('a b x')
    # diffs = [diff(a * slog10(x) + b, a), diff(a * slog10(x) + b, b)]
    # print (diffs)
    ## [log10(x), 1]

    # fit model using ODR params
    xn = linspace(min(x_data), max(x_data), 1000)
    yn = log_func(out.beta, xn)

    # calculate curve and confidence bands
    pl1 = prediction_interval(log_func, [log10(xn), 1], xn, yn, max(out.eps), 68., out.beta, out.cov_beta)
    pl2 = prediction_interval(log_func, [log10(xn), 1], xn, yn, max(out.eps), 95., out.beta, out.cov_beta)

    # create a figure to draw on and add a subplot
    fig, ax = subplots(1)

    # plot y calculated from px against de-logged x (and 1 and 2 sigma prediction intervals)
    ax.plot(xn, yn, '#EC472F', label='Orthogonal Distance Regression (log)')
    ax.plot(xn, yn + pl1, '#0076D4', dashes=[9, 4.5], label=f'Prediction limit (68%)', linewidth=0.8)
    ax.plot(xn, yn - pl1, '#0076D4', dashes=[9, 4.5], linewidth=0.8)
    ax.plot(xn, yn + pl2, '0.5', dashes=[9, 3], label='2 Sigma Prediction Limit (95%)', linewidth=0.5)
    ax.plot(xn, yn - pl2, '0.5', dashes=[9, 3], linewidth=0.5)

    # plot points and error bars
    ax.plot(x_data, y_data, 'k.', label='Age Control Points', markersize=3.5)
    ax.errorbar(x_data, y_data, ecolor='k', xerr=x_err, yerr=y_err, fmt=" ", linewidth=0.5, capsize=0)

    # labels, extents etc.
    ax.set_ylim(0, 60)
    ax.set_xlabel('Mean R-Value')
    ax.set_ylabel('Age (ka)')
    ax.set_title('Orthogonal Distance Regression and Prediction Limits')

    # configure legend
    ax.legend(bbox_to_anchor=(0.99, 0.99), borderaxespad=0., frameon=False)

    # export the figure
    savefig('pyrenees.png')


# load data from Pyrenees curve
x_data = array([54.034,57.601,61.934,59.234,51.135,57.802,49.502,48.602,47.202,46.603,46.904,51.638,55.372,51.905,53.072,42.904,42.705,39.971,39.671,38.905,52.140,49.506,53.507,41.950,23.409,26.477,25.811,40.584,45.152,45.820,45.820,41.585,40.084,45.487,40.318,38.451,45.488,42.920,46.755,45.955,48.857,47.223,51.025,47.790,48.224,50.092,48.124,49.258,47.524,48.836,59.836,49.603])
y_data = array([12.866,12.436,4.138,5.285,11.740,8.944,13.828,13.600,14.761,15.644,15.982,12.130,8.714,12.178,11.783,20.982,20.609,22.634,23.234,23.858,10.967,11.988,11.987,21.307,51.862,43.874,42.212,20.431,17.226,17.374,17.453,20.949,22.503,18.887,22.052,24.974,18.005,19.417,18.157,18.720,16.550,16.414,15.022,16.475,16.699,16.452,16.406,15.151,17.410,16.924,8.673,17.910])
x_err  = array([0.65,0.67,0.65,0.71,0.49,0.73,1.04,1.18,0.91,0.89,0.94,0.95,0.76,0.9,0.99,0.99,1.07,0.94,0.86,0.94,1.41,1.38,1.09,1.21,0.98,1.02,1.12,1.06,0.92,0.98,1.02,1.2,1.03,1.22,1.1,1.15,1.05,1.44,1.26,1.34,1.07,0.68,1.01,1.09,1.22,0.82,1.09,0.76,1.05,0.82,0.74,0.87])
y_err  = array([1.071,1.014,0.341,0.446,0.959,0.821,1.2,1.614,1.49,1.52,1.437,1.131,0.838,1.44,1.24,3.91,3.292,3.393,3.969,3.371,1.972,1.734,2.856,4.633,4.171,3.524,3.388,1.659,1.389,1.417,1.434,1.693,1.815,1.516,2.267,2.002,1.448,1.557,1.456,1.502,1.331,1.322,1.207,1.327,1.336,2.692,2.592,2.704,2.352,2.713,1.901,4.756])

# plot the curve
plot_shed_curve(x_data, y_data, x_err, y_err)
