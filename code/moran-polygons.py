"""
Perform Spatial Autocorrelation and LISA analysis on voroni polygons around sample
    locations on each moraine

author: jonnyhuck

Command to run:
    python Spatial_Autocorrelation.py > ../data/out/moran-polygons.txt
"""

from pandas import read_csv
from numpy.random import seed
from os import path, makedirs
from geopandas import read_file
from pysal.lib.weights import Queen
from pysal.explore.esda import Moran, Moran_Local
from pysal.viz.splot.esda import moran_scatterplot
from matplotlib.pyplot import savefig, close as plt_close


def getQuadrants(qs, sigs, acceptableSig):
    """
    * Return list of quadrant codes depending upon specified significance level
    """

    '''
    # this is just used to convert the output into quad names:
        NA = insignificant
        HH = cluster of high value
        HL = high value outlier amongst low values
        LH = low value outlier amongst high values
        LL = cluster of low values
    '''
    quadList = ["NA", "HH", "LH", "LL", "HL"]

    # return quad code rather than number
    out = []
    for q in range(len(qs)):
        # overrride non-significant values as N/A
        if sigs[q] < acceptableSig:
            out.append(quadList[qs[q]])
        else:
            out.append(quadList[0])
    return out


# set seed for reproducibility
seed(1824)

# make sure output directory is there
if not path.exists('../data/out/shapefiles/moran/polygons'):
    makedirs('../data/out/shapefiles/moran/polygons')
if not path.exists('../data/out/figures/moran/polygons'):
    makedirs('../data/out/figures/moran/polygons')

# open csv file of ages
ages = read_csv('../data/Supplementary_Table_3_SH.csv', encoding='latin-1')[['Sample_name',
    'General_Class_1sigma', 'General_Class_2sigma', 'General_Class_3sigma']]

# loop through moraines
for f in ['Arànser_Left', 'Arànser_Right', 'Outer_Pleta_Naua', 'Soum_dEch', 'Tallada']:

    # open file and create copy to write results
    moraine = read_file(f'../data/Shapefiles/Voronoi/{f}_Voronoi.shp')

    # merge (join) the age data
    moraine = moraine.merge(ages, how='inner', left_on="Sample_nam", right_on="Sample_name")

    result = moraine.copy()

    # calculate and row standardise weights matrix
    W = Queen.from_dataframe(moraine)
    W.transform = 'r'

    # loop through the columns
    for s in ['General_Class_1sigma', 'General_Class_2sigma', 'General_Class_3sigma']:

        # calculate and report global I
        mi = Moran(moraine[s].apply(lambda x : 1 if x == 'Good' else 0), W, permutations=9999)
        print(f"\nGlobal Moran's I Results for {f}: {s}")
        print("I:\t\t\t", mi.I)					   # value of Moran's I
        print("Expected I:\t\t", mi.EI)			   # expected Moran's I
        print("Simulated p:\t\t", mi.p_sim, "\n")  # simulated p

        # scatterplot for global moran (plot, save, close)
        try:
            fig, ax = moran_scatterplot(mi)
            savefig(f'../data/out/figures/moran/polygons/moran_{f}_{s}.png')
            plt_close(fig)
        except:
            # lazily ignore error and carry on - this is caused by nan value for
            #  I - which is because all of the data are the same ('Good')
            pass

        # calculate local I
        lisa = Moran_Local(moraine[s].apply(lambda x : 1 if x == 'Good' else 0),
            W, transformation='R', permutations=9999)

        # update GeoDataFrame
        result[s.replace('General_Class', 'Quad')] = getQuadrants(lisa.q, lisa.p_sim, 0.05) # quadrant (HH, HL, LH, LL)

        # combined plot for local moran (plot, save, close)
        try:
            fig, ax = moran_scatterplot(lisa, p=0.05)
            savefig(f'../data/out/figures/moran/polygons/lisa_{f}_{s}.png')
            plt_close(fig)
        except:
            # lazily ignore error and carry on - this is caused by nan value for
            #  I - which is because all of the data are the same ('Good')
            pass

    # output shapefile
    result.to_file("../data/out/shapefiles/moran/polygons/" + f + ".shp")

print("done")
