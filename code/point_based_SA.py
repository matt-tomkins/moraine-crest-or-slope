"""
Perform Spatial Autocorrelation and LISA analysis on points at sample locations
    on each moraine

author: jonnyhuck

Command to run:
    python Spatial_Autocorrelation.py > ../data/out/moran-points.txt
"""

from pandas import read_csv
from numpy.random import seed
from os import path, makedirs
from geopandas import GeoDataFrame
from shapely.geometry import Point
from pysal.lib.weights import DistanceBand
from pysal.explore.esda import Moran, Moran_Local
from pysal.viz.splot.esda import moran_scatterplot
from matplotlib.pyplot import savefig, close as plt_close
from pysal.lib.weights.util import min_threshold_distance


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
if not path.exists('../data/out/shapefiles/points'):
    makedirs('../data/out/shapefiles/points')
if not path.exists('../data/out/figures/points'):
    makedirs('../data/out/figures/points')


# open csv file of ages
ages = read_csv('../data/Supplementary_Table_3_SH.csv', encoding='latin-1')[['Sample_name',
    'Landform', 'Longitude_DD', 'Latitude_DD', 'General_Class_1sigma', 'General_Class_2sigma',
    'General_Class_3sigma']]

# create geodataframe from the csv dataset
moraines = GeoDataFrame( ages, crs={'init': 'epsg:4326'},
    geometry=[Point(xy) for xy in zip(ages.Longitude_DD, ages.Latitude_DD)])

# loop through moraines
for f in ages.Landform.unique():

    # select just the current moraine
    moraine = moraines[moraines.Landform == f]

    # make a copy for writing results to
    result = moraine.copy()

    # calculate weights using minimum nearest neighbour distance threshold
    W = DistanceBand.from_dataframe(moraine, threshold=min_threshold_distance(
        [[x, y] for x, y in zip(moraine.Longitude_DD, moraine.Latitude_DD)]), binary=False)

    # perform row standardisation (so all weights in a row add up to 1)
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
            savefig(f'../data/out/figures/points/moran_{f}_{s}.png')
            plt_close(fig)
        except:
            # lazily ignore error and carry on - this is caused by nan value for
            #  I - which is because all of the data are the same ('Good')
            pass

        # calculate local I
        lisa = Moran_Local(moraine[s].apply(lambda x : 1 if x == 'Good' else 0),
            W, transformation='R', permutations=9999)

        # update GeoDataFrame with resulting quadrant
        result[s.replace('General_Class', 'Quad')] = getQuadrants(lisa.q, lisa.p_sim, 0.05)

        # plot local moran (plot, save, close)
        try:
            fig, ax = moran_scatterplot(lisa, p=0.05)
            savefig(f'../data/out/figures/points/lisa_{f}_{s}.png')
            plt_close(fig)
        except:
            # lazily ignore error and carry on - this is caused by nan value for
            #  I - which is because all of the data are the same ('Good')
            pass

    # output shapefile
    result.to_file("../data/out/shapefiles/points/" + f + ".shp")

print("done")
