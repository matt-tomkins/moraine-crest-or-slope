"""
Perform Spatial Autocorrelation and LISA analysis on points at sample locations
    on each moraine

author: jonnyhuck

Command to run:
    python moran_points.py > ../data/out/moran-points.txt
"""

from pandas import read_csv
from numpy.random import seed
from os import path, makedirs
from operator import itemgetter
from geopandas import GeoDataFrame
from shapely.geometry import Point
from pointpats import PointPattern
from pysal.lib.weights import DistanceBand
from pysal.explore.esda import Moran, Moran_Local
from pysal.viz.splot.esda import moran_scatterplot
from matplotlib.pyplot import savefig, close as plt_close
# from pysal.lib.weights.util import min_threshold_distance


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
if not path.exists('../data/Moran/shapefiles'):
    makedirs('../data/Moran/shapefiles')
if not path.exists('../data/Moran/figures'):
    makedirs('../data/Moran/figures')


# open csv file of ages
ages = read_csv('../data/Supplementary_Table_3_SH.csv', encoding='latin-1')[['Sample_name',
    'Landform', 'Longitude_DD', 'Latitude_DD', 'General_Class_1sigma', 'General_Class_2sigma',
    'General_Class_3sigma']]

# create geodataframe from the csv dataset
moraines = GeoDataFrame( ages, crs='epsg:4326',
    geometry=[Point(xy) for xy in zip(ages.Longitude_DD, ages.Latitude_DD)])

# loop through moraines
for f in ages.Landform.unique():

    # get the appropriate projection code for each moraine
    proj = "32630" if f == "Soum d'Ech" else "32631"

    # select just the current moraine and project as appropriate
    moraine = moraines[moraines.Landform == f].to_crs("epsg:"+proj)

    # make a copy for writing results to
    result = moraine.copy()

    # calculate weights using minimum nearest neighbour distance threshold with one neighbour
    # W = DistanceBand.from_dataframe(moraine, threshold=min_threshold_distance(
    #     [[x, y] for x, y in zip(moraine.geometry.x, moraine.geometry.y)], binary=False)

    # calculate weights using minimum nearest neighbour distance threshold with knn
    W = DistanceBand.from_dataframe(moraine, threshold=max(PointPattern(
        [[x, y] for x, y in zip(moraine.geometry.x, moraine.geometry.y)]).knn(2)[1],
        key=itemgetter(1))[1], binary=False)

    # print(W.cardinalities)  # you can use this to see how many neighbours each observation has

    # perform row standardisation (so all weights in a row add up to 1)
    W.transform = 'r'

    # loop through the columns
    for s in ['General_Class_1sigma', 'General_Class_2sigma', 'General_Class_3sigma']:

        # calculate and report global I
        mi = Moran(moraine[s].apply(lambda x : 1 if x == 'Good' else 0), W, permutations=9999)
        print(f"\nGlobal Moran's I Results for {f}: {s}")
        print("I:\t\t\t", mi.I)					   # value of Moran's I
        print("Expected I:\t\t", mi.EI)			   # expected Moran's I·
        print("Simulated p:\t\t", mi.p_sim, "\n")  # simulated p

        # scatterplot for global moran (plot, save, close)
        try:
            fig, ax = moran_scatterplot(mi)
            savefig(f'../data/Moran/figures/moran_{f}_{s}.png')
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
            savefig(f'../data/Moran/figures/lisa_{f}_{s}.png')
            plt_close(fig)
        except:
            # lazily ignore error and carry on - this is caused by nan value for
            #  I - which is because all of the data are the same ('Good')
            pass

    # output shapefile
    result.to_file("../data/Moran/shapefiles/" + f + ".shp")

print("done")
