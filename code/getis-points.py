"""
Perform Spatial Autocorrelation and LISA analysis on points at sample locations
    on each moraine

author: jonnyhuck

Command to run:
    python getis-points.py > ../data/out/getis-points.txt
"""

from pandas import read_csv
from numpy.random import seed
from os import path, makedirs
from geopandas import GeoDataFrame
from shapely.geometry import Point
from pysal.lib.weights import DistanceBand
from pysal.explore.esda import G, G_Local
from pysal.lib.weights.util import min_threshold_distance


def getQuadrants(zs, sigs, acceptableSig):
    """
    * Return list of quadrant codes depending upon specified significance level
    * Quadrant codes not really relevant for Getis-Ord, just maintaining the
    *   terminology for simiplicity
    *
    * NA = insignificant
    * HH = cluster of high values
    * LL = cluster of low values
    """

    # return quad code rather than number
    out = []
    for q in range(len(zs)):
        # overrride non-significant values as N/A
        if sigs[q] < acceptableSig:
            # check if high or low based on sim
            if zs[q] >= 0:
                out.append("HH")
            else:
                out.append("LL")
        else:
            out.append("NA")
    return out


# set seed for reproducibility
seed(1824)

# make sure output directory is there
if not path.exists('../data/out/shapefiles/getis/points'):
    makedirs('../data/out/shapefiles/getis/points')
if not path.exists('../data/out/figures/getis/points'):
    makedirs('../data/out/figures/getis/points')


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
        go = G(moraine[s].apply(lambda x : 1 if x == 'Good' else 0), W, permutations=9999)
        print(f"\nGetis Ord G Results for {f}: {s}")
        print("G:\t\t\t", go.G)					   # value of G
        print("Expected G:\t\t", go.EG)			   # expected G
        print("Simulated p:\t\t", go.p_sim, "\n")  # simulated p

        # calculate local I
        lgo = G_Local(moraine[s].apply(lambda x : 1 if x == 'Good' else 0),
            W, transform='R', permutations=9999, star=True)

        # update GeoDataFrame with resulting quadrant
        result[s.replace('General_Class', 'Quad')] = getQuadrants(lgo.Zs, lgo.p_sim, 0.05)

    # output shapefile
    result.to_file("../data/out/shapefiles/getis/points/" + f + ".shp")

print("done")
