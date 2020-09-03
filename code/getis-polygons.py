"""
Perform Spatial Autocorrelation and LISA analysis on voroni polygons around sample
    locations on each moraine

author: jonnyhuck

Command to run:
    python getis-polygons.py > ../data/out/getis-polygons.txt
"""

from pandas import read_csv
from numpy.random import seed
from os import path, makedirs
from geopandas import read_file
from pysal.lib.weights import Queen
from pysal.explore.esda import G, G_Local


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
if not path.exists('../data/out/shapefiles/getis/polygons'):
    makedirs('../data/out/shapefiles/getis/polygons')
if not path.exists('../data/out/figures/getis/polygons'):
    makedirs('../data/out/figures/getis/polygons')

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
        go = G(moraine[s].apply(lambda x : 1 if x == 'Good' else 0), W, permutations=9999)
        print(f"\nGetis Ord G Results for {f}: {s}")
        print("G:\t\t\t", go.G)					   # value of G
        print("Expected G:\t\t", go.EG)			   # expected G
        print("Simulated p:\t\t", go.p_sim, "\n")  # simulated p

        # calculate local Gi* (include focal point)
        lgo = G_Local(moraine[s].apply(lambda x : 1 if x == 'Good' else 0),
            W, transform='R', permutations=9999, star=True)

        # update GeoDataFrame
        result[s.replace('General_Class', 'Quad')] = getQuadrants(lgo.Zs, lgo.p_sim, 0.05)

    # output shapefile
    result.to_file("../data/out/shapefiles/getis/polygons/" + f + ".shp")

print("done")
