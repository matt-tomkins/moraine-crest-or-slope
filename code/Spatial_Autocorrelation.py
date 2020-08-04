'''
author: 

Code to perform iterative global and local Moran spatial autocorrelation on
MC simulated datasets.

Analysis of 6 landforms, each with 1000 simulated datasets.

Intended outcomes:
    - For each dataset, the results of global Morans I i.e. is there stat.
    signif. spatial clustering?
    - If yes, the results of local Morans I with a record of cluster locations
    (Voronoi polygons).
    - For each landform, a record of clustering frequency (i.e. # signif. / 1000). 

Sources:
    - 

'''

# Loops through each landform

    # Loops through each simulated dataset (n = 1000)

        # Performs global Morans I

        # Stores result and modelled p value

        # If simulated p is < 0.05:

            # Add to result Significant += 1

            # Perform local Morans I

            # Record the locations of significant clusters

        # Else (simulated p > 0.05:

            # Add to result Not-Significant += 1

    
'''

At the end, we should we able to ask the following:

    (1) Does the degree of clustering vary between landforms?

    (2) Where are the good/bad boulder clusters?

        - This can either be presented as a true value (e.g. this voronoi polygon
        was a cluster n times.
        - OR proportionally (e.g. the polygon was a cluster in n % of iterations). 



'''
