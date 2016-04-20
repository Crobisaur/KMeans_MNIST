__author__= 'Christo Robison'

import numpy as np
import h5py
import matplotlib.pyplot as plt


f = h5py.File('mnist40_Centroids.h5','r')
temp = f['mnist40_Centroids_numpy'][:]
f.close()

centroids = temp


# take centroids use them as value for each node (one for each centroid)
# calculate beta values (standard deviation of each centroid) and weights for output nodes
# one for each class, 10 in this case.  One for each digit.

