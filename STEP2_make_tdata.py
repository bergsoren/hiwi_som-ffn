"""
To create the multidimensional dataset from specific datasets, one for
each parameter. Datasets for each parameter should be rescaled by this
point. Training dataset structure must be the same as labelling dataset.
This code follows the work of Maciek Telszewski and loads SST, MLD, CHL, 
BATHY, etc. data from e.g. reanalysis products to train the NN.

Original matlab code by:
Peter Landschutzer 17.01.2012
University of East Anglia, Norwich

Python version by:
Søren Jakob Berger 29.08.2024
Max-Planck-Institut für Meteorologie, Hamburg
"""

#========IMPORTS====
import numpy as np
import scipy.io

import time

import settings
import debug


def run() -> None:
    """_summary_
    """
    #========Load the data which are in (264,180,360) format====
    """TODO. The data is in the format [months
    latitude longitude] = 480x180x360. V2 used: ECCO, Raynolds, Globalviev, Taka, Globecolor.
    """
    data_aco2: np.ndarray = settings.data_aco2
    data_mld: np.ndarray = settings.data_mld
    data_pco2_taka: np.ndarray = settings.data_pco2_taka
    data_sss: np.ndarray = settings.data_sss
    data_sst: np.ndarray = settings.data_sst
    data_chl: np.ndarray = settings.data_chl

    data_STEP1_biomes: np.ndarray = scipy.io.loadmat('SOM_biome_4x4.mat', appendmat=False)['biomes']

    data_lat: np.ndarray = settings.data_lat
    data_lon: np.ndarray = settings.data_lon
    """data_lat and data_lon are both 180x360 arrays with data_lat[i, :]
    containing the latidude in degrees from -89.5 to 89.5 and data_lon[:, i]
    containing the longitude in degrees from -179.5 to 179.5.
    """

    for year2go in settings.year_output:
        print(year2go)
    #========The first parameter is SST====
        






if __name__ == '__main__':
    print('-----------------------------------------------------------------')
    print('Thanks for using this script!')
    print('While this script works just fine by directly running it,')
    print('it is optimised to be used in i.e. a jupyter notebook!', end='\n\n')
    print('Try')
    print('import name')
    print('where name.py is the name of this script.')
    print('-----------------------------------------------------------------')
    run()
if __name__ != '__main__':
    print('-----------------------------------------------------------------')
    print('Thanks for importing the ' + __name__ + '.py script!')
    print('Usage:', end='\n\n')
    print('import ' + __name__ + ' as step2')
    print('step2.run()')
    print('-----------------------------------------------------------------')