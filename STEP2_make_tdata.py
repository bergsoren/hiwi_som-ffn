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
    data_aco2 = scipy.io.loadmat(settings.PATH_DATA_ACO2, appendmat=False)['ac02']
    data_mld = scipy.io.loadmat(settings.PATH_DATA_MLD, appendmat=False)['mld']
    data_pco2_taka = scipy.io.loadmat(settings.PATH_DATA_PCO2TAKA,
                                    appendmat=False)['pco2_taka']
    data_sss = scipy.io.loadmat(settings.PATH_DATA_SSS, appendmat=False)['sss']
    data_sst = scipy.io.loadmat(settings.PATH_DATA_SST, appendmat=False)['sst']
    data_chl = scipy.io.loadmat(settings.PATH_DATA_CHL, appendmat=False)['chl']
    #TODO: load('output/BIOMEoutput_SOCAT/networks/SOM_biome_4x4.mat');

    # data_lat = scipy.io.loadmat(settings.PATH_DATA_LAT, appendmat=False)['latsst']
    # data_lon = scipy.io.loadmat(settings.PATH_DATA_LON, appendmat=False)['lonsst']
    # TODO: add from STEP1


    #========2) take 20-year average====






if __name__ == '__main__':
    print('-----------------------------------------------------------------')
    print('Thanks for using this script!')
    print('-----------------------------------------------------------------')
    run()
if __name__ != '__main__':
    print('-----------------------------------------------------------------')
    print('Thanks for importing the ' + __name__ + '.py script!')
    print('Usage:', end='\n\n')
    print('import ' + __name__ + ' as step2')
    print('step2.run()')
    print('-----------------------------------------------------------------')