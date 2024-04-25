"""
Welcome to the simple things: this code uses PyTorch
(https://github.com/pytorch/pytorch) and creates monthly generated basin wide
fCO2 maps derrived from the SOM. The input file specifies the year and month
and the lat-lon area. 

Original matlab code by:
Peter Landschutzer 17.01.2012
University of East Anglia, Norwich

Python version by:
Søren Jakob Berger 24.04.2024
Max-Planck-Institut für Meteorologie, Hamburg
"""

#========IMPORTS====
import numpy as np
import scipy.io

import settings
import debug


#========Input_Training_and_Labelling_new_GUI_v2021.m====
year_min = settings.INPUT_TIME['year_min']
year_max = settings.INPUT_TIME['year_max']
year2find = settings.INPUT_TIME['year2find']
month2plot = settings.INPUT_TIME['month2plot']

year_output = np.arange(year_min, year_max+1)
#timevec = np.arange(1980, year_max+1+1/12, 1/12)

loncropmin = settings.INPUT_GEOBORDERS['lonmin']
loncropmax = settings.INPUT_GEOBORDERS['lonmax']
latcropmin = settings.INPUT_GEOBORDERS['latmin']
latcropmax = settings.INPUT_GEOBORDERS['latmax']
invareamin = settings.INPUT_GEOBORDERS['invareamin']
invareamax = settings.INPUT_GEOBORDERS['invareamax']
lonmin = settings.INPUT_GEOBORDERS['lonplotmin']
lonmax = settings.INPUT_GEOBORDERS['lonplotmax']
latmin = settings.INPUT_GEOBORDERS['latplotmin']
latmax = settings.INPUT_GEOBORDERS['latplotmax']
version = settings.INPUT_GEOBORDERS['version']

FFN_go = settings.INPUT_METHOD['FFN']
SOM_go = settings.INPUT_METHOD['SOM']
CL_go = settings.INPUT_METHOD['Clustering']
MLR_go = settings.INPUT_METHOD['MLR']
BIOME_go = settings.INPUT_METHOD['BIOME']
SSOM_go = settings.INPUT_METHOD['SSOM']

if(FFN_go == True or BIOME_go == True):
    net2take = settings.INPUT_METHOD_FFN_OR_BIOME['net2take']
    netlayer = settings.INPUT_METHOD_FFN_OR_BIOME['netlayer']
    layer2take = settings.INPUT_METHOD_FFN_OR_BIOME['layer2take']
    nnnumber = settings.INPUT_METHOD_FFN_OR_BIOME['nnnumber']
    load_trained_net = settings.INPUT_METHOD_FFN_OR_BIOME['load_trained_net']

if(SOM_go == True or BIOME_go == True):
    SOMnr = settings.INPUT_METHOD_SOM_OR_BIOME['SOMnr']
    maplength = settings.INPUT_METHOD_SOM_OR_BIOME['maplength']
    maphight = settings.INPUT_METHOD_SOM_OR_BIOME['maphight']
    epochnr = settings.INPUT_METHOD_SOM_OR_BIOME['epochnr']
    load_trained_SOM = settings.INPUT_METHOD_SOM_OR_BIOME['load_trained_SOM']
    hc_clusters = settings.INPUT_METHOD_SOM_OR_BIOME['hc_clusters']

if(CL_go == True):
    clusternr = settings.INPUT_METHOD_CL['clusternr']
    nb_cluster = settings.INPUT_METHOD_CL['nb_cluster']

if(MLR_go == True):
    MLRnr = settings.INPUT_METHOD_MLR['MLRnr']

if(SSOM_go == True):
    SSOMnr = settings.INPUT_METHOD_SSOM['SSOMnr']
    nnnumber = settings.INPUT_METHOD_SSOM['nnnumber']


#========1) load cluster data====
"""Loading the data needed for clustering. The data is in the format [months
latitude longitude] = 480x180x360."""
data_mld = scipy.io.loadmat(settings.PATH_DATA_MLD, appendmat=False)['mld']
data_pco2_taka = scipy.io.loadmat(settings.PATH_DATA_PCO2TAKA,
                                  appendmat=False)['pco2_taka']
data_sss = scipy.io.loadmat(settings.PATH_DATA_SSS, appendmat=False)['sss']
data_sst = scipy.io.loadmat(settings.PATH_DATA_SST, appendmat=False)['sst']


#========2) take 20-year average====
timevec = np.arange(settings.TWENTYYEARAVERAGE_TIMEVEC_MIN,
                    settings.TWENTYYEARAVERAGE_TIMEVEC_MAX, 1/12)
"""timevec is the time dimension (first dimension of the data, e.g. months)
in years"""

for i in range(12):
    """Takes annual mean of the data ignoring NaNs, therefore the
    data_annual[0, :, :] is for january, data_annual[1, :, :] for
    february, ..."""
    data_mld_annual = np.nanmean(data_mld[i::12, :, :])
    data_pco2_taka_annual = np.nanmean(data_pco2_taka[i::12, :, :])
    data_sss_annual = np.nanmean(data_sss[i::12, :, :])
    data_sst_annual = np.nanmean(data_sst[i::12, :, :])

