"""
This file includes all settings and configurables.
Variables in capital letters are to be configured.
-> Everything that needs to be changed for the code to work will be found in
this central location.
"""

#========Input_Training_and_Labelling_new_GUI_v2021.m====
INPUT_TIME = {'year_min': 1982, 'year_max': 2020, 'year2find': 2005,
              'month2plot': 1}
"""Configure here what was before in the input dialog 'please enter time'."""

INPUT_GEOBORDERS = {'latmin': -90, 'latmax': 90, 'lonmin': -180,
                    'lonmax': 180, 'invareamin': 1, 'invareamax': 11,
                    'latplotmin': -90, 'latplotmax': 90, 'lonplotmin': -180,
                    'lonplotmax': 180, 'version': 2};
"""Configure here what was before in the input dialog 'please enter geographic
boarders'."""

INPUT_METHOD = {'FFN': False, 'SOM': False, 'Clustering': False, 'MLR': False,
                'BIOME': True, 'SSOM': False}
"""Configure here what was before in the input dialog
'please enter method'."""

INPUT_METHOD_FFN_OR_BIOME = {
    'net2take': 'GCBv2020_GCBv2020a-combined_smoothed', 'netlayer': '2layer',
    'layer2take': '2layer_plus_coast_v1p5', 'nnnumber': 50,
    'load_trained_net': 'yes or no'}
"""If INPUT_METHOD['FFN'] == True or INPUT_METHOD['BIOME'] == True further
enter method."""

INPUT_METHOD_SOM_OR_BIOME = {'SOMnr': 'SOM', 'maplength': 4, 'maphight': 4,
                             'epochnr': int(1e6), 'load_trained_SOM': 'yes or no',
                             'hc_clusters': 10}
"""If INPUT_METHOD['SOM'] == True or INPUT_METHOD['BIOME'] == True further
enter method."""

INPUT_METHOD_CL = {'clusternr': 'cl100', 'nb_cluster': 100}
"""If INPUT_METHOD['CL'] == True further enter method."""

INPUT_METHOD_MLR = {'MLRnr': 'MLR100'}
"""If INPUT_METHOD['MLR'] == True further enter method."""

INPUT_METHOD_SSOM = {'SSOMnr': 'fCO2_Super_SOM_biome_1', 'nnnumber': 2}
"""If INPUT_METHOD['SSOM'] == True further enter method."""


#========1) load cluster data====
PATH_DATA_ACO2 = 'data/aco2.mat'
PATH_DATA_CHL = 'data/chl.mat'
PATH_DATA_MLD = 'data/mld.mat'
PATH_DATA_PCO2TAKA = 'data/pco2_taka.mat'
PATH_DATA_PRESSURE = 'data/pressure.mat'
PATH_DATA_SEAICE = 'data/seaice.mat'
PATH_DATA_SOCAT = 'data/socat.mat'
PATH_DATA_SSS = 'data/sss.mat'
PATH_DATA_SST = 'data/sst.mat'
PATH_DATA_WIND = 'data/wind.mat'
"""These are the paths to the data, at the moment as .mat files."""


#========2) take 20-year average====
TWENTYYEARAVERAGE_TIMEVEC_MIN = 1982
TWENTYYEARAVERAGE_TIMEVEC_MAX = 2022
"""timevec is the time dimension (first dimension of the data, e.g. months)
in years"""


#========3) reshape and rearrange for SOM====



#-----------------------------------------------------------------------------
import numpy as np
"""Loading the configured variables from the settings script.
Please don't change anything below this point.
"""
year_min = INPUT_TIME['year_min']
year_max = INPUT_TIME['year_max']
year2find = INPUT_TIME['year2find']
month2plot = INPUT_TIME['month2plot']

year_output = np.arange(year_min, year_max+1)
# timevec = np.arange(1980, year_max+1+1/12, 1/12)
# TODO: delete line?

loncropmin = INPUT_GEOBORDERS['lonmin']
loncropmax = INPUT_GEOBORDERS['lonmax']
latcropmin = INPUT_GEOBORDERS['latmin']
latcropmax = INPUT_GEOBORDERS['latmax']
invareamin = INPUT_GEOBORDERS['invareamin']
invareamax = INPUT_GEOBORDERS['invareamax']
lonmin = INPUT_GEOBORDERS['lonplotmin']
lonmax = INPUT_GEOBORDERS['lonplotmax']
latmin = INPUT_GEOBORDERS['latplotmin']
latmax = INPUT_GEOBORDERS['latplotmax']
version = INPUT_GEOBORDERS['version']

FFN_go = INPUT_METHOD['FFN']
SOM_go = INPUT_METHOD['SOM']
CL_go = INPUT_METHOD['Clustering']
MLR_go = INPUT_METHOD['MLR']
BIOME_go = INPUT_METHOD['BIOME']
SSOM_go = INPUT_METHOD['SSOM']

if(FFN_go == True or BIOME_go == True):
    net2take = INPUT_METHOD_FFN_OR_BIOME['net2take']
    netlayer = INPUT_METHOD_FFN_OR_BIOME['netlayer']
    layer2take = INPUT_METHOD_FFN_OR_BIOME['layer2take']
    nnnumber = INPUT_METHOD_FFN_OR_BIOME['nnnumber']
    load_trained_net = INPUT_METHOD_FFN_OR_BIOME['load_trained_net']

if(SOM_go == True or BIOME_go == True):
    SOMnr = INPUT_METHOD_SOM_OR_BIOME['SOMnr']
    maplength = INPUT_METHOD_SOM_OR_BIOME['maplength']
    maphight = INPUT_METHOD_SOM_OR_BIOME['maphight']
    epochnr = INPUT_METHOD_SOM_OR_BIOME['epochnr']
    load_trained_SOM = INPUT_METHOD_SOM_OR_BIOME['load_trained_SOM']
    hc_clusters = INPUT_METHOD_SOM_OR_BIOME['hc_clusters']

if(CL_go == True):
    clusternr = INPUT_METHOD_CL['clusternr']
    nb_cluster = INPUT_METHOD_CL['nb_cluster']

if(MLR_go == True):
    MLRnr = INPUT_METHOD_MLR['MLRnr']

if(SSOM_go == True):
    SSOMnr = INPUT_METHOD_SSOM['SSOMnr']
    nnnumber = INPUT_METHOD_SSOM['nnnumber']