"""
This file includes all settings and configurables.
Variables in capital letters are to be configured.
-> Everything that needs to be changed for the code to work will be found in
this central location.
"""

import typing

#========Input_Training_and_Labelling_new_GUI_v2021.m====
INPUT_TIME: dict[str, int] = {'year_min': 1982, 'year_max': 2020, 'year2find': 2005,
              'month2plot': 1}
"""Configure here what was before in the input dialog 'please enter time'."""

INPUT_GEOBORDERS: dict[str, int] = {'latmin': -90, 'latmax': 90, 'lonmin': -180,
                    'lonmax': 180, 'invareamin': 1, 'invareamax': 11,
                    'latplotmin': -90, 'latplotmax': 90, 'lonplotmin': -180,
                    'lonplotmax': 180, 'version': 2};
"""Configure here what was before in the input dialog 'please enter geographic
boarders'."""

INPUT_METHOD: dict[str, bool] = {'FFN': False, 'SOM': False, 'Clustering': False, 'MLR': False,
                'BIOME': True, 'SSOM': False}
"""Configure here what was before in the input dialog
'please enter method'."""

INPUT_METHOD_FFN_OR_BIOME: dict[str, typing.Any] = {
    'net2take': 'GCBv2020_GCBv2020a-combined_smoothed', 'netlayer': '2layer',
    'layer2take': '2layer_plus_coast_v1p5', 'nnnumber': 50,
    'load_trained_net': 'yes or no'}
"""If INPUT_METHOD['FFN'] == True or INPUT_METHOD['BIOME'] == True further
enter method."""

INPUT_METHOD_SOM_OR_BIOME: dict[str, typing.Any] = {'SOMnr': 'SOM', 'maplength': 4, 'maphight': 4,
                             'epochnr': int(1e6), 'load_trained_SOM': 'yes or no',
                             'hc_clusters': 10}
"""If INPUT_METHOD['SOM'] == True or INPUT_METHOD['BIOME'] == True further
enter method."""

INPUT_METHOD_CL: dict[str, typing.Any] = {'clusternr': 'cl100', 'nb_cluster': 100}
"""If INPUT_METHOD['CL'] == True further enter method."""

INPUT_METHOD_MLR: dict[str, str] = {'MLRnr': 'MLR100'}
"""If INPUT_METHOD['MLR'] == True further enter method."""

INPUT_METHOD_SSOM: dict[str, typing.Any] = {'SSOMnr': 'fCO2_Super_SOM_biome_1', 'nnnumber': 2}
"""If INPUT_METHOD['SSOM'] == True further enter method."""


#========PATHS====
PATH_DATA_ACO2: str = 'data/aco2.mat'
PATH_DATA_CHL: str = 'data/chl.mat'
PATH_DATA_MLD: str = 'data/mld.mat'
PATH_DATA_PCO2TAKA: str = 'data/pco2_taka.mat'
PATH_DATA_PRESSURE: str = 'data/pressure.mat'
PATH_DATA_SEAICE: str = 'data/seaice.mat'
PATH_DATA_SOCAT: str = 'data/socat.mat'
PATH_DATA_SSS: str = 'data/sss.mat'
PATH_DATA_SST: str = 'data/sst.mat'
PATH_DATA_WIND: str = 'data/wind.mat'
"""These are the paths to the data, at the moment as .mat files."""


#========STEP1====
STEP1_TWENTYYEARAVERAGE_TIMEVEC_MIN: int = 1982
STEP1_TWENTYYEARAVERAGE_TIMEVEC_MAX: int = 2022
"""timevec is the time dimension (first dimension of the data, e.g. months)
in years."""


#========STEP2====
"""Define basin.
"""
STEP2_OCEAN_GLOBAL: bool = True
"""Please enter 'False' for Atlantic or 'True' for Global."""
STEP2_DEACTIVATE_TESTNUM: str = 'no'
"""'yes' or 'no'"""
STEP2_VAL_CHECK_ONLY: str = 'no'
"""'yes' or 'no'"""
STEP2_PERFORMANCE_FUNCTION: str = 'mse'
"""'mse' or 'sse'"""


#-----------------------------------------------------------------------------
"""Loading the configured variables from the settings script.
Please don't change anything below this point.
"""
import matplotlib.colors
import numpy as np
import scipy.io

year_min: int = INPUT_TIME['year_min']
year_max: int = INPUT_TIME['year_max']
year2find: int = INPUT_TIME['year2find']
month2plot: int = INPUT_TIME['month2plot']

year_output: np.ndarray = np.arange(year_min, year_max+1)
# timevec = np.arange(1980, year_max+1+1/12, 1/12)
# TODO: delete line?

loncropmin: int = INPUT_GEOBORDERS['lonmin']
loncropmax: int = INPUT_GEOBORDERS['lonmax']
latcropmin: int = INPUT_GEOBORDERS['latmin']
latcropmax: int = INPUT_GEOBORDERS['latmax']
invareamin: int = INPUT_GEOBORDERS['invareamin']
invareamax: int = INPUT_GEOBORDERS['invareamax']
lonmin: int = INPUT_GEOBORDERS['lonplotmin']
lonmax: int = INPUT_GEOBORDERS['lonplotmax']
latmin: int = INPUT_GEOBORDERS['latplotmin']
latmax: int = INPUT_GEOBORDERS['latplotmax']
version: int = INPUT_GEOBORDERS['version']

FFN_go: bool = INPUT_METHOD['FFN']
SOM_go: bool = INPUT_METHOD['SOM']
CL_go: bool = INPUT_METHOD['Clustering']
MLR_go: bool = INPUT_METHOD['MLR']
BIOME_go: bool = INPUT_METHOD['BIOME']
SSOM_go: bool = INPUT_METHOD['SSOM']

if(FFN_go == True or BIOME_go == True):
    net2take: str = INPUT_METHOD_FFN_OR_BIOME['net2take']
    netlayer: str = INPUT_METHOD_FFN_OR_BIOME['netlayer']
    layer2take: str = INPUT_METHOD_FFN_OR_BIOME['layer2take']
    nnnumber: int = INPUT_METHOD_FFN_OR_BIOME['nnnumber']
    load_trained_net: str = INPUT_METHOD_FFN_OR_BIOME['load_trained_net']

if(SOM_go == True or BIOME_go == True):
    SOMnr: str = INPUT_METHOD_SOM_OR_BIOME['SOMnr']
    maplength: int = INPUT_METHOD_SOM_OR_BIOME['maplength']
    maphight: int = INPUT_METHOD_SOM_OR_BIOME['maphight']
    epochnr: int = INPUT_METHOD_SOM_OR_BIOME['epochnr']
    load_trained_SOM: str = INPUT_METHOD_SOM_OR_BIOME['load_trained_SOM']
    hc_clusters: int = INPUT_METHOD_SOM_OR_BIOME['hc_clusters']

if(CL_go == True):
    clusternr: str = INPUT_METHOD_CL['clusternr']
    nb_cluster: int = INPUT_METHOD_CL['nb_cluster']

if(MLR_go == True):
    MLRnr: str = INPUT_METHOD_MLR['MLRnr']

if(SSOM_go == True):
    SSOMnr: str = INPUT_METHOD_SSOM['SSOMnr']
    nnnumber: int = INPUT_METHOD_SSOM['nnnumber']


#========LOAD DATA====
data_aco2: np.ndarray = scipy.io.loadmat(PATH_DATA_ACO2, appendmat=False)['aco2']
data_mld: np.ndarray = scipy.io.loadmat(PATH_DATA_MLD, appendmat=False)['mld']
data_pco2_taka: np.ndarray = scipy.io.loadmat(PATH_DATA_PCO2TAKA,
                                    appendmat=False)['pco2_taka']
data_sss: np.ndarray = scipy.io.loadmat(PATH_DATA_SSS, appendmat=False)['sss']
data_sst: np.ndarray = scipy.io.loadmat(PATH_DATA_SST, appendmat=False)['sst']
data_chl: np.ndarray = scipy.io.loadmat(PATH_DATA_CHL, appendmat=False)['chl']

data_lat: np.ndarray = np.tile(np.linspace(-89.5, 89.5, 180), (360, 1)).T
data_lon: np.ndarray = np.tile(np.linspace(-179.5, 179.5, 360), (180, 1))


colormap = matplotlib.colors.ListedColormap(np.loadtxt('cm.txt'))