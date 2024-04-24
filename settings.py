"""
This file includes all settings and configurables.
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
"""Configure here what was before in the input dialog 'please enter method'."""

INPUT_METHOD_FFN_OR_BIOME = {
    'net2take': 'GCBv2020_GCBv2020a-combined_smoothed', 'netlayer': '2layer',
    'layer2take': '2layer_plus_coast_v1p5', 'nnnumber': 50,
    'load_trained_net': 'yes or no'}
"""If INPUT_METHOD['FFN'] == True or INPUT_METHOD['BIOME'] == True further
enter method."""

INPUT_METHOD_SOM_OR_BIOME = {'SOMnr': 'SOM', 'maplength': 2, 'maphight': 2,
                             'epochnr': 5, 'load_trained_SOM': 'yes or no',
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
TWENTYYEARAVERAGE_TIMEVEC_MIN = 1980
TWENTYYEARAVERAGE_TIMEVEC_MAX = 2023