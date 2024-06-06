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
import torch
import quicksom.som as qsom
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import settings
import debug

"""In quicksom.som one change is required:
    clusterer = AgglomerativeClustering(affinity='precomputed', linkage='average', n_clusters=n_local_min)
needs to be changed to
    clusterer = AgglomerativeClustering(metric='precomputed', linkage='average', n_clusters=n_local_min)
as sklearn.cluster has changed the keyword argument name.
"""

def step1() -> None:
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

    #data_lat = scipy.io.loadmat(settings.PATH_DATA_LAT, appendmat=False)['latsst']
    #data_lon = scipy.io.loadmat(settings.PATH_DATA_LON, appendmat=False)['lonsst']


    #========2) take 20-year average====
    timevec = np.arange(settings.TWENTYYEARAVERAGE_TIMEVEC_MIN,
                        settings.TWENTYYEARAVERAGE_TIMEVEC_MAX, 1/12)
    """timevec is the time dimension (first dimension of the data, e.g. months)
    in years."""
    data_lat = np.tile(np.linspace(-89.5, 89.5, 180), (360, 1)).T
    data_lon = np.tile(np.linspace(-179.5, 179.5, 360), (180, 1))
    """data_lat and data_lon are both 180x360 arrays with data_lat[i, :]
    containing the latidude in degrees from -89.5 to 89.5 and data_lon[:, i]
    containing the longitude in degrees from -179.5 to 179.5.
    """

    data_mld_annual = np.empty((12, 180, 360))
    data_pco2_taka_annual = np.empty((12, 180, 360))
    data_sss_annual = np.empty((12, 180, 360))
    data_sst_annual = np.empty((12, 180, 360))
    #data_months_annual = np.empty((12, 180, 360))

    for i in range(12):
        """Takes annual mean of the data ignoring NaNs, therefore the
        data_annual[0, :, :] is for january, data_annual[1, :, :] for
        february, ..."""
        data_mld_annual[i, :, :] = np.nanmean(data_mld[i::12, :, :], axis=0)
        data_pco2_taka_annual[i, :, :] = np.nanmean(data_pco2_taka[i::12, :, :], axis=0)
        data_sss_annual[i, :, :] = np.nanmean(data_sss[i::12, :, :], axis=0)
        data_sst_annual[i, :, :] = np.nanmean(data_sst[i::12, :, :], axis=0)
        
        #data_months_annual[i, :, :] = i+1
        """data_months_annual[0, :, :] is a 180x360 array of 1s for january,
        data_months_annual[1, :, :] is a 180x360 array of 2s for february etc.
        """


    #data_lat_annual = np.tile(data_lat, (12, 1, 1))
    #data_lon_annual = np.tile(data_lon, (12, 1, 1))
    """adding another dimension for data_lat and data_lon to be in the same
    shape as the other data.
    """


    #========3) reshape and rearrange for SOM====
    data_mld_annual_flatten = data_mld_annual.flatten()
    data_pco2_taka_annual_flatten = data_pco2_taka_annual.flatten()
    data_sss_annual_flatten = data_sss_annual.flatten()
    data_sst_annual_flatten = data_sst_annual.flatten()

    nan_index = (np.isnan(data_mld_annual_flatten) 
                 | np.isnan(data_pco2_taka_annual_flatten) 
                 | np.isnan(data_sss_annual_flatten) 
                 | np.isnan(data_sst_annual_flatten))

    som_input = np.array([data_mld_annual_flatten[~nan_index],
                        data_pco2_taka_annual_flatten[~nan_index],
                        data_sss_annual_flatten[~nan_index], data_sst_annual_flatten[~nan_index]]).T

    debug.message(som_input.shape)
    #========5) SOM part to identify biomes====
    net = qsom.SOM(maphight, maplength, som_input.shape[1], n_epoch=epochnr,
                   device=('cuda' if torch.cuda.is_available()
                           else 'cpu'))
    debug.message("started fitting")
    # learning_error = net.fit(som_input)
    # debug.message("training completed")
    # net.save_pickle('som.p')
    # predicted_clusts, errors = net.predict_cluster(som_input)
    # debug.message(predicted_clusts)
    # debug.message(errors)

    predicted_clusts = scipy.io.loadmat('classes.mat', appendmat=False)['classes']
    debug.message(predicted_clusts.shape)
    
    debug.message(predicted_clusts.shape)

    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    
    #plt.show()

    #========6) Smoothing of biomes====

if __name__ == '__main__':
    debug.message("name == main")
    step1()
if __name__ != '__main__':
    #debug.message("name != main")
    pass