"""
Welcome to the simple things: this code uses MiniSom
(https://github.com/JustGlowing/minisom) and PyTorch
(https://github.com/pytorch/pytorch), and creates monthly generated basin wide
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
import minisom
import matplotlib.pyplot as plt
import matplotlib
import cartopy.crs as ccrs
import cartopy

import time
import warnings

import settings
import debug


def run(som_epochnr: int = settings.INPUT_METHOD_SOM_OR_BIOME['epochnr'],
        som_sigma: float = 2.0, som_learning_rate: float = 0.5,
        som_neighborhood_function: str = 'gaussian',
        plt_show: bool = True) -> list:
    """_summary_ TODO

    Args:
        som_epochnr (int, optional): _description_. Defaults to settings.INPUT_METHOD_SOM_OR_BIOME['epochnr'].
        som_sigma (float, optional): _description_. Defaults to 2.0.
        som_learning_rate (float, optional): _description_. Defaults to 0.5.
        som_neighborhood_function (str, optional): _description_. Defaults to 'gaussian'.
        plt_show (bool, optional): _description_. Defaults to True.

    Returns:
        list: _description_
    """
    #========1) load cluster data====
    """Loading the data needed for clustering. The data is in the format [months
    latitude longitude] = 480x180x360.
    """
    data_mld: np.ndarray = settings.data_mld
    data_pco2_taka: np.ndarray = settings.data_pco2_taka
    data_sss: np.ndarray = settings.data_sss
    data_sst: np.ndarray = settings.data_sst


    #========2) take 20-year average====
    timevec: np.ndarray = np.arange(settings.STEP1_TWENTYYEARAVERAGE_TIMEVEC_MIN,
                        settings.STEP1_TWENTYYEARAVERAGE_TIMEVEC_MAX, 1/12)
    """timevec is the time dimension (first dimension of the data, e.g. months)
    in years.
    """
    data_lat: np.ndarray = settings.data_lat
    data_lon: np.ndarray = settings.data_lon
    """data_lat and data_lon are both 180x360 arrays with data_lat[i, :]
    containing the latidude in degrees from -89.5 to 89.5 and data_lon[:, i]
    containing the longitude in degrees from -179.5 to 179.5.
    """

    data_mld_annual: np.ndarray = np.empty((12, 180, 360))
    data_pco2_taka_annual: np.ndarray = np.empty((12, 180, 360))
    data_sss_annual: np.ndarray = np.empty((12, 180, 360))
    data_sst_annual: np.ndarray = np.empty((12, 180, 360))
    """Initialize empty arrays to be filled in the following for loop.
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for i in range(12):
            """Takes annual mean of the data ignoring NaNs, therefore the
            data_annual[0, :, :] is for january, data_annual[1, :, :] for
            february, ...
            """
            data_mld_annual[i, :, :] = np.nanmean(data_mld[i::12, :, :], axis=0)
            data_pco2_taka_annual[i, :, :] = np.nanmean(data_pco2_taka[i::12, :, :], axis=0)
            data_sss_annual[i, :, :] = np.nanmean(data_sss[i::12, :, :], axis=0)
            data_sst_annual[i, :, :] = np.nanmean(data_sst[i::12, :, :], axis=0)
            

    #========3) reshape and rearrange for SOM====
    data_mld_annual_flatten: np.ndarray = data_mld_annual.flatten()
    data_pco2_taka_annual_flatten: np.ndarray = data_pco2_taka_annual.flatten()
    data_sss_annual_flatten: np.ndarray = data_sss_annual.flatten()
    data_sst_annual_flatten: np.ndarray = data_sst_annual.flatten()
    """Flattening the data to just have one dimension for the training of the
    SOM.
    """

    nan_index: np.ndarray = (np.isnan(data_mld_annual_flatten) 
                 | np.isnan(data_pco2_taka_annual_flatten) 
                 | np.isnan(data_sss_annual_flatten) 
                 | np.isnan(data_sst_annual_flatten))
    """nan_index is True where either one of the data sets has a NaN.
    """

    som_input: np.ndarray = np.array([data_mld_annual_flatten[~nan_index],
                        data_pco2_taka_annual_flatten[~nan_index],
                        data_sss_annual_flatten[~nan_index],
                        data_sst_annual_flatten[~nan_index]]).T
    """som_input is the flattened data sets with removed NaNs.
    """


    #========5) SOM part to identify biomes====
    t0: float = time.time()
    """Starting time to time SOM.
    """
    print('-----------------------------------------------------------------')
    print('SOM training started with ' + str(som_epochnr) + ' total epochs.')
    print('...')

    som: minisom.MiniSom = minisom.MiniSom(settings.maplength, settings.maphight, som_input.shape[1],
                          sigma=som_sigma, learning_rate=som_learning_rate,
                          neighborhood_function=som_neighborhood_function, random_seed=0)
    print(f'{som_sigma=}, {som_learning_rate=}, {som_neighborhood_function=}')
    som.train_random(som_input, som_epochnr)
    """Creating and training the SOM.
    """

    predicted_clusts: np.ndarray = np.array([som.winner(x) for x in som_input])
    predicted_clusts: np.ndarray = predicted_clusts[:, 0] * settings.maplength + predicted_clusts[:, 1]
    """TODO: Documentation.
    """

    t1: float = time.time()
    total_time: float = t1-t0
    print('...')
    print('SOM training ended with a total time of ' + str(total_time) + ' seconds.')
    print('-----------------------------------------------------------------')
    """Second step of timing the SOM.
    """
    
    # predicted_clusts = scipy.io.loadmat('classes.mat', appendmat=False)['classes'].squeeze()
    # debug.message(predicted_clusts.shape)
    # debug.message(predicted_clusts)

    #========6) Smoothing of biomes====
    biomes: np.ndarray = np.full((12, 180, 360), np.nan)
    biomes: np.ndarray = biomes.flatten()
    biomes[np.logical_not(nan_index)] = predicted_clusts
    biomes: np.ndarray = biomes.reshape((12, 180, 360))
    """Creates a 12x180x360 array of NaNs, flattens them to one dimension
    as it was done with the data sets, adds the cluster data (which is the
    same shape/size as the data with removed NaNs) and the reshapes the array
    to be the original size of 12x180x360. The result is an array with the
    size of the original data sets, but with the cluster number instead of the
    data. NaNs in the original data sets stay NaNs.
    """

    #========7) save and plot 3-D biomes====
    biomes: np.ndarray = biomes[0]
    """Plot just january at the moment. TODO.
    """
    # biomes = scipy.io.loadmat('array_test.mat', appendmat=False)['array_test'].squeeze()

    if(plt_show):
        ax: cartopy.mpl.geoaxes.GeoAxes = plt.axes(projection=ccrs.PlateCarree())
        ax.coastlines()
        #cmap: matplotlib.colors.ListedColormap = plt.colormaps['viridis'].with_extremes(under='white')
        cmap: matplotlib.colors.ListedColormap = settings.colormap.with_extremes(under='white')
        plot: cartopy.mpl.contour.GeoContourSet = ax.contourf(data_lon[0], data_lat[:, 0], biomes, np.arange(0, 16.1, 1), cmap=cmap)
        plt.colorbar(plot)
        plt.show()
    """Plot the 16 clusters with different colours, white shows the NaNs, therefore the land.
    """

    
    return [data_lon[0], data_lat[:, 0], biomes]

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
    print('import ' + __name__ + ' as step1')
    print('step1.run(som_epochnr, som_sigma,', end='')
    print(' som_learning_rate, som_neighborhood_function, plt_show)')
    print('\nwhere som_epochnr is the number of total epochs,')
    print('som_sigma is the initial/maximum radius,')
    print('som_learning_rate is the learning rate,')
    print('som_neighborhood_function is one of "gaussian",', end='')
    print(' "mexican_hat", "bubble" or "triangle" and')
    print('plt_show is a boolean flag for showing the plot instead of just', end='')
    print(' returning data in format of [lon, lat, biomes].')
    print('-----------------------------------------------------------------')