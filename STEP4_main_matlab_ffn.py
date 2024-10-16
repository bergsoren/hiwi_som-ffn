"""
Welcome to the simple things: this code uses the NN matlab package and
creates monthly generated basin wide fCO2 maps derrived from the SOM. The
input file specifies the year and month and the lat-lon area. 

Peter Landschutzer 17.01.2012
University of East Anglia, Norwich

Python version by:
Søren Jakob Berger 13.09.2024
Max-Planck-Institut für Meteorologie, Hamburg
"""

#========IMPORTS====
import numpy as np
import scipy.io
import torch
import matplotlib.pyplot as plt
import matplotlib
import cartopy.crs as ccrs
import cartopy

import time

import settings
import debug


def run() -> None:
    """_summary_
    """
    #========1) define basin====
    """_summary_
    """
    ocean_global: bool = settings.STEP2_OCEAN_GLOBAL
    deactivate_testnum: str = settings.STEP2_DEACTIVATE_TESTNUM
    val_check_only: str = settings.STEP2_VAL_CHECK_ONLY
    performance_function: str = settings.STEP2_PERFORMANCE_FUNCTION


    #========2) load the labelling and training data====
    """_summary_
    """
    def clear_dataset(dataset: np.ndarray, latcrop_range: list = [settings.latcropmin, settings.latcropmax], loncrop_range: list = [settings.loncropmin, settings.loncropmax], mediterranean_crop: bool = False, coast_crop: bool = False, arctic_crop: bool = False) -> np.ndarray:
        """_summary_

        Args:
            dataset (np.ndarray): _description_
            latcrop_range (list, optional): _description_. Defaults to [settings.latcropmin, settings.latcropmax].
            loncrop_range (list, optional): _description_. Defaults to [settings.loncropmin, settings.loncropmax].
            mediterranean_crop (bool, optional): _description_. Defaults to False.
            coast_crop (bool, optional): _description_. Defaults to False.
            arctic_crop (bool, optional): _description_. Defaults to False.

        Returns:
            np.ndarray: _description_
        """
        if(latcrop_range != False):
            #empty lists are False
            dataset[:, 3][dataset[:, 3] < latcrop_range[0]] = np.nan
            dataset[:, 3][dataset[:, 3] > latcrop_range[1]] = np.nan
        if(loncrop_range != False):
            #empty lists are False
            dataset[:, 4][dataset[:, 4] < loncrop_range[0]] = np.nan
            dataset[:, 4][dataset[:, 4] > loncrop_range[1]] = np.nan
        if(mediterranean_crop == True):
            dataset[:, 3][dataset[:, 3] < 45] = np.nan
            dataset[:, 3][dataset[:, 3] > 10] = np.nan
            dataset[:, 4][dataset[:, 4] < 45] = np.nan
            dataset[:, 4][dataset[:, 4] > -2] = np.nan
        if(coast_crop == True):
            print('this has not been implimented yet, sorry')
        if(arctic_crop == True):
            #TODO: doppelt gemobbelt?
            dataset[:, 3][dataset[:, 3] > 80] = np.nan

            dataset[:, 3][dataset[:, 3] > 65] = np.nan
            dataset[:, 4][dataset[:, 4] < -90] = np.nan
            dataset[:, 4][dataset[:, 4] > 30] = np.nan

        return dataset
    
    """LData
    data[:, 0] = year, data[:, 1] = month, data[:, 2] = lat, data[:, 3] = lon,
    data[:, 4] = sst, data[:, 5] = mld, data[:, 6] = chl, data[:, 7] = sss, 
    data[:, 8] = atm co2, data[:, 9] = takahashi clim, data[:, 10] = sst anom,
    data[:, 11] = mld anom, data[:, 12] = chl anom, data[:, 13] = sss anom,
    data[:, 14] = atm co2 anom, data[:, 15] = pco2 socat, data[:, 16] = biomes.
    """
    print('-----------------------------------------------------------------')
    print('Loading LData.')
    print('...')
    ldata = np.empty((0,17))
    for year in np.arange(settings.year_min, settings.year_max):#TODO: change back to for year in settings.year_output:
        print(year)
        if(ocean_global == True):
            ldata_path = f'interim_output/Ldata/Ldata_{year}_v2023.mat'
        else:
            ldata_path = f'interim_output/Ldata/Ldata_{year}_coincided_v2.mat'

        ldata_unfiltered = scipy.io.loadmat(ldata_path, appendmat=False)[f'Ldata_{year}_coincided_v2']
        ldata = np.append(ldata, clear_dataset(ldata_unfiltered), axis=0)
    
    print('LData loaded.')
    print('-----------------------------------------------------------------')

    """TData
    data[:, 0] = year, data[:, 1] = month, data[:, 2] = lat, data[:, 3] = lon,
    data[:, 4] = sst, data[:, 5] = mld, data[:, 6] = chl, data[:, 7] = sss, 
    data[:, 8] = atm co2, data[:, 9] = takahashi clim, data[:, 10] = sst anom,
    data[:, 11] = mld anom, data[:, 12] = chl anom, data[:, 13] = sss anom,
    data[:, 14] = atm co2 anom, data[:, 15] = pco2 socat, data[:, 16] = biomes.
    """
    print('-----------------------------------------------------------------')
    print('Loading TData.')
    print('...')
    tdata = np.empty((0,17))
    for year in settings.year_output:
        print(year)
        if(ocean_global == True):
            tdata_path = f'interim_output/Tdata/Tdata_{year}_v2024.mat'
        else:
            tdata_path = f'interim_output/Tdata/Tdata_{year}V2.mat'

        tdata_unfiltered = scipy.io.loadmat(tdata_path, appendmat=False)[f'Tdata_{year}']
        tdata = np.append(tdata, clear_dataset(tdata_unfiltered), axis=0)
    
    print('TData loaded.')
    print('-----------------------------------------------------------------')


    #========3) NAN's of both labelling and training data have to be removed====
    # debug.message(f"{ldata.shape=}, {tdata.shape=}")

    ldata_nan_index: np.ndarray = (np.isnan(ldata[:, 4]) #sst
                 | np.isnan(ldata[:, 5]) #mld
                 | np.isnan(ldata[:, 6]) #chl
                 | np.isnan(ldata[:, 7]) #sss
                 | np.isnan(ldata[:, 8]) #atm co2
                 | np.isnan(ldata[:, 10]) #sst anom
                 | np.isnan(ldata[:, 12]) #chl anom
                 | np.isnan(ldata[:, 13]) #sss aom
                 | np.isnan(ldata[:, 14])) #atm co2 anom
    """ldata_nan_index is True where either one of the data sets has a NaN.
    """
    data_label: np.ndarray = np.array([ldata[:, 4][~ldata_nan_index], #sst
                        ldata[:, 5][~ldata_nan_index], #mld
                        ldata[:, 6][~ldata_nan_index], #chl
                        ldata[:, 7][~ldata_nan_index], #sss
                        ldata[:, 8][~ldata_nan_index], #atm co2
                        ldata[:, 10][~ldata_nan_index], #sst anom
                        ldata[:, 12][~ldata_nan_index], #chl anom
                        ldata[:, 13][~ldata_nan_index], #sss aom
                        ldata[:, 14][~ldata_nan_index]]).T #atm co2 anom
    """data_label is the labelling data sets with removed NaNs.
    """

    tdata_nan_index: np.ndarray = (np.isnan(tdata[:, 4]) #sst
                 | np.isnan(tdata[:, 5]) #mld
                 | np.isnan(tdata[:, 6]) #chl
                 | np.isnan(tdata[:, 7]) #sss
                 | np.isnan(tdata[:, 8]) #atm co2
                 | np.isnan(tdata[:, 10]) #sst anom
                 | np.isnan(tdata[:, 12]) #chl anom
                 | np.isnan(tdata[:, 13]) #sss aom
                 | np.isnan(tdata[:, 14])) #atm co2 anom
    """tdata_nan_index is True where either one of the data sets has a NaN.
    """
    data_train: np.ndarray = np.array([tdata[:, 4][~tdata_nan_index], #sst
                        tdata[:, 5][~tdata_nan_index], #mld
                        tdata[:, 6][~tdata_nan_index], #chl
                        tdata[:, 7][~tdata_nan_index], #sss
                        tdata[:, 8][~tdata_nan_index], #atm co2
                        tdata[:, 10][~tdata_nan_index], #sst anom
                        tdata[:, 12][~tdata_nan_index], #chl anom
                        tdata[:, 13][~tdata_nan_index], #sss aom
                        tdata[:, 14][~tdata_nan_index]]).T #atm co2 anom
    """data_train is the training data sets with removed NaNs.
    """

    data_fCO2 = ldata[:, 15][~ldata_nan_index]

    data_l_month = ldata[:, 1][~ldata_nan_index]
    data_l_year = ldata[:, 0][~ldata_nan_index]
    data_t_month = tdata[:, 1][~tdata_nan_index]
    data_t_year = tdata[:, 0][~tdata_nan_index]

    data_l_lat = ldata[:, 2][~ldata_nan_index]
    data_l_lon = ldata[:, 3][~ldata_nan_index]
    data_t_lat = tdata[:, 2][~tdata_nan_index]
    data_t_lon = tdata[:, 3][~tdata_nan_index]

    data_l_classes = ldata[:, 16][~ldata_nan_index]
    data_t_classes = tdata[:, 16][~tdata_nan_index]

    # debug.message(f"{data_label.shape=}, {data_train.shape=}")
    # debug.message(f"{data_fCO2.shape=}")
    # debug.message(f"{data_l_month.shape=}, {data_l_year.shape=}, {data_l_lat.shape=}, {data_l_lon.shape=}, {data_l_classes.shape=}")
    # debug.message(f"{data_t_month.shape=}, {data_t_year.shape=}, {data_t_lat.shape=}, {data_t_lon.shape=}, {data_t_classes.shape=}")
    
    content = np.unique(data_l_classes)
    content = content[~np.isnan(content)]
    # debug.message(content)
    # debug.message(content.shape)
    

    #========4) Backprop part for every Neuron====
    for biome in content:
        ffn_training_classes = data_t_classes[biome==data_t_classes]
        ffn_training_data = data_train[biome==data_t_classes]
        ffn_training_month = data_t_month[biome==data_t_classes]
        ffn_training_year = data_t_year[biome==data_t_classes]
        ffn_training_lat = data_t_lat[biome==data_t_classes]
        ffn_training_lon = data_t_lon[biome==data_t_classes]

        ffn_labelling_data = data_label[biome==data_l_classes]
        ffn_fCO2 = data_fCO2[biome==data_l_classes]
        
        # batch_size = 100
        # n_iters = 44640 #thats for num_epochs to be 200 as it was in the SOM in the matlab code
        # num_epochs = n_iters / (ffn_training_data.size / batch_size)
        # num_epochs = int(num_epochs)

        # train_loader = torch.utils.data.DataLoader(dataset=ffn_training_data, 
        #                                         batch_size=batch_size, 
        #                                         shuffle=True)

        # test_loader = torch.utils.data.DataLoader(dataset=ffn_labelling_data, 
        #                                         batch_size=batch_size, 
        #                                         shuffle=False)

        # ffn_input_dim = ffn_labelling_data.size
        # ffn_hidden_dim = 512#in matlab 25 is used
        # ffn_output_dim = 1

        # device = ("cuda" if torch.cuda.is_available() else "cpu")
        # print(f"Using {device} device")
        # class FFN(torch.nn.Module):
        #     def __init__(self):
        #         super().__init__()
        #         self.flatten = torch.nn.Flatten()
        #         self.ffn_stack = torch.nn.Sequential(
        #             torch.nn.Linear(ffn_input_dim, ffn_hidden_dim),
        #             torch.nn.ReLU(),
        #             torch.nn.Linear(ffn_hidden_dim, ffn_output_dim)
        #         )

        #     def forward(self, x):
        #         x = self.flatten(x)
        #         logits = self.ffn_stack(x)
        #         return logits
            
        # net = FFN().to(device)
        # debug.message(ffn_labelling_data.shape)
        # net([ffn_labelling_data])
        # criterion = torch.nn.MSELoss()

    #matlab_pco2_sim = scipy.io.loadmat('ffnoutput_pCO2.mat', appendmat=False)['data_all']
    step4_plot_data_pco2 = scipy.io.loadmat('step4_plot_data_pco2.mat', appendmat=False)['step4_plot_data_pco2']
    step4_plot_data_bgcmean = scipy.io.loadmat('step4_plot_data_bgcmean.mat', appendmat=False)['step4_plot_data_bgcmean']

    # lon = settings.data_lon[0]
    # lat = settings.data_lat[:, 0]
    # double_lon = np.empty(360*2)
    # double_lon[0:360] = lon
    # double_lon[360:360*2] = lon+360
    # double_data = np.empty([180, 360*2])
    # double_data[:, 0:360] = step4_plot_data_pco2
    # double_data[:, 360:360*2] = step4_plot_data_pco2
    # debug.message(double_data.shape)

    plt_show = True
    plot_data = [step4_plot_data_pco2, step4_plot_data_bgcmean]
    if(plt_show):
        # pco2 plot
        ax: cartopy.mpl.geoaxes.GeoAxes = plt.axes(projection=ccrs.InterruptedGoodeHomolosine(
            central_longitude=-160, globe=None, emphasis='ocean'))
        ax.coastlines()
        #cmap: matplotlib.colors.ListedColormap = plt.colormaps['viridis'].with_extremes(under='white')
        cmap: matplotlib.colors.ListedColormap = settings.colormap.with_extremes(under='white')
        plot: cartopy.mpl.contour.GeoContourSet = ax.contourf(
            settings.data_lon[0], settings.data_lat[:, 0], plot_data[0], 
            np.arange(250, 400, 1), cmap=cmap, transform=ccrs.PlateCarree())
        plt.colorbar(plot)
        plt.show()

        # bgcmean plot
        ax: cartopy.mpl.geoaxes.GeoAxes = plt.axes(projection=ccrs.InterruptedGoodeHomolosine(
            central_longitude=-160, globe=None, emphasis='ocean'))
        ax.coastlines()
        #cmap: matplotlib.colors.ListedColormap = plt.colormaps['viridis'].with_extremes(under='white')
        cmap: matplotlib.colors.ListedColormap = settings.colormap.with_extremes(under='white')
        plot: cartopy.mpl.contour.GeoContourSet = ax.contourf(
            settings.data_lon[0], settings.data_lat[:, 0], plot_data[1], 
            np.arange(0, 16.1, 1), cmap=cmap, transform=ccrs.PlateCarree())
        plt.colorbar(plot)
        plt.show()

    return plot_data




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
    print('import ' + __name__ + ' as step4')
    print('step4.run()')
    print('-----------------------------------------------------------------')