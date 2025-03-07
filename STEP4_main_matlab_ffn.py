"""
Welcome to the simple things: this code uses the NN matlab package and
creates monthly generated basin wide fCO2 maps derrived from the SOM. The
input file specifies the year and month and the lat-lon area. 

Peter Landschutzer 17.01.2012
University of East Anglia, Norwich

Python version by:
Søren Jakob Berger 13.09.2024
Max-Planck-Institut für Meteorologie, Hamburg

with code by and help for the NN from
Maurie Keppens 05.12.2024
Vlaams Instituut voor de Zee, Oostende
"""

#========IMPORTS====
import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
import matplotlib
import cartopy.crs as ccrs
import cartopy
import sklearn.model_selection

import tensorflow as tf

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
    # label -> training/validation
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
    # train -> generate maps at end
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
        print(f'\nBiome {biome}-------------------------------------------')
        ffn_training_classes = data_t_classes[biome==data_t_classes]
        debug.message(ffn_training_classes.shape)
        ffn_training_data = data_train[biome==data_t_classes]
        ffn_training_month = data_t_month[biome==data_t_classes]
        ffn_training_year = data_t_year[biome==data_t_classes]
        ffn_training_lat = data_t_lat[biome==data_t_classes]
        ffn_training_lon = data_t_lon[biome==data_t_classes]

        ffn_labelling_data = data_label[biome==data_l_classes]
        ffn_fCO2 = data_fCO2[biome==data_l_classes]
        
        tf.random.set_seed(1)
        #tf.debugging.set_log_device_placement(True)
        """https://www.youtube.com/watch?v=YAJ5XBwlN4o
        0) Prepare data
        1) Design model (input, output size, forward pass)
        2) Construct loss and optimizer
        3) Training loop
            - forward pass: compute prediction and loss
            - backward pass: gradients
            - update weights
        """
        #0) Prepare data
        data_ffn_training = ffn_labelling_data.astype(np.float32)
        data_ffn_pco2 = ffn_fCO2.astype(np.float32)

        data_ffn_estimation = ffn_training_data.astype(np.float32)

        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(data_ffn_training, data_ffn_pco2, test_size=0.2)

        # ffn_x_mean = ffn_x.mean(dim=0)
        # ffn_x_std = ffn_x.std(dim=0)
        # ffn_x = ffn_x - ffn_x_mean
        # ffn_x = ffn_x / ffn_x_std

        # ffn_y = ffn_y.view(ffn_y.shape[0], 1)

        #1) Design model (input, output size, forward pass)
        ffn_n_samples = X_train.shape[0]
        ffn_n_features = X_train.shape[1]
        ffn_hidden_dim = 60#in matlab 25 is used
        ffn_output_dim = 1

        ffn_learning_rate = 0.001
        ffn_num_epochs = 200

        ffn_normalizer = tf.keras.layers.Normalization(axis=-1)
        ffn_normalizer.adapt(X_train)

        def build_and_compile_model():
            model = tf.keras.Sequential([
                ffn_normalizer,
                tf.keras.layers.Dense(ffn_hidden_dim, activation='relu'),
                tf.keras.layers.Dense(ffn_output_dim, activation= 'linear')
            ])
            #2) Construct loss and optimizer
            model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(learning_rate=ffn_learning_rate), metrics=['R2Score'])
            return model
            
        ffn = build_and_compile_model()
        ffn.summary()


        ffn_callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0,
            patience=6,
            verbose=0,
            mode="auto",
            baseline=None,
            restore_best_weights=True,
            start_from_epoch=0
        )

        #3) Training loop
        print('\n\nTraining started')
        start_time = time.time()

        ffn_history = ffn.fit(
            X_train,
            y_train,
            validation_split=0.2,
            verbose=2,
            callbacks = [ffn_callback],
            epochs=ffn_num_epochs
        )
        print(f'Training completed, {time.time()-start_time}s passed.\n\n')

        def plot_loss(history):
            plt.plot(history.history['loss'], label='Training')
            plt.plot(history.history['val_loss'], label='Internal validation')
            #plt.ylim([25, 40])
            plt.xlabel('Epoch')
            plt.ylabel('Mean Squared Error')
            plt.legend()
            plt.grid(True)
            plt.show()
        plot_loss(ffn_history)

        # debug.message(f'{ffn.metrics_names=}')

        ffn_test_results = {}
        ffn_loss, ffn_rsquared = ffn.evaluate(X_test, y_test, verbose=2)
        ffn_test_results['FFN'] = {
            'Mean Squared error': ffn_loss,
            'R2': ffn_rsquared
        }
        
        results_df = pd.DataFrame(ffn_test_results).T
        test_predictions = ffn.predict(X_test)

        # debug.message(f'{results_df=}')
        # debug.message(f'{test_predictions=}')

        axs = plt.axes(aspect='equal')
        plt.scatter(y_test, test_predictions)
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        lims = [0,500]
        plt.xlim(lims)
        plt.ylim(lims)
        plt.plot(lims, lims)
        plt.show()

        error = y_test - test_predictions
        plt.hist(error, bins=25)
        plt.xlabel('Error = Target - Outputs')
        plt.ylabel('Count')
        plt.show()


        pco2_estimate = ffn.predict(data_ffn_estimation)
        pco2_estimate[pco2_estimate<0] = np.nan

        debug.message(pco2_estimate.shape)
        debug.message(ffn_training_classes.shape)

        #TODO Dropout check how many percent is used for backward?

        def vec_to_array2_from_orgmatlabcode(vec, lon, lat, LON, LAT):
            arr = np.zeros((len(lat), len(lon)))

            for a in range(len(vec)):
                i = np.where(LAT[a] == lat)[0]
                j = np.where(LON[a] == lon)[0]
                if i.size > 0 and j.size > 0:
                    arr[i[0], j[0]] = vec[a]
    
            arr[arr == 0] = np.nan
            return arr

        final_pco2 = np.empty()
        final_biomes = np.empty()
        for year in settings.year_output:

            pco2_oneyear = np.empty(12)
            biomes_oneyear = np.empty(12)
            for month in range(11):

                debug.message(np.array(np.where((ffn_training_year == year) & (ffn_training_month == month))).shape)
                pco2 = pco2_estimate[np.where((ffn_training_year == year) & (ffn_training_month == month))]
                biomes = ffn_training_classes[np.where((ffn_training_year == year) & (ffn_training_month == month))]

                pco2 = vec_to_array2_from_orgmatlabcode(pco2, settings.data_lon[0], settings.data_lat[:, 0], ffn_training_lon, ffn_training_lat)
                biomes = vec_to_array2_from_orgmatlabcode(biomes, settings.data_lon[0], settings.data_lat[:, 0], ffn_training_lon, ffn_training_lat)

                pco2_oneyear[month] = pco2
                biomes_oneyear[month] = biomes
                pass


            final_pco2 = np.append(final_pco2, pco2_oneyear, axis=0)
            final_biomes = np.append(final_biomes, biomes_oneyear, axis=0)

        print(f'Biome {biome} finished----------------------------------\n\n')


    
    debug.message(1/0)
    nan_index = 0
    data_pco2: np.ndarray = np.full((516, 180, 360), np.nan)
    data_pco2: np.ndarray = data_pco2.flatten()
    data_pco2[np.logical_not(nan_index)] = final_pco2
    data_pco2: np.ndarray = data_pco2.reshape((516, 180, 360))

    data_biomes: np.ndarray = np.full((516, 180, 360), np.nan)
    data_biomes: np.ndarray = data_biomes.flatten()
    data_biomes[np.logical_not(nan_index)] = final_biomes
    data_biomes: np.ndarray = data_biomes.reshape((516, 180, 360))

    # step4_ffn_output = ffn(ffn_x).detach().numpy()
    # matlab_pco2_sim = scipy.io.loadmat('ffnoutput_pCO2.mat', appendmat=False)['data_all']
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
        cmap: matplotlib.colors.ListedColormap = plt.colormaps['viridis'].with_extremes(under='white')
        #cmap: matplotlib.colors.ListedColormap = settings.colormap.with_extremes(under='white')
        plot: cartopy.mpl.contour.GeoContourSet = ax.contourf(
            settings.data_lon[0], settings.data_lat[:, 0], plot_data[0], 
            np.arange(250, 400, 1), cmap=cmap, transform=ccrs.PlateCarree())
        plt.colorbar(plot)
        plt.show()

        # bgcmean plot
        ax: cartopy.mpl.geoaxes.GeoAxes = plt.axes(projection=ccrs.InterruptedGoodeHomolosine(
            central_longitude=-160, globe=None, emphasis='ocean'))
        ax.coastlines()
        cmap: matplotlib.colors.ListedColormap = plt.colormaps['viridis'].with_extremes(under='white')
        #cmap: matplotlib.colors.ListedColormap = settings.colormap.with_extremes(under='white')
        plot: cartopy.mpl.contour.GeoContourSet = ax.contourf(
            settings.data_lon[0], settings.data_lat[:, 0], plot_data[1], 
            np.arange(0.1, 16.1, 1), cmap=cmap, transform=ccrs.PlateCarree())
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