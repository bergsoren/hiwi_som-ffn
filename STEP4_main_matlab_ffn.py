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
    
    """LData"""
    print('-----------------------------------------------------------------')
    print('Loading LData.')
    print('...')
    ldata = []
    for year in settings.year_output:
        print(year)
        if(ocean_global == True):
            ldata_path = f'interim_output/Ldata/Ldata_{year}_v2023.mat'
        else:
            ldata_path = f'interim_output/Ldata/Ldata_{year}_coincided_v2.mat'

    ldata_unfiltered = scipy.io.loadmat(ldata_path, appendmat=False)[f'Ldata_{year}_coincided_v2']
    ldata.append(clear_dataset(ldata_unfiltered))
    
    print('LData loaded.')
    print('-----------------------------------------------------------------')

    """TData"""
    print('-----------------------------------------------------------------')
    print('Loading TData.')
    print('...')
    tdata = []
    for year in settings.year_output:
        print(year)
        if(ocean_global == True):
            tdata_path = f'interim_output/Tdata/Tdata_{year}_v2024.mat'
        else:
            tdata_path = f'interim_output/Tdata/Tdata_{year}V2.mat'

    tdata_unfiltered = scipy.io.loadmat(tdata_path, appendmat=False)[f'Tdata_{year}']
    tdata.append(clear_dataset(tdata_unfiltered))
    
    print('TData loaded.')
    print('-----------------------------------------------------------------')


    #========3) NAN's of both labelling and training data have to be removed====
    



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