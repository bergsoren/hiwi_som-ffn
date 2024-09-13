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