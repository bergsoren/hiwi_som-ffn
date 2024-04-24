"""
This file is for debugging and development functions, which will not be in the final project.
"""

debug = True

def message(debugmessage):
    """a version of print(), which can be turned off and on in a central location for debugging

    Args:
        debugmessage (String): The debug message to be printed in the console.
    """
    if debug:
        print('~~~~DEBUG~~~~', end='\n\n')
        print(debugmessage, end='\n\n')
        print('~~~~~~~~~~~~~')
