"""
safep

Tools for Analyzing and Debugging (SA)FEP calculations
"""

__version__ = "0.1.0"
__author__ = 'Jérôme Hénin, Ezry Santiago, Tom Joseph, Grace Brannigan'
__credits__ = 'Rutgers University - Camden'
__all__=['processing', 'plotting', 'fileIO', 'estimators', 'helpers']

from .processing import *
from .plotting import *
from .fileIO import *
from .estimators import *

