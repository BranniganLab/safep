# Import block
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from scipy.stats import linregress as lr
from scipy.stats import norm
from scipy.special import erfc
from scipy.optimize import curve_fit as scipyFit
from scipy.stats import skew

import pandas as pd

from alchemlyb.visualisation.dF_state import plot_dF_state
from alchemlyb.parsing import namd
from alchemlyb.estimators import BAR
from alchemlyb.visualisation.dF_state import plot_dF_state
from alchemlyb.visualisation import plot_convergence

import re
from tqdm import tqdm #for progress bars
from natsort import natsorted #for sorting "naturally" instead of alphabetically
from glob import glob #file regexes

from .helpers import *



#Guess lambda based on file name (last number in the filename divided by 100)
def guessLambda(fname):
    L = int(re.findall(r'\d+', fname)[-1])/100
    return L
    
def saveUNK(u_nk, filepath):
    u_nk.to_csv(filepath)
     
def readUNK(filepath):
    u_nk = pd.read_csv(filepath)
    u_nk['fep-lambda'] = u_nk['fep-lambda'].astype(str)
    u_nk = u_nk.set_index(['time', 'fep-lambda'])
    
    return u_nk.copy()
    
#redFEPOUT reads each file in a single pass: keeping track of lambda values and appending each line to an array. 
#The array is cast to a dataframe at the end to avoid appending to a dataframe
def readFEPOUT(fileName, step=1):
    colNames = ["type",'step', 'Elec_l', 'Elec_ldl', 'vdW_l', 'vdW_ldl', 'dE', 'dE_avg', 'Temp', 'dG', 'FromLambda', "ToLambda"]

    data = []

    L = np.nan
    L2 = np.nan
    LIDWS = np.nan
    
    frame = 0
    with open(fileName) as fs:
        for line in fs:
            if line[0] == '#':
                frame = 0
                #print(line)
                Lambda = re.search('LAMBDA SET TO (\d+(\.\d+)*)', line)
                Lambda2 = re.search('LAMBDA2 (\d+(\.\d+)*)', line)
                LambdaIDWS = re.search('LAMBDA_IDWS (\d+(\.\d+)*)', line)
                if Lambda:
                    L = Lambda.group(1)
                    #print(f'L={L}')
                if Lambda2:
                    L2 = Lambda2.group(1)
                    #print(f'L2={L2}')
                if LambdaIDWS:
                    LIDWS = LambdaIDWS.group(1)
                    #print(f'LIDWS={LIDWS}')
            elif frame % step <= 1:
                if np.isnan(L):
                    print("WARNING: lambda is not defined!")
                    L = guessLambda(fileName)
                    print("Guessing lambda to be {L} based on file name.")


                lineList = line.split()
                lineList.append(L)
                if lineList[0] == "FepEnergy:":
                    lineList.append(L2)
                elif lineList[0] == "FepE_back:":
                    lineList.append(LIDWS)
                else:
                    print(f'Unexpected line start: {lineList[0]}')
                    return 0
                data.append(lineList)
                frame = frame + 1
            else:
                frame = frame + 1

            stashL = L
            stashL2 = L2
            stashLIDWS = LIDWS

    fs.close()
    
    df = pd.DataFrame(data).dropna()
    df.columns = colNames
    df = df.iloc[:,1:].astype(float)
    df["window"]=np.mean([df.FromLambda,df.ToLambda], axis=0)
    df["up"]=df.ToLambda>df.FromLambda

    df = df.sort_index()
    return df
    
def readFiles(files, step=1):
    fileList = []
    for file in files:
        df = readFEPOUT(file, step)
        fileList.append(df)
    data = pd.concat(fileList)
    
    data.index = data.window
    data["dVdW"] = data.vdW_ldl - data.vdW_l
    
    return data