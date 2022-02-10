#Large datasets can be difficult to parse on a workstation due to inefficiencies in the way data is represented for pymbar. When possible, reduce the size of your dataset.
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import linregress as lr
from scipy.stats import norm

from glob import glob #file regexes
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm #for progress bars
import re #regex
from natsort import natsorted #for sorting "naturally" instead of alphabetically

from alchemlyb.visualisation.dF_state import plot_dF_state

from alchemlyb.parsing import namd
from alchemlyb.estimators import BAR
from alchemlyb.visualisation.dF_state import plot_dF_state
from alchemlyb.visualisation import plot_convergence

import re

#Guess lambda based on file name (last number in the filename divided by 100)
def guessLambda(fname):
    L = int(re.findall(r'\d+', fname)[-1])/100
    return L

#redFEPOUT uses reads each file in a single pass: keeping track of lambda values and appending each line to an array. 
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


# In[5]:


def readFiles(files, step=1):
    fileList = []
    for file in files:
        df = readFEPOUT(file, step)
        fileList.append(df)
    data = pd.concat(fileList)
    
    data.index = data.window
    data["dVdW"] = data.vdW_ldl - data.vdW_l
    
    return data


# In[13]:

def u_nk_fromDF(data, temperature, eqTime, warnings=True):
    from scipy.constants import R, calorie
    beta = 1/(R/(1000*calorie) * temperature) #So that the final result is in kcal/mol
    u_nk = pd.pivot_table(data, index=["step", "FromLambda"], columns="ToLambda", values="dE")
    #u_nk = u_nk.sort_index(level=0).sort_index(axis='columns') #sort the data so it can be interpreted by the BAR estimator
    u_nk = u_nk*beta
    u_nk.index.names=['time', 'fep-lambda']
    u_nk.columns.names = ['']
    u_nk = u_nk.loc[u_nk.index.get_level_values('time')>=eqTime]

    
    #Shift and align values to be consistent with alchemlyb standards
    lambdas = list(set(u_nk.index.get_level_values(1)).union(set(u_nk.columns)))
    lambdas.sort()
    warns = set([])
            
    for L in lambdas:
        try:
            u_nk.loc[(slice(None), L), L] = 0
        except:
            if warnings:
                warns.add(L)
    
    prev = lambdas[0]
    for L in lambdas[1:]:
        try:
            u_nk.loc[(slice(None), L), prev] = u_nk.loc[(slice(None), L), prev].shift(1)
        except:
            if warnings:
                warns.add(L)
            
        prev = L
    
    if len(warns)>0:
        print(f"Warning: lambdas={warns} not found in indices/columns")
    u_nk = u_nk.dropna(thresh=2)
    u_nk = u_nk.sort_index(level=1).sort_index(axis='columns') #sort the data so it can be interpreted by the BAR estimator
    return u_nk


# In[7]:


def get_dG(u_nk):
    #the data frame is organized from index level 1 (fep-lambda) TO column
    #dG will be FROM column TO index
    groups = u_nk.groupby(level=1)
    dG=pd.DataFrame([]) 
    for name, group in groups:
        dG[name] = np.log(np.mean(np.exp(-1*group)))
        dG = dG.copy() # this is actually faster than having a fragmented dataframe
        
    return dG
    
    
def plot_cumsum(f, errors, l, units):
    plt.errorbar(l, f, yerr=errors, marker='.')
    plt.xlabel('lambda')
    plt.ylabel(f'DeltaG(lambda) ({units})')
    return plt.gca()


def plot_per_window(df, l_mid, units):
    plt.errorbar(l_mid, df, yerr=ddf, marker='.')
    plt.xlabel('lambda')
    plt.ylabel(f'Delta G per window ({units})')
    return plt.gca()

def convergence_plot(u_nk, states, tau=1):
    grouped = u_nk.groupby('fep-lambda')
    data_list = [grouped.get_group(s) for s in states]

    forward = []
    forward_error = []
    backward = []
    backward_error = []
    num_points = 10
    for i in range(1, num_points+1):
        # forward
        partial = pd.concat([data[:int(len(data)/num_points*i)] for data in data_list])
        estimate = BAR().fit(partial)
        forward.append(estimate.delta_f_.iloc[0,-1])
        # For BAR, the error estimates are off-diagonal
        ddf = [estimate.d_delta_f_.iloc[i+1,i] * np.sqrt(tau) for i in range(len(states)-1)]
        error = np.sqrt((np.array(ddf)**2).sum())
        forward_error.append(error)

        # backward
        partial = pd.concat([data[-int(len(data)/num_points*i):] for data in data_list])
        estimate = BAR().fit(partial)
        backward.append(estimate.delta_f_.iloc[0,-1])
        ddf = [estimate.d_delta_f_.iloc[i+1,i] * np.sqrt(tau) for i in range(len(states)-1)]
        error = np.sqrt((np.array(ddf)**2).sum())
        backward_error.append(error)

    ax = plot_convergence(forward, forward_error, backward, backward_error)

    return plt.gca()

def get_MBAR(bar):
    
    # Extract data for plotting
    states = bar.states_

    f = bar.delta_f_.iloc[0,:] # dataframe
    l = np.array([float(s) for s in states])
    # lambda midpoints for each window
    l_mid = 0.5*(l[1:] + l[:-1])

    # FE differences are off diagonal
    df = np.diag(bar.delta_f_, k=1)
    

    # error estimates are off diagonal
    ddf = np.array([bar.d_delta_f_.iloc[i, i+1] for i in range(len(states)-1)])

    # Accumulate errors as sum of squares
    errors = np.array([np.sqrt((ddf[:i]**2).sum()) for i in range(len(states))])
    
    
    return l, l_mid, f, df, ddf, errors

def get_EXP(u_nk):
    #the data frame is organized from index level 1 (fep-lambda) TO column
    #dG will be FROM column TO index
    groups = u_nk.groupby(level=1)
    dG=pd.DataFrame([])
    for name, group in groups:
        dG[name] = np.log(np.mean(np.exp(-1*group)))

    dG_f=np.diag(dG, k=1)
    dG_b=np.diag(dG, k=-1)

    l=dG.columns.to_list()
    l_mid = np.mean([l[1:],l[:-1]], axis=0)

    return l, l_mid, dG_f, dG_b

def fb_discrepancy_plot(l_mid, dG_f, dG_b):
    plt.vlines(l_mid, np.zeros(len(l_mid)), dG_f + np.array(dG_b), label="fwd - bwd", linewidth=3)
    plt.legend()
    plt.title('Fwd-bwd discrepancies by lambda')
    plt.xlabel('Lambda')
    plt.ylabel('Diff. in delta-G')
    #Return the figure
    return plt.gca() 

def fb_discrepancy_hist(dG_f, dG_b):
    plt.hist(dG_f + np.array(dG_b));
    plt.title('Distribution of fwd-bwd discrepancies')
    plt.xlabel('Difference in delta-G')
    plt.ylabel('Count')
    #return the figure
    return plt.gca()


#Light-weight exponential estimator
def get_dG_fromData(data, temperature):
    from scipy.constants import R, calorie
    beta = 1/(R/(1000*calorie) * temperature) #So that the final result is in kcal/mol
    
    groups = data.groupby(level=0)
    dG=[]
    for name, group in groups:
        isUp = group.up
        dE = group.dE
        toAppend = [name, -1*np.log(np.mean(np.exp(-beta*dE[isUp]))), 1]
        dG.append(toAppend)
        toAppend=[name, -1*np.log(np.mean(np.exp(-beta*dE[~isUp]))), 0]
        dG.append(toAppend)
    
    dG = pd.DataFrame(dG, columns=["window", "dG", "up"])
    dG = dG.set_index('window')
    
    dG_f = dG.loc[dG.up==1] 
    dG_b = dG.loc[dG.up==0]

    dG_f = dG_f.dG.dropna()
    dG_b = dG_b.dG.dropna()

    return dG_f, dG_b


# Additional utility functions:
# Estimate the probability density distribution from the moving slope of a CDF. i.e. using the values X and their cumulative density FX
def getMovingAveSlope(X,FX,window):
    slopes = []
    Xwindowed = sliding_window_view(X, window)
    FXwindowed = sliding_window_view(FX, window)
   
    for i in np.arange(len(Xwindowed)):
        Xwindow = Xwindowed[i]
        FXwindow = FXwindowed[i]
        result = lr(Xwindow, FXwindow)
        m = result.slope
        slopes.append(m)
    return slopes

# Calculate the coefficient of determination:
def GetRsq(X, Y, Yexpected):
    residuals = Y-Yexpected
    SSres = np.sum(residuals**2)
    SStot = np.sum((X-np.mean(X))**2)
    R = 1-SSres/SStot
    R

if __name__ == '__main__':
    fepoutFiles = glob('*.fep*')
    temperature = 300
    RT = 0.00198720650096 * temperature

    plt.rcParams['figure.dpi'] = 150

    u_nk = namd.extract_u_nk(fepoutFiles, temperature);
    u_nk = u_nk.sort_index(level=1).sort_index(axis='columns') #sort the data so it can be interpreted by the BAR estimator

    from alchemlyb.preprocessing import subsampling

    method = 'dhdl_all' #'dE' also works, 'dhdl' (the default) fails
    affix = f'decorrelated_{method}'
    #affix = 'unprocessed'

    groups = u_nk.groupby('fep-lambda')
    decorr = pd.DataFrame([])
    # Decorrelate each lambda individually
    for key, group in groups:
        test = subsampling.decorrelate_u_nk(group, method)
        decorr = decorr.append(test)

    u_nk = decorr.copy()
    
    bar = BAR()
    bar.fit(u_nk)

    plot_cumsum(bar)
    plot_per_window(bar)
    convergence_plot(u_nk)
    fb_discrepancy_plots(u_nk)


