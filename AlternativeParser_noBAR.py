#!/usr/bin/env python
# coding: utf-8

# User Settings:

# In[1]:


temperature = 300


# Imports

# In[2]:


from glob import glob #file regexes

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm #for progress bars
import re #regex
from natsort import natsorted #for sorting "naturally" instead of alphabetically


# In[3]:


#Don't work right yet
#from alchemlyb.estimators import BAR 
#from alchemlyb.visualisation.dF_state import plot_dF_state
#from alchemlyb.visualisation import plot_convergence


# Function Delcarations:

# In[4]:


#redFEPOUT uses reads each file in a single pass: keeping track of lambda values and appending each line to an array. 
#The array is cast to a dataframe at the end to avoid appending to a dataframe
def readFEPOUT(fileName, step=1):
    colNames = ["type",'step', 'Elec_l', 'Elec_ldl', 'vdW_l', 'vdW_ldl', 'dE', 'dE_avg', 'Temp', 'dG', 'FromLambda', "ToLambda"]

    data = []

    L = np.nan
    L2 = np.nan
    LIDWS = np.nan
    
    frame = 0
    with open(fileName) as downFile:
        for line in downFile:
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

    downFile.close()
    
    df = pd.DataFrame(data).dropna()
    df.columns = colNames
    df = df.iloc[:,1:].astype(float)
    df["window"]=np.mean([df.FromLambda,df.ToLambda], axis=0)
    df["up"]=df.ToLambda>df.FromLambda
   
    df = df.sort_index()
    return df


# In[5]:


def readFiles(files):
    fileList = []
    for file in tqdm(files):
        df = readFEPOUT(file, 50)
        fileList.append(df)
    data = pd.concat(fileList)
    
    data.index = data.window
    data["dVdW"] = data.vdW_ldl - data.vdW_l
    
    return data


# In[13]:


def u_nk_fromDF(data):
    u_nk = pd.pivot_table(data, index=["step", "FromLambda"], columns="ToLambda", values="dE")
    u_nk = u_nk.sort_index(level=0).sort_index(axis='columns') #sort the data so it can be interpreted by the BAR estimator
    #u_nk = u_nk.sort_index(level=1).sort_index(axis='columns') #sort the data so it can be interpreted by the BAR estimator
    
    return u_nk


# In[7]:


def get_dG(u_nk):
    #the data frame is organized from index level 1 (fep-lambda) TO column
    #dG will be FROM column TO index
    groups = u_nk.groupby(level=1)
    dG=pd.DataFrame([]) 
    equil = 50000
    for name, group in groups:
        group[group.index.get_level_values(0)>equil]
        dG[name] = np.log(np.mean(np.exp(-1*group)))
        dG = dG.copy() # this is actually faster than having a fragmented dataframe
        
    return dG


# Read files

# In[8]:


files = glob("*.fepout")
files = natsorted(files)


# In[9]:


data = readFiles(files)


# In[14]:


u_nk = u_nk_fromDF(data)


# In[15]:


dG = get_dG(u_nk)
dG_f=np.diag(dG, k=1)
dG_b=np.diag(dG, k=-1)

l=dG.columns.to_list()
l_mid = np.mean([l[1:],l[:-1]], axis=0)


# In[16]:


plt.vlines(l_mid, np.zeros(len(l_mid)), dG_f + np.array(dG_b), label="fwd - bwd", linewidth=3)

plt.legend()
plt.title('Fwd-bwd discrepencies by lambda')
plt.xlabel('Lambda')
plt.ylabel('Diff. in delta-G')
plt.savefig("figure.svg")


# In[18]:


print(f'The rough estimate for total dG (forward windows only) is: {np.sum(dG_f[~np.isnan(dG_f)])}. The backward estimate is {-np.sum(dG_b[~np.isnan(dG_b)])}')


# In[19]:


#split into forward and backward values for each window
eqTime = 10000
backward = data.dE[~(data.up) * data.step>eqTime].sort_index()*(-1)
forward = data.dE[data.up * data.step>eqTime].sort_index()
print(f'eqTime: {eqTime}\n backward: {backward.mean()}, forward: {forward.mean()}') 


# In[20]:


completeWindows = np.sort(list(set(backward.index) & set(forward.index))) #those windows which have both forward and backward data


# In[21]:


import seaborn as sns


# Plot dE for EACH complete window (may take several minutes)

# In[22]:


for i in completeWindows:
    
    # creating a figure composed of two matplotlib.Axes objects (ax_box and ax_hist)
    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})

    # assigning a graph to each ax
    tempDat = [forward.loc[i], backward.loc[i]]
    
    ax_box.boxplot(tempDat, vert=False)
    ax_box.set_yticklabels(["forward", "backward"])
    plt.title(f'[{np.round(i-0.004,3)} {np.round(i+0.004, 3)}]')
    # Remove x axis name for the boxplot
    ax_box.set(xlabel='')

    sns.histplot(backward.loc[i], bins=50, label="backward", ax=ax_hist);
    sns.histplot(forward.loc[i], bins=50, label="forward", ax=ax_hist, color="orange");
    
    plt.legend()
    plt.show()
    #plt.savefig(f'./diagnosticPlots/dE_SmallerWindows{np.round(i,3)}.svg')
    plt.clf()
    plt.close()


# In[ ]:




