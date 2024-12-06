# Import block
import numpy as np
import pandas as pd

import re

from .helpers import *


def guess_lambda(fname):
    '''
    Guess lambda based on file name (last number in the filename divided by 100). Not very good.
    Arguments: file name
    Returns: estimated lambda value
    '''
    L = int(re.findall(r'\d+', fname)[-1])/100
    return L

from pathlib import Path
def save_UNK(u_nk, filepath, safety=True):
    '''
    Write u_nk to a file
    Arguments: u_nk in the format of alchemlyb, filepath
    '''
    fpath = Path(filepath)
    if safety and fpath.is_file():
        print(f'File {filepath} already exists and safety is on. Doing nothing.')
    else:
        print(f'Writing to {filepath}')
        u_nk.to_csv(filepath)

    return
     
def read_UNK(filepath):
    '''
    Read a u_nk that was written by saveUNK.
    Arguments: filepath
    Returns: u_nk
    '''
    u_nk = pd.read_csv(filepath, index_col=[0,1], dtype=float)
    u_nk.columns = [float(x) for x in u_nk.columns]
    
    return u_nk.copy()
    

def read_FEPOUT(fileName, step=1):
    '''
    Reads all data from a NAMD fepout file unlike alchemlyb's parser.
    readFEPOUT reads each file in a single pass: keeping track of lambda values and appending each line to an array. 
    The array is cast to a dataframe at the end to avoid appending to a dataframe.
    Arguments: fileName, step (stride)
    Returns: a dataframe containing all the data in a fepout file.
    '''
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
                Lambda = re.search(r'LAMBDA SET TO (\d+(\.\d+)*)', line)
                Lambda2 = re.search(r'LAMBDA2 (\d+(\.\d+)*)', line)
                LambdaIDWS = re.search(r'LAMBDA_IDWS (\d+(\.\d+)*)', line)
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
                    L = guess_lambda(fileName)
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
    
def read_Files(files, step=1):
    '''
    Batch readFEPOUT
    Arguments: file (paths), step (stride)
    Returns: a list of dataframes, one dataframe per file
    '''
    fileList = []
    for file in files:
        df = read_FEPOUT(file, step)
        fileList.append(df)
    data = pd.concat(fileList)
    
    data.index = data.window
    data["dVdW"] = data.vdW_ldl - data.vdW_l
    
    return data


def parse_Colvars_log(filename):
    '''
    Parses Colvars standard output from a log file

    Returns dictionaries containing key-value pairs
    global_conf: top-level Colvars parameters
    colvars: list of dicts, one for each colvar
    biases: list of dicts, one for each bias

    Note: within colvars, parameters of sub-objects CVCs and atom groups are found as nested dictionaries
    in a list under the 'children' key, e.g.:
    colvars[0]['children'][0] is the first CVC of the first colvar
    colvars[0]['children'][0]['children'][0] is the first atom group of that CVC
    '''
    global_conf = {}
    global_conf['parent'] = global_conf # top-level object is own parent
    level = prev_level = 0

    colvars = list()
    biases = list()
    current = global_conf

    with open(filename) as file:
        for line in file:
            if re.match(r'^colvars:\s+Reading new configuration:', line):
                level = 0
                current = global_conf
                continue

            if re.match(r'^colvars:\s+Initializing a new collective variable.', line): # colvar
                level = 1
                current = {}
                current['children'] = list()
                colvars.append(current)
                print(f'New colvar at level {level}')
                continue

            match = re.match(r'^colvars:\s+Initializing a new "(.*)" instance.$', line) # Bias
            if match:
                key = match.group(1).strip()
                level = 1
                print(f'New bias {key} at level {level}')
                current = {}
                current['key'] = key
                biases.append(current)
                continue

            new_child = False
            match = re.match(r'^colvars:(\s+)Initializing a new "(.*)" component.$', line) # component
            if match:
                prev_level = level
                level = (len(match.group(1))-1) // 2
                if level == 1: # Top-level CVCs are not indented, fix manually
                    level = 2
                key = match.group(2).strip()
                print(f'New cvc {key} at level {level}')
                new_child = True

            match = re.match(r'^colvars:(\s+)Initializing atom group "(.*)".$', line) # atom group
            if match:
                prev_level = level
                level = (len(match.group(1))-1) // 2
                key = match.group(2).strip()
                print(f'New atom group {key} at level {level}')
                new_child = True

            if new_child: # Common to new CVCs and atom groups
                if level > prev_level:
                    parent = current
                elif level < prev_level:
                    parent = parent['parent']
                parent['children'].append({})
                current = parent['children'][-1]
                current['key'] = key
                current['children'] = list()
                continue

            match = re.match(r'^colvars:\s+#\s+(\w+) = (.*?)\s*(?:\[default\])?$', line) # key-value pair
            if match:
                key = match.group(1)
                value = match.group(2).strip(' "')  # Extract key and value, remove extra spaces
                current[key] = value  # Add to dictionary
                continue

    return global_conf, colvars, biases