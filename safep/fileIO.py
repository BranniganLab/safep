# Import block
from pathlib import Path
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
    u_nk = pd.read_csv(filepath, index_col=[0, 1], dtype=float)
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
    colNames = ["type", 'step', 'Elec_l', 'Elec_ldl', 'vdW_l',
                'vdW_ldl', 'dE', 'dE_avg', 'Temp', 'dG', 'FromLambda', "ToLambda"]

    data = []

    L = np.nan
    L2 = np.nan
    LIDWS = np.nan

    frame = 0
    with open(fileName) as fs:
        for line in fs:
            if line[0] == '#':
                frame = 0
                # print(line)
                Lambda = re.search(r'LAMBDA SET TO (\d+(\.\d+)*)', line)
                Lambda2 = re.search(r'LAMBDA2 (\d+(\.\d+)*)', line)
                LambdaIDWS = re.search(r'LAMBDA_IDWS (\d+(\.\d+)*)', line)
                if Lambda:
                    L = Lambda.group(1)
                    # print(f'L={L}')
                if Lambda2:
                    L2 = Lambda2.group(1)
                    # print(f'L2={L2}')
                if LambdaIDWS:
                    LIDWS = LambdaIDWS.group(1)
                    # print(f'LIDWS={LIDWS}')
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
    df = df.iloc[:, 1:].astype(float)
    df["window"] = np.mean([df.FromLambda, df.ToLambda], axis=0)
    df["up"] = df.ToLambda > df.FromLambda

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
    TI_traj dict(dict(list(numerical))): dA/dLambda and stage information

    Note: within colvars, parameters of sub-objects (components and atom groups) are found as nested dictionaries
    in a list under the 'children' key, e.g.:
    colvars[0]['children'][0] is the first component of the first colvar
    colvars[0]['children'][0]['children'][0] is the first atom group of that component
    '''
    global_conf = {}
    colvars = list()
    biases = list()

    TI_traj = {}

    with open(filename) as file:
        lines = file.readlines()

    # Header: get version and output prefix, then break
    for line in lines:
        match = re.match(
            r'^colvars: Initializing the collective variables module, version (.*).$', line)
        if match:
            global_conf['version'] = match.group(1).strip()
            break

    for line in lines:
        match = re.match(
            r'^colvars: The final output state file will be "(.+).colvars.state".$', line)
        if match:
            global_conf['output_prefix'] = match.group(1).strip()
            break

    cv_lines = [l for l in lines if line.startswith('colvars:')]

    # Parse rest of file for more config data
    TI_traj = parse_cv_lines(global_conf, colvars, biases, TI_traj, cv_lines)

    return global_conf, colvars, biases, TI_traj


def parse_cv_lines(global_conf, colvars, biases, TI_traj, cv_lines):
    for line in cv_lines:
        new_config = re.match(r'^colvars:\s+Reading new configuration:', line)
        new_CV = re.match(
            r'^colvars:\s+Initializing a new collective variable\.', line)
        new_bias = re.match(
            r'^colvars:\s+Initializing a new "(.*)" instance\.$', line)
        new_child = False
        new_component = re.match(
            r'^colvars:(\s+)Initializing a new "(.*)" component\.$', line)
        new_atom_group = re.match(
            r'^colvars:(\s+)Initializing atom group "(.*)"\.$', line)
        new_key_value = re.match(
            r'^colvars:\s+#\s+(\w+) = (.*?)\s*(?:\[default\])?$', line)
        new_RFEP_stage = re.match(
            r'^colvars:\s+Restraint (\S+), stage (\S+) : lambda = (\S+), k = (\S+)$', line)
        end_of_RFEP_stage = re.match(
            r'^colvars:\s+Restraint (\S+) Lambda= (\S+) dA/dLambda= (\S+)$', line)
        cv_traj_file = re.match(
            r'^colvars: Synchronizing \(emptying the buffer of\) trajectory file "(.+)"\.$', line)

        if new_config:
            level, current = start_cv_config(global_conf)
        elif new_CV:
            level, current = create_cv(colvars)
        elif new_bias:
            level, current = add_bias(biases, new_bias)
        elif new_key_value:
            current = add_new_key_value_pair(current, new_key_value)
        elif new_RFEP_stage:
            name, stage, L, k = parse_RFEP_stage(new_RFEP_stage)
            if not name in TI_traj:
                TI_traj = start_new_RFEP(TI_traj, name, stage, L, k)
            else:
                TI_traj = continue_RFEP(TI_traj, name, stage, L, k)
        elif end_of_RFEP_stage:
            TI_traj = terminate_RFEP_stage(TI_traj, line, end_of_RFEP_stage)
        elif cv_traj_file:
            global_conf['traj_file'] = cv_traj_file.group(1).strip()
        elif new_component:
            prev_level, level, new_child, key = add_new_component(
                level, new_component)
        elif new_atom_group:
            prev_level, level, new_child, key = add_new_atom_group(
                level, new_atom_group)

        if new_child:  # Common to new CVCs and atom groups
            if level > prev_level:
                parent = current
            elif level < prev_level:
                parent = parent['parent']
            parent['children'].append({})
            current = add_child(key, parent)
    return TI_traj


def add_child(key, parent):
    current = parent['children'][-1]
    current['key'] = key
    current['children'] = list()
    return current


def add_new_atom_group(level, new_atom_group):
    prev_level = level
    level = (len(new_atom_group.group(1))-1) // 2
    key = new_atom_group.group(2).strip()
    new_child = True
    return prev_level, level, new_child, key


def add_new_component(level, new_component):
    prev_level = level
    level = (len(new_component.group(1))-1) // 2
    if level == 1:  # Top-level CVCs are not indented, fix manually
        level = 2
    key = new_component.group(2).strip()
    new_child = True
    return prev_level, level, new_child, key


def terminate_RFEP_stage(TI_traj, line, end_of_RFEP_stage):
    # Parse free energy derivative estimates - end of stage: add dAdL value
    name = end_of_RFEP_stage.group(1).strip()
    L = float(end_of_RFEP_stage.group(2).strip())
    dAdL = float(end_of_RFEP_stage.group(3).strip())
    if TI_traj[name]['L'][-1] != L:
        bad_lambda_msg = f'Error: mismatched lambda value in log: expected lambda = {L} and read:\n{line}'
        raise RuntimeError(bad_lambda_msg)
    TI_traj[name]['dAdL'][-1] = dAdL

    return TI_traj


def continue_RFEP(TI_traj, name, stage, L, k):
    TI_traj[name]['stage'].append(stage)
    TI_traj[name]['L'].append(L)
    TI_traj[name]['k'].append(k)
    # NaN to be replaced by actual value if present
    TI_traj[name]['dAdL'].append(np.nan)
    return TI_traj


def start_new_RFEP(TI_traj, name, stage, L, k):
    TI_traj[name] = {'stage': [stage], 'L': [L], 'k': [k], 'dAdL': [None]}
    return TI_traj


def parse_RFEP_stage(new_RFEP_stage):
    # Parse free energy derivative estimates - beginning of stage
    name = new_RFEP_stage.group(1).strip()
    stage = int(new_RFEP_stage.group(2).strip())
    L = float(new_RFEP_stage.group(3).strip())
    k = float(new_RFEP_stage.group(4).strip())
    return name, stage, L, k


def add_new_key_value_pair(current, new_key_value):
    key = new_key_value.group(1)
    # Extract key and value, remove extra spaces
    value = new_key_value.group(2).strip(' "')
    # Add to dictionary
    current[key] = value

    return current


def add_bias(biases, new_bias):
    key = new_bias.group(1).strip()
    level = 1
    current = {}
    current['key'] = key
    biases.append(current)
    return level, current


def create_cv(colvars):
    level = 1
    current = {}
    current['children'] = list()
    colvars.append(current)
    return level, current


def start_cv_config(global_conf):
    level = 0
    current = global_conf
    return level, current
