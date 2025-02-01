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
    L = int(re.findall(r'\d+', fname)[-1]) / 100
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
    colNames = [
        "type", 'step', 'Elec_l', 'Elec_ldl', 'vdW_l', 'vdW_ldl', 'dE', 'dE_avg', 'Temp', 'dG',
        'FromLambda', "ToLambda"
    ]

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
    with open(filename) as file:
        log = file.read()

    global_conf = GlobalConfig(log)

    cv_lines = re.findall(r'\n(colvars:.*)', log)
    colvars, biases, TI_traj = parse_cv_lines(global_conf, cv_lines)

    return global_conf, colvars, biases, TI_traj


class CVConfig(dict):
    def __init__(self):
        self['children'] = list()
        self.level = 0

class GlobalConfig(CVConfig):
    def __init__(self, log):
        self.get_colvars_version(log)
        self.get_output_prefix(log)
        self.get_cv_traj_file(log)

    def get_output_prefix(self, log):
        match = re.search(r'\ncolvars: The final output state file will be "(.+).colvars.state".\n',
                        log)
        self['output_prefix'] = match.group(1).strip()

    def get_colvars_version(self, log):
        match = re.search(r'\ncolvars: Initializing the collective variables module, version (.*).\n',
                        log)
        self['version'] = match.group(1).strip()

    def get_cv_traj_file(self, log):
        cv_traj_file = re.compile(r'\ncolvars: Synchronizing \(emptying the buffer of\) trajectory file "(.+)"\.\n')
        self['traj_file'] = cv_traj_file.search(log).group(1).strip()

class CVLine():
    grammar = {'new_config': re.compile(r'^colvars:\s+Reading new configuration:'),
                'new_CV': re.compile(r'^colvars:\s+Initializing a new collective variable\.'),
                'new_bias': re.compile(r'^colvars:\s+Initializing a new "(.*)" instance\.$'),
                'new_component': re.compile(r'^colvars:(\s+)Initializing a new "(.*)" component\.$'),
                'new_atom_group': re.compile(r'^colvars:(\s+)Initializing atom group "(.*)"\.$'),
                'new_key_value': re.compile(r'^colvars:\s+#\s+(\w+) = (.*?)\s*(?:\[default\])?$'),
                'new_RFEP_stage': re.compile(r'^colvars:\s+Restraint (\S+), stage (\S+) : lambda = (\S+), k = (\S+)$'),
                'end_of_RFEP_stage': re.compile(r'^colvars:\s+Restraint (\S+) Lambda= (\S+) dA/dLambda= (\S+)$'),
                }
    def __init__(self, line):
        for name, regex in self.grammar.items():
            matched = regex.match(line)
            self.__setattr__(name, matched)



def parse_cv_lines(global_conf, cv_lines):
    biases = list()
    TI_traj = {}
    colvars = list()
    for line in cv_lines:
        cv_line = CVLine(line)

        if cv_line.new_config:
            # QUESTION: start_cv_config doesn't actually depend on new_config line being found. Should we have a different check?
            level, current = start_cv_config(global_conf)
        elif cv_line.new_CV:
            level, current = create_cv(colvars)
        elif cv_line.new_bias:
            level, current = add_bias(biases, cv_line.new_bias)
        elif cv_line.new_key_value:
            current = add_new_key_value_pair(current, cv_line.new_key_value)
        elif cv_line.new_RFEP_stage:
            name, stage, L, k = parse_RFEP_stage(cv_line.new_RFEP_stage)
            if not name in TI_traj:
                TI_traj = start_new_RFEP(TI_traj, name, stage, L, k)
            else:
                TI_traj = continue_RFEP(TI_traj, name, stage, L, k)
        elif cv_line.end_of_RFEP_stage:
            TI_traj = terminate_RFEP_stage(TI_traj, line, cv_line.end_of_RFEP_stage)
        elif cv_line.new_component or cv_line.new_atom_group:
            if cv_line.new_component:
                prev_level, level, key = add_new_component(level, cv_line.new_component)
            elif cv_line.new_atom_group:
                prev_level, level, key = add_new_atom_group(level, cv_line.new_atom_group)

            if level > prev_level:
                parent = current
            elif level < prev_level:
                parent = parent['parent']
            parent['children'].append({})
            current = add_child(key, parent)
    return colvars, biases, TI_traj


def add_child(key, parent):
    current = parent['children'][-1]
    current['key'] = key
    current['children'] = list()
    return current


def add_new_atom_group(level, new_atom_group):
    prev_level = level
    level = (len(new_atom_group.group(1)) - 1) // 2
    key = new_atom_group.group(2).strip()
    return prev_level, level, key


def add_new_component(level, new_component):
    prev_level = level
    level = (len(new_component.group(1)) - 1) // 2
    if level == 1:  # Top-level CVCs are not indented, fix manually
        level = 2
    key = new_component.group(2).strip()
    return prev_level, level, key


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
    current = CVConfig()
    colvars.append(current)
    return level, current


def start_cv_config(global_conf):
    level = 0
    current = global_conf
    return level, current
