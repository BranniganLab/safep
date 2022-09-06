import warnings
import logging
import copy
from AFEP_parse import *
import matplotlib as mpl

def checkPaths(paths, nDone):
    goodpaths = []
    for path in paths:
        feps = glob(path+pattern)
        countDone = 0
        for fep in feps:
            with open(fep) as f:
                if 'Free' in f.read():
                    countDone+=1
        if countDone == nDone:
            goodpaths.append(path)
    
    return goodpaths


def processLeg(paths, RT, decorrelate, pattern, temperature, detectEQ, lambdas):
    u_nks, cumulative, perWindow, affix = batchProcess(paths, RT, decorrelate, pattern, temperature, detectEQ)
    
    reps = set(perWindow.columns.get_level_values(0))
    
    #Do convergence
    convergence = {}
    for l in reps:
        forward, forward_error, backward, backward_error = doConvergence(u_nks[l], lambdas)
        convergence[l] = {}
        convergence[l]['forward'] = forward
        convergence[l]['forward_error'] = forward_error
        convergence[l]['backward'] = backward
        convergence[l]['backward_error'] = backward_error
        
    dfs = {x:pd.DataFrame.from_dict(convergence[x]) for x in convergence}
    df_converge = pd.concat(dfs, axis=1)
    
    #Get means
    perWindow[('mean', 'df')] = np.mean(perWindow.loc[:, (slice(None), 'df')], axis=1)
    perWindow[('mean', 'ddf')] = np.mean(perWindow.loc[:, (slice(None), 'ddf')], axis=1)
    perWindow[('mean', 'dG_f')] = np.mean(perWindow.loc[:, (slice(None), 'dG_f')], axis=1)
    perWindow[('mean', 'dG_b')] = np.mean(perWindow.loc[:, (slice(None), 'dG_b')], axis=1)
    
    #do hysteresis

    for key in set(perWindow.columns.get_level_values(0)):
        perWindow[(key, 'diff')] = perWindow[(key, 'dG_f')]+perWindow[(key, 'dG_b')]

    #keys = set(cumulative.columns.get_level_values(0))
    colors = ['#0072B2', '#D55E00', '#CC79A7', '#009E73', '#E69F00', '#56B4E9', '#004455', '#ABCDFE', '#331155', '#654321']
    keyColors = {}
    i = 0
    for key in reps:
        keyColors[key] = colors[i]
        i += 1

    
    return keyColors, cumulative, perWindow, df_converge, affix


def processAllLegs(root, prefixes, pattern, lambdas, temperature, RT, decorrelate, detectEQ, checkReplicas=False, dry=False):
    meta_cumulative = {}
    meta_perWindow = {}
    meta_keyColors = {}
    meta_convergence = {}
    nDone = len(lambdas)
    
    for prefix in prefixes:
        paths = glob(root+prefix+'*/')
        if checkReplicas==True:
            paths = checkPaths(paths, nDone)
        if len(paths)==0:
            print(f"WARNING: Skipping {root+prefix}")
        elif not dry:
            print(paths)
            keyColors, cumulative, perWindow, convergence, affix = processLeg(paths, RT, decorrelate, pattern, temperature, detectEQ, lambdas)
            
            meta_keyColors[prefix] = keyColors
            meta_cumulative[prefix] = cumulative
            meta_perWindow[prefix] = perWindow
            meta_convergence[prefix] = convergence
            
    df_cumulative = pd.concat(meta_cumulative,axis=1)
    df_perWindow = pd.concat(meta_perWindow, axis=1)
    df_keyColors = pd.DataFrame.from_dict(meta_keyColors).unstack().dropna()
    df_convergence = pd.concat(meta_convergence, axis=1)
    
    df_perWindow.index=np.round(df_perWindow.index, 5)

    return {'cumulatives':df_cumulative, 'perWins':df_perWindow, 'affix':affix, 'keyColors':df_keyColors, 'convergence':df_convergence}


def saveSystem(system, path):
    system['cumulatives'].to_csv(path+"cumulatives.csv")
    system['perWins'].to_csv(path+"perWindow.csv")
    with open(path+"metadata.txt", 'w') as fs:
        fs.write(system['affix'])
    system['keyColors'].to_csv(path+"keyColors.csv")
    system['convergence'].to_csv(path+"convergence.csv")

    
def loadSystem(path):
    system = {}
    system['cumulatives']= pd.read_csv(path+"cumulatives.csv", index_col=0, header=[0,1,2])
    system['perWins']= pd.read_csv(path+"perWindow.csv", index_col=0, header=[0,1,2])
    with open(path+"metadata.txt") as fs:
        system['affix'] = fs.read()
    system['keyColors']= pd.read_csv(path+"keyColors.csv", index_col=[0,1])['0']
    system['convergence']=pd.read_csv(path+"convergence.csv", index_col=0, header=[0,1,2])
    
    return system


## unit test for reading and writing systems
# for key in tmpSys.keys():
#     for j in tmpSys[key].keys():
#         print(f"({key},{j})")
#         unitA = tmpSys[key][j]
#         unitB = allSys[key][j]
#         try:
#             roundA = np.round(unitA, 12)
#             roundB = np.round(unitB, 12)
#             cfAB = np.all(roundA == roundB) 
#             cfAA = np.all(roundA == roundA)
#             cfBB = np.all(roundB == roundB)
            
#             print(cfAB or (not cfAA and not cfBB))
#         except:
#             print(np.all(unitA == unitB))