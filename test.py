from rdkit import rdBase, Chem
from rdkit.Chem.Descriptors import NumValenceElectrons
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem, Draw
from rdkit import DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
# import pubchempy as pcp
from rdkit.Chem import rdFMCS
from matplotlib.colors import ColorConverter
import pandas as pd
from rdkit.Chem.rdMolDescriptors import GetUSRScore, GetUSRCAT
# import nglview
from rdkit.Chem import rdDepictor
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import sys
import pandas as pd
import time
from ThresholdAlgorithm import ThresholdAlgorithm
import torch
import pandas as pd
import numpy as np
import shutil, os
import rapids
import math
def CalAccuracy(fil1, file2, k):
    df1 = pd.read_csv(fil1, sep='\t', usecols=['Query_ID', 'Dec_ID', 'Rank', 'Score'])
    df2 = pd.read_csv(file2, sep='\t', usecols=['Query_ID', 'Dec_ID', 'Rank', 'Score'])
    ans = 0
    num = 0
    for id in range(0, len(df1)-1, k):
        a1 = df1.loc[id:id+k-1]
        a2 = df2.loc[id:id+k-1]
        num += k
        for i in range(0, k):
            if a1.iloc[i]['Dec_ID'] in a2['Dec_ID'].values:
                ans +=1
    print("k: ", k, "\t ans: ", ans, "\t num:", num)
    return ans/num

def func(x, k):
    # return k*1/(1 + math.exp(-1/x))*(math.exp(-x)+1/x)
    return k*math.exp(-1/x)
if __name__ == '__main__':
    # print(func(8,2))
    # print(func(3,1))
    #casf - 2016 处理数据
    data_path = './data'
    i=0
    rdDepictor.SetPreferCoordGen(True)
    for root, dirs, files in os.walk(data_path):
        for lig in dirs:
            source = os.path.join(data_path, lig, lig + '_ligand.mol2')
            target = os.path.join(r'./test3/data', lig + '_ligand.mol2')
            i+=1
            shutil.copy(source, target)

    print(i)
