from rdkit import rdBase, Chem
from rdkit.Chem.Descriptors import NumValenceElectrons
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Scaffolds import MurckoScaffold
# import pubchempy as pcp
from rdkit.Chem import rdFMCS
from matplotlib.colors import ColorConverter
import pandas as pd
from rdkit.Chem.rdMolDescriptors import GetUSRScore, GetUSRCAT
# import nglview

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


'''

fh=open('1.txt')
smis=[]
for line in fh.readlines():
    smi=line.strip()
    smis.append(smi)

# print(smis)
mols3d=[]
for smi in smis:
    # print(smi)
    mol=Chem.MolFromSmiles(smi)
    m2=Chem.AddHs(mol)
    AllChem.EmbedMolecule(m2)
    # maxIters默认是优化200次，有时不会收敛，建议设置成2000
    opt_state=AllChem.MMFFOptimizeMolecule(m2,maxIters=2000)
#     print(opt_state)
    mols3d.append(m2)

# Draw.MolToImage(mols3d[0], size=(450, 150))
img = Draw.MolToImage(mols3d)

img.save("./save_demo.png")

plt.imshow(img)

usrcats = [ GetUSRCAT( mol ) for mol in mols3d ]
for i in range( len( usrcats )):
    for j in range( len( usrcats )):
        score = GetUSRScore( usrcats[ i ], usrcats[ j ] )
        # print(i,j,"usrscroe:",score)

'''

from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AtomPairs import Pairs
from rdkit import RDConfig
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D, Generate
from rdkit.Avalon import pyAvalonTools
import os
if __name__ == '__main__':

    # 创建一个SMILES字符串表示的分子对象
    mol = Chem.MolFromSmiles('CC(=CCC/C(=C/CSC[C@H](NC(=O)C)C(=O)O)/C)C')

    mol = Chem.MolFromSmiles('c1ccccc1CC1CC1')
    rdkinfo = {}
    rdkfp = Chem.RDKFingerprint(mol, bitInfo=rdkinfo)

    # Chem.SanitizeMol(mol)
    bi = {}
    fp = AllChem.GetMorganFingerprint(mol, 2, bitInfo=bi)
    # fp = MACCSkeys.GenMACCSKeys(mol)
    # fp = pyAvalonTools.GetAvalonFP(mol, bitInfo=bi)
    # fp = Generate.Gen2DFingerprint(mol, )
    factory = Gobbi_Pharm2D.factory
    factory.skipFeats = ['Donor', 'Acceptor', 'NegIonizable', 'PosIonizable', 'ZnBinder', 'Aromatic', 'Hydrophobe', 'LumpedHydrophobe']
    factory.Init()
    fdef = AllChem.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))
    print(fdef.GetNumFeatureDefs())

    # calc 3d p4 fp
    fp = Generate.Gen2DFingerprint(mol, factory, bitInfo=bi)

    num = factory.GetSigSize()
    # x = fp.GetNonzeroElements()
    # x = list(fp.GetOnBits())
    # print(bi)
    bits = []
    # fp = AllChem.GetAtomPairFingerprint(mol)
    # bi = fp.GetNonzeroElements()
    for i in bi.keys():
        if len(bi[i]) == 2:
            bits.append(i)

    print(rdkinfo)