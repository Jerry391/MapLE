from rdkit import rdBase, Chem
from rdkit.Chem.Descriptors import NumValenceElectrons
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem, Draw
import pandas as pd
import os, math
def read_MorganFinger(file, radius = 1):
    mol = Chem.MolFromMol2File(file, sanitize=True, removeHs=False)
    # Chem.SanitizeMol(mol)
    bi = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, nBits=256, radius=radius, bitInfo=bi)
    # print(bi)
    bits = []
    for i in bi.keys():
        # print(len(bi[i]))
        if len(bi[i]) == 1:
            bits.append(i)
    return bits, fp
    # for bit in bits:
    #     mfp2_svg = Draw.DrawMorganBit(mol, bit, bi)
    #     imgs.append(mfp2_svg)


if __name__ == "__main__":

    data_path = './data'
    res = pd.DataFrame(columns=['Query_ID', 'Dec_ID', 'Rank', 'Score'])
    query_id = './test3/data/3u8k_ligand.mol2'
    mol = Chem.MolFromMol2File(query_id)
    access = 0
    tmp = pd.DataFrame(columns=['Query_ID', 'Dec_ID', 'Rank', 'Score'])
    for root, dirs, files in os.walk(data_path):
        for lig in dirs:
            path = os.path.join(data_path, lig, lig + '_ligand.mol2')
            fp1 = read_MorganFinger(query_id)
            fp2 = read_MorganFinger(path)
            # res.loc[res['Dec_ID'] == lig]['Score'].values[0] += CalSimilarity(fp1[0], fp2[0])
            score = 1
            new = pd.DataFrame({'Query_ID': 0, 'Dec_ID': lig, 'Rank': 1, 'Score': score}, index=[1])
            tmp = tmp._append(new)
            # new = pd.DataFrame({'Query_ID': 0, 'Dec_ID': lig, 'Rank': 1, 'Score': score / 6}, index=[1])
    tmp = tmp.sort_values(by=['Score'], ascending=False)
    for i in range(0, tmp.shape[0]):
        # self.result.iloc[i]['Rank'] = i + 1
        tmp.iloc[i, 2] = i + 1
        name = tmp.iloc[i, 1]
        res.iloc[i, 3] = math.exp(-res.loc[res['Dec_ID'] == name]['Rank'].values[0]) + math.exp(-i)
    # Morgan
