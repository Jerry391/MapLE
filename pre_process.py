from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDepictor
import os
import pandas as pd



def add_Finger(finger, index, ligand):
    for fig in finger:
        if index.get(fig) is not None:
            index[fig].append(ligand)
        else:
            index[fig] = [ligand]
    return index

def read_AtomPairFinger(file):
    mol = Chem.MolFromMol2File(file, sanitize=True, removeHs=False)
    bi = {}
    fp = AllChem.GetAtomPairFingerprint(mol)
    bi = fp.GetNonzeroElements()
    bits = []
    for i in bi.keys():
        if bi[i] == 1:
            bits.append(i)
    return bits, fp

def read_MorganFinger(file, radius=1):
    mol = Chem.MolFromMol2File(file, sanitize=True, removeHs=False)
    bi = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, nBits=2048, radius=radius, bitInfo=bi)
    bits = []
    for i in bi.keys():
        if len(bi[i]) == 1:
            bits.append(i)

    bi = list(fp.GetOnBits())
    return bits, fp

def read_MACCSFinger(file):
    mol = Chem.MolFromMol2File(file, sanitize=True, removeHs=False)
    bi = {}
    fp = AllChem.GetMACCSKeysFingerprint(mol)
    x = list(fp.GetOnBits())
    bits = []
    for i in bi.keys():
        if len(bi[i]) == 1:
            bits.append(i)
    return x, fp

def pre_process(data_path,res_path):
    rdDepictor.SetPreferCoordGen(True)

    # process AtomPairfinger
    index = {}
    for root, dirs, files in os.walk(data_path):
        for lig in dirs:
            path = os.path.join(data_path, lig, lig + '_ligand.mol2')
            print(path)
            Finger, ap_fps = read_AtomPairFinger(path)
            print(Finger)
            index = add_Finger(Finger, index, lig)
    Result = pd.DataFrame(columns=['Query_ID', 'Dec_ID', 'Rank', 'Score'])

    for idx in index.keys():
        i = 0
        idx_Query = pd.DataFrame(columns=['Query_ID', 'Dec_ID', 'Rank', 'Score'])
        for name in index[idx]:
            lig_path = os.path.join(data_path, name, name + '_ligand.mol2')
            Finger, MACCS_fps = read_AtomPairFinger(lig_path)
            x = float(len(Finger))
            topo_score = 1 / x
            new = pd.DataFrame({'Query_ID': idx, 'Dec_ID': name, 'Rank': i, 'Score': topo_score}, index=[1])
            idx_Query = idx_Query._append(new)
            i += 1
        idx_Query = idx_Query.sort_values(by=['Score'], ascending=False)
        for i in range(0, idx_Query.shape[0]):
            idx_Query.iloc[i, 2] = i + 1
        Result = Result._append(idx_Query)
    print(Result)
    if not os.path.exists(res_path):
        os.makedirs(res_path)
    Result.to_csv(res_path + 'AtomPair' + '.tsv', index=False, sep='\t')

    # process MACCSfinger
    index = {}
    for root, dirs, files in os.walk(data_path):
        for lig in dirs:
            path = os.path.join(data_path, lig, lig + '_ligand.mol2')
            print(path)
            Finger, MACCS_fps = read_MACCSFinger(path)
            print(Finger)
            index = add_Finger(Finger, index, lig)
    Result = pd.DataFrame(columns=['Query_ID', 'Dec_ID', 'Rank', 'Score'])

    for idx in index.keys():
        i = 0
        idx_Query = pd.DataFrame(columns=['Query_ID', 'Dec_ID', 'Rank', 'Score'])
        for name in index[idx]:
            lig_path = os.path.join(data_path, name, name + '_ligand.mol2')
            Finger, MACCS_fps = read_MACCSFinger(lig_path)
            x = float(len(Finger))
            MACCS_score = 1 / x
            new = pd.DataFrame({'Query_ID': idx, 'Dec_ID': name, 'Rank': i, 'Score': MACCS_score}, index=[1])
            idx_Query = idx_Query._append(new)
            i += 1
        idx_Query = idx_Query.sort_values(by=['Score'], ascending=False)
        for i in range(0, idx_Query.shape[0]):
            idx_Query.iloc[i, 2] = i + 1
        Result = Result._append(idx_Query)
    print(Result)
    if not os.path.exists(res_path):
        os.makedirs(res_path)
    Result.to_csv(res_path + 'MACCS' + '.tsv', index=False, sep='\t')

    # process morganfinger
    for radius in [1, 5, 10, 20]:
        index = {}
        for root, dirs, files in os.walk(data_path):
            for lig in dirs:
                path = os.path.join(data_path, lig, lig + '_ligand.mol2')
                print(path)
                Finger, morgan_fps = read_MorganFinger(path, radius)
                print(Finger)
                index = add_Finger(Finger, index, lig)
        Result = pd.DataFrame(columns=['Query_ID', 'Dec_ID', 'Rank', 'Score'])

        for idx in index.keys():
            i = 0
            idx_Query = pd.DataFrame(columns=['Query_ID', 'Dec_ID', 'Rank', 'Score'])
            for name in index[idx]:
                lig_path = os.path.join(data_path, name, name + '_ligand.mol2')
                Finger, morgan_fps = read_MorganFinger(lig_path, radius)
                x = float(len(Finger))
                morgan_score = 1 / x  # math.exp(1 / x)
                new = pd.DataFrame({'Query_ID': idx, 'Dec_ID': name, 'Rank': i, 'Score': morgan_score}, index=[1])
                idx_Query = idx_Query._append(new)
                i += 1
            idx_Query = idx_Query.sort_values(by=['Score'], ascending=False)
            for i in range(0, idx_Query.shape[0]):
                idx_Query.iloc[i, 2] = i + 1
            Result = Result._append(idx_Query)
        print(Result)
        if not os.path.exists(res_path):
            os.makedirs(res_path)
        Result.to_csv(res_path + 'Morgan_' + str(radius) + '.tsv', index=False, sep='\t')

if __name__ == '__main__':

    data_path = './mol_data/pre_process'
    res_path = './result/score_list/'
    pre_process(data_path,res_path)