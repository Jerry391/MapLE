from rdkit import rdBase, Chem
from rdkit.Chem.Descriptors import NumValenceElectrons
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem, Draw
from rdkit import DataStructs
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D, Generate
from rdkit.Chem.Pharm2D.SigFactory import SigFactory
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
import rapids
import math
import operator
from rdkit import RDConfig
from e3fp.fingerprint.generate import fp, fprints_dict_from_mol
from e3fp.conformer.generate import generate_conformers
from pmapper.pharmacophore import Pharmacophore as P

def add_Finger(finger, index, ligand):
    for fig in finger:
        if index.get(fig) is not None:
            index[fig].append(ligand)
        else:
            index[fig] = [ligand]
    return index

def read_3DFinger(file):
    mol = Chem.MolFromMol2File(file, sanitize=True, removeHs=False)
    AllChem.EmbedMolecule(mol)  # gen 3d
    # Chem.SanitizeMol(mol)
    factory = Gobbi_Pharm2D.factory
    # calc 3d p4 fp
    fp = Generate.Gen2DFingerprint(mol, factory, dMat=Chem.Get3DDistanceMatrix(mol))
    bi = []
    # fp = AllChem.GetAtomPairFingerprint(mol)
    bi = list(fp.GetOnBits())

    # bits = []
    # for i in bi.keys():
    #     if bi[i] == 1:
    #         bits.append(i)
    return bi, fp

def read_E3FPFinger(file):
    mol = Chem.MolFromMol2File(file, sanitize=True, removeHs=False)
    AllChem.EmbedMolecule(mol)  # gen 3d
    # Chem.SanitizeMol(mol)
    fp = fprints_dict_from_mol(mol)[5][0]
    binfp = fp.fold().to_rdkit()
    # calc 3d p4 fp
    bi = []
    # fp = AllChem.GetAtomPairFingerprint(mol)
    bi = list(binfp.GetOnBits())
    # bits = []
    # for i in bi.keys():
    #     if bi[i] == 1:
    #         bits.append(i)
    return bi, fp

def read_p1Finger(file):
    mol = Chem.MolFromMol2File(file, sanitize=True, removeHs=False)
    AllChem.EmbedMolecule(mol)  # gen 3d
    # Chem.SanitizeMol(mol)
    p = P()
    p.load_from_mol(mol)
    sig = p.get_signature_md5(tol=5)
    # fp = fprints_dict_from_mol(mol)[5][0]
    # binfp = fp.fold().to_rdkit()
    # calc 3d p4 fp
    bi = [sig]
    # fp = AllChem.GetAtomPairFingerprint(mol)
    # bi = list(binfp.GetOnBits())
    # bits = []
    # for i in bi.keys():
    #     if bi[i] == 1:
    #         bits.append(i)
    return bi, fp

def output(top_k, output_file):
    with open(output_file, 'a+') as f:
        top_k.to_csv(f, index=False, header=False, sep='\t')

def pre_process2(data_path, res_path):
    rdDepictor.SetPreferCoordGen(True)

    # process 3DParam
    index = {}
    for root, dirs, files in os.walk(data_path):
        for decoy in files:
            path = os.path.join(data_path, decoy)
            print(path)
            Finger, d3_fps = read_3DFinger(path)
            print(Finger)
            index = add_Finger(Finger, index, decoy.split(".")[0])
    Result = pd.DataFrame(columns=['Query_ID', 'Dec_ID', 'Rank', 'Score'])

    for idx in index.keys():
        i = 0
        idx_Query = pd.DataFrame(columns=['Query_ID', 'Dec_ID', 'Rank', 'Score'])
        for name in index[idx]:
            lig_path = os.path.join(data_path, name + '.mol2')
            Finger, d3_fps = read_3DFinger(lig_path)
            x = float(len(Finger))
            topo_score = 1 / x #math.exp(1 / x)
            new = pd.DataFrame({'Query_ID': idx, 'Dec_ID': name, 'Rank': i, 'Score': topo_score}, index=[1])
            idx_Query = idx_Query._append(new)
            i += 1
        idx_Query = idx_Query.sort_values(by=['Score'], ascending=False)
        for i in range(0, idx_Query.shape[0]):
            # self.result.iloc[i]['Rank'] = i + 1
            idx_Query.iloc[i, 2] = i + 1
        Result = Result._append(idx_Query)
    print(Result)

    Result.to_csv(res_path + 'Pharm3D' + '.tsv', index=False, sep='\t')

    # /process MACCSfinger
    index = {}
    for root, dirs, files in os.walk(data_path):
        for decoy in files:
            path = os.path.join(data_path, decoy)
            print(path)
            Finger, E3FP_fps = read_E3FPFinger(path)
            print(Finger)
            index = add_Finger(Finger, index, decoy.split(".")[0])
    Result = pd.DataFrame(columns=['Query_ID', 'Dec_ID', 'Rank', 'Score'])

    for idx in index.keys():
        i = 0
        idx_Query = pd.DataFrame(columns=['Query_ID', 'Dec_ID', 'Rank', 'Score'])
        for name in index[idx]:
            lig_path = os.path.join(data_path, name + '.mol2')
            Finger, E3FP_fps = read_E3FPFinger(lig_path)
            x = float(len(Finger))
            MACCS_score =  1 / x #math.exp(1 / x)
            new = pd.DataFrame({'Query_ID': idx, 'Dec_ID': name, 'Rank': i, 'Score': MACCS_score}, index=[1])
            idx_Query = idx_Query._append(new)
            i += 1
        idx_Query = idx_Query.sort_values(by=['Score'], ascending=False)
        for i in range(0, idx_Query.shape[0]):
            # self.result.iloc[i]['Rank'] = i + 1
            idx_Query.iloc[i, 2] = i + 1
        Result = Result._append(idx_Query)
    print(Result)
    Result.to_csv(res_path + 'E3FP' + '.tsv', index=False, sep='\t')

    # /process MACCSfinger
    index = {}
    for root, dirs, files in os.walk(data_path):
        for decoy in files:
            path = os.path.join(data_path, decoy)
            print(path)
            Finger, E3FP_fps = read_p1Finger(path)
            print(Finger)
            index = add_Finger(Finger, index, decoy.split(".")[0])
    Result = pd.DataFrame(columns=['Query_ID', 'Dec_ID', 'Rank', 'Score'])

    for idx in index.keys():
        i = 0
        idx_Query = pd.DataFrame(columns=['Query_ID', 'Dec_ID', 'Rank', 'Score'])
        for name in index[idx]:
            lig_path = os.path.join(data_path, name + '.mol2')
            Finger, E3FP_fps = read_p1Finger(lig_path)
            x = float(len(Finger))
            MACCS_score =  1 / x #math.exp(1 / x)
            new = pd.DataFrame({'Query_ID': idx, 'Dec_ID': name, 'Rank': i, 'Score': MACCS_score}, index=[1])
            idx_Query = idx_Query._append(new)
            i += 1
        idx_Query = idx_Query.sort_values(by=['Score'], ascending=False)
        for i in range(0, idx_Query.shape[0]):
            # self.result.iloc[i]['Rank'] = i + 1
            idx_Query.iloc[i, 2] = i + 1
        Result = Result._append(idx_Query)
    print(Result)
    Result.to_csv(res_path + 'p1' + '.tsv', index=False, sep='\t')

if __name__ == '__main__':
    path = './data2'
    for root, dirs, files in os.walk(path):
        for data_dir in dirs:
            res_path = './output2/'+data_dir+'/'
            if not os.path.exists(res_path):
                os.makedirs(res_path)
            pre_process2(data_path = os.path.join(path, data_dir), res_path = res_path)

            # path = "./output2/Finger/"

        data_path = './test4/data'
        for root, dirs, files in os.walk(data_path):
            for file in files:
                print("file: ", file)
                id = file.split("_")[0]

                t = 0;
                query_id = os.path.join(data_path, file)
                query_idse3, fps = read_E3FPFinger(query_id)
                query_idsp3, fps = read_3DFinger(query_id)
                query_idsp1, fps = read_p1Finger(query_id)
                output_file = os.path.join('./output2',id,'top10.tsv')
                # clean the output file
                columns = ['Query_ID', 'Dec_ID', 'Rank', 'Score']
                empty_df = pd.DataFrame(columns=columns)
                empty_df.to_csv(output_file, index=False, sep='\t')
                for root, dirs, files in os.walk(res_path):
                    df_query = []
                    weights_vector = 0
                    idx_num = 0

                    # print(query_ids)
                    weights_vector += len(query_idse3)
                    # print(query_ids)
                    weights_vector += len(query_idsp3)
                    # print(query_ids)
                    weights_vector += len(query_idsp3)

                    for filename in files:
                        if operator.contains(filename, 'E3FP'):
                            df = pd.read_csv(path + filename, sep='\t', usecols=['Query_ID', 'Dec_ID', 'Rank', 'Score'])
                            for q in query_idse3:
                                df_query.append(df.loc[df['Query_ID'] == q])
                                idx_num += 1
                        if operator.contains(filename, 'Pharm3D'):
                            df = pd.read_csv(path + filename, sep='\t', usecols=['Query_ID', 'Dec_ID', 'Rank', 'Score'])
                            for q in query_idsp3:
                                df_query.append(df.loc[df['Query_ID'] == q])
                                idx_num += 1
                        if operator.contains(filename, 'p1'):
                            df = pd.read_csv(path + filename, sep='\t', usecols=['Query_ID', 'Dec_ID', 'Rank', 'Score'])
                            for q in query_idsp1:
                                df_query.append(df.loc[df['Query_ID'] == q])
                                idx_num += 1

                        # df_cudf = cudf.from_pandas(df_query)
                    start = time.time()
                    fa = ThresholdAlgorithm(idx_num, 10)
                    top_k = fa.get_topk(datasets=df_query, weights=[1] * weights_vector)
                    output(top_k, output_file)