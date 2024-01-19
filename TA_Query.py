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
import rapids
import math


device = torch.device('cuda:0')

def set_gpu(data, device):
    data_gpu = []
    for g in data:
        data_gpu.append(g.to(device))

    return data_gpu

def show(mol):
    img = Draw.MolToImage(mol)

    img.save("./save_demo.png")

    plt.imshow(img)
    plt.show()

def displayimgsinrow(imgs, col=4):
    plt.figure(figsize=(200, 1000))
    columns = col
    for i, image in enumerate(imgs):
        # ax = plt.subplot(int(len(imgs) / columns) + 1, columns, i + 1)
        # ax.set_axis_off()
        plt.imshow(image)
    plt.show()
def add_Finger(finger, index, ligand):
    for fig in finger:
        if index.get(fig) is not None:
            index[fig].append(ligand)
        else:
            index[fig] = [ligand]
    return index

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
def output(top_k, output_file):
    with open(output_file, 'a+') as f:
            top_k.to_csv(f, index=False, header=False, sep='\t')

def pre_process(radius = 1):
    index = {}

    data_path = './refined-set'
    rdDepictor.SetPreferCoordGen(True)
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
            # new = pd.DataFrame(columns=['Query_ID', 'Dec_ID', 'Rank', 'Score'])
            Finger, morgan_fps = read_MorganFinger(lig_path, radius)
            # mol = Chem.MolFromMol2File(lig_path)
            # Chem.SanitizeMol(mol)
            # idx_mol = Chem.PathToSubmol(mol, [idx])
            # Chem.SanitizeMol(idx_mol)
            # idx_mol.RemoveAtom(0)
            # idx_fp = AllChem.GetMorganFingerprintAsBitVect(idx_mol, nBits=1024, radius=radius)
            # idx_mol = morgan_fps[idx]
            # morgan_score = DataStructs.FingerprintSimilarity(morgan_fps, morgan_fps)
            # morgan_score = DataStructs.DiceSimilarity(idx_mol, morgan_fps)
            x = float(len(Finger))
            morgan_score = 1/x
            new = pd.DataFrame({'Query_ID': idx, 'Dec_ID': name, 'Rank': i, 'Score': morgan_score}, index=[1])
            idx_Query = idx_Query._append(new)
            i += 1
        idx_Query = idx_Query.sort_values(by=['Score'], ascending=False)
        for i in range(0, idx_Query.shape[0]):
            # self.result.iloc[i]['Rank'] = i + 1
            idx_Query.iloc[i, 2] = i + 1
        Result = Result._append(idx_Query)
    print(Result)
    res_path = './output/Finger/'
    if not os.path.exists(res_path):
        os.makedirs(res_path)
        # topk = T1.get_topk(datasets=assembleRES, weights=[1, 1, 1])
    Result.to_csv(res_path + 'Morgan_' + str(radius) + '.tsv', index=False, sep='\t')

def TA_query(path, query_id, output_file, k = 20):
    t = 0;
    file_num = sum([os.path.isfile(os.path.join(path, listx)) for listx in os.listdir(path)])

    for root, dirs, files in os.walk(path):
        df_query = []
        weights_vector = 0
        idx_num = 0
        for filename in files:
            id  = filename.split(".")[0].split("_")[-1]
            # print(id)
            query_ids, fps = read_MorganFinger(query_id, radius = int(id))
            # print(query_ids)
            weights_vector += len(query_ids)
            df = pd.read_csv(path + filename, sep='\t', usecols=['Query_ID', 'Dec_ID', 'Rank', 'Score'])
            for q in query_ids:
                df_query.append(df.loc[df['Query_ID'] == q])
                idx_num += 1
            # df_cudf = cudf.from_pandas(df_query)
        start = time.time()
        fa = ThresholdAlgorithm(idx_num, k)
        top_k = fa.get_topk(datasets=df_query, weights=[1]*weights_vector)
        w_access = fa.access
        end = time.time()
        # print(top_k)
        t += end-start
        output(top_k, output_file)
    return t, w_access
def CalSimilarity(fp1, fp2):
    a = len(fp1)
    b = len(fp2)
    c = 0
    for i in fp1:
        for j in fp2:
            if i == j:
                c+=1
    if c == 0:
        return 0
    return c/(a+b-c)
def Naive_query(path, query_id, output_file, k = 20):
    file_num = sum([os.path.isfile(os.path.join(path, listx)) for listx in os.listdir(path)])

    data_path = './refined-set'
    res = pd.DataFrame(columns=['Query_ID', 'Dec_ID', 'Rank', 'Score'])

    mol = Chem.MolFromMol2File(query_id)
    access = 0
    start = time.time()
    for root, dirs, files in os.walk(data_path):
        for lig in dirs:
            path = os.path.join(data_path, lig, lig + '_ligand.mol2')
            score = 0
            moli = Chem.MolFromMol2File(path)
            access += 1
            for r in [1, 5, 10, 15]:

                fp1 = read_MorganFinger(query_id, radius=r)
                fp2 = read_MorganFinger(path, radius=r)

                # fp1 = AllChem.GetMorganFingerprint(mol, r)
                # fp2 = AllChem.GetMorganFingerprint(moli, r)
                score += CalSimilarity(fp1[0], fp2[0])
            new = pd.DataFrame({'Query_ID': 0, 'Dec_ID': lig, 'Rank': 1, 'Score': score/4}, index=[1])
            res = res._append(new)
    res.sort_values(by='Score', ascending=False, inplace=True)
    res.reset_index(drop=True, inplace=True)
    for i in range(0, res.shape[0]):
        # self.result.iloc[i]['Rank'] = i + 1
        res.iloc[i, 2] = i + 1

    top_k = res.head(k)
    t = time.time() - start
    print(top_k)
    output(top_k, output_file)
    return t, access

def Threshold_query():
    # reading all the parameters from the terminal
    # parameters' shape:
    # [k] [number of files/dataset] [weight of the i-th dataframe score separete by space] [output directory]
    # Ex:
    # 5 2 ./data/output-stopwords-BM25Scorer-title.tsv ./data/output-stopwords-BM25Scorer-text.tsv 2 1 ./data/output-threshold.tsv
    '''
    k = int(sys.argv[1])
    total_files = int(sys.argv[2])
    filenames = []
    Ls = []
    for i in range(total_files):
        filenames.append(sys.argv[3+i])
        Ls.append('L'+str(i))
    weights_vector = []
    for i in range(total_files):
        weights_vector.append(int(sys.argv[3+total_files+i]))
    output_file = sys.argv[3+total_files+i+1]

    start = time.time()
    if total_files != len(filenames) != len(weights_vector):
        raise Exception("Number of files or elements in the vector of weights don't match")
    :return:
    '''

    path = "./output/Finger/"
    # file_num = sum([os.path.isfile(listx) for listx in os.listdir(path)])

    # dfs = []
    # weights_vector = []
    # for root, dirs, files in os.walk(path):
    #     for filename in files:
    #         dfs.append(pd.read_csv(path + filename, sep='\t', usecols=['Query_ID', 'Dec_ID', 'Rank', 'Score']))
    #         weights_vector.append(1)

    # dfs = set_gpu(dfs[:-1], device)
    # parse ligand to different query

    time1 = []
    time2 = []
    access1 = []
    access2 = []
    k_list = [x for x in range(1,11)]

    for k in k_list:
        output_file = './output/top_'+str(k)+'.tsv'
        # clean the output file
        columns = ['Query_ID', 'Dec_ID', 'Rank', 'Score']
        empty_df = pd.DataFrame(columns=columns)
        empty_df.to_csv(output_file, index=False, sep='\t')
        start = time.time()
        # ligand_name = '1a30_ligand.mol2'
        data_path = './test2'
        # rdDepictor.SetPreferCoordGen(True)
        t1 = 0
        wa = 0
        for root, dirs, files in os.walk(data_path):
            for file in files:
                print("file: ", file)
                t, w = TA_query(path, os.path.join(data_path, file), output_file, k)
                t1 += t
                wa += w
        end = time.time()
        time1.append(t1)
        access1.append(wa)
        start = time.time()
        # ligand_name = '1a30_ligand.mol2'
        rdDepictor.SetPreferCoordGen(True)
        output_file = './output/topN_' + str(k) + '.tsv'
        # clean the output file
        columns = ['Query_ID', 'Dec_ID', 'Rank', 'Score']
        empty_df = pd.DataFrame(columns=columns)
        empty_df.to_csv(output_file, index=False, sep='\t')
        t2 = 0
        wa = 0
        for root, dirs, files in os.walk(data_path):
            for file in files:
                print("file: ", file)
                t, w = Naive_query(path, os.path.join(data_path, file), output_file, k)
                t2 += t
                wa += w
        end = time.time()
        time2.append(t2)
        access2.append(wa)
    bar_width = 0.2

    plt.bar(np.arange(len(time1)) - bar_width / 2, time1, lw=0.5, label='TECR', color='#DC3023', alpha=0.8,
            width=bar_width)
    plt.bar(np.arange(len(time1)) + bar_width/2, time2, lw=0.5, label='Naive', color='#FED71A', alpha=0.8, width=bar_width)
    plt.xticks(np.arange(len(time1)), k_list)
    plt.title("time (sec.)")
    plt.xlabel("k value")
    plt.ylabel("time of cand. tested")
    plt.legend()
    plt.show()
    # execute the Threshold's Algorithm for each query of each document
    # query_ids = dfs[0]['Query_ID'].unique()
    # for df in dfs:

    print("access+", access1)
    plt.bar(np.arange(len(access1)) - bar_width / 2, access1, lw=0.5, label='TECR', color='#DC3023', alpha=0.8,
            width=bar_width)
    plt.bar(np.arange(len(access1)) + bar_width / 2, access2, lw=0.5, label='Naive', color='#FED71A', alpha=0.8,
            width=bar_width)
    plt.xticks(np.arange(len(access1)), k_list)
    plt.title("candidate access")
    plt.xlabel("k value")
    plt.ylabel("num. of cand. tested ")
    plt.legend()
    plt.show()

def CalAccuracy(file1, file2, k):
    df1 = pd.read_csv(file1, sep='\t', usecols=['Query_ID', 'Dec_ID', 'Rank', 'Score'])
    df2 = pd.read_csv(file2, sep='\t', usecols=['Query_ID', 'Dec_ID', 'Rank', 'Score'])
    ans = 0
    num = 0
    a1 = df1.loc[0:k-1]
    a2 = df2.loc[0:k-1]
    num += k
    for i in range(0, k):
        if a1.iloc[i]['Dec_ID'] in a2['Dec_ID'].values:
            ans +=1
    print("k: ", k, "\t ans: ", ans, "\t num:", num)
    if num == 0:
        return 1
    return ans/num

def draw(time1, access1, time2, access2, k_list, id):
    bar_width = 0.2
    save_path = './test2/output/'

    plt.bar(np.arange(len(time1)) - bar_width / 2, time1, lw=0.5, label='TECR', color='#DC3023', alpha=0.8,
            width=bar_width)
    plt.bar(np.arange(len(time1)) + bar_width / 2, time2, lw=0.5, label='Naive', color='#FED71A', alpha=0.8,
            width=bar_width)
    plt.xticks(np.arange(len(time1)), k_list)
    plt.title("time (sec.)")
    plt.xlabel("k value\n"+id)
    plt.ylabel("time of cand. tested")
    plt.legend()
    plt.savefig(save_path+id+'_time.png', dpi=300, bbox_inches='tight')
    plt.show()
    # execute the Threshold's Algorithm for each query of each document
    # query_ids = dfs[0]['Query_ID'].unique()
    # for df in dfs:

    print("access+", access1)
    plt.bar(np.arange(len(access1)) - bar_width / 2, access1, lw=0.5, label='TECR', color='#DC3023', alpha=0.8,
            width=bar_width)
    plt.bar(np.arange(len(access1)) + bar_width / 2, access2, lw=0.5, label='Naive', color='#FED71A', alpha=0.8,
            width=bar_width)
    plt.xticks(np.arange(len(access1)), k_list)
    plt.title("candidate access")
    plt.xlabel("k value\n"+id)
    plt.ylabel("num. of cand. tested ")
    plt.legend()
    plt.savefig(save_path+id+'_access.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Draw Accuracy
    Accuracy = []
    for k in k_list:
        TA_file = './output/top_' + str(k) + '.tsv'
        Naive_file = './output/topN_' + str(k) + '.tsv'
        a = CalAccuracy(TA_file, Naive_file, k)
        Accuracy.append(a)
    color1 = "#924361"  # purple
    color2 = "#e8c51f"  # yellow
    plt.plot(k_list, Accuracy, color=color1, marker="o", label="TA", linewidth=1.5)
    plt.plot(k_list, [1]*len(k_list), color=color2, marker="s", label="Naive", linewidth=1.5, linestyle='--')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper left', frameon=True, fontsize=10)
    # 设置刻度字体和范围
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlim(1, k_list[len(k_list)-1])
    plt.ylim(0, 1.1)
    # 设置坐标轴样式
    for spine in plt.gca().spines.values():
        spine.set_edgecolor("#CCCCCC")
        spine.set_linewidth(1.5)
    plt.savefig(save_path+id+'_accuracy.png', dpi=300, bbox_inches='tight')
    # 显示图像
    plt.show()

def Threshold_query1():
    path = "./output/Finger/"

    k_list = [x for x in range(1,11)]
    data_path = './test2/data'
    for root, dirs, files in os.walk(data_path):
        for file in files:
            print("file: ", file)
            id = file.split("_")[0]
            time1 = []
            time2 = []
            access1 = []
            access2 = []

            for k in k_list:
                start = time.time()
                output_file = './output/top_'+str(k)+'.tsv'
                # clean the output file
                print("TA_top", k)
                columns = ['Query_ID', 'Dec_ID', 'Rank', 'Score']
                empty_df = pd.DataFrame(columns=columns)
                empty_df.to_csv(output_file, index=False, sep='\t')
                t, w = TA_query(path, os.path.join(data_path, file), output_file, k)
                end = time.time()
                time1.append(t)
                access1.append(w)
            for k in k_list:
                start = time.time()
                print("Naive_top", k)
                output_file = './output/topN_' + str(k) + '.tsv'
                columns = ['Query_ID', 'Dec_ID', 'Rank', 'Score']
                empty_df = pd.DataFrame(columns=columns)
                empty_df.to_csv(output_file, index=False, sep='\t')
                t, w = Naive_query(path, os.path.join(data_path, file), output_file, k)
                end = time.time()
                time2.append(t)
                access2.append(w)

            draw(time1, access1, time2, access2, k_list, id)


if __name__ == '__main__':
    # for r in [1, 5, 10, 15]:
    #     pre_process(radius = r)
    Threshold_query1()

    # mfp2_svg = Draw.DrawMorganBit(mol, 872, bi, useSVG=True)
    # rdkbi = {}
    # rdkfp = Chem.RDKFingerprint(mol, maxPath=5, bitInfo=rdkbi)
    # rdk_svg = Draw.DrawRDKitBit(mol, 1553, rdkbi, useSVG=True)
    # plt.imshow(rdk_svg)
    # plt.show()

    # SMILES = Chem.MolToSmiles(mol)
    # fingerprints = []
    # safe = []
    #
    # fingerprint = [Chem.Fingerprints.FingerprintMols.FingerprintMol(mol)]
    # fingerprints.append(fingerprint)
    #
    # print(fingerprint)
    # show(mol)

    # fh = open('1.txt')
    # smis = []
    # for line in fh.readlines():
    #     smi = line.strip()
    #     smis.append(smi)
    # mol = Chem.MolFromMol2File('file')
    # m2=Chem.AddHs(mol)
    # AllChem.EmbedMolecule(m2)
    # print(smis)
    # mols3d=[]
'''
    for smi in smis:
        # print(smi)
        mol=Chem.MolFromSmiles(smi)
        m2=Chem.AddHs(mol)
        AllChem.EmbedMolecule(m2)
        # maxIters默认是优化200次，有时不会收敛，建议设置成2000
        opt_state=AllChem.MMFFOptimizeMolecule(m2,maxIters=2000)
        # print(opt_state)
        mols3d.append(m2)
    
    # Draw.MolToImage(mols3d[0], size=(450, 150))
        img = Draw.MolToImage(m2)
    
        img.save("./save_demo.png")
        print(type(img))
        plt.imshow(img)
        plt.show()

'''
