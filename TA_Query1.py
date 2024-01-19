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
from matplotlib.font_manager import FontProperties

# 指定字体文件和路径
font_path = "/home/liaochuyue/.fonts/times.ttf"
# 创建字体属性对象
custom_font = FontProperties(fname=font_path)

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


def read_AtomPairFinger(file):
    mol = Chem.MolFromMol2File(file, sanitize=True, removeHs=False)
    # Chem.SanitizeMol(mol)
    bi = {}
    fp = AllChem.GetAtomPairFingerprint(mol)
    bi = fp.GetNonzeroElements()
    bits = []
    for i in bi.keys():
        if bi[i] == 1:
            bits.append(i)
    return bits, fp


def read_MACCSFinger(file):
    mol = Chem.MolFromMol2File(file, sanitize=True, removeHs=False)
    # Chem.SanitizeMol(mol)
    bi = {}
    fp = AllChem.GetMACCSKeysFingerprint(mol)
    x = list(fp.GetOnBits())
    # print(bi)

    # fdef = AllChem.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))
    # print(fdef.GetNumFeatureDefs())
    # sigFactory = SigFactory(fdef, useCounts=1, minPointCount=1)
    # fp = Generate.Gen2DFingerprint(mol, sigFactory, bitInfo=bi)

    bits = []
    # fp = AllChem.GetAtomPairFingerprint(mol)
    # bi = fp.GetNonzeroElements()
    for i in bi.keys():
        if len(bi[i]) == 1:
            bits.append(i)
    return x, fp


def read_MorganFinger(file, radius=1):
    mol = Chem.MolFromMol2File(file, sanitize=True, removeHs=False)
    # Chem.SanitizeMol(mol)
    bi = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, nBits=2048, radius=radius, bitInfo=bi)
    # print(bi)
    bits = []
    for i in bi.keys():
        # print(len(bi[i]))
        if len(bi[i]) == 1:
            bits.append(i)

    bi = list(fp.GetOnBits())
    return bits, fp
    # for bit in bits:
    #     mfp2_svg = Draw.DrawMorganBit(mol, bit, bi)
    #     imgs.append(mfp2_svg)


def output(top_k, output_file):
    with open(output_file, 'a+') as f:
        top_k.to_csv(f, index=False, header=False, sep='\t')


def pre_process():
    data_path = './data'
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
            topo_score = 1 / x  # math.exp(1 / x)
            new = pd.DataFrame({'Query_ID': idx, 'Dec_ID': name, 'Rank': i, 'Score': topo_score}, index=[1])
            idx_Query = idx_Query._append(new)
            i += 1
        idx_Query = idx_Query.sort_values(by=['Score'], ascending=False)
        for i in range(0, idx_Query.shape[0]):
            # self.result.iloc[i]['Rank'] = i + 1
            idx_Query.iloc[i, 2] = i + 1
        Result = Result._append(idx_Query)
    print(Result)
    res_path = './output1/Finger/'
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
            MACCS_score = 1 / x  # math.exp(1 / x)
            new = pd.DataFrame({'Query_ID': idx, 'Dec_ID': name, 'Rank': i, 'Score': MACCS_score}, index=[1])
            idx_Query = idx_Query._append(new)
            i += 1
        idx_Query = idx_Query.sort_values(by=['Score'], ascending=False)
        for i in range(0, idx_Query.shape[0]):
            # self.result.iloc[i]['Rank'] = i + 1
            idx_Query.iloc[i, 2] = i + 1
        Result = Result._append(idx_Query)
    print(Result)
    res_path = './output1/Finger/'
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
                morgan_score = 1 / x  # math.exp(1 / x)
                new = pd.DataFrame({'Query_ID': idx, 'Dec_ID': name, 'Rank': i, 'Score': morgan_score}, index=[1])
                idx_Query = idx_Query._append(new)
                i += 1
            idx_Query = idx_Query.sort_values(by=['Score'], ascending=False)
            for i in range(0, idx_Query.shape[0]):
                # self.result.iloc[i]['Rank'] = i + 1
                idx_Query.iloc[i, 2] = i + 1
            Result = Result._append(idx_Query)
        print(Result)
        res_path = './output1/Finger/'
        if not os.path.exists(res_path):
            os.makedirs(res_path)
            # topk = T1.get_topk(datasets=assembleRES, weights=[1, 1, 1])
        Result.to_csv(res_path + 'Morgan_' + str(radius) + '.tsv', index=False, sep='\t')


def TA_query(path, query_id, output_file, k=20):
    t = 0;
    file_num = sum([os.path.isfile(os.path.join(path, listx)) for listx in os.listdir(path)])

    for root, dirs, files in os.walk(path):
        df_query = []
        weights_vector = 0
        idx_num = 0
        for filename in files:
            if operator.contains(filename, 'Morgan'):
                id = filename.split(".")[0].split("_")[-1]
                # print(id)
                query_ids, fps = read_MorganFinger(query_id, radius=int(id))
                # print(query_ids)
                weights_vector += len(query_ids)
                df = pd.read_csv(path + filename, sep='\t', usecols=['Query_ID', 'Dec_ID', 'Rank', 'Score'])
                for q in query_ids:
                    df_query.append(df.loc[df['Query_ID'] == q])
                    idx_num += 1
            if operator.contains(filename, 'AtomPair'):
                query_ids, fps = read_AtomPairFinger(query_id)
                # print(query_ids)
                weights_vector += len(query_ids)
                df = pd.read_csv(path + filename, sep='\t', usecols=['Query_ID', 'Dec_ID', 'Rank', 'Score'])
                for q in query_ids:
                    df_query.append(df.loc[df['Query_ID'] == q])
                    idx_num += 1
            if operator.contains(filename, 'MACCS'):
                query_ids, fps = read_MACCSFinger(query_id)
                # print(query_ids)
                weights_vector += len(query_ids)
                df = pd.read_csv(path + filename, sep='\t', usecols=['Query_ID', 'Dec_ID', 'Rank', 'Score'])
                for q in query_ids:
                    df_query.append(df.loc[df['Query_ID'] == q])
                    idx_num += 1
            # df_cudf = cudf.from_pandas(df_query)
        start = time.time()
        fa = ThresholdAlgorithm(idx_num, k)
        top_k = fa.get_topk(datasets=df_query, weights=[1] * weights_vector)
        w_access = fa.access
        end = time.time()
        # print(top_k)
        t += end - start
        output(top_k, output_file)
    return t, w_access


def CalSimilarity(fp1, fp2):
    a = len(fp1)
    b = len(fp2)
    c = 0
    for i in fp1:
        for j in fp2:
            if i == j:
                c += 1
    if c == 0:
        return 0
    return c / (a + b - c)


def Naive_query(path, query_id, output_file, k=20):
    file_num = sum([os.path.isfile(os.path.join(path, listx)) for listx in os.listdir(path)])

    data_path = './data'
    res = pd.DataFrame(columns=['Query_ID', 'Dec_ID', 'Rank', 'Score'])

    mol = Chem.MolFromMol2File(query_id)
    access = 0
    start = time.time()
    # AtomPair
    for root, dirs, files in os.walk(data_path):
        for lig in dirs:
            path = os.path.join(data_path, lig, lig + '_ligand.mol2')
            access += 1
            fp1 = read_AtomPairFinger(query_id)
            fp2 = read_AtomPairFinger(path)
            score = CalSimilarity(fp1[0], fp2[0])  # math.exp(CalSimilarity(fp1[0], fp2[0]))
            fp1 = read_MACCSFinger(query_id)
            fp2 = read_MACCSFinger(path)
            # score += CalSimilarity(fp1[0], fp2[0])
            score += CalSimilarity(fp1[0], fp2[0])  # math.exp(CalSimilarity(fp1[0], fp2[0]))
            for r in [1, 5, 10, 20]:
                fp1 = read_MorganFinger(query_id, radius=r)
                fp2 = read_MorganFinger(path, radius=r)
                # score += CalSimilarity(fp1[0], fp2[0])
                score += CalSimilarity(fp1[0], fp2[0])  # math.exp(CalSimilarity(fp1[0], fp2[0]))

            # res.loc[res['Dec_ID'] == lig]['Score'].values[0] += math.exp(CalSimilarity(fp1[0], fp2[0]))

            new = pd.DataFrame({'Query_ID': 0, 'Dec_ID': lig, 'Rank': 1, 'Score': score}, index=[1])
            res = res._append(new)
    '''
    res = res.sort_values(by=['Score'], ascending=False)
    for i in range(0, res.shape[0]):
        # self.result.iloc[i]['Rank'] = i + 1
        res.iloc[i, 2] = i + 1
        name = res.iloc[i, 1]
        res.iloc[i, 3] = math.exp(-res.loc[res['Dec_ID'] == name]['Rank'].values[0])

    # MACCS
    tmp = pd.DataFrame(columns=['Query_ID', 'Dec_ID', 'Rank', 'Score'])
    for root, dirs, files in os.walk(data_path):
        for lig in dirs:
            path = os.path.join(data_path, lig, lig + '_ligand.mol2')
            fp1 = read_MACCSFinger(query_id)
            fp2 = read_MACCSFinger(path)
            # res.loc[res['Dec_ID'] == lig]['Score'].values[0] += CalSimilarity(fp1[0], fp2[0])
            score = CalSimilarity(fp1[0], fp2[0])
            new = pd.DataFrame({'Query_ID': 0, 'Dec_ID': lig, 'Rank': 1, 'Score': score}, index=[1])
            tmp = tmp._append(new)
            # new = pd.DataFrame({'Query_ID': 0, 'Dec_ID': lig, 'Rank': 1, 'Score': score / 6}, index=[1])
    tmp = tmp.sort_values(by=['Score'], ascending=False)
    for i in range(0, tmp.shape[0]):
        # self.result.iloc[i]['Rank'] = i + 1
        tmp.iloc[i, 2] = i + 1
        name = tmp.iloc[i, 1]
        res.iloc[i, 3] = res.loc[res['Dec_ID'] == name]['Score'].values[0] + math.exp(-i)
    #Morgan
    for r in [1, 5, 10, 15]:
        tmp.drop(tmp.index, inplace=True)
        tmp = tmp.drop(index=tmp.index)
        for root, dirs, files in os.walk(data_path):
            for lig in dirs:
                path = os.path.join(data_path, lig, lig + '_ligand.mol2')
                # moli = Chem.MolFromMol2File(path)
                # access += 1
                fp1 = read_MorganFinger(query_id, radius=r)
                fp2 = read_MorganFinger(path, radius=r)
                # fp1 = AllChem.GetMorganFingerprint(mol, r)
                # fp2 = AllChem.GetMorganFingerprint(moli, r)
                # res.loc[res['Dec_ID'] == lig]['Score'].values[0] += CalSimilarity(fp1[0], fp2[0])
                score = CalSimilarity(fp1[0], fp2[0])
                new = pd.DataFrame({'Query_ID': 0, 'Dec_ID': lig, 'Rank': 1, 'Score': score}, index=[1])
                tmp = tmp._append(new)
        tmp = tmp.sort_values(by=['Score'], ascending=False)
        for i in range(0, tmp.shape[0]):
            # self.result.iloc[i]['Rank'] = i + 1
            tmp.iloc[i, 2] = i + 1
            name = tmp.iloc[i, 1]
            res.iloc[i, 3] = res.loc[res['Dec_ID'] == name]['Score'].values[0] + math.exp(-i)

    '''
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

    path = "./output1/Finger/"
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
    k_list = [x for x in range(1, 11)]
    max_naivetime = 0
    for k in k_list:
        output_file = './output1/top_' + str(k) + '.tsv'
        # clean the output file
        columns = ['Query_ID', 'Dec_ID', 'Rank', 'Score']
        empty_df = pd.DataFrame(columns=columns)
        empty_df.to_csv(output_file, index=False, sep='\t')
        start = time.time()
        # ligand_name = '1a30_ligand.mol2'
        data_path = './test3/data'
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
        output_file = './output1/topN_' + str(k) + '.tsv'
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
        if t2 > max_naivetime:
            max_naivetime = t2
        time2[:] = [max_naivetime] * k
        access2.append(wa)
    bar_width = 0.2

    draw(time1, access1, time2, access2, k_list, "15_best")


def CalAccuracy(file1, file2, k):
    df1 = pd.read_csv(file1, sep='\t', usecols=['Query_ID', 'Dec_ID', 'Rank', 'Score'])
    df2 = pd.read_csv(file2, sep='\t', usecols=['Query_ID', 'Dec_ID', 'Rank', 'Score'])
    ans = 0
    num = 0
    a1 = df1.loc[0:k - 1]
    a2 = df2.loc[0:k - 1]
    num += k
    for i in range(0, k):
        if a1.iloc[i]['Dec_ID'] in a2['Dec_ID'].values:
            ans += 1
    print("k: ", k, "\t ans: ", ans, "\t num:", num)
    if num == 0:
        return 1
    return ans / num


def draw(time1, access1, time2, access2, k_list, id):
    bar_width = 0.2
    line_style = '-'
    save_path = './test3/output/'
    # 柱状条形图
    plt.bar(np.arange(len(time1)) - bar_width / 2, time1, lw=0.5, label='MapLE', color='#DC3023', alpha=0.8,
            width=bar_width)
    plt.bar(np.arange(len(time1)) + bar_width / 2, time2, lw=0.5, label='Naive', color='#FED71A', alpha=0.8,
            width=bar_width)
    # # 折线图
    # color1 = "#924361"  # purple
    # color2 = "#e8c51f"  # yellow
    # plt.plot(np.arange(len(time1)), time1, color=color1, marker="o", label="MapLE", linewidth=1.5)
    # plt.plot(np.arange(len(time2)), time2, color=color2, marker="s", label="Naive", linewidth=1.5, linestyle='--')
    plt.xticks(np.arange(len(time1)), k_list)
    plt.ylabel("time of candidates tested", fontproperties=custom_font, fontsize=24)
    plt.legend(fontsize=15, loc='upper left')
    plt.savefig(save_path + id + '_time.png', dpi=300, bbox_inches='tight')
    plt.show()
    # execute the Threshold's Algorithm for each query of each document
    # query_ids = dfs[0]['Query_ID'].unique()
    # for df in dfs:

    print("access+", access1)
    # plt.bar(np.arange(len(access1)) - bar_width / 2, access1, lw=0.5, label='MapLE', color='#DC3023', alpha=0.8,
    #         width=bar_width)
    # plt.bar(np.arange(len(access1)) + bar_width / 2, access2, lw=0.5, label='Naive', color='#FED71A', alpha=0.8,
    #         width=bar_width)
    color1 = "#924361"  # purple
    color2 = "#e8c51f"  # yellow
    plt.plot(np.arange(len(access1)), access1, color=color1, marker="o", label="MapLE", linewidth=1.5)
    plt.plot(np.arange(len(access2)), access2, color=color2, marker="s", label="Naive", linewidth=1.5, linestyle='--')
    plt.xticks(np.arange(len(access1)), k_list)
    plt.ylabel("number of candidates tested ", fontproperties=custom_font, fontsize=24)
    plt.legend(fontsize=15, loc='lower right')
    plt.savefig(save_path + id + '_access.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Draw Accuracy
    Accuracy = []
    for k in k_list:
        TA_file = './output1/top_' + str(k) + '.tsv'
        Naive_file = './output1/topN_' + str(k) + '.tsv'
        a = CalAccuracy(TA_file, Naive_file, k)
        Accuracy.append(a)
    color1 = "#924361"  # purple
    color2 = "#e8c51f"  # yellow
    plt.plot(k_list, Accuracy, color=color1, marker="o", label="MapLE", linewidth=1.5)
    plt.plot(k_list, [1] * len(k_list), color=color2, marker="s", label="Naive", linewidth=1.5, linestyle='--')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=16, loc='lower right')
    # 设置刻度字体和范围
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlim(1, k_list[len(k_list) - 1])
    plt.ylim(0, 1.1)
    # 设置坐标轴样式
    for spine in plt.gca().spines.values():
        spine.set_edgecolor("#CCCCCC")
        spine.set_linewidth(1.5)
    plt.savefig(save_path + id + '_accuracy.png', dpi=300, bbox_inches='tight')
    print("save picture" + id + '_accuracy.png')
    # 显示图像
    plt.show()


def Threshold_query1():
    '''
    批量处理单个文件（Naive， TA），并输出其对应的time，access， accuracy
    :return:
    '''
    path = "./output1/Finger/"

    k_list = [x for x in range(1, 11)]
    data_path = './test3/data'
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
                output_file = './output1/top_' + str(k) + '.tsv'
                # clean the output file
                print("TA_top", k)
                columns = ['Query_ID', 'Dec_ID', 'Rank', 'Score']
                empty_df = pd.DataFrame(columns=columns)
                empty_df.to_csv(output_file, index=False, sep='\t')
                t, w = TA_query(path, os.path.join(data_path, file), output_file, k)
                end = time.time()
                time1.append(t)
                access1.append(w)

            max_NaiveTime = 0
            for k in k_list:
                start = time.time()
                print("Naive_top", k)
                output_file = './output1/topN_' + str(k) + '.tsv'
                columns = ['Query_ID', 'Dec_ID', 'Rank', 'Score']
                empty_df = pd.DataFrame(columns=columns)
                empty_df.to_csv(output_file, index=False, sep='\t')
                t, w = Naive_query(path, os.path.join(data_path, file), output_file, k)
                if t > max_NaiveTime:
                    max_NaiveTime = t
                end = time.time()
                time2[:] = [max_NaiveTime] * k
                access2.append(w)

            draw(time1, access1, time2, access2, k_list, id)


def draw_Aver_accuracy():
    '''
    处理优秀的文件的K下平均准确率，并画在一张图上
    '''
    average_accuracy = []
    path = "./output1/Finger/"
    # filenum = 0
    k_list = [x for x in range(1, 11)]
    data_path = './test3/data'
    for root, dirs, files in os.walk(data_path):
        for file in files:
            # filenum += 1
            print("file: ", file)
            # id = file.split("_")[0]
            for k in k_list:
                output_file = './output1/top_' + str(k) + '.tsv'
                # clean the output file
                print("TA_top", k)
                columns = ['Query_ID', 'Dec_ID', 'Rank', 'Score']
                empty_df = pd.DataFrame(columns=columns)
                empty_df.to_csv(output_file, index=False, sep='\t')
                TA_query(path, os.path.join(data_path, file), output_file, k)

            for k in k_list:
                print("Naive_top", k)
                output_file = './output1/topN_' + str(k) + '.tsv'
                columns = ['Query_ID', 'Dec_ID', 'Rank', 'Score']
                empty_df = pd.DataFrame(columns=columns)
                empty_df.to_csv(output_file, index=False, sep='\t')
                Naive_query(path, os.path.join(data_path, file), output_file, k)

            avg_acc_for_file = 0
            for k in k_list:
                TA_file = './output1/top_' + str(k) + '.tsv'
                Naive_file = './output1/topN_' + str(k) + '.tsv'
                a = CalAccuracy(TA_file, Naive_file, k)
                avg_acc_for_file += a
            avg_acc_for_file /= len(k_list)
            average_accuracy.append(avg_acc_for_file)
    filenames = [filename[:4] for filename in os.listdir(data_path) if filename.endswith('.mol2')]
    color1 = "#924361"  # purple
    color2 = "#e8c51f"  # yellow

    plt.plot(filenames, average_accuracy, color=color1, marker="o", label="MapLE", linewidth=1.5)
    plt.plot(filenames, [1] * len(filenames), color=color2, marker="s", label="Naive", linewidth=1.5, linestyle='--')

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper left', frameon=True, fontsize=10)

    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)

    plt.ylim(0, 1.1)
    for spine in plt.gca().spines.values():
        spine.set_edgecolor("#CCCCCC")
        spine.set_linewidth(1.5)

    plt.title("Accuracy Curve", fontsize=20)
    plt.xlabel("Molecules", fontsize=15)
    plt.ylabel("Accuracy", fontsize=15)

    # 获取标题的位置
    title_x, title_y = plt.gca().title.get_position()
    # plt.legend(loc='best', bbox_to_anchor=(title_x - 0.8, title_y-0.9))
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # pre_process()

    Threshold_query()
    # Threshold_query1()
    # draw_Aver_accuracy()
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
