from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDepictor
import matplotlib.pyplot as plt
import os
import time
from ThresholdAlgorithm import ThresholdAlgorithm
import pandas as pd
import numpy as np
import operator


def read_AtomPairFinger(file):
    mol = Chem.MolFromMol2File(file, sanitize=True, removeHs=False)
    fp = AllChem.GetAtomPairFingerprint(mol)
    bi = fp.GetNonzeroElements()
    bits = []
    for i in bi.keys():
        if bi[i] == 1:
            bits.append(i)
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


def read_MorganFinger(file, radius=1):
    mol = Chem.MolFromMol2File(file, sanitize=True, removeHs=False)
    bi = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, nBits=2048, radius=radius, bitInfo=bi)
    bits = []
    for i in bi.keys():
        if len(bi[i]) == 1:
            bits.append(i)
    return bits, fp


def output(top_k, output_file):
    with open(output_file, 'a+') as f:
        top_k.to_csv(f, index=False, header=False, sep='\t')

def TA_query(path, query_id, output_file, k=20):
    t = 0;

    for root, dirs, files in os.walk(path):
        df_query = []
        weights_vector = 0
        idx_num = 0
        for filename in files:
            if operator.contains(filename, 'Morgan'):
                id = filename.split(".")[0].split("_")[-1]
                query_ids, fps = read_MorganFinger(query_id, radius=int(id))
                weights_vector += len(query_ids)
                df = pd.read_csv(path + filename, sep='\t', usecols=['Query_ID', 'Dec_ID', 'Rank', 'Score'])
                for q in query_ids:
                    df_query.append(df.loc[df['Query_ID'] == q])
                    idx_num += 1
            if operator.contains(filename, 'AtomPair'):
                query_ids, fps = read_AtomPairFinger(query_id)
                weights_vector += len(query_ids)
                df = pd.read_csv(path + filename, sep='\t', usecols=['Query_ID', 'Dec_ID', 'Rank', 'Score'])
                for q in query_ids:
                    df_query.append(df.loc[df['Query_ID'] == q])
                    idx_num += 1
            if operator.contains(filename, 'MACCS'):
                query_ids, fps = read_MACCSFinger(query_id)
                weights_vector += len(query_ids)
                df = pd.read_csv(path + filename, sep='\t', usecols=['Query_ID', 'Dec_ID', 'Rank', 'Score'])
                for q in query_ids:
                    df_query.append(df.loc[df['Query_ID'] == q])
                    idx_num += 1
        start = time.time()
        fa = ThresholdAlgorithm(idx_num, k)
        top_k = fa.get_topk(datasets=df_query, weights=[1] * weights_vector)
        w_access = fa.access
        end = time.time()
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


def Naive_query(query_id, output_file, k=20):
    data_path = './mol_data/pre_process'
    res = pd.DataFrame(columns=['Query_ID', 'Dec_ID', 'Rank', 'Score'])
    access = 0
    start = time.time()
    # AtomPair
    for root, dirs, files in os.walk(data_path):
        for lig in dirs:
            path = os.path.join(data_path, lig, lig + '_ligand.mol2')
            access += 1
            fp1 = read_AtomPairFinger(query_id)
            fp2 = read_AtomPairFinger(path)
            score = CalSimilarity(fp1[0], fp2[0])
            fp1 = read_MACCSFinger(query_id)
            fp2 = read_MACCSFinger(path)
            score += CalSimilarity(fp1[0], fp2[0])
            for r in [1, 5, 10, 20]:
                fp1 = read_MorganFinger(query_id, radius=r)
                fp2 = read_MorganFinger(path, radius=r)
                score += CalSimilarity(fp1[0], fp2[0])
            new = pd.DataFrame({'Query_ID': 0, 'Dec_ID': lig, 'Rank': 1, 'Score': score}, index=[1])
            res = res._append(new)

    res.sort_values(by='Score', ascending=False, inplace=True)
    res.reset_index(drop=True, inplace=True)
    for i in range(0, res.shape[0]):
        res.iloc[i, 2] = i + 1

    top_k = res.head(k)
    t = time.time() - start
    print(top_k)
    output(top_k, output_file)
    return t, access


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
    save_path = './result/pic/'
    plt.bar(np.arange(len(time1)) - bar_width / 2, time1, lw=0.5, label='MapLE', color='#DC3023', alpha=0.8,
            width=bar_width)
    plt.bar(np.arange(len(time1)) + bar_width / 2, time2, lw=0.5, label='Naive', color='#FED71A', alpha=0.8,
            width=bar_width)
    plt.xticks(np.arange(len(time1)), k_list)
    plt.ylabel("time of candidates tested", fontsize=24)
    plt.legend(fontsize=15, loc='upper left')
    plt.savefig(save_path + id + '_time.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("access+", access1)
    color1 = "#924361"  # purple
    color2 = "#e8c51f"  # yellow
    plt.plot(np.arange(len(access1)), access1, color=color1, marker="o", label="MapLE", linewidth=1.5)
    plt.plot(np.arange(len(access2)), access2, color=color2, marker="s", label="Naive", linewidth=1.5, linestyle='--')
    plt.xticks(np.arange(len(access1)), k_list)
    plt.ylabel("number of candidates tested ", fontsize=24)
    plt.legend(fontsize=15, loc='lower right')
    plt.savefig(save_path + id + '_access.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Draw Accuracy
    Accuracy = []
    for k in k_list:
        TA_file = './result/top_' + str(k) + '.tsv'
        Naive_file = './result/topN_' + str(k) + '.tsv'
        a = CalAccuracy(TA_file, Naive_file, k)
        Accuracy.append(a)
    color1 = "#924361"  # purple
    color2 = "#e8c51f"  # yellow
    plt.plot(k_list, Accuracy, color=color1, marker="o", label="MapLE", linewidth=1.5)
    plt.plot(k_list, [1] * len(k_list), color=color2, marker="s", label="Naive", linewidth=1.5, linestyle='--')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=16, loc='lower right')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlim(1, k_list[len(k_list) - 1])
    plt.ylim(0, 1.1)
    for spine in plt.gca().spines.values():
        spine.set_edgecolor("#CCCCCC")
        spine.set_linewidth(1.5)
    plt.savefig(save_path + id + '_accuracy.png', dpi=300, bbox_inches='tight')
    print("save picture" + id + '_accuracy.png')
    plt.show()


def Threshold_query(sample_data):
    path = "./result/score_list/"

    time1 = []
    time2 = []
    access1 = []
    access2 = []
    k_list = [x for x in range(1, 11)]
    max_naivetime = 0
    for k in k_list:
        output_file = './result/top_' + str(k) + '.tsv'
        # clean the output file
        columns = ['Query_ID', 'Dec_ID', 'Rank', 'Score']
        empty_df = pd.DataFrame(columns=columns)
        empty_df.to_csv(output_file, index=False, sep='\t')
        t1 = 0
        wa = 0
        for root, dirs, files in os.walk(sample_data):
            for file in files:
                print("file: ", file)
                t, w = TA_query(path, os.path.join(sample_data, file), output_file, k)
                t1 += t
                wa += w
        time1.append(t1)
        access1.append(wa)
        rdDepictor.SetPreferCoordGen(True)
        output_file = './result/topN_' + str(k) + '.tsv'
        # clean the output file
        columns = ['Query_ID', 'Dec_ID', 'Rank', 'Score']
        empty_df = pd.DataFrame(columns=columns)
        empty_df.to_csv(output_file, index=False, sep='\t')
        t2 = 0
        wa = 0
        for root, dirs, files in os.walk(sample_data):
            for file in files:
                print("file: ", file)
                t, w = Naive_query(os.path.join(sample_data, file), output_file, k)
                t2 += t
                wa += w
        if t2 > max_naivetime:
            max_naivetime = t2
        time2[:] = [max_naivetime] * k
        access2.append(wa)
    draw(time1, access1, time2, access2, k_list, "sample")


def Threshold_query1(sample_data):

    path = "./result/score_list/"

    k_list = [x for x in range(1, 11)]
    for root, dirs, files in os.walk(sample_data):
        for file in files:
            print("file: ", file)
            id = file.split("_")[0]
            time1 = []
            time2 = []
            access1 = []
            access2 = []

            for k in k_list:
                output_file = './result/top_' + str(k) + '.tsv'
                # clean the output file
                print("TA_top", k)
                columns = ['Query_ID', 'Dec_ID', 'Rank', 'Score']
                empty_df = pd.DataFrame(columns=columns)
                empty_df.to_csv(output_file, index=False, sep='\t')
                t, w = TA_query(path, os.path.join(sample_data, file), output_file, k)
                time1.append(t)
                access1.append(w)

            max_NaiveTime = 0
            for k in k_list:
                print("Naive_top", k)
                output_file = './result/topN_' + str(k) + '.tsv'
                columns = ['Query_ID', 'Dec_ID', 'Rank', 'Score']
                empty_df = pd.DataFrame(columns=columns)
                empty_df.to_csv(output_file, index=False, sep='\t')
                t, w = Naive_query(os.path.join(sample_data, file), output_file, k)
                if t > max_NaiveTime:
                    max_NaiveTime = t
                time2[:] = [max_NaiveTime] * k
                access2.append(w)
            draw(time1, access1, time2, access2, k_list, id)


if __name__ == '__main__':


    sample_data = "./mol_data/sample"

    # # Use this code, if you want to match the selected molecules in a batch.
    Threshold_query(sample_data)

    # Use this code, if you want to match the selected molecules one by one.
    # Threshold_query1(sample_data)
