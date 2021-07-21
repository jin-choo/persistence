import math, numpy as np, networkx as nx, pandas as pd, re, sys, statistics, pickle, powerlaw, matplotlib
from matplotlib import pyplot as plt
from scipy.stats import entropy
from ast import literal_eval
from itertools import combinations, product
from collections import Counter
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score, r2_score, mean_squared_error
from scipy.optimize import curve_fit
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFECV
import statsmodels.api as sm

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

# count_time_units
def count_time_units(filepath: str, unit: int):
    timestamp_min = sys.maxsize
    timestamp_max = 0

    f_times = open(f"../dataset/{filepath}/{filepath}-times.txt", 'r')
    while True:
        try:
            timestamp = int(f_times.readline())
            if timestamp < timestamp_min:
                timestamp_min = timestamp
            if timestamp > timestamp_max:
                timestamp_max = timestamp
        except:
            break
    f_times.close()

    f = open(f"./txt/time_units/{filepath}_time_units.txt", "w")
    f.write(f"{timestamp_min} {timestamp_max} {unit}")
    f.close()

# process networkx graph
def graph(filepath: str):
    f = open(f"./txt/time_units/{filepath}_time_units.txt", "r")
    time_units_split = f.readline().split()
    f.close()
    timestamp_min = int(time_units_split[0])
    unit = int(time_units_split[2])

    H = nx.Graph()
    G = nx.MultiGraph()

    f_nverts = open(f"../dataset/{filepath}/{filepath}-nverts.txt", 'r')
    f_simplices = open(f"../dataset/{filepath}/{filepath}-simplices.txt", 'r')
    f_times = open(f"../dataset/{filepath}/{filepath}-times.txt", 'r')

    while True:
        line = f_nverts.readline()
        if not line:
            break
        line = int(line)
        if line >= 2:
            edges_list = list(combinations(sorted(set(int(f_simplices.readline().rstrip('\n')) for vert in range(line))), 2))
            H.add_edges_from(edges_list)
            G.add_edges_from(edges_list, timestamp=round((int(f_times.readline()) - timestamp_min) / unit))
        else:

            f_times.readline()

    f_nverts.close()
    f_simplices.close()
    f_times.close()

    nx.write_gpickle(H, f"../dataset/{filepath}-proj-graph/{filepath}.gpickle")
    nx.write_gpickle(G, f"../dataset/{filepath}-proj-graph/{filepath}-timestamp.gpickle")

    H = nx.Graph()
    for u, v in G.edges():
        if H.has_edge(u, v):
            H[u][v]['weight'] += 1
        else:
            H.add_edge(u, v, weight=1)

    nx.write_gpickle(H, f"../dataset/{filepath}-proj-graph/{filepath}-weight.gpickle")

# process networkx graph (bipartite)
def graph_bi(filepath: str):
    f = open(f"./txt/time_units/{filepath}_time_units.txt", "r")
    time_units_split = f.readline().split()
    f.close()
    timestamp_min = int(time_units_split[0])
    unit = int(time_units_split[2])

    G = nx.Graph()

    f_nverts = open(f"../dataset/{filepath}/{filepath}-nverts.txt", 'r')
    f_simplices = open(f"../dataset/{filepath}/{filepath}-simplices.txt", 'r')
    f_times = open(f"../dataset/{filepath}/{filepath}-times.txt", 'r')

    edge_idx = 100000000

    while True:
        line = f_nverts.readline()
        if not line:
            break
        line = int(line)
        if line >= 2:
            nodes_list = sorted(set(int(f_simplices.readline().rstrip('\n')) for vert in range(line)))
            G.add_nodes_from(nodes_list, bipartite=0)
            G.add_node(edge_idx, bipartite=1, timestamp=round((int(f_times.readline()) - timestamp_min) / unit))
            G.add_edges_from(product(nodes_list, [edge_idx]))
            edge_idx += 1
        else:
            f_times.readline()

    f_nverts.close()
    f_simplices.close()
    f_times.close()

    nx.write_gpickle(G, f"../dataset/{filepath}-proj-graph/{filepath}-bi.gpickle")

def linlaw(x, a, b):
    return a + x * b

# global analysis
def global_analysis(filepath: str, m: int, size_hoi: int, basket_max: int, observed_time_units: int):
    plt.rcParams["font.size"] = "32"
    plt.rcParams["ytick.major.pad"] = "8"

    powerlaw_exponent = []
    k_low_mid_high = []
    average_persistence = []

    df = pd.DataFrame(index=range(2, size_hoi + 1), columns=['dataset', 'k', 'r_squared', 'powerlaw_exponent', 'average_persistence'])

    for size_hoi_ in range(1, size_hoi):
        counter_time_unit = dict([list(map(int, line.rstrip('\n').split(": "))) for line in open(f"../persistence/output/{filepath}_i{size_hoi_ + 1}_{m}_{observed_time_units}_{basket_max}_c.txt", 'r')])
        df.loc[size_hoi_ + 1, 'dataset'] = filepath
        df.loc[size_hoi_ + 1, 'k'] = size_hoi_ + 1

        x = []
        y = []
        x_list = []
        low_mid_high = [0, 0, 0]
        for key, value in sorted(counter_time_unit.items()):
            x.append(key)
            y.append(value)
            x_list += [key] * value
            if np.log10(key) / np.log10(observed_time_units) < 1/3:
                low_mid_high[0] += value
            elif np.log10(key) / np.log10(observed_time_units) < 2/3:
                low_mid_high[1] += value
            else:
                low_mid_high[2] += value
        sum_low_mid_high = sum(low_mid_high)
        k_low_mid_high.append([size_hoi_ + 1, 'low', low_mid_high[0]])
        k_low_mid_high.append([size_hoi_ + 1, 'mid', low_mid_high[1]])
        k_low_mid_high.append([size_hoi_ + 1, 'high', low_mid_high[2]])
        average_persistence_value = np.dot(x, y) / sum_low_mid_high
        average_persistence.append(average_persistence_value)
        df.loc[size_hoi_ + 1, 'average_persistence'] = average_persistence_value

        plt.figure(figsize=(5, 4))
        plt.locator_params(axis='y', nbins=2)
        plt.scatter(x, y, s=150, color='tab:blue')
        plt.locator_params(axis='y', nbins=2)
        try:
            xdata_log = np.log10(x)
            ydata_log = np.log10(y)
            popt_log, pcov_log = curve_fit(linlaw, xdata_log, ydata_log)
            powerlaw_exponent.append(popt_log[1])
            df.loc[size_hoi_ + 1, 'powerlaw_exponent'] = popt_log[1]
            residuals = ydata_log - linlaw(xdata_log, *popt_log)
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((ydata_log - np.mean(ydata_log)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            df.loc[size_hoi_ + 1, 'r_squared'] = r_squared
            plt.plot(x, np.power(10, linlaw(xdata_log, *popt_log)), color='black', linewidth=1)
            plt.locator_params(axis='y', nbins=2)
        except:
            pass

        plt.xlabel('Persistence')
        plt.ylabel('Frequency')
        plt.xscale('log')
        plt.yscale('log')
        plt.minorticks_off()
        plt.gca().spines['right'].set_color('none')
        plt.gca().spines['top'].set_color('none')
        x_max = np.max(x)
        if x_max >= 10:
            plt.xticks([1, 10], [r'$10^0$', r'$10^1$'])
        else:
            plt.xticks([1, x_max], [1, x_max])
        if filepath == 'coauth-DBLP':
            plt.yticks([10, 10**4], [r'$10^1$', r'$10^4$'])
        if size_hoi_ == 1:
            plt.savefig(f"./figure/global_analysis/{filepath}_{m}_{observed_time_units}_{basket_max}_scatter_2.pgf", bbox_inches="tight")
            plt.close()
        elif size_hoi_ == 2:
            plt.savefig(f"./figure/global_analysis/{filepath}_{m}_{observed_time_units}_{basket_max}_scatter_3.pgf", bbox_inches="tight")
            plt.close()
        elif size_hoi_ == 3:
            plt.savefig(f"./figure/global_analysis/{filepath}_{m}_{observed_time_units}_{basket_max}_scatter_4.pgf", bbox_inches="tight")
            plt.close()

    df.to_csv(f"./txt/global_analysis/{filepath}_{m}_{observed_time_units}_{basket_max}.csv", sep=',')

# save local analysis result
def local_analysis_save(filepath: str, m: int, size_hoi: int, basket_max: int, observed_time_units_features: int, observed_time_units: int, what_to_do: str):

    f = open(f"./txt/time_units/{filepath}_time_units.txt", "r")
    time_units_split = f.readline().split()
    f.close()
    
    timestamp_min = int(time_units_split[0])
    timestamp_max = int(time_units_split[1])
    unit = int(time_units_split[2])
    time_unit_length = round((timestamp_max - timestamp_min) / unit) / m
    num_time_units = m - observed_time_units - observed_time_units_features + 1  # 61
    
    feature_num = 8
    feature_node_dict = dict()
    HE_time_unit_list = []

    B = nx.read_gpickle(f"../dataset/{filepath}-proj-graph/{filepath}-bi.gpickle")
    G = nx.read_gpickle(f"../dataset/{filepath}-proj-graph/{filepath}-timestamp.gpickle")

    for time_unit in range(num_time_units):  # 0-60
        if what_to_do == 'local_analysis_past':  # < 66-6
            HE_time_unit_list.append(set(n for n, d in B.nodes(data=True) if d["bipartite"] == 1 and d['timestamp'] < (m - observed_time_units - time_unit + 1) * time_unit_length))
            H = nx.Graph([(u, v) for u, v, e in G.edges(data=True) if e['timestamp'] < (m - observed_time_units - time_unit + 1) * time_unit_length])
            H_multi = nx.MultiGraph([(u, v) for u, v, e in G.edges(data=True) if e['timestamp'] < (m - observed_time_units - time_unit + 1) * time_unit_length])
            for n, d in B.nodes(data=True):
                if d["bipartite"] == 1 and d['timestamp'] < (m - observed_time_units - time_unit + 1) * time_unit_length:
                    for key in B[n]:
                        try:
                            feature_node_dict[key][time_unit][2] += 1
                        except:
                            feature_node_dict[key] = [[0 for i_feature in range(feature_num)] for time_unit in range(num_time_units)]
                            feature_node_dict[key][time_unit][2] += 1
                            pass
        else: # 60-0 <= < 66-6
            HE_time_unit_list.append(set(n for n, d in B.nodes(data=True) if d["bipartite"] == 1 and (m - observed_time_units - time_unit - observed_time_units_features) * time_unit_length <= d['timestamp'] < (m - observed_time_units - time_unit + 1) * time_unit_length))
            H = nx.Graph([(u, v) for u, v, e in G.edges(data=True) if (m - observed_time_units - time_unit - observed_time_units_features) * time_unit_length <= e['timestamp'] < (m - observed_time_units - time_unit + 1) * time_unit_length])
            H_multi = nx.MultiGraph([(u, v) for u, v, e in G.edges(data=True) if (m - observed_time_units - time_unit - observed_time_units_features) * time_unit_length <= e['timestamp'] < (m - observed_time_units - time_unit + 1) * time_unit_length])
            for n, d in B.nodes(data=True):
                if d["bipartite"] == 1 and (m - observed_time_units - time_unit - observed_time_units_features) * time_unit_length <= d['timestamp'] < (m - observed_time_units - time_unit + 1) * time_unit_length:
                    for key in B[n]:
                        try:
                            feature_node_dict[key][time_unit][2] += 1
                        except:
                            feature_node_dict[key] = [[0 for i_feature in range(feature_num)] for time_unit in range(num_time_units)]
                            feature_node_dict[key][time_unit][2] += 1
                            pass

        for key, value in H.degree():
            try:
                feature_node_dict[key][time_unit][0] = value
            except:
                feature_node_dict[key] = [[0 for i_feature in range(feature_num)] for time_unit in range(num_time_units)]
                feature_node_dict[key][time_unit][0] = value
                pass
        for key, value in H_multi.degree(weight='weight'):
            try:
                feature_node_dict[key][time_unit][1] = value
            except:
                feature_node_dict[key] = [[0 for i_feature in range(feature_num)] for time_unit in range(num_time_units)]
                feature_node_dict[key][time_unit][1] = value
                pass
        for key, value in nx.core_number(H).items():
            try:
                feature_node_dict[key][time_unit][3] = value
            except:
                feature_node_dict[key] = [[0 for i_feature in range(feature_num)] for time_unit in range(num_time_units)]
                feature_node_dict[key][time_unit][3] = value
                pass
        for key, value in nx.pagerank(H).items():
            try:
                feature_node_dict[key][time_unit][4] = value
            except:
                feature_node_dict[key] = [[0 for i_feature in range(feature_num)] for time_unit in range(num_time_units)]
                feature_node_dict[key][time_unit][4] = value
                pass
        for key, value in nx.average_neighbor_degree(H).items():
            try:
                feature_node_dict[key][time_unit][5] = value
            except:
                feature_node_dict[key] = [[0 for i_feature in range(feature_num)] for time_unit in range(num_time_units)]
                feature_node_dict[key][time_unit][5] = value
                pass
        for key, value in nx.average_neighbor_degree(H_multi, weight='weight').items():
            try:
                feature_node_dict[key][time_unit][6] = value
            except:
                feature_node_dict[key] = [[0 for i_feature in range(feature_num)] for time_unit in range(num_time_units)]
                feature_node_dict[key][time_unit][6] = value
                pass
        for key, value in nx.clustering(H).items():
            try:
                feature_node_dict[key][time_unit][7] = value
            except:
                feature_node_dict[key] = [[0 for i_feature in range(feature_num)] for time_unit in range(num_time_units)]
                feature_node_dict[key][time_unit][7] = value
                pass

    persistence_feature_list = [[[[] for i_persistence in range(observed_time_units + 1)] for size_hoi_ in range(size_hoi)] for time_unit in range(num_time_units)]
    perseverance_feature_dict = [dict() for size_hoi_ in range(size_hoi)]

    for line in open(f"../persistence/output/{filepath}_i_f1_{m}_{observed_time_units}_{observed_time_units_features}_{basket_max}.txt", 'r'):
        line_split = re.split(': | / ', line.rstrip('\n'))
        node = literal_eval(line_split[0])
        persistence = int(line_split[-2])
        first_time_unit_ = int(line_split[-1]) # 0-60
        first_time_unit = num_time_units - first_time_unit_ - 1 # 60-0

        HE_first_time_unit = HE_time_unit_list[first_time_unit]
        HE_node = set(B[node]) & HE_first_time_unit
        len_HE_co_occur = len(HE_node)

        HN_common_nbrs = set()
        HE_co_occur_size = []
        for HE in HE_node:
            HN_in_HE = set(B[HE])
            HN_common_nbrs |= HN_in_HE
            HE_co_occur_size.append(len(HN_in_HE))
        sum_HE_co_occur_size = sum(HE_co_occur_size)
        len_HN_common_nbrs = len(HN_common_nbrs) - 1

        perseverance_feature_dict[0][node] = [persistence, feature_node_dict[node][first_time_unit], first_time_unit_]
        try:
            persistence_feature_list[first_time_unit_][0][persistence].append([len_HE_co_occur, len_HE_co_occur, sum_HE_co_occur_size, len_HN_common_nbrs, len_HE_co_occur / len_HN_common_nbrs, sum_HE_co_occur_size / len_HN_common_nbrs, statistics.mean(HE_co_occur_size), entropy(list(Counter(HE_co_occur_size).values()))] + [feature_node_dict[node][first_time_unit][i_feature] for i_feature in range(feature_num)]) # 60-0
        except:
            pass

    for size_hoi_ in tqdm(range(1, size_hoi)):
        for line in open(f"../persistence/output/{filepath}_i_f{size_hoi_ + 1}_{m}_{observed_time_units}_{observed_time_units_features}_{basket_max}.txt", 'r'):
            line_split = re.split(': | / ', line.rstrip('\n'))
            node_tuple = literal_eval(line_split[0])
            persistence = int(line_split[-2])
            first_time_unit_ = int(line_split[-1]) # 0-60
            first_time_unit = num_time_units - first_time_unit_ - 1 # 60-0

            HE_first_time_unit = HE_time_unit_list[first_time_unit]
            node_0 = node_tuple[0]
            HE_node_0 = set(B[node_0]) & HE_first_time_unit
            HE_co_occur = set(HE_node_0)
            HE_union = set(HE_node_0)
            feature_node_list = [[feature_node_dict[node_0][first_time_unit][i_feature]] for i_feature in range(feature_num)]
            try:
                perseverance_feature_dict[size_hoi_][node_0][0].append(persistence)
                if perseverance_feature_dict[size_hoi_][node_0][1] < first_time_unit:
                    perseverance_feature_dict[size_hoi_][node_0][1] = first_time_unit
            except:
                perseverance_feature_dict[size_hoi_][node_0] = [[persistence], first_time_unit]
                pass

            for node in node_tuple[1:]:
                HE_node = set(B[node]) & HE_first_time_unit
                HE_co_occur &= HE_node
                HE_union |= HE_node
                for i_feature in range(feature_num):
                    feature_node_list[i_feature].append(feature_node_dict[node][first_time_unit][i_feature])
                try:
                    perseverance_feature_dict[size_hoi_][node][0].append(persistence)
                    if perseverance_feature_dict[size_hoi_][node][1] < first_time_unit:
                        perseverance_feature_dict[size_hoi_][node][1] = first_time_unit
                except:
                    perseverance_feature_dict[size_hoi_][node] = [[persistence], first_time_unit]
                    pass

            len_HE_co_occur = max(len(HE_co_occur), 1)

            HN_common_nbrs = set()
            HE_co_occur_size = []
            for HE in HE_co_occur:
                HN_in_HE = set(B[HE])
                HN_common_nbrs |= HN_in_HE
                HE_co_occur_size.append(len(HN_in_HE))
            sum_HE_co_occur_size = sum(HE_co_occur_size)
            len_HN_common_nbrs = len(HN_common_nbrs) - size_hoi_

            try:
                persistence_feature_list[first_time_unit_][size_hoi_][persistence].append([len_HE_co_occur, len_HE_co_occur / len(HE_union), sum_HE_co_occur_size / sum(len(B[HE]) for HE in HE_union), len_HN_common_nbrs, len_HE_co_occur / len_HN_common_nbrs, sum_HE_co_occur_size / len_HN_common_nbrs, statistics.mean(HE_co_occur_size), entropy(list(Counter(HE_co_occur_size).values()))] + [statistics.mean(feature_node_list[i_feature]) for i_feature in range(feature_num)])
            except:
                pass

        for node, persistence_first_time_unit in perseverance_feature_dict[size_hoi_].items():
            first_time_unit_ = persistence_first_time_unit[1]
            perseverance_feature_dict[size_hoi_][node] = [statistics.mean(persistence_first_time_unit[0]), feature_node_dict[node][first_time_unit_], num_time_units - first_time_unit_ - 1]

    with open(f"./local_analysis/{filepath}_{what_to_do}_{m}_{observed_time_units}_{observed_time_units_features}_{basket_max}_1.pickle", 'wb') as file:
        pickle.dump(persistence_feature_list, file)

    with open(f"./local_analysis/{filepath}_{what_to_do}_{m}_{observed_time_units}_{observed_time_units_features}_{basket_max}_2.pickle", 'wb') as file:
        pickle.dump(perseverance_feature_dict, file)

# local analysis: Group Features vs. Group Persistence
def local_group_group(filepath: str, m: int, size_hoi: int, basket_max: int, observed_time_units_features: int, observed_time_units: int, what_to_do: str):
    feature_dict = {0: 'Co-Occurrence-Count', 1: 'Co-Occurrence-Ratio', 2: 'Co-Occurrence-Ratio-HE-Size', 3: 'Common-Nbrs-Size', 4: 'Common-Nbrs-HE-Count', 5: 'Common-Nbrs-HE-Size', 6: 'HE-Size-Average', 7: 'HE-Size-Entropy'}
    feature_index = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}
    feature_name = {0: r'$\#$', 1: r'$\#/\cup$', 2: r'$\Sigma/\Sigma\cup$', 3: r'$\cap$', 4: r'$\#/\cap$', 5: r'$\Sigma/\cap$', 6: r'$\Sigma/\#$', 7: r'$\mathcal{H}$'}
    feature_num = len(feature_dict)

    if what_to_do[-4:] == 'past':
        what_to_do_ = 'local_analysis_past'
    else:
        what_to_do_ = 'local_analysis'

    with open(f"./local_analysis/{filepath}_{what_to_do_}_{m}_{observed_time_units}_{observed_time_units_features}_{basket_max}_1.pickle", 'rb') as file:
        persistence_feature_list = pickle.load(file)

    x_bin_box = 10
    f_corr = open(f"./txt/{what_to_do}/{filepath}_{what_to_do}_{m}_{observed_time_units}_{observed_time_units_features}_{basket_max}_corr.txt", "w")
    f_mi = open(f"./txt/{what_to_do}/{filepath}_{what_to_do}_{m}_{observed_time_units}_{observed_time_units_features}_{basket_max}_mi.txt", "w")
    f_mi_norm = open(f"./txt/{what_to_do}/{filepath}_{what_to_do}_{m}_{observed_time_units}_{observed_time_units_features}_{basket_max}_mi_norm.txt", "w")

    for i_feature in range(feature_num):
        x = [[] for k in range(1, size_hoi)]
        y = [[] for k in range(1, size_hoi)]
        x_min = [0 for k in range(1, size_hoi)]
        x_max = [0 for k in range(1, size_hoi)]
        y_min = [0 for k in range(1, size_hoi)]
        y_max = [0 for k in range(1, size_hoi)]
        y_bin = [0 for k in range(1, size_hoi)]
        x_edges = [np.array([]) for k in range(1, size_hoi)]
        y_edges = [np.array([]) for k in range(1, size_hoi)]
        feature_list = [[] for k in range(1, size_hoi)]

        for k in range(size_hoi - 1):
            for persistence_feature_list_ in persistence_feature_list:
                for persistence in range(observed_time_units + 1):
                    for pattern in persistence_feature_list_[k + 1][persistence]:
                        x[k].append(pattern[i_feature])
                        y[k].append(persistence)
            x[k] = np.array(x[k])
            y[k] = np.array(y[k])
            f_corr.write(f"{feature_dict[i_feature]}_{k + 2}: {np.corrcoef(x[k], y[k])[0, 1]}\n")
            f_mi.write(f"{feature_dict[i_feature]}_{k + 2}: {mutual_info_score(x[k], y[k])}\n")
            f_mi_norm.write(f"{feature_dict[i_feature]}_{k + 2}: {normalized_mutual_info_score(x[k], y[k])}\n")
            x_min[k] = np.min(x[k])
            if feature_dict[i_feature] not in ['Co-Occurrence-Count', 'Common-Nbrs-Size', 'HE-Size-Average'] or x_min[k] == 0:
                x[k] += 1
                x_min[k] = 1
            x_max[k] = np.max(x[k])
            y_min[k] = np.min(y[k])
            y_max[k] = np.max(y[k])
            y_bin[k] = y_max[k] - y_min[k] + 1
            x_edges[k] = np.array([np.percentile(x[k], i_percentile * 100 / x_bin_box) for i_percentile in range(x_bin_box + 1)])
            vals, idx_start, count = np.unique(x_edges[k], return_counts=True, return_index=True)
            if len(vals) > 1:
                for count_1 in np.where(count > 1)[0]:
                    idx_start_count_1 = idx_start[count_1]
                    count_count_1 = count[count_1]
                    if count_1 < len(vals) - 1:
                        x_edges[k][idx_start_count_1:idx_start_count_1 + count_count_1] = np.geomspace(vals[count_1], vals[count_1 + 1], count_count_1 + 1)[:-1]
                    else:
                        x_edges[k][-(count_count_1 + 1):] = np.geomspace(vals[count_1 - 1], vals[count_1], count_count_1 + 1)
            y_edges[k] = range(y_min[k], y_max[k] + 2)

            feature_list[k] = [[] for i in range(y_bin[k])]
            for persistence in range(observed_time_units + 1):
                i_y = np.searchsorted(y_edges[k], persistence, side='right') - 1
                for persistence_feature_list_ in persistence_feature_list:
                    for pattern in persistence_feature_list_[k + 1][persistence]:
                        if feature_dict[i_feature] not in ['Co-Occurrence-Count', 'Common-Nbrs-Size', 'HE-Size-Average']:
                            pattern[i_feature] += 1
                        feature_list[k][i_y].append(pattern[i_feature])

    plt.rcParams["font.size"] = "27"

    for i_feature in range(feature_num):  #feature_num
        # box pattern plot
        for k in range(1, size_hoi):
            plt.figure(figsize=(4, 3))
            plt.boxplot(feature_list[k - 1], showfliers=False, vert=0)  #, showmeans=True, meanline=True
            y_list = []
            mean_list = []
            median_list = []
            x_max = 0
            for i_x in range(len(feature_list[k - 1])):
                p = feature_list[k - 1][i_x]
                if len(p) > 0:
                    y_list.append(i_x + 1)
                    mean_list.append(np.mean(p))
                    median_list.append(np.median(p))
                    if x_max < np.max(p):
                        x_max = np.max(p)
            plt.plot(mean_list, y_list, label='Mean', linestyle='dashed')
            plt.plot(median_list, y_list, label='Median')
            plt.xscale('log')
            plt.minorticks_off()
            if feature_dict[i_feature] == 'HE-Size-Average':
                if filepath == 'email-Eu':
                    if k > 1:
                        plt.xticks([2 ** i_pow for i_pow in range(3, 5)], [r'$2^{' + str(i_pow) + '}$' for i_pow in range(3, 5)])
                    else:
                        plt.xticks([2 ** i_pow for i_pow in range(2, 5, 2)], [r'$2^{' + str(i_pow) + '}$' for i_pow in range(2, 5, 2)])
                elif filepath in ['contact-high-school', 'contact-primary-school']:
                    if k == 1:
                        plt.xticks([2, 3], [2, 3])
                    elif k == 2:
                        plt.xticks([3, 4], [3, 4])
                    else:
                        plt.xticks([1], [1])
                elif filepath in ['tags-ask-ubuntu', 'tags-math-sx']:
                    if k == 1:
                        plt.xticks([3, 4], [3, 4])
                    elif k == 2:
                        plt.xticks([3, 4, 5], [3, 4, 5])
                    else:
                        plt.xticks([4, 5], [4, 5])
                elif filepath in ['threads-ask-ubuntu', 'threads-math-sx']:
                    if k == 1:
                        plt.xticks([3, 4], [3, 4])
                    elif k == 2:
                        plt.xticks([4, 6], [4, 6])
                    else:
                        plt.xticks([6, 8], [6, 8])
                else:
                    if k == 3:
                        plt.xticks([2 ** i_pow for i_pow in range(3, 5)], [r'$2^{' + str(i_pow) + '}$' for i_pow in range(3, 5)])
                    else:
                        plt.xticks([2 ** i_pow for i_pow in range(2, 4)], [r'$2^{' + str(i_pow) + '}$' for i_pow in range(2, 4)])
            else:
                if x_max >= 10:
                    plt.xticks([1, 10], [r'$10^0$', r'$10^1$'])
                else:
                    plt.xticks([1, x_max], [1, x_max])

            plt.yticks(range(1, observed_time_units + 2, 2), range(0, observed_time_units + 1, 2))
            plt.xlabel(f'{feature_name[feature_index[i_feature]]}')
            plt.ylabel('Persistence')
            plt.gca().spines['right'].set_color('none')
            plt.gca().spines['top'].set_color('none')
            plt.savefig(f"./figure/{what_to_do}/{filepath}_{what_to_do}_{m}_{observed_time_units}_{observed_time_units_features}_{basket_max}_{feature_dict[i_feature]}_{k + 1}.pgf", bbox_inches="tight")
            legend = plt.legend(ncol=2, prop={'size': 9})
            plt.close()
        fig = legend.figure
        fig.canvas.draw()
        bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(f"./figure/{what_to_do}/{what_to_do}_{observed_time_units_features}_{basket_max}_legend.pgf", dpi="figure", bbox_inches=bbox)

# local analysis: Node Features vs. Group Persistence
def local_node_group(filepath: str, m: int, size_hoi: int, basket_max: int, observed_time_units_features: int, observed_time_units: int, what_to_do: str):
    feature_dict = {0: 'Degree', 1: 'Degree-Weighted', 2: 'Occurrence-Count', 3: 'Core-Number', 4: 'Page-Rank', 5: 'Avg-Nbr-Degree', 6: 'Avg-Nbr-Degree-Weighted', 7: 'Clustering-Coefficient'}
    feature_index = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}
    feature_name = {0: r'Degree $(d)$', 1: r'Weighted Degree $(w)$', 2: r'Number of Occurrences $(o)$', 3: r'Core Number $(c)$', 4: r'PageRank $(r)$', 5: r'Avg. Nei. Deg. ($\bar{d}$)', 6: r'Avg. Wei. Nei. Deg. ($\bar{w}$)', 7: r'Local Clus. Coef. $(l)$'}
    feature_num = len(feature_dict)

    if what_to_do[-4:] == 'past':
        what_to_do_ = 'local_analysis_past'
    else:
        what_to_do_ = 'local_analysis'

    with open(f"./local_analysis/{filepath}_{what_to_do_}_{m}_{observed_time_units}_{observed_time_units_features}_{basket_max}_1.pickle", 'rb') as file:
        persistence_feature_list = pickle.load(file)

    x_bin_box = 10
    f_corr = open(f"./txt/{what_to_do}/{filepath}_{what_to_do}_{m}_{observed_time_units}_{observed_time_units_features}_{basket_max}_corr.txt", "w")
    f_mi = open(f"./txt/{what_to_do}/{filepath}_{what_to_do}_{m}_{observed_time_units}_{observed_time_units_features}_{basket_max}_mi.txt", "w")
    f_mi_norm = open(f"./txt/{what_to_do}/{filepath}_{what_to_do}_{m}_{observed_time_units}_{observed_time_units_features}_{basket_max}_mi_norm.txt", "w")

    for i_feature in range(feature_num):
        x = [[] for k in range(1, size_hoi)]
        y = [[] for k in range(1, size_hoi)]
        x_min = [0 for k in range(1, size_hoi)]
        x_max = [0 for k in range(1, size_hoi)]
        y_min = [0 for k in range(1, size_hoi)]
        y_max = [0 for k in range(1, size_hoi)]
        y_bin = [0 for k in range(1, size_hoi)]
        x_edges = [np.array([]) for k in range(1, size_hoi)]
        y_edges = [np.array([]) for k in range(1, size_hoi)]
        feature_list = [[] for k in range(1, size_hoi)]

        for k in range(size_hoi - 1):
            for persistence_feature_list_ in persistence_feature_list:
                for persistence in range(observed_time_units + 1):
                    for pattern in persistence_feature_list_[k + 1][persistence]:
                        x[k].append(pattern[i_feature + 8])
                        y[k].append(persistence)
            x[k] = np.array(x[k])
            y[k] = np.array(y[k])
            f_corr.write(f"{feature_dict[i_feature]}_{k + 2}: {np.corrcoef(x[k], y[k])[0, 1]}\n")
            f_mi.write(f"{feature_dict[i_feature]}_{k + 2}: {mutual_info_score(x[k], y[k])}\n")
            f_mi_norm.write(f"{feature_dict[i_feature]}_{k + 2}: {normalized_mutual_info_score(x[k], y[k])}\n")
            x_min[k] = np.min(x[k])
            if feature_dict[i_feature] in ['Page-Rank', 'Clustering-Coefficient'] or x_min[k] == 0:
                x[k] += 1
                x_min[k] = 1
            x_max[k] = np.max(x[k])
            y_min[k] = np.min(y[k])
            y_max[k] = np.max(y[k])
            y_bin[k] = y_max[k] - y_min[k] + 1
            x_edges[k] = np.array([np.percentile(x[k], i_percentile * 100 / x_bin_box) for i_percentile in range(x_bin_box + 1)])
            vals, idx_start, count = np.unique(x_edges[k], return_counts=True, return_index=True)
            for count_1 in np.where(count > 1)[0]:
                idx_start_count_1 = idx_start[count_1]
                count_count_1 = count[count_1]
                if count_1 < len(vals) - 1:
                    x_edges[k][idx_start_count_1:idx_start_count_1 + count_count_1] = np.geomspace(vals[count_1], vals[count_1 + 1], count_count_1 + 1)[:-1]
                else:
                    x_edges[k][-(count_count_1 + 1):] = np.geomspace(vals[count_1 - 1], vals[count_1], count_count_1 + 1)
            y_edges[k] = range(y_min[k], y_max[k] + 2)

            feature_list[k] = [[] for i in range(y_bin[k])]
            for persistence in range(observed_time_units + 1):
                i_y = np.searchsorted(y_edges[k], persistence, side='right') - 1
                for persistence_feature_list_ in persistence_feature_list:
                    for pattern in persistence_feature_list_[k + 1][persistence]:
                        if feature_dict[i_feature] in ['Page-Rank', 'Clustering-Coefficient']:
                            pattern[i_feature + 8] += 1
                        feature_list[k][i_y].append(pattern[i_feature + 8])

    plt.rcParams["font.size"] = "27"

    for i_feature in range(feature_num):  # feature_num
        # box pattern plot
        for k in range(1, size_hoi):
            plt.figure(figsize=(4, 3))
            plt.boxplot(feature_list[k - 1], showfliers=False, vert=0)  # , showmeans=True, meanline=True
            y_list = []
            mean_list = []
            median_list = []
            for i_x in range(len(feature_list[k - 1])):
                p = feature_list[k - 1][i_x]
                if len(p) > 0:
                    y_list.append(i_x + 1)
                    mean_list.append(np.mean(p))
                    median_list.append(np.median(p))
            plt.plot(mean_list, y_list, label='Mean', linestyle='dashed')
            plt.plot(median_list, y_list, label='Median')
            plt.xscale('log')
            plt.minorticks_off()
            if feature_dict[i_feature] == 'Avg-Nbr-Degree-Weighted':
                if filepath in ['coauth-DBLP', 'coauth-MAG-Geology', 'coauth-MAG-History']:
                    if k > 1:
                        plt.xticks([2 ** i_pow for i_pow in range(3, 6, 2)], [r'$2^{' + str(i_pow) + '}$' for i_pow in range(3, 6, 2)])
                elif filepath == 'contact-high-school':
                    if k == 2:
                        plt.xticks([2 ** i_pow for i_pow in range(3, 6, 2)], [r'$2^{' + str(i_pow) + '}$' for i_pow in range(3, 6, 2)])
                    elif k == 3:
                        plt.xticks([2 ** i_pow for i_pow in range(3, 5)], [r'$2^{' + str(i_pow) + '}$' for i_pow in range(3, 5)])
                elif filepath == 'contact-primary-school':
                    if k == 1:
                        plt.xticks([10 ** i_pow for i_pow in range(1, 3)], [r'$10^{' + str(i_pow) + '}$' for i_pow in range(1, 3)])
                    else:
                        plt.xticks([2 ** i_pow for i_pow in range(4, 6)], [r'$2^{' + str(i_pow) + '}$' for i_pow in range(4, 6)])
                elif filepath in ['tags-math-sx', 'threads-math-sx']:
                    if k > 1:
                        plt.xticks([2 ** i_pow for i_pow in range(6, 9, 2)], [r'$2^{' + str(i_pow) + '}$' for i_pow in range(6, 9, 2)])
                elif filepath in ['tags-ask-ubuntu']:
                    if k > 2:
                        plt.xticks([2 ** i_pow for i_pow in range(6, 9, 2)], [r'$2^{' + str(i_pow) + '}$' for i_pow in range(6, 9, 2)])
            elif filepath in ['contact-high-school', 'contact-primary-school']:
                if k == 3:
                    plt.xticks([10 ** i_pow for i_pow in range(2, 4)], [r'$10^{' + str(i_pow) + '}$' for i_pow in range(2, 4)])

            plt.yticks(range(1, observed_time_units + 2, 2), range(0, observed_time_units + 1, 2))
            plt.xlabel(f'{feature_name[i_feature]}')
            plt.ylabel('Persistence')
            plt.gca().spines['right'].set_color('none')
            plt.gca().spines['top'].set_color('none')
            plt.savefig(f"./figure/{what_to_do}/{filepath}_{what_to_do}_{m}_{observed_time_units}_{observed_time_units_features}_{basket_max}_{feature_dict[i_feature]}_{k + 1}.pgf", bbox_inches="tight")
            legend = plt.legend(ncol=2, prop={'size': 9})
            plt.close()

        fig = legend.figure
        fig.canvas.draw()
        bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(f"./figure/{what_to_do}/{what_to_do}_{observed_time_units_features}_{basket_max}_legend.pgf", dpi="figure", bbox_inches=bbox)

# local analysis: Node Features vs. Node Persistence
def local_node_node(filepath: str, m: int, size_hoi: int, basket_max: int, observed_time_units_features: int, observed_time_units: int, what_to_do: str):
    feature_dict = {0: 'Degree', 1: 'Degree-Weighted', 2: 'Occurrence-Count', 3: 'Core-Number', 4: 'Page-Rank', 5: 'Avg-Nbr-Degree', 6: 'Avg-Nbr-Degree-Weighted', 7: 'Clustering-Coefficient'}
    feature_index = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}
    feature_name = {0: r'Degree $(d)$', 1: r'Weighted Degree $(w)$', 2: r'Number of Occurrences $(o)$', 3: r'Core Number $(c)$', 4: r'PageRank $(r)$', 5: r'Avg. Nei. Deg. ($\bar{d}$)', 6: r'Avg. Wei. Nei. Deg. ($\bar{w}$)', 7: r'Local Clus. Coef. $(l)$'}
    feature_num = len(feature_dict)

    if what_to_do[-4:] == 'past':
        what_to_do_ = 'local_analysis_past'
    else:
        what_to_do_ = 'local_analysis'

    with open(f"./local_analysis/{filepath}_{what_to_do_}_{m}_{observed_time_units}_{observed_time_units_features}_{basket_max}_2.pickle", 'rb') as file:
        perseverance_feature_dict = pickle.load(file)

    x_bin_box = 10
    f_corr = open(f"./txt/{what_to_do}/{filepath}_{what_to_do}_{m}_{observed_time_units}_{observed_time_units_features}_{basket_max}_corr.txt", "w")
    f_mi = open(f"./txt/{what_to_do}/{filepath}_{what_to_do}_{m}_{observed_time_units}_{observed_time_units_features}_{basket_max}_mi.txt", "w")
    f_mi_norm = open(f"./txt/{what_to_do}/{filepath}_{what_to_do}_{m}_{observed_time_units}_{observed_time_units_features}_{basket_max}_mi_norm.txt", "w")

    for i_feature in range(feature_num):
        x = [[] for k in range(1, size_hoi)]
        y = [[] for k in range(1, size_hoi)]
        x_min = [0 for k in range(1, size_hoi)]
        x_max = [0 for k in range(1, size_hoi)]
        y_min = [0 for k in range(1, size_hoi)]
        y_max = [0 for k in range(1, size_hoi)]
        y_bin = [0 for k in range(1, size_hoi)]
        x_edges = [np.array([]) for k in range(1, size_hoi)]
        y_edges = [np.array([]) for k in range(1, size_hoi)]
        feature_list = [[] for k in range(1, size_hoi)]

        for k in range(size_hoi - 1):
            for node, perseverance_feature_list in perseverance_feature_dict[k + 1].items():
                x[k].append(perseverance_feature_list[1][feature_index[i_feature]])
                y[k].append(perseverance_feature_list[0])
            x[k] = np.array(x[k])
            y[k] = np.array(y[k])
            f_corr.write(f"{feature_dict[i_feature]}_{k + 2}: {np.corrcoef(x[k], y[k])[0, 1]}\n")
            f_mi.write(f"{feature_dict[i_feature]}_{k + 2}: {mutual_info_score(x[k], y[k])}\n")
            f_mi_norm.write(f"{feature_dict[i_feature]}_{k + 2}: {normalized_mutual_info_score(x[k], y[k])}\n")
            x_min[k] = np.min(x[k])
            if feature_dict[i_feature] in ['Page-Rank', 'Clustering-Coefficient'] or x_min[k] == 0:
                x[k] += 1
                x_min[k] = 1
            x_max[k] = np.max(x[k])
            y_min[k] = math.floor(np.min(y[k]))
            y_max[k] = math.ceil(np.max(y[k]))
            y_bin[k] = y_max[k] - y_min[k] + 1
            x_edges[k] = np.array([np.percentile(x[k], i_percentile * 100 / x_bin_box) for i_percentile in range(x_bin_box + 1)])
            vals, idx_start, count = np.unique(x_edges[k], return_counts=True, return_index=True)
            for count_1 in np.where(count > 1)[0]:
                idx_start_count_1 = idx_start[count_1]
                count_count_1 = count[count_1]
                if count_1 < len(vals) - 1:
                    x_edges[k][idx_start_count_1:idx_start_count_1 + count_count_1] = np.geomspace(vals[count_1], vals[count_1 + 1], count_count_1 + 1)[:-1]
                else:
                    x_edges[k][-(count_count_1 + 1):] = np.geomspace(vals[count_1 - 1], vals[count_1], count_count_1 + 1)
            y_edges[k] = range(y_min[k], y_max[k] + 2)

            feature_list[k] = [[] for i in range(y_bin[k])]
            for node, perseverance_feature_list in perseverance_feature_dict[k + 1].items():
                perseverance = perseverance_feature_list[0]
                if feature_dict[i_feature] in ['Page-Rank', 'Clustering-Coefficient']:
                    pattern = perseverance_feature_list[1][feature_index[i_feature]] + 1
                else:
                    pattern = perseverance_feature_list[1][feature_index[i_feature]]
                i_y = np.searchsorted(y_edges[k], perseverance, side='right') - 1
                feature_list[k][i_y].append(pattern)

    plt.rcParams["font.size"] = "27"

    for i_feature in range(feature_num):  # feature_num
        # box pattern plot
        for k in range(1, size_hoi):
            plt.figure(figsize=(4, 3))
            plt.boxplot(feature_list[k - 1], showfliers=False, vert=0)  # , showmeans=True, meanline=True
            y_list = []
            mean_list = []
            median_list = []
            for i_x in range(len(feature_list[k - 1])):
                p = feature_list[k - 1][i_x]
                if len(p) > 0:
                    y_list.append(i_x + 1)
                    mean_list.append(np.mean(p))
                    median_list.append(np.median(p))
            plt.plot(mean_list, y_list, label='Mean', linestyle='dashed')
            plt.plot(median_list, y_list, label='Median')
            plt.xscale('log')
            plt.minorticks_off()
            if feature_dict[i_feature] == 'Avg-Nbr-Degree-Weighted':
                if filepath in ['coauth-DBLP', 'coauth-MAG-Geology']:
                    if k > 1:
                        plt.xticks([2 ** i_pow for i_pow in range(3, 6, 2)], [r'$2^{' + str(i_pow) + '}$' for i_pow in range(3, 6, 2)])
                elif filepath == 'coauth-MAG-History':
                    if k == 2:
                        plt.xticks([2 ** i_pow for i_pow in range(2, 4)], [r'$2^{' + str(i_pow) + '}$' for i_pow in range(2, 4)])
                    elif k > 2:
                        plt.xticks([2 ** i_pow for i_pow in range(3, 6, 2)], [r'$2^{' + str(i_pow) + '}$' for i_pow in range(3, 6, 2)])
                elif filepath == 'contact-high-school':
                    plt.xticks([2 ** i_pow for i_pow in range(3, 6, 2)], [r'$2^{' + str(i_pow) + '}$' for i_pow in range(3, 6, 2)])
                elif filepath == 'contact-primary-school':
                    if k == 1:
                        plt.xticks([10 ** i_pow for i_pow in range(1, 3)], [r'$10^{' + str(i_pow) + '}$' for i_pow in range(1, 3)])
                    elif k == 2:
                        plt.xticks([2 ** i_pow for i_pow in range(3, 6, 2)], [r'$2^{' + str(i_pow) + '}$' for i_pow in range(3, 6, 2)])
                    else:
                        plt.xticks([2 ** i_pow for i_pow in range(4, 6)], [r'$2^{' + str(i_pow) + '}$' for i_pow in range(4, 6)])
                elif filepath == 'NDC-substances':
                    if k > 2:
                        plt.xticks([2 ** i_pow for i_pow in range(3, 6, 2)], [r'$2^{' + str(i_pow) + '}$' for i_pow in range(3, 6, 2)])
                elif filepath == 'tags-math-sx':
                    if k > 1:
                        plt.xticks([2 ** i_pow for i_pow in range(6, 10, 3)], [r'$2^{' + str(i_pow) + '}$' for i_pow in range(6, 10, 3)])
                elif filepath == 'threads-math-sx':
                    if k > 1:
                        plt.xticks([2 ** i_pow for i_pow in range(4, 9, 4)], [r'$2^{' + str(i_pow) + '}$' for i_pow in range(4, 9, 4)])
                elif filepath == 'tags-ask-ubuntu':
                    if k > 2:
                        plt.xticks([2 ** i_pow for i_pow in range(6, 11, 4)], [r'$2^{' + str(i_pow) + '}$' for i_pow in range(6, 11, 4)])
            else:
                if filepath in ['coauth-DBLP', 'coauth-MAG-Geology', 'tags-math-sx']:
                    if k > 1:
                        plt.xticks([2 ** i_pow for i_pow in range(3, 6, 2)], [r'$2^{' + str(i_pow) + '}$' for i_pow in range(3, 6, 2)])
                elif filepath == 'coauth-MAG-History':
                    if k == 2:
                        plt.xticks([2 ** i_pow for i_pow in range(3, 6, 2)], [r'$2^{' + str(i_pow) + '}$' for i_pow in range(3, 6, 2)])
                elif filepath == 'contact-primary-school':
                    if k == 2:
                        plt.xticks([2 ** i_pow for i_pow in range(5, 8, 2)], [r'$2^{' + str(i_pow) + '}$' for i_pow in range(5, 8, 2)])
                elif filepath in ['tags-ask-ubuntu', 'threads-math-sx']:
                    if k > 1:
                        plt.xticks([2 ** i_pow for i_pow in range(2, 5, 2)], [r'$2^{' + str(i_pow) + '}$' for i_pow in range(2, 5, 2)])
                elif filepath == 'threads-ask-ubuntu':
                    # if k > 1:
                    plt.xticks([2 ** i_pow for i_pow in range(2, 5, 2)], [r'$2^{' + str(i_pow) + '}$' for i_pow in range(2, 5, 2)])

            plt.yticks(range(1, observed_time_units + 2, 2), range(0, observed_time_units + 1, 2))
            plt.xlabel(f'{feature_name[i_feature]}')
            plt.ylabel(r'$k$-node Persistence', fontsize=26)
            plt.gca().spines['right'].set_color('none')
            plt.gca().spines['top'].set_color('none')
            plt.savefig(f"./figure/{what_to_do}/{filepath}_{what_to_do}_{m}_{observed_time_units}_{observed_time_units_features}_{basket_max}_{feature_dict[i_feature]}_{k + 1}.pgf", bbox_inches="tight")
            legend = plt.legend(ncol=2, prop={'size': 9})
            plt.close()

        fig = legend.figure
        fig.canvas.draw()
        bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(f"./figure/{what_to_do}/{what_to_do}_{observed_time_units_features}_{basket_max}_legend.pgf", dpi="figure", bbox_inches=bbox)

# global analysis stat
def global_stat(basket_max: int):
    filepath = ['NDC-substances', 'NDC-classes', 'email-Enron', 'email-Eu', 'contact-high-school', 'contact-primary-school', 'tags-ask-ubuntu', 'tags-math-sx', 'threads-ask-ubuntu', 'threads-math-sx', 'coauth-DBLP', 'coauth-MAG-Geology', 'coauth-MAG-History']
    m = [59, 59, 43, 38, 84, 108, 104, 89, 92, 85, 83, 219, 219]
    observed_time_units = [15, 20, 20, 15, 30, 30, 30, 30, 30, 30, 30, 30, 30]
    df = pd.read_csv(f"./txt/global_analysis/{filepath[0]}_{m[0]}_{observed_time_units[0]}_{basket_max}.csv", index_col=0)
    for i in range(1, len(m)):
        df = pd.concat([df, pd.read_csv(f"./txt/global_analysis/{filepath[i]}_{m[i]}_{observed_time_units[i]}_{basket_max}.csv", index_col=0)])
    df.to_csv(f"./txt/global_analysis/global_stat.csv", sep=',', index=False)

# local analysis stat
def local_stat(basket_max: int, observed_time_units_features: int, observed_time_units: int, what_to_do: str, read: str):
    filepath = ['NDC-substances', 'NDC-classes', 'email-Enron', 'email-Eu', 'contact-high-school', 'contact-primary-school', 'tags-ask-ubuntu', 'tags-math-sx', 'threads-ask-ubuntu', 'threads-math-sx', 'coauth-DBLP', 'coauth-MAG-Geology', 'coauth-MAG-History']
    m = [59, 59, 43, 38, 84, 108, 104, 89, 92, 85, 83, 219, 219]

    index_list = []
    data_dict = [[] for i in range(len(filepath))]
    for stat in ['corr', 'mi', 'mi_norm']:
        for line in open(f"./txt/{read}/{filepath[0]}_{read}_{m[0]}_{observed_time_units}_{observed_time_units_features}_{basket_max}_{stat}.txt", "r"):
            line_split = line.rstrip('\n').split(': ')
            index_list.append(line_split[0])
            data_dict[0].append(line_split[1])
        for i in range(1, len(m)):
            try:
                for line in open(f"./txt/{read}/{filepath[i]}_{read}_{m[i]}_{observed_time_units}_{observed_time_units_features}_{basket_max}_{stat}.txt", "r"):
                    line_split = line.rstrip('\n').split(': ')
                    data_dict[i].append(line_split[1])
            except:
                data_dict[i] = ['' for j in range(len(index_list))]
                pass
    data_dict = {filepath[i] + '_' + str(m[i]): data_dict[i] for i in range(len(m))}
    df = pd.DataFrame(data_dict, index=index_list)
    df.to_csv(f"./txt/{what_to_do}/{what_to_do}_{observed_time_units}_{observed_time_units_features}_{basket_max}_{read}.csv")

# prediction model 1
def pred_1(filepath: str, m: int, size_hoi: int, basket_max: int, observed_time_units_features: int, observed_time_units: int, what_to_do: str):
    features = ['Co-Occurrence-Count', 'Co-Occurrence-Ratio', 'Co-Occurrence-Ratio-HE-Size', 'Common-Nbrs-Size', 'Common-Nbrs-HE-Count', 'Common-Nbrs-HE-Size', 'HE-Size-Average', 'HE-Size-Entropy', 'Degree', 'Degree-Weighted', 'Occurrence-Count', 'Core-Number', 'Page-Rank', 'Avg-Nbr-Degree', 'Avg-Nbr-Degree-Weighted', 'Clustering-Coefficient']
    len_features = len(features)

    num_time_units = m - observed_time_units - observed_time_units_features + 1

    if what_to_do[-4:] == 'past':
        what_to_do_ = 'local_analysis_past'
    else:
        what_to_do_ = 'local_analysis'

    with open(f"./local_analysis/{filepath}_{what_to_do_}_{m}_{observed_time_units}_{observed_time_units_features}_{basket_max}_1.pickle", 'rb') as file:
        persistence_feature_list = pickle.load(file)

    df_result = []
    for size_hoi_ in tqdm(range(1, size_hoi)):
        X = []
        y = []
        for time_unit in range(num_time_units):
            for i_persistence in range(observed_time_units + 1):
                for x in persistence_feature_list[time_unit][size_hoi_][i_persistence]:
                    X.append(x)
                    y.append(i_persistence)
        X = pd.DataFrame(X, columns=features)
        y = pd.Series(y, name='Persistence')

        kfold_n = 3
        kf = KFold(n_splits=kfold_n, shuffle=True, random_state=1)

        # LinearRegression
        regr = LinearRegression()
        rfecv = RFECV(estimator=regr, cv=kf, scoring='neg_root_mean_squared_error')
        rfecv.fit(X, y)
        df_scores = pd.DataFrame({'LR': -rfecv.grid_scores_}, index=range(1, len_features + 1))
        df_ranking = pd.DataFrame({'LR_r': rfecv.ranking_}, index=features)
        df_ranking['LR_s'] = rfecv.support_
        df_result.append([filepath, size_hoi_ + 1, 'LR', rfecv.score(X, y), rfecv.n_features_, df_scores['LR'].min()])

        # RandomForestRegressor
        regr = RandomForestRegressor(n_estimators=30, max_depth=10, random_state=1)
        rfecv = RFECV(estimator=regr, cv=kf, scoring='neg_root_mean_squared_error')
        rfecv.fit(X, y)
        df_scores['RF'] = -rfecv.grid_scores_
        df_ranking['RF_r'] = rfecv.ranking_
        df_ranking['RF_s'] = rfecv.support_
        df_result.append([filepath, size_hoi_ + 1, 'RF', rfecv.score(X, y), rfecv.n_features_, df_scores['RF'].min()])
        regr.fit(X, y)
        regr_feature_importances = regr.feature_importances_

        # LinearSVR
        regr = LinearSVR(random_state=1, max_iter=10000)
        rfecv = RFECV(estimator=regr, cv=kf, scoring='neg_root_mean_squared_error')
        rfecv.fit(X, y)
        df_scores['SVR'] = -rfecv.grid_scores_
        df_result.append([filepath, size_hoi_ + 1, 'SVR', rfecv.score(X, y), rfecv.n_features_, df_scores['SVR'].min()])

        coef_list = []
        std_err_list = []
        p_values_list = []
        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X.loc[train_index, :], X.loc[test_index, :]
            y_train, y_test = y.loc[train_index], y.loc[test_index]

            # Mean
            y_pred = np.full(y_test.shape, y.loc[train_index].mean())
            df_result.append([filepath, size_hoi_ + 1, 'Mean', r2_score(y_test, y_pred), 0, mean_squared_error(y_test, y_pred, squared=False)])

            # MLP
            regr = MLPRegressor(random_state=1, hidden_layer_sizes=(2 * X.shape[1] + 1,), activation='tanh', learning_rate='adaptive', max_iter=1000, early_stopping=True)
            regr.fit(X_train, y_train)
            df_result.append([filepath, size_hoi_ + 1, 'MLP', regr.score(X, y), len_features, mean_squared_error(y_test, regr.predict(X_test), squared=False)])

            results = sm.OLS(y_train, X_train).fit()
            coef_list.append(results.params.values)
            std_err_list.append(results.bse.values)
            p_values_list.append(results.pvalues.values)

        pd.DataFrame({'coef': np.mean(coef_list, axis=0), 'std_err': np.mean(std_err_list, axis=0), 'p_values': np.mean(p_values_list, axis=0)}, index=range(1, len_features + 1)).to_csv(f"./txt/pred/lr/{filepath}_{what_to_do}_{m}_{observed_time_units}_{observed_time_units_features}_{basket_max}_{size_hoi_ + 1}_lr.csv")
        df_scores.to_csv(f"./txt/pred/rmse/{filepath}_{what_to_do}_{m}_{observed_time_units}_{observed_time_units_features}_{basket_max}_{size_hoi_ + 1}_rmse.csv")
        df_ranking.to_csv(f"./txt/pred/ranking/{filepath}_{what_to_do}_{m}_{observed_time_units}_{observed_time_units_features}_{basket_max}_{size_hoi_ + 1}_ranking.csv")
        pd.DataFrame({'RF': regr_feature_importances}, index=features).to_csv(f"./txt/pred/feature_importance/{filepath}_{what_to_do}_{m}_{observed_time_units}_{observed_time_units_features}_{basket_max}_{size_hoi_ + 1}_feature_importance.csv")
        pd.DataFrame(df_result, columns=['Dataset', 'k', 'Model', 'R^2', 'Num_RMSE', 'RMSE']).groupby(['Dataset', 'k', 'Model']).mean().to_csv(f"./txt/pred/predictability/{filepath}_{what_to_do}_{m}_{observed_time_units}_{observed_time_units_features}_{basket_max}_predictability.csv")

# prediction model 2
def pred_2(filepath: str, m: int, size_hoi: int, basket_max: int, observed_time_units_features: int, observed_time_units: int, what_to_do: str):
    features = ['Degree', 'Degree-Weighted', 'Occurrence-Count', 'Core-Number', 'Page-Rank', 'Avg-Nbr-Degree', 'Avg-Nbr-Degree-Weighted', 'Clustering-Coefficient']
    len_features = len(features)

    if what_to_do[-4:] == 'past':
        what_to_do_ = 'local_analysis_past'
    else:
        what_to_do_ = 'local_analysis'

    with open(f"./local_analysis/{filepath}_{what_to_do_}_{m}_{observed_time_units}_{observed_time_units_features}_{basket_max}_2.pickle", 'rb') as file:
        perseverance_feature_dict = pickle.load(file)

    df_result = []
    for size_hoi_ in tqdm(range(1, size_hoi)):
        X = []
        y = []
        for node, persistence_first_time_unit in perseverance_feature_dict[size_hoi_].items():
            X.append(persistence_first_time_unit[1])
            y.append(persistence_first_time_unit[0])
        X = pd.DataFrame(X, columns=features)
        y = pd.Series(y, name='Perseverance')

        kfold_n = 5
        kf = KFold(n_splits=kfold_n, shuffle=True, random_state=1)

        # LinearRegression
        regr = LinearRegression()
        rfecv = RFECV(estimator=regr, cv=kf, scoring='neg_root_mean_squared_error')
        rfecv.fit(X, y)
        df_scores = pd.DataFrame({'LR': -rfecv.grid_scores_}, index=range(1, len_features + 1))
        df_ranking = pd.DataFrame({'LR_r': rfecv.ranking_}, index=features)
        df_ranking['LR_s'] = rfecv.support_
        df_result.append([filepath, size_hoi_ + 1, 'LR', rfecv.score(X, y), rfecv.n_features_, df_scores['LR'].min()])

        # RandomForestRegressor
        regr = RandomForestRegressor(n_estimators=30, max_depth=10, random_state=1)
        rfecv = RFECV(estimator=regr, cv=kf, scoring='neg_root_mean_squared_error')
        rfecv.fit(X, y)
        df_scores['RF'] = -rfecv.grid_scores_
        df_ranking['RF_r'] = rfecv.ranking_
        df_ranking['RF_s'] = rfecv.support_
        df_result.append([filepath, size_hoi_ + 1, 'RF', rfecv.score(X, y), rfecv.n_features_, df_scores['RF'].min()])
        regr.fit(X, y)
        regr_feature_importances = regr.feature_importances_

        # LinearSVR
        regr = LinearSVR(random_state=1, max_iter=10000)
        rfecv = RFECV(estimator=regr, cv=kf, scoring='neg_root_mean_squared_error')
        rfecv.fit(X, y)
        df_scores['SVR'] = -rfecv.grid_scores_
        df_result.append([filepath, size_hoi_ + 1, 'SVR', rfecv.score(X, y), rfecv.n_features_, df_scores['SVR'].min()])

        coef_list = []
        p_values_list = []
        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X.loc[train_index, :], X.loc[test_index, :]
            y_train, y_test = y.loc[train_index], y.loc[test_index]

            # Mean
            y_pred = np.full(y_test.shape, y.loc[train_index].mean())
            df_result.append([filepath, size_hoi_ + 1, 'Mean', r2_score(y_test, y_pred), 0, mean_squared_error(y_test, y_pred, squared=False)])

            # MLP
            regr = MLPRegressor(random_state=1, hidden_layer_sizes=(2 * X.shape[1] + 1,), activation='tanh', learning_rate='adaptive', max_iter=1000, early_stopping=True)
            regr.fit(X_train, y_train)
            df_result.append([filepath, size_hoi_ + 1, 'MLP', regr.score(X, y), len_features, mean_squared_error(y_test, regr.predict(X_test), squared=False)])

            results = sm.OLS(y_train, X_train).fit()
            coef_list.append(results.params.values)
            p_values_list.append(results.pvalues.values)

        pd.DataFrame({'coef': np.mean(coef_list, axis=0), 'p_values': np.mean(p_values_list, axis=0)}, index=range(1, len_features + 1)).to_csv(f"./txt/pred/lr/{filepath}_{what_to_do}_{m}_{observed_time_units}_{observed_time_units_features}_{basket_max}_{size_hoi_ + 1}_lr.csv")
        df_scores.to_csv(f"./txt/pred/rmse/{filepath}_{what_to_do}_{m}_{observed_time_units}_{observed_time_units_features}_{basket_max}_{size_hoi_ + 1}_rmse.csv")
        df_ranking.to_csv(f"./txt/pred/ranking/{filepath}_{what_to_do}_{m}_{observed_time_units}_{observed_time_units_features}_{basket_max}_{size_hoi_ + 1}_ranking.csv")
        pd.DataFrame({'RF': regr_feature_importances}, index=features).to_csv(f"./txt/pred/feature_importance/{filepath}_{what_to_do}_{m}_{observed_time_units}_{observed_time_units_features}_{basket_max}_{size_hoi_ + 1}_feature_importance.csv")
        pd.DataFrame(df_result, columns=['Dataset', 'k', 'Model', 'R^2', 'Num_RMSE', 'RMSE']).groupby(['Dataset', 'k', 'Model']).mean().to_csv(f"./txt/pred/predictability/{filepath}_{what_to_do}_{m}_{observed_time_units}_{observed_time_units_features}_{basket_max}_predictability.csv")

# pred_feature_selection
def pred_feature_selection(size_hoi: int, basket_max: int, observed_time_units_features: int, observed_time_units: int, what_to_do: str):
    filepath = ['NDC-classes', 'NDC-substances', 'email-Enron', 'email-Eu', 'contact-high-school', 'contact-primary-school', 'tags-ask-ubuntu', 'tags-math-sx', 'threads-ask-ubuntu', 'threads-math-sx', 'coauth-DBLP', 'coauth-MAG-Geology', 'coauth-MAG-History']
    m = [59, 59, 43, 38, 84, 108, 104, 89, 92, 85, 83, 219, 219]

    if what_to_do[-4:] == 'past':
        what_to_do_ = 'pred_1_past'
    else:
        what_to_do_ = 'pred_1'

    plt.rcParams["font.size"] = "32"
    plt.rcParams["ytick.major.pad"] = "8"
    plt.figure(figsize=(5, 4))
    for size_hoi_ in range(1, size_hoi):
        df_scores = pd.read_csv(f"./txt/pred/rmse/{filepath[0]}_{what_to_do_}_{m[0]}_{observed_time_units}_{observed_time_units_features}_{basket_max}_{size_hoi_ + 1}_rmse.csv", index_col=0)
        for i in range(1, len(m)):
            df_scores_ = pd.read_csv(f"./txt/pred/rmse/{filepath[i]}_{what_to_do_}_{m[i]}_{observed_time_units}_{observed_time_units_features}_{basket_max}_{size_hoi_ + 1}_rmse.csv", index_col=0)
            df_scores = pd.concat([df_scores, df_scores_])
        df_scores = df_scores.groupby(df_scores.index).mean()
        if size_hoi_ == 1:
            plt.plot(df_scores.index, df_scores.RF, label=r'$|S|=2$', marker='o')
        elif size_hoi_ == 2:
            plt.plot(df_scores.index, df_scores.RF, '--', label=r'$|S|=3$', marker='x')
        else:
            plt.plot(df_scores.index, df_scores.RF, ':', label=r'$|S|=4$', marker='^')
    plt.xlabel(r"\# Features Selected")
    plt.ylabel("RMSE")
    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    plt.xticks([4, 8, 12, 16], [4, 8, 12, 16])
    plt.savefig(f"./figure/pred/rmse/{what_to_do}_1_{observed_time_units}_{observed_time_units_features}_{basket_max}.pgf", bbox_inches="tight")
    plt.close()

    if what_to_do[-4:] == 'past':
        what_to_do_ = 'pred_2_past'
    else:
        what_to_do_ = 'pred_2'

    plt.figure(figsize=(5, 4))
    for size_hoi_ in range(1, size_hoi):
        df_scores = pd.read_csv(f"./txt/pred/rmse/{filepath[0]}_{what_to_do_}_{m[0]}_{observed_time_units}_{observed_time_units_features}_{basket_max}_{size_hoi_ + 1}_rmse.csv", index_col=0)
        for i in range(1, len(m)):
            df_scores = pd.concat([df_scores, pd.read_csv(f"./txt/pred/rmse/{filepath[i]}_{what_to_do_}_{m[i]}_{observed_time_units}_{observed_time_units_features}_{basket_max}_{size_hoi_ + 1}_rmse.csv", index_col=0)])
        df_scores = df_scores.groupby(df_scores.index).mean()
        if size_hoi_ == 1:
            plt.plot(df_scores.index, df_scores.RF, label=r'$|S|=2$', marker='o')
        elif size_hoi_ == 2:
            plt.plot(df_scores.index, df_scores.RF, '--', label=r'$|S|=3$', marker='x')
        else:
            plt.plot(df_scores.index, df_scores.RF, ':', label=r'$|S|=4$', marker='^')
    plt.xlabel(r"\# Features Selected")
    plt.ylabel("RMSE")
    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    plt.xticks([2, 4, 6, 8], [2, 4, 6, 8])
    plt.savefig(f"./figure/pred/rmse/{what_to_do}_2_{observed_time_units}_{observed_time_units_features}_{basket_max}_2.pgf", bbox_inches="tight")
    legend = plt.legend(ncol=1, prop={'size': 10})
    plt.close()
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(f"./figure/pred/rmse/{what_to_do}_{observed_time_units}_{observed_time_units_features}_{basket_max}_legend.pgf", dpi="figure", bbox_inches=bbox)

# predictability
def predictability(basket_max: int, observed_time_units: int):
    filepath = ['NDC-substances', 'NDC-classes', 'email-Enron', 'email-Eu', 'contact-high-school', 'contact-primary-school', 'tags-ask-ubuntu', 'tags-math-sx', 'threads-ask-ubuntu', 'threads-math-sx', 'coauth-DBLP', 'coauth-MAG-Geology', 'coauth-MAG-History']
    m = [59, 59, 43, 38, 84, 108, 104, 89, 92, 85, 83, 219, 219]

    for what_to_do in ['pred_1']:#, 'pred_1_past']:
        for observed_time_units_features in [5]:#, 3, 1]:
            for i in range(len(m)):
                df_pred_ = pd.read_csv(f"./txt/pred/predictability/{filepath[i]}_{what_to_do}_{m[i]}_{observed_time_units}_{observed_time_units_features}_{basket_max}_predictability.csv")
                df_pred_['method'] = what_to_do
                df_pred_['time_units'] = observed_time_units_features
                try:
                    df_pred = pd.concat([df_pred, df_pred_])
                except:
                    df_pred = df_pred_

        df_pred[['method', 'time_units', 'Dataset', 'k', 'Model', 'R^2', 'RMSE']].to_csv(f"./txt/pred/predictability/predictability_1.csv")

    for what_to_do in ['pred_2', 'pred_2_past']:
        for observed_time_units_features in [5, 3, 1]:
            for i in range(len(m)):
                df_pred_ = pd.read_csv(f"./txt/pred/predictability/{filepath[i]}_{what_to_do}_{m[i]}_{observed_time_units}_{observed_time_units_features}_{basket_max}_predictability.csv")
                df_pred_['method'] = what_to_do
                df_pred_['time_units'] = observed_time_units_features
                try:
                    df_pred = pd.concat([df_pred, df_pred_])
                except:
                    df_pred = df_pred_

        df_pred[['method', 'time_units', 'Dataset', 'k', 'Model', 'R^2', 'RMSE']].to_csv(f"./txt/pred/predictability/predictability_2.csv")