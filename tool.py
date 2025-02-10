import pandas as pd
import os
import networkx as nx
import scipy
from scipy.linalg import expm, sqrtm
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import numpy as np
np.set_printoptions(threshold=np.inf)

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from configparser import ConfigParser
from scipy.spatial.distance import euclidean


cfg = ConfigParser()
cfg.read("./params.ini")  # Load the configuration file named "params.ini"

node_to_module = [1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 4, 1, 7, 6, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4, 7, 3, 4, 6, 7, 7, 7, 7, 7, 7, 2, 4, 4, 2, 2, 2, 3, 4, 4, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 7, 7, 6, 7, 7, 7, 6, 7, 7, 7, 7, 7, 6, 6, 7, 7, 6, 4, 6, 3, 6, 6, 6, 6, 6, 6, 7, 5, 6, 5, 6, 5, 5, 7, 3, 3, 6, 6, 2, 2, 2, 2, 2, 2, 4, 4, 2, 4, 4, 5, 6, 4, 4, 4, 2, 3, 3, 5, 1, 5, 1, 5, 7, 2, 2, 1, 1, 7, 7, 7, 5, 7, 6, 5, 5, 3, 3, 3, 4, 3, 3, 1, 3, 6, 6, 3, 4, 4, 6, 7, 7, 1, 1, 1, 1, 1, 3, 1, 1, 1, 7, 7, 1, 5, 7, 5, 4, 2, 4, 6, 6, 5, 2, 2, 2, 7, 7, 4, 6, 7, 1, 3, 1, 1, 1, 1, 1, 2, 2, 3, 3, 7, 1, 7, 6, 1, 3, 1, 1, 1, 1, 1, 1, 2, 4, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 2, 4, 4, 2, 2, 2, 3, 4, 4, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 4, 6, 4, 4, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 4, 6, 6, 6, 6, 6, 6, 6, 6, 7, 5, 7, 5, 6, 5, 5, 7, 3, 3, 6, 7, 4, 2, 2, 2, 2, 2, 4, 4, 2, 4, 4, 5, 6, 7, 4, 4, 4, 3, 3, 5, 1, 5, 1, 5, 7, 2, 2, 1, 1, 7, 7, 7, 5, 7, 7, 5, 5, 3, 3, 3, 2, 4, 3, 1, 3, 6, 6, 1, 4, 4, 7, 7, 7, 1, 1, 1, 1, 1, 3, 1, 1, 1, 7, 7, 1, 5, 5, 5, 4, 2, 4, 7, 6, 5, 2, 2, 2, 7, 7, 4, 6, 7]
hcp_to_yeo_mapping = {180: 1.0, 181: 1.0, 182: 1.0, 183: 1.0, 184: 1.0, 185: 1.0, 186: 1.0, 187: 2.0, 188: 2.0,
                      189: 3.0, 190: 3.0, 191: 4.0, 192: 1.0, 193: 7.0, 194: 6.0, 195: 1.0, 196: 1.0, 197: 1.0,
                      198: 1.0, 199: 1.0, 200: 1.0, 201: 1.0, 202: 1.0, 203: 2.0, 204: 4.0, 205: 7.0, 206: 3.0,
                      207: 4.0, 208: 6.0, 209: 7.0, 210: 7.0, 211: 7.0, 212: 7.0, 213: 7.0, 214: 7.0, 215: 2.0,
                      216: 4.0, 217: 4.0, 218: 2.0, 219: 2.0, 220: 2.0, 221: 3.0, 222: 4.0, 223: 4.0, 224: 3.0,
                      225: 3.0, 226: 3.0, 227: 3.0, 228: 3.0, 229: 3.0, 230: 2.0, 231: 2.0, 232: 2.0, 233: 2.0,
                      234: 2.0, 235: 2.0, 236: 4.0, 237: 4.0, 238: 4.0, 239: 4.0, 240: 7.0, 241: 7.0, 242: 6.0,
                      243: 7.0, 244: 7.0, 245: 7.0, 246: 6.0, 247: 7.0, 248: 7.0, 249: 7.0, 250: 7.0, 251: 7.0,
                      252: 6.0, 253: 6.0, 254: 7.0, 255: 7.0, 256: 6.0, 257: 4.0, 258: 6.0, 259: 3.0, 260: 6.0,
                      261: 6.0, 262: 6.0, 263: 6.0, 264: 6.0, 265: 6.0, 266: 7.0, 267: 5.0, 268: 6.0, 269: 5.0,
                      270: 6.0, 271: 5.0, 272: 5.0, 273: 7.0, 274: 3.0, 275: 3.0, 276: 6.0, 277: 6.0, 278: 2.0,
                      279: 2.0, 280: 2.0, 281: 2.0, 282: 2.0, 283: 2.0, 284: 4.0, 285: 4.0, 286: 2.0, 287: 4.0,
                      288: 4.0, 289: 5.0, 290: 6.0, 291: 4.0, 292: 4.0, 293: 4.0, 294: 2.0, 295: 3.0, 296: 3.0,
                      297: 5.0, 298: 1.0, 299: 5.0, 300: 1.0, 301: 5.0, 302: 7.0, 303: 2.0, 304: 2.0, 305: 1.0,
                      306: 1.0, 307: 7.0, 308: 7.0, 309: 7.0, 310: 5.0, 311: 7.0, 312: 6.0, 313: 5.0, 314: 5.0,
                      315: 3.0, 316: 3.0, 317: 3.0, 318: 4.0, 319: 3.0, 320: 3.0, 321: 1.0, 322: 3.0, 323: 6.0,
                      324: 6.0, 325: 3.0, 326: 4.0, 327: 4.0, 328: 6.0, 329: 7.0, 330: 7.0, 331: 1.0, 332: 1.0,
                      333: 1.0, 334: 1.0, 335: 1.0, 336: 3.0, 337: 1.0, 338: 1.0, 339: 1.0, 340: 7.0, 341: 7.0,
                      342: 1.0, 343: 5.0, 344: 7.0, 345: 5.0, 346: 4.0, 347: 2.0, 348: 4.0, 349: 6.0, 350: 6.0,
                      351: 5.0, 352: 2.0, 353: 2.0, 354: 2.0, 355: 7.0, 356: 7.0, 357: 4.0, 358: 6.0, 359: 7.0,
                      0: 8.0, 1: 10.0, 2: 8.0, 3: 8.0, 4: 8.0, 5: 8.0, 6: 8.0, 7: 9.0, 8: 9.0, 9: 10.0, 10: 10.0,
                      11: 14.0, 12: 8.0, 13: 14.0, 14: 13.0, 15: 8.0, 16: 10.0, 17: 8.0, 18: 8.0, 19: 8.0, 20: 8.0,
                      21: 8.0, 22: 8.0, 23: 9.0, 24: 11.0, 25: 14.0, 26: 14.0, 27: 14.0, 28: 13.0, 29: 14.0,
                      30: 14.0, 31: 14.0, 32: 14.0, 33: 14.0, 34: 14.0, 35: 9.0, 36: 11.0, 37: 11.0, 38: 9.0,
                      39: 9.0, 40: 9.0, 41: 10.0, 42: 11.0, 43: 11.0, 44: 10.0, 45: 10.0, 46: 10.0, 47: 10.0,
                      48: 10.0, 49: 10.0, 50: 9.0, 51: 9.0, 52: 9.0, 53: 9.0, 54: 9.0, 55: 9.0, 56: 11.0, 57: 13.0,
                      58: 11.0, 59: 11.0, 60: 14.0, 61: 14.0, 62: 14.0, 63: 14.0, 64: 14.0, 65: 14.0, 66: 14.0,
                      67: 14.0, 68: 14.0, 69: 14.0, 70: 14.0, 71: 14.0, 72: 13.0, 73: 14.0, 74: 14.0, 75: 14.0,
                      76: 14.0, 77: 11.0, 78: 13.0, 79: 13.0, 80: 13.0, 81: 13.0, 82: 13.0, 83: 13.0, 84: 13.0,
                      85: 13.0, 86: 14.0, 87: 12.0, 88: 14.0, 89: 12.0, 90: 13.0, 91: 12.0, 92: 12.0, 93: 14.0,
                      94: 10.0, 95: 10.0, 96: 13.0, 97: 14.0, 98: 11.0, 99: 9.0, 100: 9.0, 101: 9.0, 102: 9.0,
                      103: 9.0, 104: 11.0, 105: 11.0, 106: 9.0, 107: 11.0, 108: 11.0, 109: 12.0, 110: 13.0,
                      111: 14.0, 112: 11.0, 113: 11.0, 114: 11.0, 115: 10.0, 116: 10.0, 117: 12.0, 118: 8.0,
                      119: 12.0, 120: 8.0, 121: 12.0, 122: 14.0, 123: 9.0, 124: 9.0, 125: 8.0, 126: 8.0, 127: 14.0,
                      128: 14.0, 129: 14.0, 130: 12.0, 131: 14.0, 132: 14.0, 133: 12.0, 134: 12.0, 135: 10.0,
                      136: 10.0, 137: 10.0, 138: 9.0, 139: 11.0, 140: 10.0, 141: 8.0, 142: 10.0, 143: 13.0,
                      144: 13.0, 145: 8.0, 146: 11.0, 147: 11.0, 148: 14.0, 149: 14.0, 150: 14.0, 151: 8.0,
                      152: 8.0, 153: 8.0, 154: 8.0, 155: 8.0, 156: 10.0, 157: 8.0, 158: 8.0, 159: 8.0, 160: 14.0,
                      161: 14.0, 162: 8.0, 163: 12.0, 164: 12.0, 165: 12.0, 166: 11.0, 167: 9.0, 168: 11.0,
                      169: 14.0, 170: 13.0, 171: 12.0, 172: 9.0, 173: 9.0, 174: 9.0, 175: 14.0, 176: 14.0,
                      177: 11.0, 178: 13.0, 179: 14.0}


def calculate_accuracy(row_ind, col_ind, node_to_module):

    correct_matches = sum(1 for r, c in zip(row_ind, col_ind) if node_to_module[r] == node_to_module[c])

    return correct_matches / len(row_ind)


def calculate_system_accuracies(row_ind, col_ind, node_to_module):

    system_correct_matches = [0] * 7
    system_total_matches = [0] * 7

    for r, c in zip(row_ind, col_ind):
        module_r = node_to_module[r] - 1
        module_c = node_to_module[c] - 1
        if 0 <= module_r < 7 and 0 <= module_c < 7:
            system_total_matches[module_r] += 1
            if module_r == module_c:
                system_correct_matches[module_r] += 1

    system_accuracies = [correct / total if total > 0 else 0 for correct, total in
                         zip(system_correct_matches, system_total_matches)]
    
    return system_accuracies


def calculate_DC(connectivity_matrix):

    return connectivity_matrix


def calculate_USP(connectivity_matrix):

    G = nx.from_numpy_array(connectivity_matrix, create_using=nx.DiGraph())
    shortest_paths = dict(nx.floyd_warshall(G))
    new_connectivity_matrix = np.zeros_like(connectivity_matrix)
    for i in range(len(connectivity_matrix)):
        for j in range(len(connectivity_matrix)):
            try:
                new_connectivity_matrix[i][j] = shortest_paths[i][j]
            except KeyError:
                new_connectivity_matrix[i][j] = 0
    new_connectivity_matrix[new_connectivity_matrix == np.inf] = 0
    return new_connectivity_matrix


def calculate_WSP(connectivity_matrix):

    inverted_matrix = np.where(connectivity_matrix == np.inf, np.inf, 1 / connectivity_matrix)
    G = nx.from_numpy_array(inverted_matrix, create_using=nx.DiGraph())
    shortest_paths = dict(nx.floyd_warshall(G))
    wsp_matrix = np.zeros_like(connectivity_matrix)
    for i in range(len(connectivity_matrix)):
        for j in range(len(connectivity_matrix)):
            try:
                wsp_matrix[i][j] = shortest_paths[i][j]
            except KeyError:
                wsp_matrix[i][j] = 0
    wsp_matrix[wsp_matrix == 0] = np.inf
    wsp_matrix = np.where(wsp_matrix == np.inf, 0, 1 / wsp_matrix)

    return wsp_matrix


def calculate_si_matrix1(connectivity_matrix):

    with np.errstate(divide='ignore', invalid='ignore'):
        edge_length_matrix = -np.log(connectivity_matrix)
        edge_length_matrix[np.isinf(edge_length_matrix)] = np.inf

    G = nx.Graph()
    num_nodes = len(connectivity_matrix)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if connectivity_matrix[i, j] > 0:
                G.add_edge(i, j, weight=edge_length_matrix[i, j])

    shortest_paths = dict(nx.all_pairs_dijkstra_path(G, weight='weight'))
    shortest_paths_lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))

    def calculate_si(source, target):
        path_nodes = shortest_paths[source][target]
        if len(path_nodes) < 2:
            return 0

        path_probability = 1
        for i in range(len(path_nodes) - 1):
            current_node = path_nodes[i]
            next_node = path_nodes[i + 1]
            w = connectivity_matrix[current_node, next_node]
            node_strength = np.sum(connectivity_matrix[current_node])
            path_probability *= w / node_strength

        return -np.log2(path_probability)

    SI_matrix = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if i != j and connectivity_matrix[i, j] > 0:
                si_st = calculate_si(i, j)
                si_ts = calculate_si(j, i)
                SI_matrix[i, j] = (si_st + si_ts) / 2
                SI_matrix[j, i] = SI_matrix[i, j]

    return SI_matrix


def calculate_matching_index(G, i, j):

    def indicator_function(weight):
        return 1 if weight > 0 else 0
    sum_i = sum(G.get_edge_data(i, k)["weight"] for k in G[i] if k != j)
    sum_j = sum(G.get_edge_data(j, k)["weight"] for k in G[j] if k != i)
    numerator = sum(G.get_edge_data(i, k)["weight"] * indicator_function(G.get_edge_data(i, k)["weight"])
                    for k in G[i] if k != j)
    denominator = sum(G.get_edge_data(k, j)["weight"] * indicator_function(G.get_edge_data(k, j)["weight"])
                      for k in G[j] if k != i)
    if denominator == 0:
        return 0
    else:
        return numerator / denominator


def PT(connectivity_matrix):

    np.fill_diagonal(connectivity_matrix, 0)
    sparse_matrix = csr_matrix(connectivity_matrix)
    shortest_paths = shortest_path(sparse_matrix, directed=False)
    PT_matrix = np.zeros_like(connectivity_matrix)

    n = connectivity_matrix.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if shortest_paths[i, j] > 0:

                match_index = np.sum(
                    (connectivity_matrix[i, :] > 0) & (connectivity_matrix[j, :] > 0)
                )

                PT_ij = 2 * match_index / (
                        np.sum(connectivity_matrix[i, :] > 0) +
                        np.sum(connectivity_matrix[j, :] > 0)
                )

                PT_matrix[i, j] = PT_matrix[j, i] = PT_ij

    return PT_matrix


def calculate_PT(connectivity_matrix):

    G = nx.from_numpy_array(connectivity_matrix, create_using=nx.DiGraph())
    all_pairs_shortest_paths = dict(nx.all_pairs_dijkstra_path(G))
    pt_matrix = np.zeros_like(connectivity_matrix)

    for i in range(len(connectivity_matrix)):
        for j in range(len(connectivity_matrix)):

            shortest_path = all_pairs_shortest_paths[i][j]

            if len(shortest_path) <= 1:
                pt_matrix[i][j] = 0
                continue

            num_nodes = len(shortest_path)
            pt_value = 0

            for idx_i, node_i in enumerate(shortest_path[:-1]):
                for node_j in shortest_path[idx_i + 1:]:
                    m_ij = calculate_matching_index(G, node_i, node_j)
                    pt_value += m_ij

            if num_nodes <= 1:
                pt_matrix[i][j] = 0
            else:
                pt_matrix[i][j] = 2 / (num_nodes * (num_nodes - 1)) * pt_value

    return pt_matrix


def calculate_uc_matrix(A):

    binary_matrix = (A > 0).astype(int)

    def matrix_exponential(A, max_k=1, tol=1e-10):


        result = np.eye(A.shape[0])
        current_term = np.eye(A.shape[0])

        for k in range(max_k + 1):
            current_term = current_term.dot(A) / (k + 1)
            result += current_term
            if np.linalg.norm(current_term) < tol:
                break
        return result
    UC_matrix = matrix_exponential(binary_matrix)
    np.fill_diagonal(UC_matrix, 0)
    return UC_matrix


def calculate_wc_matrix(W, epsilon=1e-10):

    S = np.diag(np.sum(W, axis=1) + epsilon)
    S_sqrt = np.sqrt(S)

    if np.any(np.diag(S_sqrt) == 0):
        raise ValueError("Diagonal elements of S_sqrt should not be zero after adding epsilon.")

    S_inv_sqrt = np.linalg.inv(S_sqrt)
    exp_W = expm(S_inv_sqrt @ W @ S_inv_sqrt)
    WC_matrix = exp_W
    np.fill_diagonal(WC_matrix, 0)

    return WC_matrix


def load_and_process_csv(filepath):

    df = pd.read_csv(filepath, header=None)
    fc = df.values
    np.fill_diagonal(fc, 0)

    return fc


def calculate_cost(fc1, fc2):

    nROIs = fc1.shape[0]
    costmat = np.zeros((nROIs, nROIs))
    fc1_cleaned = np.nan_to_num(fc1)
    fc2_cleaned = np.nan_to_num(fc2)
    for x in range(nROIs):
        for y in range(nROIs):
            costmat[x, y] = euclidean(fc1_cleaned[x], fc2_cleaned[y])

    return costmat


def graph_matching(adj_matrix_1, adj_matrix_2):

    remap_matrix = np.zeros((360, 360))
    costmat = calculate_cost(adj_matrix_1, adj_matrix_2)
    row_ind, col_ind = linear_sum_assignment(costmat)
    for src_node, tgt_node in zip(row_ind, col_ind):
        remap_matrix[src_node, tgt_node] = 1
    network_remap_matrix = np.zeros((14, 14))
    for i in range(360):
        for j in range(360):
            if remap_matrix[i, j] == 1:
                yeo_i = int(hcp_to_yeo_mapping.get(i, -1)) - 1
                yeo_j = int(hcp_to_yeo_mapping.get(j, -1)) - 1
                if 0 <= yeo_i < 14 and 0 <= yeo_j < 14:
                    network_remap_matrix[yeo_i, yeo_j] += 1

    accuracy = calculate_system_accuracies(row_ind, col_ind, node_to_module)

    return remap_matrix, accuracy


def graph_matching_node(adj_matrix_1, adj_matrix_2):

    remap_matrix = np.zeros((360, 360))
    costmat = calculate_cost(adj_matrix_1, adj_matrix_2)
    row_ind, col_ind = linear_sum_assignment(costmat)
    for src_node, tgt_node in zip(row_ind, col_ind):
        remap_matrix[src_node, tgt_node] = 1
    accuracy = calculate_system_accuracies(row_ind, col_ind, node_to_module)

    return remap_matrix, accuracy


def save_matrix_to_csv(matrix, save_path):

    df = pd.DataFrame(matrix)
    df.to_csv(save_path, index=False)


def load_and_process_mat(filepath):

    data = scipy.io.loadmat(filepath)
    fc = data['ROICorrelation_FisherZ']
    np.fill_diagonal(fc, 0)

    return fc


def reorder_matrix(matrix, node_to_module):

    sort_indices = sorted(range(len(node_to_module)), key=lambda k: node_to_module[k])
    reordered_matrix = matrix[sort_indices, :][:, sort_indices]

    return reordered_matrix


def calculate_matrixs_in_folder_fc_sc(folder_path, pattern_key):

    accuracies = []
    function_map = {
        'DC': calculate_DC,
        'USP': calculate_USP,
        'WSP': calculate_WSP,
        'SI': calculate_si_matrix1,
        'PT': PT,
        'UC': calculate_uc_matrix,
        'WC': calculate_wc_matrix,
    }

    normalized_remap_matrices = []
    subfolders = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                  os.path.isdir(os.path.join(folder_path, f))]
    for subfolder in tqdm(subfolders, desc="Processing subfolders"):
        csv_files = [os.path.join(subfolder, f) for f in os.listdir(subfolder) if f.endswith('.csv')]
        mat_files = [os.path.join(subfolder, f) for f in os.listdir(subfolder) if f.endswith('.mat')]
        if len(csv_files) == len(mat_files):
            fc = load_and_process_mat(mat_files[0])
            sc = load_and_process_csv(csv_files[0])
            sc = function_map[pattern_key](sc)
            file_name_with_extension = os.path.basename(csv_files[0])

            file_name_without_extension = os.path.splitext(file_name_with_extension)[0]

            normalized_remap_matrix, accuracy = graph_matching(fc, sc)
            output_folder = r'D:\Project\20231201-fMRI\result\output_folder'
            os.makedirs(output_folder, exist_ok=True)  # 确保目录存在
            new_file_path = os.path.join(output_folder, f'{file_name_without_extension}_{pattern_key}.csv')
            print(new_file_path )
            save_matrix_to_csv(normalized_remap_matrix, new_file_path)
            print(accuracy)

            accuracies.append(accuracy)
            normalized_remap_matrices.append(normalized_remap_matrix)

    return normalized_remap_matrices, accuracies


def calculate_matrixs_in_folder_fc_sc_node(folder_path, pattern_key):

    accuracies = []
    function_map = {
        'DC': calculate_DC,
        'USP': calculate_USP,
        'WSP': calculate_WSP,
        'SI': calculate_si_matrix1,
        'PT': PT,
        'UC': calculate_uc_matrix,
        'WC': calculate_wc_matrix,
    }

    normalized_remap_matrices = []
    subfolders = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                  os.path.isdir(os.path.join(folder_path, f))]
    for subfolder in tqdm(subfolders, desc="Processing subfolders"):
        csv_files = [os.path.join(subfolder, f) for f in os.listdir(subfolder) if f.endswith('.csv')]
        mat_files = [os.path.join(subfolder, f) for f in os.listdir(subfolder) if f.endswith('.mat')]
        if len(csv_files) == len(mat_files):
            fc = load_and_process_mat(mat_files[0])
            sc = load_and_process_csv(csv_files[0])
            sc = function_map[pattern_key](sc)

            fc = reorder_matrix(fc, node_to_module)
            sc = reorder_matrix(sc, node_to_module)

            sc_flattened = sc.reshape(-1, 1)
            normalized_remap_matrix, accuracy = graph_matching_node(fc, sc)
            print(accuracy)

            accuracies.append(accuracy)
            normalized_remap_matrices.append(normalized_remap_matrix)

    return normalized_remap_matrices, accuracies