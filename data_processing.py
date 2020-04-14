import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.io import loadmat
from sklearn.metrics.pairwise import cosine_similarity
# import matplotlib.pyplot as plt
from random import sample


def load_data(opt):
    """Load raw data to be used features, labels and links."""
    data_dir = opt['data_dir']
    dataset_str = opt['dataset_str']

    if dataset_str == 'cnn':
        features_file = loadmat(data_dir+'/CNN_ImageFeatureNormalizedMat.mat')
        labels_file = loadmat(data_dir+'/CNN_LabelMat.mat')
        links_file = loadmat(data_dir+'/CNN_TFIDF_Normalized.mat')

    elif dataset_str == 'fox':
        features_file = loadmat(data_dir+'/FOX_ImageFeatureNormalizedMat.mat')
        labels_file = loadmat(data_dir+'/FOX_LabelMat.mat')
        links_file = loadmat(data_dir+'/FOX_TFIDF_Normalized.mat')

    features_all = sp.lil_matrix(features_file['imageFeatureNormalizedMat'].transpose())
    labels_all = labels_file['labelMat'].toarray()
    links_all = links_file['TFIDF_Normalized'].transpose()

    return features_all, labels_all, links_all


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def graph_construction(opt):
    """Construct graph and return giant component."""
    features_all, labels_all, links_all = load_data(opt)
    similarity_threshold = opt['similarity_threshold']

    # adjacency matrix
    adj_all = cosine_similarity(links_all)
    adj_all = 1 * (adj_all > similarity_threshold)
    g_all = nx.from_numpy_matrix(adj_all, create_using=None)

    # giant component
    g_cc = sorted(nx.connected_component_subgraphs(g_all), key=len, reverse=True)
    adj_cc = nx.to_scipy_sparse_matrix(g_cc[0], format='csr')
    idx_cc = list(g_cc[0].node)
    mask_cc = sample_mask(idx_cc, len(g_all))
    features_cc = features_all[mask_cc, :]
    labels_cc = labels_all[mask_cc, :]

    return features_cc, labels_cc, adj_cc, g_cc


def graph_visualization(opt):
    """Visualize giant component."""
    features_cc, labels_cc, adj_cc, g_cc = graph_construction(opt)
    g0 = g_cc[0]
    pos = nx.spring_layout(g0)
    full_color_map = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    color_map = labels_cc*full_color_map[:labels_cc.shape[1]]
    plt_opt = {
        "node_color": color_map,
        "node_size": 20,
        "line_color": "grey",
        "linewidths": 0,
        "width": 0.1,
        "cmap": plt.cm.jet,
    }
    nx.draw(g0, pos, **plt_opt)
    plt.show()


def gcn_input(opt):
    """Prepare inputs for gcn: data split and formatting"""
    features_cc, labels_cc, adj_cc, _ = graph_construction(opt)

    num_nodes = len(labels_cc)
    idx_rem = list(range(num_nodes))
    idx_train = sample(idx_rem, int(num_nodes * 0.5))
    idx_rem = list(set(idx_rem) - set(idx_train))
    idx_val = sample(idx_rem, int(num_nodes * 0.1))
    idx_test = list(set(idx_rem) - set(idx_val))

    train_mask = sample_mask(idx_train, num_nodes)
    val_mask = sample_mask(idx_val, num_nodes)
    test_mask = sample_mask(idx_test, num_nodes)

    y_train = np.zeros(labels_cc.shape)
    y_val = np.zeros(labels_cc.shape)
    y_test = np.zeros(labels_cc.shape)
    y_train[train_mask, :] = labels_cc[train_mask, :]
    y_val[val_mask, :] = labels_cc[val_mask, :]
    y_test[test_mask, :] = labels_cc[test_mask, :]

    adj = adj_cc
    features = features_cc

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask

