#!/usr/bin/env python
# coding: utf-8
import logging
import os
import warnings
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy
import scipy.sparse
from scipy.sparse.linalg import eigsh
from sklearn.cluster import k_means
from sklearn.cluster.spectral import discretize
from sklearn.manifold import spectral_embedding
from sklearn.neighbors import kneighbors_graph

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 14
fig_size[1] = 10

warnings.filterwarnings("ignore")
# np.random.seed(1995)  # Fixed seed for testing reproducability
SAVE_PLOTS = True
DATASET_DIRECTORY = './datasets/'
NUM_CLUSTERS = 2
EXPERIMENT_NUM_ITERATIONS = 10
EXPERIMENTS = {
    'vanilla',
    'regularized',
    'regularized_with_kmeans',
    'sklearn_spectral_embedding',
    'sklearn_kmeans',
    'tau_p30',
    'tau_p40',
    'tau_p50',
    'tau_p90',
    'tau_p99',
    'tau_max',
}
SAVE_SUMMARY_GRAPH = True

'''
Reconstructing [1]
[1] Understanding Regularized Spectral Clustering via Graph Conductance Yilin Zhang, Karl Rohe: NIPS'18
https://arxiv.org/pdf/1806.01468.pdf

'''
W
# Create enough colors for each cluster
COLORS = np.array(list(['#377eb8', '#FF0000', '#4daf4a',
                        '#f781bf', '#a65628', '#984ea3',
                        '#999999', '#e41a1c', '#dede00',
                        '#ff7f00',
                        ]))
BLACK = '#000000'
GREY = '#D3D3D3'
TRAINING_NODE_COLOR = BLACK

# Disable logging once evaluating
logging.basicConfig(filename='app.log', filemode='a', format='%(levelname)s - %(message)s', level=logging.INFO)

logger = logging.getLogger()
is_logging_enabled = logger.isEnabledFor(logging.DEBUG)


def regularized_laplacian_matrix(adj_matrix, tau):
    """
    Using ARPACK solver, compute the first K eigen vector.
    The laplacian is computed using the regularised formula from [2]
    [2]Kamalika Chaudhuri, Fan Chung, and Alexander Tsiatas 2018.
        Spectral clustering of graphs with general degrees in the extended planted partition model.

    L = I - D^-1/2 * A * D ^-1/2

    :param adj_matrix: adjacency matrix representation of graph where [m][n] >0 if there is edge and [m][n] = weight
    :param tau: the regularisation constant
    :return: the first K eigenvector
    """
    # Code inspired from nx.normalized_laplacian_matrix, with changes to allow regularisation
    n, m = adj_matrix.shape
    I = np.eye(n, m)
    diags = adj_matrix.sum(axis=1).flatten()
    # add tau to the diags to produce a regularised diags
    if tau != 0:
        diags = np.add(diags, tau)

    # diags will be zero at points where there is no edge and/or the node you are at
    #  ignore the error and make it zero later
    with scipy.errstate(divide='ignore'):
        diags_sqrt = 1.0 / scipy.sqrt(diags)
    diags_sqrt[scipy.isinf(diags_sqrt)] = 0
    D = scipy.sparse.spdiags(diags_sqrt, [0], m, n, format='csr')

    L = I - (D.dot(adj_matrix.dot(D)))
    return L


def eigen_solver(laplacian, n_clusters):
    """
    ARPACK eigen solver in Shift-Invert Mode based on http://docs.scipy.org/doc/scipy/reference/tutorial/arpack.html
    """
    lap = laplacian * -1
    v0 = np.random.uniform(-1, 1, lap.shape[0])
    eigen_values, eigen_vectors = eigsh(lap, k=n_clusters, sigma=1.0, v0=v0)
    eigen_vectors = eigen_vectors.T[n_clusters::-1]
    return eigen_values, eigen_vectors[:n_clusters].T


def regularized_spectral_clustering(adj_matrix, tau, n_clusters, algo='scan'):
    """
    :param adj_matrix: adjacency matrix representation of graph where [m][n] >0 if there is edge and [m][n] = weight
    :param n_clusters: cluster partitioning constant
    :param algo: the clustering separation algorithm, possible value kmeans++ or scan
    :return: labels, number of clustering iterations needed, smallest set of cluster found, execution time
    """
    start = timer()
    regularized_laplacian = regularized_laplacian_matrix(adj_matrix, tau)
    eigen_values, eigen_vectors = eigen_solver(regularized_laplacian, n_clusters=n_clusters)
    if algo == 'kmeans++':
        _, labels, _, num_iterations = k_means(eigen_vectors,
                                               n_clusters=n_clusters,
                                               return_n_iter=True)
    else:
        if n_clusters == 2:  # cluster based on sign
            second_eigen_vector_index = np.argsort(eigen_values)[1]
            second_eigen_vector = eigen_vectors.T[second_eigen_vector_index]
            labels = [0 if val <= 0 else 1 for val in second_eigen_vector]  # use only the second eigenvector
            num_iterations = 1
        else:  # bisecting it into k-ways, use all eigenvectors
            labels = discretize(eigen_vectors)
            num_iterations = 20  # assume worst case scenario that it tooks 20 restarts
    end = timer()
    execution_time = end - start
    smallest_cluster_size = min(np.sum(labels), abs(np.sum(labels) - len(labels)))
    return labels, num_iterations, smallest_cluster_size, execution_time


def sklearn_kmeans(adj_matrix, n_clusters):
    start = timer()
    _, labels, _, num_iterations = k_means(adj_matrix,
                                           n_clusters=n_clusters,
                                           return_n_iter=True)

    end = timer()
    execution_time = end - start
    smallest_cluster_size = min(np.sum(labels), abs(np.sum(labels) - labels.size))
    return labels, num_iterations, smallest_cluster_size, execution_time


def sklearn_spectral_clustering(adj_matrix, n_clusters):
    """
    :param adj_matrix: adjacency matrix representation of graph where [m][n] >0 if there is edge and [m][n] = weight
    :param n_clusters: cluster partitioning constant
    :return: labels, number of clustering iterations needed, smallest set of cluster found, execution time
    """
    start = timer()
    connectivity = kneighbors_graph(adj_matrix, n_neighbors=10,
                                    include_self=True)
    affinity_matrix_ = 0.5 * (connectivity + connectivity.T)

    eigen_vectors = spectral_embedding(affinity_matrix_,
                                       n_components=n_clusters,
                                       eigen_solver='arpack',
                                       eigen_tol=0.0,
                                       norm_laplacian=True,
                                       drop_first=False)

    _, labels, _, num_iterations = k_means(eigen_vectors,
                                           n_clusters=n_clusters,
                                           return_n_iter=True)

    end = timer()
    execution_time = end - start
    smallest_cluster_size = min(np.sum(labels), abs(np.sum(labels) - labels.size))
    return labels, num_iterations, smallest_cluster_size, execution_time


def split_graph_edges(graph: nx.Graph):
    """Utility method to split the graph edges into two sets"""
    logging.debug('Splitting the graph into training edges set and a testing edges set')

    edges_list = list(graph.edges(data=False))
    np.random.shuffle(edges_list)
    training_edges, testing_edges = edges_list[:len(edges_list) // 2], edges_list[len(edges_list) // 2:]

    training_graph: nx.Graph = graph.edge_subgraph(training_edges).copy()
    testing_graph: nx.Graph = graph.edge_subgraph(testing_edges).copy()
    return training_graph, testing_graph


def evaluate_conductance(graph: nx.Graph, subgraphs, tau):
    """
    :param graph: the graph being evaluated
    :param subgraphs: K cluster of Subsets of the main graph
    :param tau: tuning parameter, tau = 0 = vanilla conductance
    :return: core_cut, vanilla_conductance
    """

    vanilla_conductances = []
    core_cuts = []
    for _, nodes in subgraphs.items():
        subgraph = graph.subgraph(nodes).copy()
        subgraph_complement = set(graph) - set(subgraph)
        cut = nx.cut_size(graph, subgraph, subgraph_complement)
        volume_subgraph = nx.volume(graph, subgraph)
        volume_subgraph_complement = nx.volume(graph, subgraph_complement)
        volume_div = min(volume_subgraph, volume_subgraph_complement)
        vanilla_conductances.append((cut / volume_div))
        core_cuts.append((cut + ((tau / len(graph)) * len(subgraph) * len(subgraph_complement))) / (
                volume_div + (tau * len(subgraph))))
    vanilla_conductance = min(vanilla_conductances)
    core_cut = min(core_cuts)
    logging.debug('Vanilla graph conductance: %f', vanilla_conductance)
    logging.debug('CoreCut graph conductance: %f', core_cut)

    return core_cut, vanilla_conductance


def plot_graph(graph, nodes_color, nodes_size, plot_name):
    print('Drawing')
    pos = nx.spring_layout(graph)

    logging.debug('Saving plot %s to file', plot_name)
    options = {
        'line_color': 'grey',
        'linewidths': 0,
        'width': 0.1,
        'with_labels': False,
        'node_color': nodes_color,
        'pos': pos,
        # 'node_size': np.squeeze(np.asarray(np.sum(nx.to_scipy_sparse_matrix(graph, format='csr'), axis=1).T)),
        'node_size': nodes_size
    }

    figure: plt.Figure = plt.figure()
    figure.suptitle(plot_name.replace('_', ' ').capitalize())

    nx.draw(graph, ax=figure.add_subplot(111), **options)

    figure.savefig(plot_name)


def evaluate_graph(graph: nx.Graph, n_clusters, graph_name):
    """
    Reconsutrction of [1]Understanding Regularized Spectral Clustering via Graph Conductance, Yilin Zhang, Karl Rohe

    :param graph: Graph to be evaluated
    :param n_clusters: How many clusters to look at
    :param graph_name: the graph name used to create checkpoints and figures
    :return:
    """
    # Experiment only on undirected graphs
    if graph.is_directed():
        logging.debug('Graph is directed graph, mirroring the edges to undirected')
        graph = graph.to_undirected()

    graph_size_before_trimming = graph.number_of_nodes()
    # Before computing anything, largest connected component identified and used
    graph = graph.subgraph(max(nx.connected_components(graph), key=len)).copy()

    if is_logging_enabled:  # avoid needless mathematical operations
        logging.debug('Removing all components not connected to largest subgraph')
        graph_size_after_trimming = graph.number_of_nodes()
        logging.debug(
            'Total nodes removed: {}'.format(graph_size_after_trimming - graph_size_before_trimming))

    training_graph, testing_graph = split_graph_edges(graph)

    training_graph_size_before_removing_dangling_set = training_graph.number_of_nodes()
    training_graph: nx.Graph = training_graph.subgraph(max(nx.connected_components(training_graph), key=len)).copy()

    if is_logging_enabled:
        logging.debug('Total nodes removed from training set: {}'.format(
            training_graph_size_before_removing_dangling_set - training_graph.number_of_nodes()))

    results = {
        'graph_size': graph.number_of_nodes(),
        'vanilla': {},
        'regularized': {},
        'regularized_with_kmeans': {},
        'sklearn_spectral_embedding': {},
        'sklearn_kmeans': {},
        'tau_p30': {},
        'tau_p40': {},
        'tau_p50': {},
        'tau_p90': {},
        'tau_p99': {},
        'tau_max': {},
    }

    graph_degree = graph.degree()
    graph_average_degree = np.sum(val for (node, val) in graph_degree) / graph.number_of_nodes()
    logging.debug('Average degree regularisation: %f', graph_average_degree)

    adj_matrix = nx.to_scipy_sparse_matrix(training_graph, format='csr')
    tau_values = {
        'vanilla': graph_average_degree,
        'regularized': graph_average_degree,
        'regularized_with_kmeans': graph_average_degree,
        'sklearn_spectral_embedding': graph_average_degree,
        'sklearn_kmeans': graph_average_degree,
        'tau_p30': np.percentile(graph_degree, 30),
        'tau_p40': np.percentile(graph_degree, 40),
        'tau_p50': np.percentile(graph_degree, 50),
        'tau_p90': np.percentile(graph_degree, 90),
        'tau_p99': np.percentile(graph_degree, 99),
        'tau_max': np.percentile(graph_degree, 100)
    }

    for method, tau in tau_values.items():
        if method not in EXPERIMENTS:
            continue
        # tau = np.sum(adj_matrix) / adj_matrix.shape[0] Unclear from the paper if they recalculate tau after or before
        logging.info('Spectral clustering with tau = %f', tau)

        if method == 'sklearn_spectral_embedding':
            labels, num_iterations, smallest_cluster_size, execution_time = sklearn_spectral_clustering(adj_matrix,
                                                                                                        n_clusters)
        elif method == 'sklearn_kmeans':
            labels, num_iterations, smallest_cluster_size, execution_time = sklearn_kmeans(adj_matrix,
                                                                                           n_clusters)

        elif method == 'vanilla':
            labels, num_iterations, smallest_cluster_size, execution_time = regularized_spectral_clustering(adj_matrix,
                                                                                                            0,
                                                                                                            n_clusters)
        elif method == 'regularized_with_kmeans':
            labels, num_iterations, smallest_cluster_size, execution_time = regularized_spectral_clustering(adj_matrix,
                                                                                                            tau,
                                                                                                            n_clusters,
                                                                                                            'kmeans++')
        else:
            labels, num_iterations, smallest_cluster_size, execution_time = regularized_spectral_clustering(adj_matrix,
                                                                                                            tau,
                                                                                                            n_clusters,
                                                                                                            'scan')
        # Create subgraphs based on clusters identified using spectral clustering
        training_nodes = list(training_graph.nodes())
        subgraphs = {i: [] for i in range(n_clusters)}

        nodes_color = dict()
        for idx, label in enumerate(labels):
            node_id = training_nodes[idx]
            subgraphs[label].append(node_id)
            if SAVE_PLOTS:
                nodes_color[node_id] = COLORS[label]

        logging.debug('Evaluating Training Edges')
        training_core_cut, training_vanilla_conductance = evaluate_conductance(training_graph,
                                                                               subgraphs,
                                                                               tau)
        logging.debug('Evaluating Testing Edges')
        testing_core_cut, testing_vanilla_conductance = evaluate_conductance(testing_graph,
                                                                             subgraphs,
                                                                             tau)

        results[method]['training_core_cut'] = training_core_cut
        results[method]['training_vanilla_conductance'] = training_vanilla_conductance

        results[method]['testing_core_cut'] = testing_core_cut
        results[method]['testing_vanilla_conductance'] = testing_vanilla_conductance

        results[method]['num_iterations'] = num_iterations
        results[method]['smallest_cluster_size'] = smallest_cluster_size
        results[method]['execution_time'] = execution_time
        logging.info(method, results[method])

        print(method, results[method])
        if SAVE_PLOTS:
            nodes_color_list = []  # because networkx is weird...
            nodes_size = []
            for node_id in set(graph.nodes()):
                nodes_color_list.append(nodes_color.get(node_id, TRAINING_NODE_COLOR))
                nodes_size.append(5 if node_id in nodes_color else 1)
            plot_name = '{}_graph_{}_clusters_{}'.format(graph_name, method, n_clusters)
            plot_graph(graph, nodes_color_list, nodes_size,
                       plot_name)

    return results


def plot_summary_graph(results, dataset_name):
    # TODO this is very hacky but I don't have much time to clean up, clean it up later
    exps_performance = dict()
    summary_values = dict()
    for experiment in EXPERIMENTS:
        exps_performance[experiment] = {
            'training_core_cut': [],
            'training_vanilla_conductance': [],
            'testing_core_cut': [],
            'testing_vanilla_conductance': [],
            'num_iterations': [],
            'smallest_cluster_size': [],
            'execution_time': [],
        }
        summary_values[experiment] = {
            'training_core_cut': (0.0, 0.0),
            'training_vanilla_conductance': (0.0, 0.0),
            'testing_core_cut': (0.0, 0.0),
            'testing_vanilla_conductance': (0.0, 0.0),
            'num_iterations': (0.0, 0.0),
            'smallest_cluster_size': (0.0, 0.0),
            'execution_time': (0.0, 0.0),
        }
    graph_size = 0
    for iteration_result in results:
        graph_size = iteration_result['graph_size']  # TODO do something with the size here
        # aggregate
        for experiment in EXPERIMENTS:
            exps_performance[experiment]['training_core_cut'].append(
                iteration_result[experiment]['training_core_cut'])
            exps_performance[experiment]['training_vanilla_conductance'].append(iteration_result[experiment][
                                                                                    'training_vanilla_conductance'])
            exps_performance[experiment]['testing_core_cut'].append(
                iteration_result[experiment]['testing_core_cut'])
            exps_performance[experiment]['testing_vanilla_conductance'].append(iteration_result[experiment][
                                                                                   'testing_vanilla_conductance'])
            exps_performance[experiment]['num_iterations'].append(iteration_result[experiment]['num_iterations'])
            exps_performance[experiment]['smallest_cluster_size'].append(
                iteration_result[experiment]['smallest_cluster_size'])
            exps_performance[experiment]['execution_time'].append(iteration_result[experiment]['execution_time'])

            # analyse it
            for key, method_performance in exps_performance.items():
                for method_name, method_results in method_performance.items():
                    exp_results = exps_performance[experiment][method_name]
                    average_performance = np.mean(exp_results)
                    std_performance = np.std(exp_results)
                    # store tuple of mean and std
                    summary_values[experiment][method_name] = (average_performance, std_performance)

    # plot it, really big plot, break it down man

    plot_group_conductance_training = {'training_core_cut',
                                       'training_vanilla_conductance',
                                       'testing_core_cut',
                                       'testing_vanilla_conductance',
                                       }
    n_groups = len(EXPERIMENTS)
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.20
    opacity = 0.4
    error_config = {'ecolor': '0.3'}
    stds_conductance = {'training_core_cut': [],
                        'training_vanilla_conductance': [],
                        'testing_core_cut': [],
                        'testing_vanilla_conductance': [],
                        }
    means_conductance = {'training_core_cut': [],
                         'training_vanilla_conductance': [],
                         'testing_core_cut': [],
                         'testing_vanilla_conductance': [],
                         }
    std_execution = []
    mean_execution = []
    std_smallest_cluster_size = []
    mean_smallest_cluster_size = []

    for bar_method in EXPERIMENTS:
        mean_execution.append(summary_values[bar_method]['execution_time'][0])
        std_execution.append(summary_values[bar_method]['execution_time'][1])

        mean_smallest_cluster_size.append(summary_values[bar_method]['smallest_cluster_size'][0])
        std_smallest_cluster_size.append(summary_values[bar_method]['smallest_cluster_size'][1])

        for y_axis_eval in plot_group_conductance_training:
            summaries = summary_values[bar_method][y_axis_eval]
            means_conductance[y_axis_eval].append(summaries[0])
            stds_conductance[y_axis_eval].append(summaries[1])

    # Plot Conductive
    plt.cla()
    for id, comparison_method in enumerate(plot_group_conductance_training):
        ax.bar(index + (bar_width * id),
               means_conductance[comparison_method],
               bar_width,
               alpha=opacity,
               color=COLORS[id],
               yerr=stds_conductance[comparison_method],
               error_kw=error_config,
               label=comparison_method.replace('_', '\n').capitalize(),
               log=False)

    ax.set_title('Graph conductance evaluation of dataset: {} with K_clusters {}, graph size: {}'
                 .format(dataset_name.replace('.txt', ''), NUM_CLUSTERS, graph_size))
    ax.set_xticks(index + bar_width)

    x_labels = [exp.replace('_', '\n') for exp in EXPERIMENTS]
    ax.set_xticklabels(x_labels)

    ax.legend()
    fig.tight_layout()
    fig.savefig("{}_conductance_summary_evaluation".format(dataset_name))

    # plot execution
    plt.cla()
    ax.bar(index,
           mean_execution,
           bar_width + 0.3,
           alpha=opacity,
           color=COLORS[0],
           yerr=std_execution,
           error_kw=error_config,
           label='Execution Performance in MS',
           log=False)

    ax.set_title('Execution Performance in MS: {} with K_clusters {}, graph size: {}'
                 .format(dataset_name.replace('.txt', ''), NUM_CLUSTERS, graph_size))
    ax.set_xticks(index + bar_width)

    x_labels = [exp.replace('_', '\n') for exp in EXPERIMENTS]
    ax.set_xticklabels(x_labels)

    ax.legend()
    fig.tight_layout()
    fig.savefig("{}_execution_time_summary_evaluation".format(dataset_name))

    # plot cluster size
    plt.cla()
    ax.bar(index,
           mean_smallest_cluster_size,
           bar_width + 0.3,
           alpha=opacity,
           color=COLORS[0],
           yerr=std_smallest_cluster_size,
           error_kw=error_config,
           label='Smallest Cluster Size',
           log=False)

    ax.set_title('Smallest Cluster Size: {} with K_clusters {}, graph size: {}'
                 .format(dataset_name.replace('.txt', ''), NUM_CLUSTERS, graph_size))
    ax.set_xticks(index + bar_width)

    x_labels = [exp.replace('_', '\n') for exp in EXPERIMENTS]
    ax.set_xticklabels(x_labels)

    ax.legend()
    fig.tight_layout()
    fig.savefig("{}_smallest_cluster_size_evaluation".format(dataset_name))


def main():
    logging.info('Number of clusters %d', NUM_CLUSTERS)
    for dataset in os.listdir(DATASET_DIRECTORY):
        if dataset.endswith('.txt'):
            results = []
            for num_itr in range(EXPERIMENT_NUM_ITERATIONS):
                try:
                    print('Evaluating dataset: {}, iteration: {}'.format(dataset, num_itr))
                    dataset_name = os.path.join(DATASET_DIRECTORY, dataset)
                    graph = nx.read_edgelist(dataset_name, create_using=nx.Graph(), nodetype=int,
                                             data=(('weight', int),))
                    graph_eval = evaluate_graph(graph, NUM_CLUSTERS,
                                                '{}_{}_k_{}'.format(dataset.replace('.txt', ''), num_itr, NUM_CLUSTERS))
                    results.append(graph_eval)
                    logging.info(graph_eval)
                except MemoryError:
                    print('Failed to evaluate dataset {} due to running out of memory'.format(dataset))
                    continue
            if SAVE_SUMMARY_GRAPH:
                plot_summary_graph(results, '{}_k_{}'.format(dataset.replace('.txt', ''), NUM_CLUSTERS))
    print('Finished')
    logging.info('Finished')


if __name__ == "__main__":
    main()
