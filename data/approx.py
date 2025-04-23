import argparse
from pathlib import Path
import pickle

import networkx as nx
from networkx.algorithms import approximation

tasks = ['MaxIndependentSet', 'MaxClique', 'MinDominateSet', 'MaxCut']

approximator_dict = {
    'MaxIndependentSet': approximation.maximum_independent_set,
    'MaxClique': approximation.max_clique,
    'MinDominateSet': approximation.min_weighted_dominating_set,
    'MaxCut': lambda g: approximation.one_exchange(g)[1][0],
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    graph_paths = sorted(list(data_dir.rglob("*.pickle")))
    for path in graph_paths:
        with open(path, 'rb') as graph_file:
            x = pickle.load(graph_file)
            g = x['graph']

        for task in tasks[:-1]: # TODO: Decide whether to include MaxCut (way slower)
            print("Approximating solution for task:", task)
            s = approximator_dict[task](g)
            node_attr = {i: float(i in s) for i in g.nodes}
            nx.set_node_attributes(g, node_attr, task)

        x['graph'] = g

        with open(path, 'wb') as graph_file:
            pickle.dump(x, graph_file)