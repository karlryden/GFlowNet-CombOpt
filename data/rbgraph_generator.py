import os
import sys
import pickle
import random
import numpy as np
import shutil
from pathlib import Path
import json
import argparse
from tqdm import tqdm
import multiprocessing
import networkx as nx

from xu_util import get_random_instance

"""
python rbgraph_generator.py --num_graph 4000 --graph_type small --save_dir rb200-300/train
python rbgraph_generator.py --num_graph 500 --graph_type small --save_dir rb200-300/test  
"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_graph', type=int, default=10)
    parser.add_argument('--graph_type', type=str, default='small')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="data")
    parser.add_argument("--constrain", action="store_true")
    args = parser.parse_args()
    np.random.seed(seed=args.seed)

    if not os.path.isdir("{}".format(args.save_dir)):
        os.makedirs("{}".format(args.save_dir))
    print("Final Output: {}".format(args.save_dir))
    print("Generating graphs...")

    if args.graph_type == "small":
        min_n, max_n = 200, 300
    elif args.graph_type == "large":
        min_n, max_n = 800, 1200
    else:
        raise NotImplementedError

    for i, num_g in enumerate(tqdm(range(args.num_graph))):
        path = Path(f'{args.save_dir}')
        stub = f"GR_{min_n}_{max_n}_{num_g}"
        while True:
            g, _ = get_random_instance(args.graph_type)
            g.remove_nodes_from(list(nx.isolates(g)))
            if min_n <= g.number_of_nodes() <= max_n:
                break

        x = {
            'graph': g,
        }

        if args.constrain:  # NOTE: Should unconstrained examples also be generated? 
            # TODO: Implement and train a 'critic' to skip indicators
            constraint_type = random.choice(['inclusion', 'exclusion'])
            constrained_node = random.choice(list(g.nodes()))
            constraint = f'This is constraint #{i}.' # TODO: Call LLM-API(type, node; g) here

            x['constraint'] = constraint
            x['signature'] = {
                'type': constraint_type,
                'node': constrained_node
            }

        output_file = path / (f'{stub}.pickle')

        with open(output_file, 'wb') as f:
            pickle.dump(x, f, pickle.HIGHEST_PROTOCOL)
        print(f"Generated graph {path}")

