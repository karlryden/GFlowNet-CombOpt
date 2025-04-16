import os, sys
import pickle
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import networkx as nx

from xu_util import get_random_instance

"""
python rbgraph_generator.py --num_graph 4000 --graph_type small --save_dir rb200-300/train
python rbgraph_generator.py --num_graph 500 --graph_type small --save_dir rb200-300/test  

add --constrain to generate softly constrained graphs
"""

# TODO: Add more templates
constraint_templates = lambda w: [
    "inclusion constraint on nodes " + ", ".join(map(str, w)),
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_graph', type=int, default=10)
    parser.add_argument('--graph_type', type=str, default='small')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="data")
    parser.add_argument("--constrain", type=float, default=0.0)
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

    assert 0 <= args.constrain <= 1, "constrain should be between 0 and 1."

    for i, num_g in enumerate(tqdm(range(args.num_graph))):
        path = Path(f'{args.save_dir}')
        stub = f"GR_{min_n}_{max_n}_{num_g}"
        while True:
            g, _ = get_random_instance(args.graph_type)
            g.remove_nodes_from(list(nx.isolates(g)))
            if min_n <= g.number_of_nodes() <= max_n:
                break

        x = {}
        wanted = []

        if args.constrain:
            # TODO: Implement and train a 'critic' to skip indicators

            constrain = np.random.binomial(1, args.constrain) # Randomly choose whether to add a constraint or not

            if constrain:
                num_wanted = max(1, g.number_of_nodes() // 10)
                wanted = np.random.choice(   # Pick a small-ish number of preferred nodes
                    g.number_of_nodes(),
                    size=num_wanted,
                    replace=False
                ).tolist()

                c = np.random.choice(  # Generate a random constraint string
                    constraint_templates(wanted)
                )

            else:
                c = ""
        
            x['constraint'] = c

        nx.set_node_attributes(g, {i: float(i in wanted) for i in range(len(g))}, 'wanted')
        x['graph'] = g

        output_file = path / (f'{stub}.pickle')

        with open(output_file, 'wb') as f:
            pickle.dump(x, f, pickle.HIGHEST_PROTOCOL)
        print(f"Generated graph {path}")