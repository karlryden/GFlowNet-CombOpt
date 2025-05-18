import os, sys
import argparse
import pickle
from pathlib import Path

from tqdm import tqdm

import numpy as np
import networkx as nx
from networkx.algorithms import approximation

from xu_util import get_random_instance

"""
python rbgraph_generator.py --num_graph 4000 --graph_type small --save_dir rb200-300/train
python rbgraph_generator.py --num_graph 500 --graph_type small --save_dir rb200-300/test  

add --constrain to generate softly constrained graphs,
"""

constraint_templates = lambda w: [
    "inclusion constraint on nodes " + ", ".join(map(str, w)),
    "prefer to include nodes " + ", ".join(map(str, w)),
    "favor selecting nodes " + ", ".join(map(str, w)),
    "encourage inclusion of nodes " + ", ".join(map(str, w)),
    "try to select nodes " + ", ".join(map(str, w)),
    "guide solution toward including nodes " + ", ".join(map(str, w)),
    "selection should prioritize nodes " + ", ".join(map(str, w)),
    "soft requirement: include nodes " + ", ".join(map(str, w)),
    "suggested inclusion: nodes " + ", ".join(map(str, w)),
    "aim to include nodes " + ", ".join(map(str, w)),
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_graph', type=int, default=10)
    parser.add_argument('--graph_type', type=str, default='small')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="data")
    parser.add_argument("--constrain", type=float, default=0.0)
    parser.add_argument("--want", type=float, default=0.1)
    args = parser.parse_args()
    np.random.seed(seed=args.seed)

    if not os.path.isdir("{}".format(args.save_dir)):
        os.makedirs("{}".format(args.save_dir))
    print("Final Output: {}".format(args.save_dir))
    print(f"Generating {args.num_graph} {args.graph_type} graphs" \
        + f"with {int(100*args.constrain)}% chance" \
        + f"of preferring {int(100*args.want)}% of nodes.")

    if args.graph_type == "tiny":
        min_n, max_n = 10, 20
    elif args.graph_type == "small":
        min_n, max_n = 200, 300
    elif args.graph_type == "large":
        min_n, max_n = 800, 1200
    else:
        raise NotImplementedError

    assert 0 <= args.constrain <= 1, "constrain should be between 0 and 1."
    assert 0 <= args.want <= 1, "want should be between 0 and 1."

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
            constrain = np.random.binomial(1, args.constrain) # Randomly choose whether to add a constraint or not
            if constrain:
                num_wanted = int(max(1, g.number_of_nodes() * args.want))  # Number of nodes to be preferred
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

        # TODO: Implement and train a network to predict w = critic(g, c)
        nx.set_node_attributes(g, {i: float(i in wanted) for i in range(len(g))}, 'wanted')
        x['graph'] = g

        output_file = path / (f'{stub}.pickle')

        with open(output_file, 'wb') as f:
            pickle.dump(x, f, pickle.HIGHEST_PROTOCOL)
        print(f"Generated graph {path}")