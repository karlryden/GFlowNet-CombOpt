import multiprocessing as mp
mp.set_start_method("spawn", force=True)

import sys, os
import pathlib
from pathlib import Path
import functools
import pickle

import networkx as nx
import torch
from torch.utils.data import Dataset, DataLoader
import dgl

def read_dgl_from_graph(graph_path):
    with open(graph_path, 'rb') as graph_file:
        _g = pickle.load(graph_file)
    labelled = "optimal" in graph_path.name or "non-optimal" in graph_path.name
    if labelled:
        g = dgl.from_networkx(_g, node_attrs=['label'])
    else:
        g = dgl.from_networkx(_g)
    return g

class GraphDataset(Dataset):
    def __init__(self, data_dir=None, size=None):
        assert data_dir is not None
        self.data_dir = data_dir
        self.graph_paths = sorted(list(self.data_dir.rglob("*.graph")))
        self.constraint_paths = sorted(list(self.data_dir.rglob("*.pt")))
        if self.constraint_paths:
            assert len(self.graph_paths) == len(self.constraint_paths)

        if size is not None:
            assert size > 0
            self.graph_paths = self.graph_paths[:size]
        self.num_graphs = len(self.graph_paths)

    def __getitem__(self, idx):
        g = read_dgl_from_graph(self.graph_paths[idx])

        if self.constraint_paths:
            pt = torch.load(self.constraint_paths[idx])
            c = pt['constraint']
            e = pt['embedding']
            s = pt['signature']

            return g, {'constraint': c, 'embedding': e, 'signature': s}

        else:
            return g, None

    def __len__(self):
        return self.num_graphs

def _prepare_instances(instance_directory: pathlib.Path, cache_directory: pathlib.Path, **kwargs):
    cache_directory.mkdir(parents=True, exist_ok=True)
    resolved_graph_paths = [graph_path.resolve() for graph_path in instance_directory.glob("*.pickle")]
    prepare_instance = functools.partial(
        _prepare_instance,
        cache_directory=cache_directory,
        **kwargs,
    )
    # imap_unordered_bar(prepare_instance, resolved_graph_paths, n_processes=None)  # NOTE: Causes a huge headache on GPU cluster for some reason

    # Use a simple for loop instead of imap_unordered_bar to avoid multiprocessing issues
    for path in tqdm(resolved_graph_paths):
        _prepare_instance(path, cache_directory=cache_directory, **kwargs)


from tqdm import tqdm
def imap_unordered_bar(func, args, n_processes=2):
    p = mp.Pool(n_processes)
    args = list(args)
    res_list = []
    with tqdm(total=len(args)) as pbar:
        for i, res in tqdm(enumerate(p.imap_unordered(func, args))):
            pbar.update()
            res_list.append(res)
    pbar.close()
    p.close()
    p.join()
    return res_list

def _prepare_instance(source_instance_path: pathlib.Path, cache_directory: pathlib.Path):
    cache_directory.mkdir(parents=True, exist_ok=True)
    dest_stem = cache_directory / source_instance_path.stem

    if os.path.exists(dest_stem.with_suffix(".graph")):
        source_mtime = os.path.getmtime(source_instance_path)
        last_updated = os.path.getmtime(dest_stem.with_suffix(".graph"))

        if source_mtime <= last_updated:
            return  # we already have an up2date version of that file as matrix

    try:
        with open(source_instance_path, 'rb') as source_instance_file:
            x = pickle.load(source_instance_file)
            g = x['graph']
            c = None if 'constraint' not in x.keys() else x['constraint']
            e = None if 'embedding' not in x.keys() else x['embedding'].cpu()
            s = None if 'signature' not in x.keys() else x['signature']

    except Exception as e:
        print(f"Failed to read {source_instance_path}: {e}.")
        return

    g.remove_edges_from(nx.selfloop_edges(g)) # remove self loops

    with open(dest_stem.with_suffix(".graph"), 'wb') as graph_file:
        pickle.dump(g, graph_file, pickle.HIGHEST_PROTOCOL)
    print(f"Updated graph file: {source_instance_path}.")

    if c is not None:
        torch.save({
            'constraint': c,
            'embedding': e,
            'signature': s
        }, dest_stem.with_suffix(".pt"))

        print(f'Updated constraint: {source_instance_path}.')

def collate_fn(samples):
    graphs = [x[0] for x in samples]
    gbatch = dgl.batch(graphs)

    if samples[0][1] is not None:
        constbatch = [x[1] for x in samples]
    else:
        constbatch = None

    return gbatch, constbatch

def get_data_loaders(cfg):
    data_path = Path(__file__).parent.parent / "data"
    data_path = data_path / Path(cfg.input)  # string to pathlib.Path
    print(f"Loading data from {data_path}.")

    preprocessed_name = "gfn"
    train_data_path = data_path / "train"
    train_cache_directory = train_data_path / "preprocessed" / preprocessed_name
    _prepare_instances(train_data_path, train_cache_directory)

    test_data_path = data_path / "test"
    test_cache_directory = test_data_path / "preprocessed" / preprocessed_name
    _prepare_instances(test_data_path, test_cache_directory)

    trainset = GraphDataset(train_cache_directory, size=cfg.trainsize)
    testset = GraphDataset(test_cache_directory,  size=cfg.testsize)

    train_batch_size = 1 if cfg.same_graph_across_batch else cfg.batch_size_interact
    train_loader = DataLoader(trainset, batch_size=train_batch_size,
            shuffle=cfg.shuffle, collate_fn=collate_fn, drop_last=False,
            num_workers=cfg.num_workers, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=cfg.test_batch_size,
             shuffle=False, collate_fn=collate_fn, num_workers=cfg.num_workers, pin_memory=True)
    return train_loader, test_loader