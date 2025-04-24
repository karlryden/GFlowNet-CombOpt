import os, sys
from omegaconf import open_dict
import random
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import dgl

def refine_cfg(cfg):
    with open_dict(cfg):
        cfg.work_directory = os.getcwd()

        if cfg.task in ["mis", "maxindset", "maxindependentset",]:
            cfg.task = "MaxIndependentSet"
            cfg.wandb_project_name = "MIS"
        elif cfg.task in ["mds", "mindomset", "mindominateset",]:
            cfg.task = "MinDominateSet"
            cfg.wandb_project_name = "MDS"
        elif cfg.task in ["mc", "maxclique",]:
            cfg.task = "MaxClique"
            cfg.wandb_project_name = "MaxClique"
        elif cfg.task in ["mcut", "maxcut",]:
            cfg.task = "MaxCut"
            cfg.wandb_project_name = "MaxCut"
        else:
            raise NotImplementedError

        # architecture
        assert cfg.arch in ["gin"]

        # data
        if "rb" in cfg.input:
            cfg.data_type = cfg.input.upper()
        elif "ba" in cfg.input:
            cfg.data_type = cfg.input.upper()
        else:
            raise NotImplementedError

    return cfg

######### Reward utils

def get_logr_scaler(cfg, process_ratio=1., reward_exp=None):
    if reward_exp is None:
        reward_exp = float(cfg.reward_exp)

    if cfg.anneal == "linear":
        process_ratio = max(0., min(1., process_ratio)) # from 0 to 1
        reward_exp = reward_exp * process_ratio +\
                     float(cfg.reward_exp_init) * (1 - process_ratio)
    elif cfg.anneal == "none":
        pass
    else:
        raise NotImplementedError

    # (R/T)^beta -> (log R - log T) * beta
    def logr_scaler(sol_size, gbatch=None):
        logr = sol_size
        return logr * reward_exp
    return logr_scaler

def get_sat_fn():
    def sat_fn(gbatch, state):
        dec = ~(state == 2)      # decided
        inc = (state == 1)      # included

        want = gbatch.ndata['wanted'].to(dtype=torch.bool)

        liable = dec & want     # liable to be penalized
        sat = want & inc        # satisfied

        cum_num_node = gbatch.batch_num_nodes().cumsum(dim=0)

        start = 0
        sat_rates = torch.empty(gbatch.batch_size, device=state.device)

        for k, end in enumerate(cum_num_node):
            w = liable[start:end]
            s = sat[start:end]

            sat_rates[k] = 1.0 if not w.sum() else s.sum() / w.sum()

            start = end

        return sat_rates

    return sat_fn

######### Pytorch Utils

def seed_torch(seed, verbose=True):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    if verbose:
        print("==> Set seed to {:}".format(seed))

def pad_batch(vec, dim_per_instance, padding_value, dim=0, batch_first=True):
    # dim_per_instance: list of int
    tupllle = torch.split(vec, dim_per_instance, dim=dim)
    pad_tensor = pad_sequence(tupllle, batch_first=batch_first, padding_value=padding_value)
    return pad_tensor

def ema_update(model, ema_model, alpha=0.999):
    for param, ema_param in zip(model.parameters(), ema_model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

######### Replay Buffer Utils

from multiprocessing import Pool
def imap_unordered_bar(func, args, n_processes=2):
    p = Pool(n_processes)
    args = list(args)
    res_list = []
    for i, res in enumerate(p.imap_unordered(func, args)):
        if isinstance(res, list):
            res_list.extend(res)
        else:
            res_list.append(res)
    p.close()
    p.join()
    return res_list

class TransitionBuffer(object):
    def __init__(self, size, cfg):
        self.size = size
        self.buffer = []
        self.pos = 0

    def reset(self):
        self.buffer = []
        self.pos = 0

    def add_batch(self, batch):
        gb, cb, traj_s, traj_a, traj_d, traj_r, traj_len = batch
        numnode_per_graph = gb.batch_num_nodes().tolist()
        batch_size = gb.batch_size  # num_graph
        g_list = dgl.unbatch(gb)
        traj_s_tuple = torch.split(traj_s, numnode_per_graph, dim=0)

        for b_idx in range(batch_size):
            g_bidx = g_list[b_idx]
            c_bidx = None if cb is None else cb[b_idx]
            traj_len_bidx = traj_len[b_idx]
            traj_s_bidx = traj_s_tuple[b_idx][..., :traj_len_bidx]
            traj_a_bidx = traj_a[b_idx, :traj_len_bidx - 1]
            traj_d_bidx = traj_d[b_idx, 1:traj_len_bidx] # "done" after transition
            traj_r_bidx = traj_r[b_idx, :traj_len_bidx]

            for i in range(traj_len_bidx - 1):
                transition = (g_bidx, c_bidx, traj_s_bidx[:, i], traj_r_bidx[i], traj_a_bidx[i],
                              traj_s_bidx[:, i + 1], traj_r_bidx[i+1], traj_d_bidx[i])
                self.add_single_transition(transition)

    def add_single_transition(self, inp):
        if len(self.buffer) < self.size:
            self.buffer.append(None)
        self.buffer[self.pos] = inp
        self.pos = (self.pos + 1) % self.size

    @staticmethod
    def transition_collate_fn(transition_ls):
        gbatch, cbatch, s_batch, logr_batch, a_batch, s_next_batch, logr_next_batch, d_batch = \
            zip(*transition_ls)  # s_batch is a list of tensors
        gbatch = dgl.batch(gbatch)

        conditioned = cbatch and (cbatch[0] is not None and isinstance(cbatch[0], torch.Tensor))
        cbatch = None if not conditioned else torch.stack(cbatch, dim=0)
        s_batch = torch.cat(s_batch, dim=0)  # (sum of # nodes in batch, )
        s_next_batch = torch.cat(s_next_batch, dim=0)

        logr_batch = torch.stack(logr_batch, dim=0)
        logr_next_batch = torch.stack(logr_next_batch, dim=0)
        a_batch = torch.stack(a_batch, dim=0)
        d_batch = torch.stack(d_batch, dim=0)

        return gbatch, cbatch, s_batch, logr_batch, a_batch, s_next_batch, logr_next_batch, d_batch

    def sample(self, batch_size):
        # random.sample: without replacement
        batch = random.sample(self.buffer, batch_size) # list of transition tuple
        return self.transition_collate_fn(batch)

    def sample_from_indices(self, indices):
        batch = [self.buffer[i] for i in indices]
        return self.transition_collate_fn(batch)

    def __len__(self):
        return len(self.buffer)