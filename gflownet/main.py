import sys, os
import gzip, pickle
from time import time, sleep
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, open_dict, OmegaConf

import random
import numpy as np
import torch
import dgl
from einops import rearrange, reduce, repeat

from data import get_data_loaders
from util import seed_torch, TransitionBuffer, get_mdp_class
from algorithm import DetailedBalanceTransitionBuffer

from transformers import AutoModelForCausalLM, AutoTokenizer

torch.backends.cudnn.benchmark = True


def get_alg_buffer(cfg, device):
    assert cfg.alg in ["db", "fl"]
    buffer = TransitionBuffer(cfg.tranbuff_size, cfg)
    alg = DetailedBalanceTransitionBuffer(cfg, device)
    return alg, buffer

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

# TODO: Move these 3 to util? 
# NOTE: I am worried about the tensors ending up on a different device to state.
def get_indicator_fn(signature):
    constraint_type = signature['type']
    constrained_node = signature['node']

    if constraint_type == 'none':
        return lambda s: torch.tensor([1.0])
    
    elif constraint_type == 'inclusion':
        return lambda s: (s[constrained_node] == 1).float()
    
    elif constraint_type == 'exclusion':
        return lambda s: (s[constrained_node] == 0).float()

def batch_indicators(gbatch, ibatch):
    cum_num_node = gbatch.batch_num_nodes().cumsum(dim=0)

    def indicator_fn(state):
        sat = torch.empty(gbatch.batch_size, device=gbatch.device)

        for k, indicator in enumerate(ibatch):
            start = 0 if k == 0 else cum_num_node[k-1]
            end = cum_num_node[k]

            sat[k] = indicator(state[start:end])

        return sat
    
    return indicator_fn

def get_penalty_fn(cfg, gbatch, critic):
    def linear_penalty(state):
        penalty_weight, = cfg.pargs
        return penalty_weight * gbatch.batch_num_nodes() * critic(state)

    if cfg.penalty == "none":
        return lambda state: torch.tensor(0., device=gbatch.device)
    
    elif cfg.penalty == "linear":
        return linear_penalty

    else:
        raise NotImplementedError

def get_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

def get_llm(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype="auto", 
        low_cpu_mem_usage=True, 
        device_map="auto"
    )

    model.eval()

    return model

def get_last_hidden_layer(prompts, tokenizer, model):
    print(f'prompt: {prompts}')

    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)

    def mean_pool(hidden_states, attention_mask):
        # hidden_states: [batch_size, seq_len, hidden_dim]
        # attention_mask: [batch_size, seq_len]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
        sum_embeddings = (hidden_states * input_mask_expanded).sum(dim=1)
        sum_mask = input_mask_expanded.sum(dim=1)
        return sum_embeddings / sum_mask

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]

    return mean_pool(last_hidden, inputs["attention_mask"])

def refine_cfg(cfg):
    with open_dict(cfg):
        cfg.device = cfg.d
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

        # log reward shape
        cfg.reward_exp = cfg.rexp
        cfg.reward_exp_init = cfg.rexpit
        if cfg.anneal in ["lin"]:
            cfg.anneal = "linear"
        if cfg.penalty in ["lin"]:
            cfg.penalty = "linear"

        # training
        cfg.batch_size = cfg.bs
        cfg.batch_size_interact = cfg.bsit
        cfg.leaf_coef = cfg.lc
        cfg.same_graph_across_batch = cfg.sameg

        # data
        cfg.test_batch_size = cfg.tbs
        if "rb" in cfg.input:
            cfg.data_type = cfg.input.upper()
        elif "ba" in cfg.input:
            cfg.data_type = cfg.input.upper()
        else:
            raise NotImplementedError

    del cfg.d, cfg.rexp, cfg.rexpit, cfg.bs, cfg.bsit, cfg.lc, cfg.sameg, cfg.tbs
    return cfg

@torch.no_grad()
def rollout(
    gbatch: dgl.DGLGraph, 
    cfg, alg, reward_exp=None, 
    cbatch: torch.Tensor=None, 
    penalty_fn=None,
):
    if cbatch is not None:
        cbatch = alg.proj(cbatch)
        if cbatch.ndimension() == 1:
            assert gbatch.batch_size == 1

        elif cbatch.ndimension() == 2:
            assert len(cbatch) == gbatch.batch_size

        else:
            raise ValueError

    if penalty_fn is None:
        p = 0

    env = get_mdp_class(cfg.task)(gbatch, cfg)
    state = env.state

    ##### sample traj
    traj_s, traj_r, traj_a, traj_d = [], [], [], []
    while not all(env.done):
        action = alg.sample(gbatch, state, env.done, cb=cbatch, rand_prob=cfg.randp, reward_exp=None)
        if penalty_fn is not None:
            p = penalty_fn(state)

        traj_s.append(state)
        traj_r.append(env.get_log_reward(penalty=p))
        traj_a.append(action)
        traj_d.append(env.done)
        state = env.step(action)

    ##### save last state
    traj_s.append(state)
    traj_r.append(env.get_log_reward(penalty=(p if penalty_fn is None else penalty_fn(state))))
    traj_d.append(env.done)
    assert len(traj_s) == len(traj_a) + 1 == len(traj_r) == len(traj_d)

    traj_s = torch.stack(traj_s, dim=1) # (sum of #node per graph in batch, max_traj_len)
    traj_r = torch.stack(traj_r, dim=1) # (batch_size, max_traj_len)
    traj_a = torch.stack(traj_a, dim=1) # (batch_size, max_traj_len-1)
    """
    traj_a is tensor like 
    [ 4, 30, 86, 95, 96, 29, -1, -1],
    [47, 60, 41, 11, 55, 64, 80, -1],
    [26, 38, 13,  5,  9, -1, -1, -1]
    """
    traj_d = torch.stack(traj_d, dim=1) # (batch_size, max_traj_len)
    """
    traj_d is tensor like 
    [False, False, False, False, False, False,  True,  True,  True],
    [False, False, False, False, False, False, False,  True,  True],
    [False, False, False, False, False,  True,  True,  True,  True]
    """
    traj_len = 1 + torch.sum(~traj_d, dim=1) # (batch_size, )

    ##### graph, state, action, done, reward, trajectory length
    batch = gbatch.cpu(), traj_s.cpu(), traj_a.cpu(), traj_d.cpu(), traj_r.cpu(), traj_len.cpu()
    return batch, env.batch_metric(state)


@hydra.main(config_path="configs", config_name="main") # for hydra-core==1.1.0
# @hydra.main(version_base=None, config_path="configs", config_name="main") # for newer hydra
def main(cfg: DictConfig):
    cfg = refine_cfg(cfg)
    device = torch.device(f"cuda:{cfg.device:d}" if torch.cuda.is_available() and cfg.device>=0 else "cpu")
    print(f"Device: {device}")
    alg, buffer = get_alg_buffer(cfg, device)
    tokenizer = None if cfg.llm == 'none' else get_tokenizer(cfg.llm)
    llm = None if cfg.llm == 'none' else get_llm(cfg.llm)

    seed_torch(cfg.seed)
    print(str(cfg))
    print(f"Work directory: {os.getcwd()}")

    train_loader, test_loader = get_data_loaders(cfg)
    trainset_size = len(train_loader.dataset)
    print(f"Trainset size: {trainset_size}")
    alg_save_path = os.path.abspath("./alg.pt")
    alg_save_path_best = os.path.abspath("./alg_best.pt")
    train_data_used = 0
    train_step = 0
    train_logr_scaled_ls = []
    train_metric_ls = []
    metric_best = 0.
    result = {"set_size": {}, "logr_scaled": {}, "train_data_used": {}, "train_step": {}, }

    @torch.no_grad()
    def evaluate(ep, train_step, train_data_used, logr_scaler):
        torch.cuda.empty_cache()
        num_repeat = 20
        mis_ls, mis_top20_ls = [], []
        logr_ls = []
        pbar = tqdm(enumerate(test_loader))
        pbar.set_description(f"Test Epoch {ep:2d} Data used {train_data_used:5d}")
        for batch_idx, gbatch in pbar:
            gbatch = gbatch.to(device)
            gbatch_rep = dgl.batch([gbatch] * num_repeat)

            env = get_mdp_class(cfg.task)(gbatch_rep, cfg)
            state = env.state
            while not all(env.done):
                action = alg.sample(gbatch_rep, state, env.done, rand_prob=0.)
                state = env.step(action)

            logr_rep = logr_scaler(env.get_log_reward())
            logr_ls += logr_rep.tolist()
            curr_mis_rep = torch.tensor(env.batch_metric(state))
            curr_mis_rep = rearrange(curr_mis_rep, "(rep b) -> b rep", rep=num_repeat).float()
            mis_ls += curr_mis_rep.mean(dim=1).tolist()
            mis_top20_ls += curr_mis_rep.max(dim=1)[0].tolist()
            pbar.set_postfix({"Metric": f"{np.mean(mis_ls):.2f}+-{np.std(mis_ls):.2f}"})

        print(f"Test Epoch{ep:2d} Data used{train_data_used:5d}: "
              f"Metric={np.mean(mis_ls):.2f}+-{np.std(mis_ls):.2f}, "
              f"top20={np.mean(mis_top20_ls):.2f}, "
              f"LogR scaled={np.mean(logr_ls):.2e}+-{np.std(logr_ls):.2e}")

        result["set_size"][ep] = np.mean(mis_ls)
        result["logr_scaled"][ep] = np.mean(logr_ls)
        result["train_step"][ep] = train_step
        result["train_data_used"][ep] = train_data_used
        pickle.dump(result, gzip.open("./result.json", 'wb'))

    for ep in range(cfg.epochs):
        for batch_idx, (gbatch, constbatch) in enumerate(train_loader):
            if constbatch:
                if (tokenizer is not None) and (llm is not None):
                    cbatch = get_last_hidden_layer(
                        [const["constraint"] for const in constbatch], tokenizer, llm
                    )
                    cbatch = cbatch.to(dtype=torch.float32) # on CPU
                else:
                    cbatch = torch.rand((len(constbatch), cfg.condition_dim))

                ibatch = [get_indicator_fn(const["signature"]) for const in constbatch]
                indicator = batch_indicators(gbatch, ibatch)    # NOTE: When using hard-coded indicators
            else:
                cbatch = None
                indicator = None

            penalty_fn = None if indicator is None else get_penalty_fn(cfg, gbatch, indicator)

            reward_exp = None
            process_ratio = max(0., min(1., train_data_used / cfg.annend))
            logr_scaler = get_logr_scaler(cfg, process_ratio=process_ratio, reward_exp=reward_exp)

            train_logr_scaled_ls = train_logr_scaled_ls[-5000:]
            train_metric_ls = train_metric_ls[-5000:]
            gbatch = gbatch.to(device)
            if cfg.same_graph_across_batch:
                gbatch = dgl.batch([gbatch] * cfg.batch_size_interact)
            train_data_used += gbatch.batch_size

            # indicator = lambda s: critic(gbatch, s, cbatch) # NOTE: When critic network is implemented

            ###### rollout
            batch, metric_ls = rollout(
                gbatch, 
                cfg, alg, 
                cbatch=cbatch, 
                penalty_fn=penalty_fn)

            buffer.add_batch(batch)

            logr = logr_scaler(batch[-2][:, -1])
            train_logr_scaled_ls += logr.tolist()
            train_logr_scaled = logr.mean().item()
            train_metric_ls += metric_ls
            train_traj_len = batch[-1].float().mean().item()

            ##### train
            batch_size = min(len(buffer), cfg.batch_size)
            indices = list(range(len(buffer)))
            for _ in range(cfg.tstep):
                if len(indices) == 0:
                    break
                curr_indices = random.sample(indices, min(len(indices), batch_size))
                batch = buffer.sample_from_indices(curr_indices)
                train_info = alg.train_step(*batch, cbatch=cbatch, reward_exp=reward_exp, logr_scaler=logr_scaler)  # TODO: Batch c with batch?
                indices = [i for i in indices if i not in curr_indices]

            if cfg.onpolicy:
                buffer.reset()

            if train_step % cfg.print_freq == 0:
                print(f"Epoch {ep:2d} Data used {train_data_used:.3e}: loss={train_info['train/loss']:.2e}, "
                      + (f"LogZ={train_info['train/logZ']:.2e}, " if cfg.alg in ["tb", "tbbw"] else "")
                      + f"metric size={np.mean(train_metric_ls):.2f}+-{np.std(train_metric_ls):.2f}, "
                      + f"LogR scaled={train_logr_scaled:.2e} traj_len={train_traj_len:.2f}")

            train_step += 1

            ##### eval
            if batch_idx == 0 or train_step % cfg.eval_freq == 0:
                alg.save(alg_save_path)
                metric_curr = np.mean(train_metric_ls[-1000:])
                if metric_curr > metric_best:
                    metric_best = metric_curr
                    print(f"best metric: {metric_best:.2f} at step {train_data_used:.3e}")
                    alg.save(alg_save_path_best)
                if cfg.eval:
                    evaluate(ep, train_step, train_data_used, logr_scaler)

    evaluate(cfg.epochs, train_step, train_data_used, logr_scaler)
    alg.save(alg_save_path)

def test():
    import networkx as nx
    from network import GIN

    from dgl.nn.pytorch.conv import GINConv

    g = dgl.DGLGraph()
    g.add_nodes(4)
    g.add_edges([0, 1, 2, 3], [1, 2, 3, 0])

    n = g.to_networkx()
    nx.draw(n, with_labels=True)

    task = "mis"
    one_hot = {
        "mis": [1, 0, 0, 0],
        "mc": [0, 1, 0, 0],
        "mcut": [0, 0, 1, 0],
        "mds": [0, 0, 0, 1],
    }

    # gin = GIN(3, g.num_nodes(), hidden_dim=8)
    policy = GIN(
        3, 
        g.num_nodes(), 
        hidden_dim=8, 
        condition_dim=4, 
        modulation_type="film"
    )

if __name__ == "__main__":
    main()

    # test()