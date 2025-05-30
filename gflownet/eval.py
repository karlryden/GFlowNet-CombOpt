import gzip, pickle
import numpy as np
import torch
from tqdm import tqdm
import dgl
from einops import rearrange
from util import get_sat_fn
from combopt import get_mdp_class

def batch_jaccard(S):
    A = S.unsqueeze(1)
    B = S.unsqueeze(0)

    cap = torch.logical_and(A, B).float().sum(dim=2)
    cup = torch.logical_or(A, B).float().sum(dim=2).clamp(min=1)
    tril_mask = torch.tril(torch.ones(len(S), len(S), dtype=torch.bool, device=S.device), diagonal=-1)

    return (cap[tril_mask] / cup[tril_mask]).mean().item()

@torch.no_grad()
def evaluate(cfg, device, test_loader, alg, train_step, train_data_used, logr_scaler, result, ep):
    torch.cuda.empty_cache()
    num_repeat = 20
    mis_ls, mis_top20_ls, logr_ls, sat_ls = [], [], [], []

    pbar = tqdm(enumerate(test_loader), desc=f"Test Epoch {ep}")
    for batch_idx, (gbatch, constbatch) in pbar:
        gbatch = gbatch.to(device)
        cbatch, ebatch, sat_fn = None, None, None
        if constbatch:
            sat_fn = get_sat_fn()
            if alg.conditioned:
                cbatch = torch.stack([const['embedding'] for const in constbatch]).to(device)
                # cbatch = torch.randn(len(constbatch), cfg.condition_dim).to(device)   # NOTE: For testing

        if cbatch is not None:
            cbatch_proj = alg.proj(cbatch)
            cbatch_proj_rep = cbatch_proj.repeat(num_repeat, 1)

            ebatch = gbatch.ndata['encoding'].to(device)
            ebatch_proj = alg.proj(ebatch)
            gbatch.ndata['encoding_proj'] = ebatch_proj
        else:
            cbatch_proj_rep = None

        gbatch_rep = dgl.batch([gbatch] * num_repeat)

        env = get_mdp_class(cfg.task)(gbatch_rep, cfg)
        state = env.state

        while not all(env.done):
            action = alg.sample(gbatch_rep, state, env.done, cb=cbatch_proj_rep, rand_prob=0.)
            state = env.step(action)

        #### report Jaccard similarity between solutions: https://en.wikipedia.org/wiki/Jaccard_index
        splits = state.split(env.numnode_per_graph)
        reps = [torch.stack(
            splits[i::gbatch.batch_size]
            ) for i in range(gbatch.batch_size)]
        jaccard_rep = [batch_jaccard(rep) for rep in reps]

        logr_rep = logr_scaler(env.get_log_reward(critic=None if sat_fn is None else lambda gb, s: sat_fn(gb, s)[0]))    # unpenalized reward is reported for evaluation
        logr_ls += logr_rep.tolist()
        curr_mis_rep = torch.tensor(env.batch_metric(state))
        curr_mis_rep = rearrange(curr_mis_rep, "(rep b) -> b rep", rep=num_repeat).float()
        mis_ls += curr_mis_rep.mean(dim=1).tolist()
        mis_top20_ls += curr_mis_rep.max(dim=1)[0].tolist()

        if sat_fn is not None:
            sat_rep, mask_rep = sat_fn(gbatch_rep, state)
            sat_rep = rearrange(sat_rep, "(rep b) -> b rep", rep=num_repeat).float()
            mask_rep = rearrange(mask_rep, "(rep b) -> b rep", rep=num_repeat).float()
            sat_ls += (sat_rep * mask_rep).mean(dim=1).tolist()
        else:
            sat_ls += [1.0] * gbatch.batch_size

        pbar.set_postfix(
            metric=f"{np.mean(mis_ls):.2f}",
            sat_rate=f"{np.mean(sat_ls):.2f}"
        )

    print(f"[Eval] Epoch {ep}: ",
          f"Metric={np.mean(mis_ls):.2f}±{np.std(mis_ls):.2f}, "
          f"Top20={np.mean(mis_top20_ls):.2f}, ",
          f"Jaccard={np.mean(jaccard_rep):.2f}±{np.std(jaccard_rep):.2f}, ",
          f"SatRate={np.mean(sat_ls):.2f}±{np.std(sat_ls):.2f}, ",
          f"LogR={np.mean(logr_ls):.2e}±{np.std(logr_ls):.2e}")

    result["set_size"][ep] = np.mean(mis_ls)
    result["logr_scaled"][ep] = np.mean(logr_ls)
    result["jaccard"][ep] = np.mean(jaccard_rep)
    result["sat_rate"][ep] = np.mean(sat_ls)
    result["train_step"][ep] = train_step
    result["train_data_used"][ep] = train_data_used
    pickle.dump(result, gzip.open("./result.json", 'wb'))