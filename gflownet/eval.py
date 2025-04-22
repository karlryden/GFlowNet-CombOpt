import gzip, pickle
import numpy as np
import torch
from tqdm import tqdm
import dgl
from einops import rearrange
from util import get_sat_fn
from combopt import get_mdp_class

@torch.no_grad()
def evaluate(cfg, device, test_loader, alg, train_step, train_data_used, logr_scaler, result, ep):
    torch.cuda.empty_cache()
    num_repeat = 20
    mis_ls, mis_top20_ls, logr_ls, sat_ls = [], [], [], []

    pbar = tqdm(enumerate(test_loader), desc=f"Test Epoch {ep}")
    for batch_idx, (gbatch, constbatch) in pbar:
        gbatch = gbatch.to(device)
        if constbatch:
            cbatch = [const['constraint'] for const in constbatch]
            ebatch = torch.stack([c['embedding'] for c in constbatch]).to(device)
            # ebatch = torch.randn(len(constbatch), cfg.condition_dim).to(device) # NOTE: For testing
            sat_fn = get_sat_fn()
        else:
            cbatch, ebatch, sat_fn = None, None, None

        gbatch_rep = dgl.batch([gbatch] * num_repeat)
        ebatch_rep = None if ebatch is None else torch.repeat_interleave(ebatch, num_repeat, dim=0)

        env = get_mdp_class(cfg.task)(gbatch_rep, cfg)
        state = env.state

        while not all(env.done):
            proj_rep = ebatch if not constbatch else alg.proj(ebatch_rep)
            action = alg.sample(gbatch_rep, state, env.done, cb=proj_rep, rand_prob=0.)
            state = env.step(action)

        logr_rep = logr_scaler(env.get_log_reward())    # unpenalized reward is reported for evaluation
        logr_ls += logr_rep.tolist()
        curr_mis_rep = torch.tensor(env.batch_metric(state))
        curr_mis_rep = rearrange(curr_mis_rep, "(rep b) -> b rep", rep=num_repeat).float()
        mis_ls += curr_mis_rep.mean(dim=1).tolist()
        mis_top20_ls += curr_mis_rep.max(dim=1)[0].tolist()

        if sat_fn is not None:
            sat_rep = sat_fn(gbatch_rep, state)
            sat_rep = rearrange(sat_rep, "(rep b) -> b rep", rep=num_repeat).float()
            sat_ls += sat_rep.mean(dim=1).tolist()
        else:
            sat_ls += [1.0] * gbatch.batch_size

        pbar.set_postfix(
            metric=f"{np.mean(mis_ls):.2f}",
            sat_rate=f"{np.mean(sat_ls):.2f}"
        )

    print(f"[Eval] Epoch {ep}: ",
          f"Metric={np.mean(mis_ls):.2f}±{np.std(mis_ls):.2f}, "
          f"Top20={np.mean(mis_top20_ls):.2f}, ",
          f"SatRate={np.mean(sat_ls):.2f}±{np.std(sat_ls):.2f}, ",
          f"LogR={np.mean(logr_ls):.2e}±{np.std(logr_ls):.2e}")

    result["set_size"][ep] = np.mean(mis_ls)
    result["logr_scaled"][ep] = np.mean(logr_ls)
    result["sat_rate"][ep] = np.mean(sat_ls)
    result["train_step"][ep] = train_step
    result["train_data_used"][ep] = train_data_used
    pickle.dump(result, gzip.open("./result.json", 'wb'))