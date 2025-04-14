import gzip, pickle
import numpy as np
import torch
from tqdm import tqdm
import dgl
from einops import rearrange
from constrain import get_indicator_fn, batch_indicators, get_penalty_fn
from combopt import get_mdp_class

@torch.no_grad()
def evaluate(cfg, device, test_loader, alg, train_step, train_data_used, logr_scaler, result, ep):
    torch.cuda.empty_cache()
    num_repeat = 20
    mis_ls, mis_top20_ls, logr_ls = [], [], []

    pbar = tqdm(enumerate(test_loader), desc=f"Test Epoch {ep}")
    for batch_idx, (gbatch, constbatch) in pbar:
        gbatch = gbatch.to(device)
        if constbatch:
            ebatch = torch.stack([c['embedding'] for c in constbatch]).to(device)
            ibatch = [get_indicator_fn(c['signature']) for c in constbatch]
            indicator_fn = batch_indicators(gbatch, ibatch)
        else:
            ebatch, indicator_fn = None, None

        penalty_fn = (lambda s: torch.tensor(0., device=s.device)) if indicator_fn is None else \
                     get_penalty_fn(cfg, gbatch, indicator_fn)

        gbatch_rep = dgl.batch([gbatch] * num_repeat)
        ebatch_rep = None if ebatch is None else torch.repeat_interleave(ebatch, num_repeat, dim=0)

        env = get_mdp_class(cfg.task)(gbatch_rep, cfg)
        state = env.state

        while not all(env.done):
            action = alg.sample(gbatch_rep, state, env.done, cb=alg.proj(ebatch_rep), rand_prob=0.)
            state = env.step(action)

        logr_rep = logr_scaler(env.get_log_reward(penalty=penalty_fn(state)))
        logr_ls += logr_rep.tolist()
        curr_mis_rep = torch.tensor(env.batch_metric(state))
        curr_mis_rep = rearrange(curr_mis_rep, "(rep b) -> b rep", rep=num_repeat).float()
        mis_ls += curr_mis_rep.mean(dim=1).tolist()
        mis_top20_ls += curr_mis_rep.max(dim=1)[0].tolist()

        pbar.set_postfix(metric=np.mean(mis_ls))

    print(f"[Eval] Epoch {ep}: Metric={np.mean(mis_ls):.2f}±{np.std(mis_ls):.2f}, "
          f"Top20={np.mean(mis_top20_ls):.2f}, "
          f"LogR={np.mean(logr_ls):.2e}±{np.std(logr_ls):.2e}")

    result["set_size"][ep] = np.mean(mis_ls)
    result["logr_scaled"][ep] = np.mean(logr_ls)
    result["train_step"][ep] = train_step
    result["train_data_used"][ep] = train_data_used
    pickle.dump(result, gzip.open("./result.json", 'wb'))