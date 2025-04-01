import sys, os
import gzip, pickle
from time import time, sleep
from tqdm import tqdm
import hydra

import random
import numpy as np
import torch
import dgl
from einops import rearrange

from data import get_data_loaders
from util import refine_cfg, seed_torch, get_logr_scaler
from combopt import get_mdp_class
from algorithm import get_alg_buffer
from constrain import get_indicator_fn, batch_indicators, get_penalty_fn

torch.backends.cudnn.benchmark = True

@hydra.main(config_path="configs", config_name="main") # for hydra-core==1.1.0
# @hydra.main(version_base=None, config_path="configs", config_name="main") # for newer hydra
def main(cfg):
    cfg = refine_cfg(cfg)
    device = torch.device(f"cuda:{cfg.device:d}" if torch.cuda.is_available() and cfg.device>=0 else "cpu")
    print(f"Device: {device}")
    alg, buffer = get_alg_buffer(cfg, device)

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
        for batch_idx, (gbatch, constbatch) in pbar:
            gbatch = gbatch.to(device)
            if constbatch:
                cbatch = [const["constraint"] for const in constbatch]
                ibatch = [get_indicator_fn(const["signature"]) for const in constbatch]
                indicator = batch_indicators(gbatch, ibatch)    # NOTE: When using hard-coded indicators
            else:
                cbatch = None
                indicator = None

            penalty_fn = None if indicator is None else get_penalty_fn(cfg, gbatch, indicator)

            gbatch_rep = dgl.batch([gbatch] * num_repeat)   # TODO: Mirror this for constbatch

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
                cbatch = [const["constraint"] for const in constbatch]
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
            batch, metric_ls = alg.rollout(
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

    # evaluate(cfg.epochs, train_step, train_data_used, logr_scaler)
    alg.save(alg_save_path)

if __name__ == "__main__":
    main()