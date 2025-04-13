import os
import random
import numpy as np
import torch
import dgl

from constrain import get_indicator_fn, batch_indicators, get_penalty_fn
from util import get_logr_scaler
from eval import evaluate

def train_loop(cfg, device, alg, buffer, train_loader, test_loader):
    train_logr_scaled_ls = []
    train_metric_ls = []
    train_data_used = 0
    train_step = 0
    metric_best = 0.
    alg_save_path = os.path.abspath("./alg.pt")
    alg_save_path_best = os.path.abspath("./alg_best.pt")
    result = {"set_size": {}, "logr_scaled": {}, "train_data_used": {}, "train_step": {}}

    for ep in range(cfg.epochs):
        for batch_idx, (gbatch, constbatch) in enumerate(train_loader):
            gbatch = gbatch.to(device)
            if constbatch:
                cbatch = [const['constraint'] for const in constbatch]
                ebatch = torch.stack([const['embedding'] for const in constbatch])
                ibatch = [get_indicator_fn(const['signature']) for const in constbatch]
                indicator_fn = batch_indicators(gbatch, ibatch)
            else:
                cbatch, ebatch, indicator_fn = None, None, None

            penalty_fn = get_penalty_fn(cfg, gbatch, indicator_fn) if indicator_fn else None
            process_ratio = max(0., min(1., train_data_used / cfg.annend))
            logr_scaler = get_logr_scaler(cfg, process_ratio)

            if cfg.same_graph_across_batch:
                gbatch = dgl.batch([gbatch] * cfg.batch_size_interact)
            train_data_used += gbatch.batch_size

            # rollout
            batch, metric_ls = alg.rollout(gbatch, cfg, cbatch=ebatch, penalty_fn=penalty_fn)
            buffer.add_batch(batch)
            logr = logr_scaler(batch[-2][:, -1])
            train_logr_scaled_ls += logr.tolist()
            train_metric_ls += metric_ls
            train_traj_len = batch[-1].float().mean().item()

            # train step(s)
            batch_size = min(len(buffer), cfg.batch_size)
            indices = list(range(len(buffer)))
            for _ in range(cfg.tstep):
                if not indices: break
                curr_indices = random.sample(indices, min(len(indices), batch_size))
                sampled_batch = buffer.sample_from_indices(curr_indices)
                train_info = alg.train_step(*sampled_batch, logr_scaler=logr_scaler)
                indices = [i for i in indices if i not in curr_indices]

            if cfg.onpolicy:
                buffer.reset()

            if train_step % cfg.print_freq == 0:
                print(f"Epoch {ep} Data used {train_data_used:.3e}: "
                      f"loss={train_info['train/loss']:.2e}, "
                      f"metric={np.mean(train_metric_ls):.2f}±{np.std(train_metric_ls):.2f}, "
                      f"logR={logr.mean():.2e}, traj_len={train_traj_len:.2f}")

            train_step += 1

            # evaluation
            if batch_idx == 0 or train_step % cfg.eval_freq == 0:
                alg.save(alg_save_path)
                metric_curr = np.mean(train_metric_ls[-1000:])
                if metric_curr > metric_best:
                    metric_best = metric_curr
                    print(f"New best metric: {metric_best:.2f}")
                    alg.save(alg_save_path_best)
                if cfg.eval:
                    evaluate(cfg, device, test_loader, alg, train_step, train_data_used, logr_scaler, result, ep)

    alg.save(alg_save_path)