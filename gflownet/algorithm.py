import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

from util import TransitionBuffer, pad_batch
from combopt import get_decided, get_parent, get_mdp_class
from network import GIN

def sample_from_logits(pf_logits, gb, state, done, rand_prob=0.):
    numnode_per_graph = gb.batch_num_nodes().tolist()
    pf_logits[get_decided(state)] = -np.inf
    pf_logits = pad_batch(pf_logits, numnode_per_graph, padding_value=-np.inf)

    # use -1 to denote impossible action (e.g. for done graphs)
    action = torch.full([gb.batch_size,], -1, dtype=torch.long, device=gb.device)
    pf_undone = pf_logits[~done].softmax(dim=1)
    action[~done] = torch.multinomial(pf_undone, num_samples=1).squeeze(-1) # 1 state chosen per graph

    if rand_prob > 0.:
        unif_pf_undone = torch.isfinite(pf_logits[~done]).float()
        rand_action_unodone = torch.multinomial(unif_pf_undone, num_samples=1).squeeze(-1)
        rand_mask = torch.rand_like(rand_action_unodone.float()) < rand_prob
        action[~done][rand_mask] = rand_action_unodone[rand_mask]
    return action

def get_alg_buffer(cfg, device):
    assert cfg.alg in ["db", "fl"]
    buffer = TransitionBuffer(cfg.tranbuff_size, cfg)
    alg = DetailedBalanceTransitionBuffer(cfg, device)
    return alg, buffer

class DetailedBalance(object):
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.task = cfg.task
        self.device = device

        assert cfg.arch in ["gin"]
        gin_dict = {"hidden_dim": cfg.hidden_dim, "num_layers": cfg.hidden_layer,
                    "dropout": cfg.dropout, "learn_eps": cfg.learn_eps,
                    "aggregator_type": cfg.aggr, "modulation_type": cfg.condition}
        self.model = GIN(3, 1, graph_level_output=0, **gin_dict).to(device)
        self.model_flow = GIN(3, 0, graph_level_output=1, **gin_dict).to(device)

        if cfg.condition != 'none': assert cfg.condition_dim > 0, "Conditioning dimension should be greater than 0 -- check config."
        if cfg.condition_dim > 0: assert cfg.condition != 'none', "Conditioning method not defined -- check config."

        self.conditioned = (cfg.condition_dim > 0) and (cfg.condition != 'none')
        if self.conditioned:
            self.proj = nn.Linear(cfg.condition_dim, cfg.hidden_dim)
            self.proj = self.proj.to(self.device)

        self.params = [
            {"params": self.model.parameters(), "lr": cfg.lr},
            {"params": self.model_flow.parameters(), "lr": cfg.zlr},
        ]
        self.optimizer = torch.optim.Adam(self.params)
        self.leaf_coef = cfg.leaf_coef

    def parameters(self):
        return list(self.model.parameters()) + list(self.model_flow.parameters())

    @torch.no_grad()
    def sample(self, 
            gb, state, done, 
            cb=None, 
            rand_prob=0., temperature=1.
        ):
        self.model.eval()

        pf_logits = self.model(gb, state, c=cb)[..., 0]   # [..., 0] selects the logits in case of graph level output

        return sample_from_logits(pf_logits / temperature, gb, state, done, rand_prob=rand_prob)

    def save(self, path):
        save_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        save_dict.update({"model_flow": self.model_flow.state_dict()})
        torch.save(save_dict, path)
        print(f"Saved to {path}")

    def load(self, path):
        save_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(save_dict["model"])
        self.model_flow.load_state_dict(save_dict["model_flow"])
        self.optimizer.load_state_dict(save_dict["optimizer"])
        print(f"Loaded from {path}")

    def train_step(self, *batch):
        raise NotImplementedError

    def rollout(self, 
            gbatch: dgl.DGLGraph, 
            cfg,
            cbatch: torch.Tensor=None, 
            critic=None
        ):
        env = get_mdp_class(cfg.task)(gbatch, cfg)
        state = env.state

        if cbatch is not None:
            cbatch = cbatch.to(self.device)
            cbatch_proj = self.proj(cbatch)

            ebatch = gbatch.ndata['encoding'].to(self.device)
            ebatch_proj = self.proj(ebatch)
            gbatch.ndata['encoding_proj'] = ebatch_proj
        else:
            cbatch_proj = None

        ##### sample traj
        # state, action, done, reward
        traj_s, traj_r, traj_a, traj_d = [], [], [], []
        while not all(env.done):
            action = self.sample(gbatch, state, env.done, cb=cbatch_proj, rand_prob=cfg.randp)
            traj_s.append(state)
            traj_r.append(env.get_log_reward(critic=critic))
            traj_a.append(action)
            traj_d.append(env.done)
            state = env.step(action)

        ##### save last state
        traj_s.append(state)
        traj_r.append(env.get_log_reward(critic=critic))
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

        if cbatch is not None:
            cbatch = cbatch.cpu()

        batch = gbatch.cpu(), cbatch, traj_s.cpu(), traj_a.cpu(), traj_d.cpu(), traj_r.cpu(), traj_len.cpu()

        return batch, env.batch_metric(state)

class DetailedBalanceTransitionBuffer(DetailedBalance):
    def __init__(self, cfg, device):
        assert cfg.alg in ["db", "fl"]
        self.forward_looking = (cfg.alg == "fl")
        super(DetailedBalanceTransitionBuffer, self).__init__(cfg, device)

    def train_step(self, *batch, logr_scaler=None):
        self.model.train()
        self.model_flow.train()
        if self.conditioned:
            self.proj.train()

        gb, cb, s, logr, a, s_next, logr_next, d = batch

        if cb is not None:
            cb = cb.to(self.device)
            cb_proj = self.proj(cb)

            eb = gb.ndata['encoding'].to(self.device)
            eb_proj = self.proj(eb)
            gb.ndata['encoding_proj'] = eb_proj
        else:
            cb_proj = None

        gb, s, logr, a, s_next, logr_next, d = gb.to(self.device), s.to(self.device), logr.to(self.device), \
                    a.to(self.device), s_next.to(self.device), logr_next.to(self.device), d.to(self.device)
        logr, logr_next = logr_scaler(logr), logr_scaler(logr_next)
        numnode_per_graph = gb.batch_num_nodes().tolist()
        batch_size = gb.batch_size

        total_num_nodes = gb.num_nodes()
        gb_two = dgl.batch([gb, gb])
        s_two = torch.cat([s, s_next], dim=0)
        cb_proj_two = None if cb_proj is None else torch.repeat_interleave(cb_proj, repeats=2, dim=0)

        logits = self.model(gb_two, s_two, c=cb_proj_two)
        _, flows_out = self.model_flow(gb_two, s_two, c=cb_proj_two) # (2 * num_graphs, 1)
        flows, flows_next = flows_out[:batch_size, 0], flows_out[batch_size:, 0]

        pf_logits = logits[:total_num_nodes, ..., 0]
        pf_logits[get_decided(s)] = -np.inf
        pf_logits = pad_batch(pf_logits, numnode_per_graph, padding_value=-np.inf)
        log_pf = F.log_softmax(pf_logits, dim=1)[torch.arange(batch_size), a]

        log_pb = torch.tensor([torch.log(1 / get_parent(s_, self.task).sum())
         for s_ in torch.split(s_next, numnode_per_graph, dim=0)]).to(self.device)

        if self.forward_looking:
            flows_next.masked_fill_(d, 0.) # \tilde F(x) = F(x) / R(x) = 1, log 1 = 0
            lhs = logr + flows + log_pf # (bs,)
            rhs = logr_next + flows_next + log_pb
            loss = (lhs - rhs).pow(2)
            loss = loss.mean()
        else:
            flows_next = torch.where(d, logr_next, flows_next)
            lhs = flows + log_pf # (bs,)
            rhs = flows_next + log_pb
            losses = (lhs - rhs).pow(2)
            loss = (losses[d].sum() * self.leaf_coef + losses[~d].sum()) / batch_size

        return_dict = {"train/loss": loss.item()}
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return return_dict