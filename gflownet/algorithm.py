import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

from util import TransitionBuffer
from combopt import get_decided, pad_batch, get_parent, get_mdp_class
from network import GIN
from llm import get_tokenizer, get_llm, get_last_hidden_layer

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
                    "condition_dim": cfg.condition_dim, "modulation_type": cfg.modulation, 
                    "dropout": cfg.dropout, "learn_eps": cfg.learn_eps, "aggregator_type": cfg.aggr}
        self.model = GIN(3, 1, graph_level_output=0, **gin_dict).to(device)
        self.model_flow = GIN(3, 0, graph_level_output=1, **gin_dict).to(device)
        if cfg.llm != 'none' and cfg.llm_dim > 0:
            self.tokenizer = get_tokenizer(cfg.llm)
            self.llm = get_llm(cfg.llm, low_cpu_mem_usage=True)
            self.proj = nn.Linear(cfg.llm_dim, cfg.condition_dim)

        self.params = [
            {"params": self.model.parameters(), "lr": cfg.lr},
            {"params": self.model_flow.parameters(), "lr": cfg.zlr},
        ]
        self.optimizer = torch.optim.Adam(self.params)
        self.leaf_coef = cfg.leaf_coef

    def parameters(self):
        return list(self.model.parameters()) + list(self.model_flow.parameters())

    def parse_condition(self, cb):
        if self.cfg.llm != 'none' and self.cfg.llm_dim > 0:
            if not isinstance(cb, torch.Tensor):
                assert isinstance(cb, list), "Condition batch must be either a tensor or a list of strings."
                assert all(isinstance(c, str) for c in cb), "All conditions in list must be strings."
                H = get_last_hidden_layer(cb, self.tokenizer, self.llm).to(self.proj.weight)

                return self.proj(H)
        else:
            return cb

    @torch.no_grad()
    def sample(self, 
            gb, state, done, 
            cb=None, 
            rand_prob=0., temperature=1., reward_exp=None
        ):
        self.model.eval()

        pf_logits = self.model(gb, state, c=cb, reward_exp=reward_exp)[..., 0]   # [..., 0] selects the logits in case of graph level output

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

    @torch.no_grad()
    def rollout(self, 
            gbatch: dgl.DGLGraph, 
            cfg, reward_exp=None, 
            cbatch: torch.Tensor=None, 
            penalty_fn=None
        ):
        cbatch = self.parse_condition(cbatch)

        if penalty_fn is None:
            penalty_fn = lambda s: 0

        env = get_mdp_class(cfg.task)(gbatch, cfg)
        state = env.state

        ##### sample traj
        # state, action, done, reward
        traj_s, traj_r, traj_a, traj_d = [], [], [], []
        while not all(env.done):
            action = self.sample(gbatch, state, env.done, cb=cbatch, rand_prob=cfg.randp, reward_exp=None)
            traj_s.append(state)
            traj_r.append(env.get_log_reward(penalty=penalty_fn(state)))
            traj_a.append(action)
            traj_d.append(env.done)
            state = env.step(action)

        ##### save last state
        traj_s.append(state)
        traj_r.append(env.get_log_reward(penalty=penalty_fn(state)))
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

        batch = gbatch.cpu(), traj_s.cpu(), traj_a.cpu(), traj_d.cpu(), traj_r.cpu(), traj_len.cpu()

        return batch, env.batch_metric(state)

class DetailedBalanceTransitionBuffer(DetailedBalance):
    def __init__(self, cfg, device):
        assert cfg.alg in ["db", "fl"]
        self.forward_looking = (cfg.alg == "fl")
        super(DetailedBalanceTransitionBuffer, self).__init__(cfg, device)

    def train_step(self, *batch, cbatch=None, reward_exp=None, logr_scaler=None):
        self.model.train()
        self.model_flow.train()
        torch.cuda.empty_cache()

        gb, s, logr, a, s_next, logr_next, d = batch
        gb, s, logr, a, s_next, logr_next, d = gb.to(self.device), s.to(self.device), logr.to(self.device), \
                    a.to(self.device), s_next.to(self.device), logr_next.to(self.device), d.to(self.device)
        logr, logr_next = logr_scaler(logr), logr_scaler(logr_next)
        numnode_per_graph = gb.batch_num_nodes().tolist()
        batch_size = gb.batch_size

        total_num_nodes = gb.num_nodes()
        gb_two = dgl.batch([gb, gb])
        s_two = torch.cat([s, s_next], dim=0)
        cbatch = self.parse_condition(cbatch)
        cbatch_two = None if cbatch is None else cbatch.repeat(gb_two.batch_size // cbatch.shape[0], 1)

        logits = self.model(gb_two, s_two, c=cbatch_two, reward_exp=reward_exp)
        _, flows_out = self.model_flow(gb_two, s_two, c=cbatch_two, reward_exp=reward_exp) # (2 * num_graphs, 1)
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