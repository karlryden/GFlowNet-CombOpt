import torch
import dgl
import dgl.function as fn

from util import pad_batch

######### MDP Utils

def get_decided(state, task="MaxIndependentSet") -> torch.bool:
    # assert isinstance(state, torch.LongTensor) or isinstance(state, torch.cuda.LongTensor) # cannot be used in jit
    assert state.dtype == torch.long
    if task in ["MaxIndependentSet", "MinDominateSet", "MaxClique", "MaxCut"]:
        return state != 2
    else:
        raise NotImplementedError

# in unif pb, to calculate number of parents
# also the number of steps taken (i.e. reward)
def get_parent(state, task="MaxIndependentSet") -> torch.bool:
    assert state.dtype == torch.long
    if task in ["MaxIndependentSet", "MaxClique", "MaxCut"]:
        return state == 1
    elif task in ["MinDominateSet"]:
        return state == 0
    else:
        raise NotImplementedError

# Parent class for all combinatorial optimization MDPs. 
# Inheriting are defined by post-step-processing (pruning) and reward calculation.
class GraphCombOptMDP(object):
    def __init__(self, gbatch, cfg):
        self.cfg = cfg
        self.task = cfg.task
        self.device = gbatch.device
        self.gbatch = gbatch
        self.batch_size = gbatch.batch_size
        self.numnode_per_graph = gbatch.batch_num_nodes().tolist()
        cum_num_node = gbatch.batch_num_nodes().cumsum(dim=0)
        self.cum_num_node = torch.cat([torch.tensor([0]).to(cum_num_node), cum_num_node])[:-1]
        self._state = torch.full((gbatch.num_nodes(),), 2, dtype=torch.long, device=self.device)

    @property
    def state(self):
        return self._state

    @property
    def done(self):
        decided_tensor = pad_batch(self.get_decided_mask(self.state), self.numnode_per_graph, padding_value=True)
        return torch.all(decided_tensor, dim=1)

    def set_state(self, state):
        self._state = state

    def step(self, action):
        state = self.state.clone()

        # label the selected node to be "1"
        action_node_idx = (self.cum_num_node + action)[~self.done]
        # make sure the idx of action hasn't been decided
        assert torch.all(~self.get_decided_mask(state[action_node_idx]))
        state[action_node_idx] = 1

        state = self._prune(state)

        self.set_state(state)

        return state

    def _prune(self, state):
        # Problem specific post-processing from the appendix

        raise NotImplementedError

    def get_log_reward(self, state=None, critic=None):
        if state is None:
            state = self.state.clone()

        padded_state = pad_batch(state, self.numnode_per_graph, padding_value=2)

        E = self.energy_fn(padded_state)
        if critic is not None:
            pc = self.cfg.penalty_coef
            score = critic(self.gbatch, state)

            E = E * (1 - pc * score)

        return -E

    def energy_fn(self, state):
        # Problem-specific energy function implemented as in the appendix

        raise NotImplementedError

    def batch_metric(self, state): # return a list of metric
        return self.get_log_reward(state).tolist()

    def get_decided_mask(self, state=None):
        state = self.state if state is None else state
        return get_decided(state, self.task)

def get_mdp_class(task):
    if task == "MaxIndependentSet":
        return MaxIndSetMDP
    elif task == "MaxClique":
        return MaxCliqueMDP
    elif task == "MinDominateSet":
        return MinDominateSetMDP
    elif task == "MaxCut":
        return MaxCutMDP
    else:
        raise NotImplementedError

class MaxIndSetMDP(GraphCombOptMDP):
    # MDP conditioned on a batch of graphs
    # state: 0: not selected, 1: selected, 2: undecided
    def __init__(self, gbatch, cfg):
        assert cfg.task == "MaxIndependentSet"
        super(MaxIndSetMDP, self).__init__(gbatch, cfg)

    def _prune(self, state):
        # label all nodes near the selected node ("1") to be "0"
        with self.gbatch.local_scope():
            self.gbatch.ndata["h"] = (state == 1).float()
            self.gbatch.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
            x1_deg = self.gbatch.ndata.pop('h')  # (#node, # of 1-labeled neighbour node)
        undecided = ~get_decided(state)
        state[undecided & (x1_deg > 0)] = 0

        return state

    def energy_fn(self, state):
        return -(state == 1).sum(dim=1).float()

class MaxCliqueMDP(GraphCombOptMDP):
    # initial state: all nodes = "2" (all nodes are undecided)
    # 1: selected, 0: not selected, 2: undecided
    def __init__(self, gbatch, cfg,):
        super(MaxCliqueMDP, self).__init__(gbatch, cfg)

    def _prune(self, state):
        # calculate num of "1" for each grpah
        num1 = pad_batch(state == 1, self.numnode_per_graph, padding_value=0).sum(dim=1)
        num1 = [num * torch.ones(count).to(self.device) for count, num in zip(self.numnode_per_graph, num1)]
        num1 = torch.cat(num1) # same shape with state
        # if a node is not connected to all "1" nodes, label it to be "0"
        with self.gbatch.local_scope():
            self.gbatch.ndata["h"] = (state == 1).float()
            self.gbatch.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
            x1_deg = self.gbatch.ndata.pop('h')
        undecided = ~get_decided(state)
        state[undecided & (x1_deg < num1)] = 0

        return state

    def energy_fn(self, state):
        return -(state == 1).sum(dim=1).float()

# TODO: Reformulate this CO to select ones instead of zeros, so that self.step can be moved up to GraphCombOptMDP
class MinDominateSetMDP(GraphCombOptMDP):
    # initial state: all nodes = "2" (all nodes are in the set)
    # 0: already deleted from the set, 1: in set, can't be deleted from the set,
    # 2: in set, might be deleted from the set in future steps
    def __init__(self, gbatch, cfg):
        super(MinDominateSetMDP, self).__init__(gbatch, cfg)
        assert not cfg.back_trajectory

    def step(self, action):
        state = self.state.clone()

        # action: delete a node from set (label it to be "0" from "2")
        action_node_idx = (self.cum_num_node + action)[~self.done]
        # assert torch.all(state[action_node_idx] == 2)
        assert torch.all(~self.get_decided_mask(state[action_node_idx]))
        state[action_node_idx] = 0

        state = self._prune(state)

        self.set_state(state)

        return state

    def _prune(self, state):
        undecided = ~get_decided(state)
        with self.gbatch.local_scope():
            self.gbatch.ndata["h"] = ((state == 1) | (state == 2)).float()
            self.gbatch.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
            x12_deg = self.gbatch.ndata.pop('h').int()
        # or if a "2" has no neighbour in the set, it must stay in the set, too
        state[undecided & (x12_deg == 0)] = 1

        # this kinds of special "0" needs to have a neighbour stay in the set
        special0 = (state == 0) & (x12_deg <= 1)
        with self.gbatch.local_scope():
            self.gbatch.ndata["h"] = special0.float()
            self.gbatch.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
            xsp0_deg = self.gbatch.ndata.pop('h').int()
        state[undecided & (xsp0_deg >= 1)] = 1

        # for the rest "2" node: it and all its neighbours are connected to the set
        # thus can be deleted from the set

        return state

    def energy_fn(self, state):
        return (state == 1).sum(dim=1).float()

class MaxCutMDP(GraphCombOptMDP):
    # initial state: all nodes = "2" (all nodes are NOT in the set)
    # 0: chosen to not be in the set
    # 1: chosen to be in the set
    # 2: undecided, also not in the set
    def __init__(self, gbatch, cfg):
        super(MaxCutMDP, self).__init__(gbatch, cfg)
        assert not cfg.back_trajectory

    def _prune(self, state):
        undecided = ~get_decided(state)
        # if a "2" has more "1" neighbours than "0"or"2" neighbours
        # it must NOT be in the set, thus label it to be "0"
        with self.gbatch.local_scope():
            self.gbatch.ndata["h"] = (state == 1).float()
            self.gbatch.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
            x1_deg = self.gbatch.ndata.pop('h').int()
        with self.gbatch.local_scope():
            self.gbatch.ndata["h"] = ((state == 0) | (state == 2)).float()
            self.gbatch.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
            x02_deg = self.gbatch.ndata.pop('h').int()
        state[undecided & (x1_deg > x02_deg)] = 0

        return state

    def energy_fn(self, state): # calculate the cut
        state[state == 2] = 0 # "0" for "not in the set"
        with self.gbatch.local_scope():
            self.gbatch.ndata["h"] = state.float()
            self.gbatch.apply_edges(fn.u_add_v("h", "h", "e"))
            # 0 + 0 = 0 (not cut), 0 + 1 = 1 (cut), 1 + 1 = 2 (not cut)
            self.gbatch.edata["e"] = (self.gbatch.edata["e"] == 1).float()
            cut = dgl.sum_edges(self.gbatch, 'e') # (bs, )
        cut = cut / 2 # each edge is counted twice
        return -cut