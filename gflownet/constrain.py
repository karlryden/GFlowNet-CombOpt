from functools import partial
import torch

indicators = {  # add custom indicators here
    'none': lambda s, nodes: torch.tensor([1.0], device=s.device),
    'inclusion': lambda s, nodes: (s[nodes] == 1).float().all(),
    'exclusion': lambda s, nodes: (s[nodes] == 0).float().all(),
}

def get_indicator_fn(signature):
    constraint_type = signature['type']
    constrained_nodes = signature['node']

    return partial(indicators[constraint_type], nodes=constrained_nodes)

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