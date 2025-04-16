import torch

def get_sat_fn():
    def sat_fn(gb, s):
        want = gb.ndata['wanted'].to(dtype=torch.bool)
        inc = (s == 1).flatten()
        sat = inc & want

        cum_num_node = gb.batch_num_nodes().cumsum(dim=0)
        
        start = 0
        sat_rates = torch.empty(gb.batch_size, device=s.device)

        for k, end in enumerate(cum_num_node):
            w = want[start:end]
            s = sat[start:end]

            sat_rates[k] = 1.0 if not w.sum() else s.sum() / w.sum()

            start = end

        return sat_rates

    return sat_fn