import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn.pytorch.conv import GINConv, GATConv, GraphConv
from dgl.nn.pytorch.glob import MaxPooling

"""
GIN architecture
from https://github.com/dmlc/dgl/blob/master/examples/pytorch/gin/train.py
"""

# TODO: Restructure nets into a hierarchy:
# GNN -> ConditionalGNN

class MLP_GIN(nn.Module):
    """Construct two-layer MLP-type aggreator for GIN model"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(input_dim, hidden_dim, bias=False))
        self.linears.append(nn.Linear(hidden_dim, output_dim, bias=False))
        self.batch_norm = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        h = F.relu(self.batch_norm(self.linears[0](x)))
        return self.linears[1](h)

class GIN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=5,
                 condition_dim=0, graph_level_output=0, learn_eps=False, dropout=0.,
                 aggregator_type="sum", modulation_type="concat"):
        super().__init__()

        self.inp_embedding = nn.Embedding(input_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        self.condition_dim = condition_dim

        # TODO: Factor out modulation to be composed with the GIN from outside?
        if modulation_type == "none":
            ...

        elif modulation_type == "concat":
            hidden_dim += condition_dim

        elif modulation_type == "film":
            # TODO: Investigate having a film layer for each layer of the GIN
            self.films = nn.ModuleList()
            for _ in range(num_layers - 1):
                self.films.append(nn.Linear(condition_dim, 2*hidden_dim))

        elif modulation_type == "attend":
            raise NotImplementedError
            # TODO: Keys and values need to be of the same dimension. Reduce c? 
            self.attention = nn.MultiheadAttention(hidden_dim, 1)

        else:
            raise ValueError("Invalid modulation type.")

        self.modulation_type = modulation_type

        self.inp_transform = nn.Identity()

        self.ginlayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        assert aggregator_type in ["sum", "mean", "max"]
        for _ in range(num_layers - 1):  # excluding the input layer
            mlp = MLP_GIN(hidden_dim, hidden_dim, hidden_dim)
            self.ginlayers.append(GINConv(mlp, aggregator_type=aggregator_type, learn_eps=learn_eps))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        self.output_dim = output_dim
        self.graph_level_output = graph_level_output
        # linear functions for graph poolings of output of each layer
        self.linear_prediction = nn.ModuleList()
        for _ in range(num_layers):
            self.linear_prediction.append(
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, output_dim+graph_level_output))
            )
        self.readout = nn.Sequential(
            nn.Linear(num_layers * hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim+graph_level_output)
        )
        self.drop = nn.Dropout(dropout)
        self.pool = MaxPooling()

    # optionally conditioned on signal c
    def forward(
            self, 
            g: dgl.DGLGraph, 
            state: torch.Tensor, 
            c: torch.Tensor=None,   # pass rows of c's in case of graph batch
            reward_exp=None
        ):
        assert reward_exp is None

        if c is None:
            assert self.condition_dim == 0, "Conditioning signal is not provided."

        else:
            c = c.to(g.device)
            if c.ndimension() == 1: # global (1D) conditioning (one graph)
                assert g.batch_size == 1
                C = c.unsqueeze(0).expand(g.num_nodes(), -1)

            elif c.ndimension() == 2:   # per-graph (2D) conditioning (batch of graphs)
                assert len(c) == g.batch_size
                C = torch.repeat_interleave(
                    c, g.batch_num_nodes(), dim=0
                )
            else:
                raise ValueError("Conditioning signal must be either 1D or 2D.")

        h = self.inp_embedding(state)
        h = self.inp_transform(h)

        if c is not None and self.modulation_type == "concat":
            h = torch.cat([h, C], dim=1)

        # list of hidden representations at each layer
        hidden_rep = [h]
        for i, layer in enumerate(self.ginlayers):
            if c is not None:
                if self.modulation_type == "film":                      
                    film = self.films[i](C)
                    gamma, beta = film.chunk(2, dim=1)
                    h = gamma * h + beta

                elif self.modulation_type == "attend":
                    
                    # TODO: Project c and h to a common dimension,
                    # then call nn.MultiHeadAttention with h as Q, C as K and V.
                    
                    # return h + attention(Q, K, V)? 

                    raise NotImplementedError

            h = layer(g, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)

        score_over_layer = self.readout(torch.cat(hidden_rep, dim=-1))

        if self.graph_level_output > 0:
            return score_over_layer[..., :self.output_dim], \
                   self.pool(g, score_over_layer[..., self.output_dim:])
        else:
            return score_over_layer


if __name__ == '__main__':
    import networkx as nx
    
    g = dgl.DGLGraph()
    g.add_nodes(4)
    g.add_edges([0, 1, 2, 3], [1, 2, 3, 0])

    n = g.to_networkx()
    nx.draw(n, with_labels=True)

    gin = GIN(3, g.num_nodes(), hidden_dim=8)
    cin = GIN(3, g.num_nodes(), hidden_dim=8, condition_dim=3, modulation_type="film")

    s = torch.full((g.num_nodes(),), 2, dtype=torch.long)
    c = torch.tensor([1, 2, 3], dtype=torch.float32)

    h = cin(g, s, c)