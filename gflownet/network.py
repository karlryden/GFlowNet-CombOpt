import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch.conv import GINConv, GATConv, GraphConv
from dgl.nn.pytorch.glob import MaxPooling

from util import pad_batch

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
                 graph_level_output=0, learn_eps=False, dropout=0.,
                 aggregator_type="sum", modulation_type="concat"):
        super().__init__()

        self.inp_embedding = nn.Embedding(input_dim, hidden_dim)
        self.hidden_dim = hidden_dim

        # TODO: Factor out modulation to be composed with the GIN from outside?
        if modulation_type == "none":
            ...

        elif modulation_type == "concat":
            hidden_dim *= 2

        elif modulation_type == "film":
            self.films = nn.ModuleList()
            for _ in range(num_layers - 1):
                self.films.append(nn.Linear(hidden_dim, 2*hidden_dim))

        elif modulation_type == "attend":
            self.query = nn.Linear(hidden_dim, hidden_dim)
            self.key = nn.Linear(hidden_dim, hidden_dim)
            self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)

        else:
            raise ValueError("Invalid modulation type.")

        self.modulation_type = modulation_type

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
        ):
        if c is None:
            assert self.modulation_type == 'none', "Conditioning signal is not provided."

        else:
            c = c.to(g.device)
            if c.ndimension() == 1: # global (1D) conditioning (one graph)
                assert g.batch_size == 1, "Conditioning signal must be 1D for a single graph."
                assert len(c) == self.hidden_dim, "Conditioning signal dimension must match GNN hidden dimension."
                C = c.unsqueeze(0).expand(g.num_nodes(), -1)

            elif c.ndimension() == 2:   # per-graph (2D) conditioning (batch of graphs)
                assert c.shape[0] == g.batch_size, "Conditioning signal batch must match graph batch size."
                assert c.shape[1] == self.hidden_dim, f"Conditioning signal dimension ({c.shape[1]}) must match GNN hidden dimension ({self.hidden_dim})."
                C = torch.repeat_interleave(
                    c, g.batch_num_nodes(), dim=0
                )
            else:
                raise ValueError(f"Conditioning signal must be either 1D or 2D but has {c.ndimension()} dimensions.")

        h = state
        h = self.inp_embedding(h)
        if 'encoding_proj' in g.ndata:
            e_proj = g.ndata['encoding_proj'].to(h)
            assert e_proj.shape[1] == self.hidden_dim, f"Positional encoding dimension ({e_proj.shape[1]}) must be projected to GNN hidden dimension ({self.hidden_dim})."
            h = h + e_proj   # add (projected) positional encoding to embedded state

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
                    Q = self.query(h)
                    K = self.key(C)
                    V = K

                    num_nodes_per_graph = g.batch_num_nodes().tolist()
                    padded_Q = pad_batch(Q, num_nodes_per_graph, padding_value=0.)
                    padded_K = pad_batch(K, num_nodes_per_graph, padding_value=0.)
                    padded_V = pad_batch(V, num_nodes_per_graph, padding_value=0.)

                    B, N, _ = padded_Q.shape
                    pad_mask = torch.ones(B, N, dtype=torch.bool, device=h.device)
                    for j, n in enumerate(num_nodes_per_graph):
                        pad_mask[j, :n] = False

                    o, _ = self.attention(padded_Q, padded_K, padded_V, key_padding_mask=pad_mask)
                    h = h + torch.cat(
                        [o[i,:n] for i, n in enumerate(num_nodes_per_graph)], dim=0
                    )

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