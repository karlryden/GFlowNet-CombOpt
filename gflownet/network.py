import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
import dgl
import dgl.function as fn
from dgl.nn.pytorch.conv import GINConv, GATConv
from dgl.nn.pytorch.glob import MaxPooling

"""
GIN architecture
from https://github.com/dmlc/dgl/blob/master/examples/pytorch/gin/train.py
"""

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
        assert modulation_type in ["concat", "attend", "film"]
        match modulation_type:
            case "concat":
                hidden_dim += condition_dim

            case "film":
                # TODO: Investigate having a film layer for each layer of the GIN
                self.film = nn.Linear(condition_dim, 2*hidden_dim)

            case "attend":
                raise NotImplementedError
                # TODO: Keys and values need to be of the same dimension. Reduce c? 
                self.attention = nn.MultiheadAttention(hidden_dim, 1)

        self.modulation_type = modulation_type

        self.inp_transform = nn.Identity()

        self.ginlayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        assert aggregator_type in ["sum", "mean", "max"]
        for layer in range(num_layers - 1):  # excluding the input layer
            mlp = MLP_GIN(hidden_dim, hidden_dim, hidden_dim)
            self.ginlayers.append(GINConv(mlp, aggregator_type=aggregator_type, learn_eps=learn_eps))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        self.output_dim = output_dim
        self.graph_level_output = graph_level_output
        # linear functions for graph poolings of output of each layer
        self.linear_prediction = nn.ModuleList()
        for layer in range(num_layers):
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
            c: torch.Tensor=None, 
            reward_exp=None
        ):
        assert reward_exp is None

        h = self.inp_embedding(state)
        h = self.inp_transform(h)

        if c is None:
            # TODO: Investigate ways to allow unconditioned forward pass. 
            assert self.condition_dim == 0, "Conditioning signal is not provided."

        else:
            assert len(c) == self.condition_dim, "Dimension error in conditioning signal."
            match self.modulation_type:
                case "concat":
                    C = c.unsqueeze(0).expand(g.num_nodes(), -1)    # expand to match the number of nodes
                    h = torch.cat([h, C], dim=1)
            
                case "film":
                    gamma, beta = self.film(c).view(2, self.hidden_dim)
                    h = gamma*h + beta

                case "attend":
                    raise NotImplementedError

        # list of hidden representations at each layer
        hidden_rep = [h]
        for i, layer in enumerate(self.ginlayers):
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
