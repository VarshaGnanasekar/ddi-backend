"""
DDI Model Definition — MemoryEfficientRGCN
Reusable module for both training and inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, BatchNorm, LayerNorm


class MemoryEfficientRGCN(nn.Module):
    def __init__(self, metadata, in_dims, hidden_dim=192, num_layers=3,
                 num_output_classes=105, dropout=0.2, **kwargs):
        super().__init__()
        self.node_types = metadata[0]
        self.edge_types  = metadata[1]
        self.hidden_dim  = hidden_dim
        self.dropout     = dropout

        # --- Input projections (drug + protein have real features; ATC/disease/SE use embeddings)
        self.proj  = nn.ModuleDict()
        self.norms = nn.ModuleDict()
        for ntype in self.node_types:
            input_dim = in_dims.get(ntype)
            if input_dim:
                self.proj[ntype]  = nn.Linear(input_dim, hidden_dim)
                self.norms[ntype] = LayerNorm(hidden_dim)

        # --- R-GCN layers
        self.rgcn_convs = nn.ModuleList()
        self.rgcn_bns   = nn.ModuleList()
        num_relations   = len(self.edge_types)
        for _ in range(num_layers):
            self.rgcn_convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations))
            self.rgcn_bns.append(BatchNorm(hidden_dim))

        # --- Edge MLP (classifier head)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_output_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.edge_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_dict, edge_index_dict):
        # Project each node type to hidden_dim
        h_dict = {}
        for ntype, x in x_dict.items():
            if ntype in self.proj:
                h = self.proj[ntype](x)
                h = self.norms[ntype](h)
                h_dict[ntype] = F.silu(h)
            else:
                h_dict[ntype] = x          # already in hidden_dim (embeddings)

        # Flatten all node types into one big tensor (R-GCN needs a global index)
        node_offsets = {}
        total_nodes  = 0
        all_nodes    = []
        for ntype in self.node_types:
            node_offsets[ntype] = total_nodes
            all_nodes.append(h_dict[ntype])
            total_nodes += h_dict[ntype].size(0)
        x_all = torch.cat(all_nodes, dim=0)

        # Build global edge index + relation type vector
        edge_indices = []
        edge_types   = []
        for i, (etype_tuple, edge_index) in enumerate(edge_index_dict.items()):
            src_type, _, dst_type = etype_tuple
            offset = torch.tensor(
                [[node_offsets[src_type]], [node_offsets[dst_type]]],
                device=edge_index.device
            )
            edge_indices.append(edge_index + offset)
            edge_types.append(
                torch.full((edge_index.size(1),), i, dtype=torch.long, device=edge_index.device)
            )
        edge_index_all = torch.cat(edge_indices, dim=1)
        edge_type_all  = torch.cat(edge_types,   dim=0)

        # Message-passing with residual connections
        h = x_all
        for conv, bn in zip(self.rgcn_convs, self.rgcn_bns):
            h_in = h
            h    = conv(h, edge_index_all, edge_type_all)
            h    = bn(h)
            h    = F.silu(h)
            h    = F.dropout(h, p=self.dropout, training=self.training)
            h    = h + h_in

        # Slice back to per-type tensors
        h_dict_out = {}
        for ntype in self.node_types:
            start = node_offsets[ntype]
            count = h_dict[ntype].size(0)
            h_dict_out[ntype] = h[start : start + count]

        return h_dict_out

    def predict(self, drug_embeddings, pairs):
        """
        pairs : LongTensor of shape (N, 2)  [src_drug_idx, dst_drug_idx]
        returns logits of shape (N, num_output_classes)
        """
        src_emb  = drug_embeddings[pairs[:, 0]]
        dst_emb  = drug_embeddings[pairs[:, 1]]
        edge_emb = torch.cat([src_emb, dst_emb], dim=1)
        return self.edge_mlp(edge_emb)
