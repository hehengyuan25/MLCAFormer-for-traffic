import torch
import torch.nn as nn

from .layers import (NodePositionAwareSpatialAttention,
                     MultiLevelTemporalAttention,
                     apply_node_position_aware_encoding)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MLCAFormer(nn.Module):
    def __init__(
            self,
            num_nodes,
            in_steps=12,
            out_steps=12,
            steps_per_day=288,
            input_dim=3,
            output_dim=1,
            input_embedding_dim=24,
            tod_embedding_dim=24,
            dow_embedding_dim=24,
            nid_embedding_dim=24,
            col_embedding_dim=80,
            feed_forward_dim=256,
            num_heads=4,
            num_layers=3,
            dropout=0.1,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.nid_embedding_dim = nid_embedding_dim
        self.col_embedding_dim = col_embedding_dim
        self.model_dim = (
                input_embedding_dim
                + tod_embedding_dim
                + dow_embedding_dim
                + col_embedding_dim
        )

        self.model_dim1 = self.model_dim + self.nid_embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.input_proj = nn.Linear(input_dim, input_embedding_dim)
        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        if col_embedding_dim > 0:
            self.col_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(in_steps, num_nodes, col_embedding_dim))
            )

      
        self.output_proj = nn.Linear(
            in_steps * self.model_dim1, out_steps * output_dim
            )

        self.spatial_attention_layers = nn.ModuleList(
            [
                NodePositionAwareSpatialAttention(self.model_dim1, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )
        self.temporal_attention_layers = nn.ModuleList(
            [
                MultiLevelTemporalAttention(self.model_dim, feed_forward_dim, num_heads, dropout, seq_len=in_steps,
                                            device=device)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        batch_size = x.shape[0]
        features = []
        x_input = self.input_proj(x[..., : self.input_dim])  
        features.append(x_input)
        if self.tod_embedding_dim > 0:
            tod = x[..., 1]
            tod_emb = self.tod_embedding((tod * self.steps_per_day).long())
            features.append(tod_emb)
        if self.dow_embedding_dim > 0:
            dow = x[..., 2]
            dow_emb = self.dow_embedding(dow.long())
            features.append(dow_emb)
        if self.col_embedding_dim > 0:
            col_emb = self.col_embedding.expand(
                size=(batch_size, *self.col_embedding.shape))
            features.append(col_emb)
        x = torch.cat(features, dim=-1)

        for attn in self.temporal_attention_layers:
            x = attn(x, dim=1)

        x = apply_node_position_aware_encoding(x, self.nid_embedding_dim)
        for attn in self.spatial_attention_layers:
            x = attn(x, dim=2)

        # Output Projection
        out = x.transpose(1, 2)
        out = out.reshape(batch_size, self.num_nodes, self.in_steps * self.model_dim1)
        out = self.output_proj(out).view(batch_size, self.num_nodes, self.out_steps, self.output_dim)
        out = out.transpose(1, 2)

        return out
