import torch
from torch import nn
from torch_scatter import scatter_mean

def build_simplexes(edge_index_cpu: torch.Tensor) -> torch.Tensor:
    adj = {}
    for u, v in edge_index_cpu.t().tolist():
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)

    tris = []
    for u, neigh_u in adj.items():
        if len(neigh_u) < 2:
            continue
        for v in neigh_u:
            if v <= u:
                continue
            inter = neigh_u & adj[v]
            for w in inter:
                if w <= v:
                    continue
                tris.append([u, v, w])

    if not tris:
        return torch.empty((3, 0), dtype=torch.long)
    return torch.tensor(tris, dtype=torch.long).t()

class SimplexConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.5):
        super().__init__()
        self.node_proj = nn.Linear(in_dim, out_dim, bias=False)
        self.tri_proj  = nn.Linear(out_dim, out_dim, bias=False)
        self.act = nn.ELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, triangles: torch.Tensor) -> torch.Tensor:
        if triangles.numel() == 0:
            return self.drop(self.act(self.tri_proj(self.node_proj(x))))
        tri_feat = self.node_proj(x[triangles])  
        tri_feat = tri_feat.mean(dim=0)          
        tri_msg  = self.tri_proj(tri_feat)       
        tri_msg_rep = tri_msg.repeat_interleave(3, dim=0)     
        index = triangles.t().reshape(-1)                        
        node_msg = scatter_mean(tri_msg_rep, index, dim=0, dim_size=x.size(0))
        return self.drop(self.act(node_msg))


class HeteroSimplexLayer(nn.Module):

    def __init__(self, num_rel: int, in_dim: int, out_dim: int, dropout=0.5):
        super().__init__()
        self.num_rel = num_rel
        self.convs = nn.ModuleList([SimplexConvLayer(in_dim, out_dim, dropout) for _ in range(num_rel)])
        self.res_proj = nn.Linear(in_dim, out_dim, bias=False)
        self.gate = nn.Sequential(nn.Linear(in_dim + out_dim, out_dim), nn.Sigmoid())
        nheads = 2 if num_rel >= 2 else 1
        self.rel_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=out_dim, nhead=nheads, dim_feedforward=out_dim * 2,
                dropout=dropout, activation='gelu'
            ),
            num_layers=1
        )

    def forward(self, x: torch.Tensor, edge_index_list, triangles_list):
        x_res = self.res_proj(x)  
        h_list = []
        for i in range(self.num_rel):
            u = self.convs[i](x, triangles_list[i])         
            a = self.gate(torch.cat([x, u], dim=1))         
            h = torch.tanh(u) * a + x_res * (1.0 - a)       
            h_list.append(h.unsqueeze(1))                   
        sem = torch.cat(h_list, dim=1)                      
        trans = self.rel_transformer(sem.permute(1, 0, 2)).permute(1, 0, 2)
        return trans.mean(dim=1)                            


class MGTABModel(nn.Module):
    def __init__(self, in_dim, hidden=32, proj_channels=32, dropout=0.1, num_rel=2):
        super().__init__()
        self.in_lin = nn.Linear(in_dim, hidden)
        self.act = nn.LeakyReLU()
        self.drop = nn.Dropout(dropout)

        self.s1 = HeteroSimplexLayer(num_rel=num_rel, in_dim=hidden, out_dim=hidden, dropout=dropout)
        self.s2 = HeteroSimplexLayer(num_rel=num_rel, in_dim=hidden, out_dim=hidden, dropout=dropout)

        self.out1 = nn.Linear(hidden, proj_channels)
        self.out2 = nn.Linear(proj_channels, 2)

        self.last_hidden = None
        self._reset()

    def _reset(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, edge_index_list, triangles_list):
        h = self.drop(self.act(self.in_lin(x)))
        h = self.s1(h, edge_index_list, triangles_list)
        h = self.s2(h, edge_index_list, triangles_list)
        h = self.drop(self.act(self.out1(h)))
        self.last_hidden = h.detach()
        return self.out2(h)


class TwiBot22Model(nn.Module):
    def __init__(self, cat_num, numeric_num, tweet_channel, des_channel,
                 hidden=128, proj_channels=64, dropout=0.5, num_rel=2):
        super().__init__()
        self.cat_num, self.numeric_num = cat_num, numeric_num
        self.tweet_channel, self.des_channel = tweet_channel, des_channel
        self.act = nn.LeakyReLU(); self.drop = nn.Dropout(dropout)

        quart = hidden // 4
        self.lin_cat = nn.Linear(cat_num,       quart)
        self.lin_num = nn.Linear(numeric_num,   quart)
        self.lin_tw  = nn.Linear(tweet_channel, quart)
        self.lin_des = nn.Linear(des_channel,   quart)
        self.lin_merge = nn.Linear(hidden, hidden)

        self.s1 = HeteroSimplexLayer(num_rel=num_rel, in_dim=hidden, out_dim=hidden, dropout=dropout)
        self.s2 = HeteroSimplexLayer(num_rel=num_rel, in_dim=hidden, out_dim=hidden, dropout=dropout)

        self.out1 = nn.Linear(hidden, proj_channels)
        self.out2 = nn.Linear(proj_channels, 2)

    def encode_modal(self, x_all):
        i0 = self.cat_num; i1 = i0 + self.numeric_num; i2 = i1 + self.tweet_channel
        cat = x_all[:, :i0]; num = x_all[:, i0:i1]; tw = x_all[:, i1:i2]; des = x_all[:, i2:]
        z = torch.cat([
            self.drop(self.act(self.lin_cat(cat))),
            self.drop(self.act(self.lin_num(num))),
            self.drop(self.act(self.lin_tw(tw))),
            self.drop(self.act(self.lin_des(des))),
        ], dim=1)
        return self.drop(self.act(self.lin_merge(z)))

    def forward(self, batch, edge_index_list, triangles_list):
        x = self.encode_modal(batch["user"].x)
        h = self.s1(x, edge_index_list, triangles_list)
        h = self.s2(h, edge_index_list, triangles_list)
        h = self.drop(self.act(self.out1(h)))
        return self.out2(h)

