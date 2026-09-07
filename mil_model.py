"""Stage-2 MIL model: gated attention (Ilse 2018) + additive per-tile contributions (Javed 2022).

The slide logit is an exact sum of per-tile contributions  a_i * psi(h_i)  — so the localisation
map IS the decomposition of the prediction (Additive-MIL), not a post-hoc saliency. Attention
weights a_i are kept as the comparison baseline localiser. Scope: research/scoping/05.
"""
import torch
import torch.nn as nn


class GatedAdditiveMIL(nn.Module):
    def __init__(self, in_dim=2048, hid=384, att=192, drop=0.25):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(), nn.Dropout(drop))
        self.att_v = nn.Sequential(nn.Linear(hid, att), nn.Tanh())
        self.att_u = nn.Sequential(nn.Linear(hid, att), nn.Sigmoid())
        self.att_w = nn.Linear(att, 1)
        self.psi = nn.Linear(hid, 1)

    def forward(self, bags):
        """bags: (B, N, in_dim) -> slide_logit (B,), contributions (B, N), attention (B, N)."""
        h = self.phi(bags)                                   # (B,N,hid)
        a = self.att_w(self.att_v(h) * self.att_u(h))        # (B,N,1) gated attention scores
        a = torch.softmax(a, dim=1)                          # normalised over the bag
        c = a.squeeze(-1) * self.psi(h).squeeze(-1)          # (B,N) additive contributions
        return c.sum(dim=1), c, a.squeeze(-1)
