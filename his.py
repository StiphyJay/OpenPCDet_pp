import torch
import pdb
pdb.set_trace()
h=torch.randn(1,32) # shape [n_pillars, n_points_in_pillars]
scale = (h.max() - h.min()) / 32
h_int=(h/scale).round().unsqueeze(-1)
coord = torch.arange((h.min()/scale).round(), (h.max()/scale).round())
coord = coord.view(1,1,-1)
h_hist=(h_int==coord).float().sum(-2)

pdb.set_trace()
h_int1 = torch.clamp(((32-1) * (h-h.min()) / (h.max() - h.min())).round().long(), 0, 32-1)
histogram = torch.zeros(32)
unique_indices, counts = h_int1.view(-1).unique(return_counts=True)
histogram[unique_indices] = counts.float()

h_his = []
for t in h:
    h_his.append(torch.histc(t, bins=32, min=h.min(), max=h.max()))