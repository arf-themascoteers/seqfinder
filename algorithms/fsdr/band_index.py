import torch.nn as nn
import torch
import torch.nn.functional as F
from algorithms.fsdr.pe import PositionalEncoding


class BandIndex(nn.Module):
    def __init__(self, val=None):
        super().__init__()
        if val is None:
            val = torch.rand(1)
            val = (val*10)-5
        self.raw_index = nn.Parameter(val)
        #self.pe = PositionalEncoding(5,0.1, 10)
        self.linear = nn.Sequential(
            nn.Linear(5,5),
            nn.LeakyReLU(),
            nn.Linear(5,1)
        )

    def forward(self, spline):
        scale = 0.005
        idx_0 = self.index_value()
        idx_1 = idx_0+ scale*1
        idx_2 = idx_0+ scale*2
        idx_3 = idx_0+ scale*3
        idx_4 = idx_0+ scale*4
        idx = torch.hstack((idx_0, idx_1, idx_2, idx_3, idx_4))
        outs = spline.evaluate(idx)
        outs = outs.permute(1, 0)
        #outs = self.pe(outs)
        return self.linear(outs).reshape(-1)

    def index_value(self):
        return F.sigmoid(self.raw_index)

    def range_loss(self):
        if self.sigmoid:
            return 0
        loss_l_lower = F.relu(-1 * self.raw_index)
        loss_l_upper = F.relu(self.raw_index - 1)
        return loss_l_lower + loss_l_upper
