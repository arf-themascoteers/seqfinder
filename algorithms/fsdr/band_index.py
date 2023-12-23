import torch.nn as nn
import torch
import torch.nn.functional as F


class BandIndex(nn.Module):
    def __init__(self, val, original_feature_size, seq, mode):
        super().__init__()
        if val is None:
            val = torch.rand(1)
            val = (val*10)-5
        self.raw_index = nn.Parameter(val)
        self.original_feature_size = original_feature_size
        self.seq = seq
        self.mode = mode
        self.sequence_length = 5
        self.distance = 5
        self.normalized_distance = self.distance/original_feature_size
        if seq:
            if self.mode == "linear_multi":
                self.linear = nn.Sequential(
                    nn.Linear(self.sequence_length,5),
                    nn.LeakyReLU(),
                    nn.Linear(5,1)
                )

    def forward(self, spline):
        idx = self.index_value()
        if self.seq:
            if self.mode == "linear_multi":
                for i in range(self.sequence_length):
                    idxs = torch.cat([idx + (i*self.normalized_distance)], dim=0)
                    outs = spline.evaluate(idxs)
                    outs = outs.permute(1, 0)
                    return self.linear(outs).reshape(-1)
        else:
            return spline.evaluate(self.index_value())

    def index_value(self):
        return F.sigmoid(self.raw_index)
