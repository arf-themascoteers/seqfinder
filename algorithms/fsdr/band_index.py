import torch.nn as nn
import torch
import torch.nn.functional as F


class BandIndex(nn.Module):
    def __init__(self, val, original_feature_size, seq, mode, seq_size=6, distance=5):
        super().__init__()
        if val is None:
            val = torch.rand(1)
            val = (val*10)-5
        self.raw_index = nn.Parameter(val)
        self.original_feature_size = original_feature_size
        self.seq = seq
        self.mode = mode
        self.sequence_length = 5
        self.distance = distance
        self.normalized_distance = self.distance/original_feature_size
        self.seq_size = seq_size
        if seq:
            if self.mode == "linear_multi":
                self.linear = nn.Sequential(
                    nn.Linear(self.sequence_length,5),
                    nn.LeakyReLU(),
                    nn.Linear(5,1)
                )
                self.distance_vector = torch.full((seq_size,), self.normalized_distance) * torch.linspace(0,self.seq_size-1,self.seq_size)

    def forward(self, spline):
        main_index = self.index_value()
        if self.seq:
            if self.mode == "linear_multi":
                return self.forward_fsdr_seq_linear_multi(spline, main_index)
        else:
            return self.forward_fsdr(spline, main_index)

    def forward_fsdr(self, spline, main_index):
        return spline.evaluate(main_index)

    def forward_fsdr_seq_linear_multi(self, spline, main_index):
        indices = None
        for i in range(self.sequence_length):
            idxs = torch.cat([idx + (i * self.normalized_distance)], dim=0)
            outs = spline.evaluate(idxs)
            outs = outs.permute(1, 0)
            return self.linear(outs).reshape(-1)

    def index_value(self):
        return F.sigmoid(self.raw_index)

