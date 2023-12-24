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
        self.distance = distance
        self.normalized_distance = self.distance/original_feature_size
        self.seq_size = seq_size
        if seq:
            if self.mode == "linear_multi":
                self.linear = nn.Sequential(
                    nn.Linear(self.seq_size,self.seq_size)
                )
        else:
            self.seq_size = 1
        self.distance_vector = torch.full((seq_size,), self.normalized_distance) * torch.linspace(0,self.seq_size-1,self.seq_size)

    def forward(self, spline):
        main_index = self.index_value()
        indices = self.distance_vector + main_index
        if self.seq:
            if self.mode == "linear_multi":
                return self.forward_fsdr_seq_linear_multi(spline, indices)
        else:
            return self.forward_fsdr(spline, indices)

    def forward_fsdr(self, spline, indices):
        return spline.evaluate(indices).reshape(-1)

    def forward_fsdr_seq_linear_multi(self, spline, indices):
        band_values = spline.evaluate(indices)
        band_values = band_values.permute(1, 0)
        return self.linear(band_values).reshape(-1)

    def index_value(self):
        return F.sigmoid(self.raw_index)

