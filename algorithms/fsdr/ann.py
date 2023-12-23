import torch.nn as nn
import torch
from algorithms.fsdr.band_index import BandIndex
import my_utils


class ANN(nn.Module):
    def __init__(self, target_feature_size, original_feature_size, seq, mode):
        super().__init__()
        self.device = my_utils.get_device()
        self.target_feature_size = target_feature_size
        self.original_feature_size = original_feature_size
        self.seq = seq
        self.mode = mode
        self.linear = nn.Sequential(
            nn.Linear(self.target_feature_size, 15),
            nn.LeakyReLU(),
            nn.Linear(15, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 1)
        )
        init_vals = torch.linspace(0.001,0.99, target_feature_size+2)
        modules = []
        for i in range(self.target_feature_size):
            val = my_utils.inverse_sigmoid_torch(init_vals[i + 1])
            bi = BandIndex(val, self.original_feature_size, self.seq, self.mode)
            modules.append(bi)
        self.machines = nn.ModuleList(modules)

    def forward(self, spline):
        size = spline._a.shape[1]
        outputs = torch.zeros(size, self.target_feature_size, dtype=torch.float32).to(self.device)
        for i,machine in enumerate(self.machines):
            outputs[:,i] = machine(spline)
        soc_hat = self.linear(outputs)
        soc_hat = soc_hat.reshape(-1)
        return soc_hat

    def get_indices(self):
        return [machine.index_value() for machine in self.machines]
