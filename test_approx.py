import torch

from ds_manager import DSManager
from approximator import get_splines


ds = DSManager(reduced_features=True, reduced_rows=True)
X,y = ds.get_X_y()
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)
splines = get_splines(X)
tv = X[:,0]
sv = splines.evaluate(torch.tensor(0))
print(tv)
print(sv)

print(tv.shape)
print(sv.shape)