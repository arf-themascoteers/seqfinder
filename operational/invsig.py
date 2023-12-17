import torch
import torch.nn.functional as F

def inverse_sigmoid(y):
    return -torch.log(1.0 / y - 1.0)

probabilities = torch.tensor([0.2, 0.7, 0.9,0.0000000001,0.5])
inverse_sigmoid_values = inverse_sigmoid(probabilities)

print(inverse_sigmoid_values)