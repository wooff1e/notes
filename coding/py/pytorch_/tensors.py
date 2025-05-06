import numpy as np
import torch
from torch import nn




tensor =torch.tensor([[1., -1.], [1., -1.]])
tensor =torch.tensor(np.array([[1, 2, 3], [4, 5, 6]]))

# Access a tensor's NumPy (tensors on CPU share memory location with the underlying arrays)
underlying_np = tensor.numpy()

# By default, new tensors are created on the CPU
if torch.cuda.is_available():
    tensor = torch.tensor.to('cuda')

# Operations that have a `_` suffix are in-place
tensor.add_(5)

# Ways to flatten a tensor:
tensor = nn.Flatten()(tensor)
tensor = tensor.reshape(tensor.shape[0], -1) 
tensor.view(tensor.size(0), -1)