import torch
import numpy as np

from unet_model import UNet

input_np = np.random.rand(1, 3, 32, 32, 32)

# input = torch.arange(0, 1*3*32*32*32, dtype=torch.float32).reshape((1, 3, 32, 32, 32))
input = torch.Tensor(input_np)

# print(input)

net = UNet(n_channels=3, n_classes=10)

# device = torch.device('cuda:0')
# net.to(device)
# input.to(device)

output = net(input)

print(output.shape)
print(output)

