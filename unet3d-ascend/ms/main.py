import mindspore as ms
import numpy as np

from unet_model import UNet
from mindspore import nn as nn
from mindspore import context


context.set_context(mode=context.GRAPH_MODE, device_target="GPU", device_id=0)

# input = ms.Tensor(np.arange(0, 1*3*32*32*32).reshape((1, 3, 32, 32, 32)), ms.float32)
input_np = np.random.rand(1, 3, 32, 32, 32)
input = ms.Tensor(input_np, ms.float32)

net = UNet(n_channels=3, n_classes=10)

output = net(input)

print(output)
print(output.shape)

