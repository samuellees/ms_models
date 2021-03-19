
from mindspore.ops import operations as P
from mindspore import nn, context
import mindspore as ms
import numpy as np


context.set_context(mode=context.PYNATIVE_MODE)
# context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

input = ms.Tensor(np.ones((32, 3, 32, 32)), ms.float32)

# print(input.shape)

conv = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3)

out = conv(input)

print(out)
