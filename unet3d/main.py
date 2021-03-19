import torch
import numpy as np
from unet_model_torch import UNet as UNet_torch

input_np = np.random.rand(1, 3, 32, 32, 32)
input_torch = torch.Tensor(input_np)
net_torch = UNet_torch(n_channels=3, n_classes=10)
net_torch.train()
# net_torch.eval()
optimizer = torch.optim.RMSprop(net_torch.parameters(), 
                                lr=0.01, 
                                eps=1e-5,
                                alpha=0.1)
optimizer.zero_grad()
output_torch = net_torch(input_torch)
loss = torch.sum(output_torch)
loss.backward()
optimizer.step()
output_torch = net_torch(input_torch)


import mindspore as ms
from unet_model_ms import UNet as UNet_ms
import mindspore.ops.composite as C
import mindspore.ops.operations as P
import mindspore.ops.functional as F
from mindspore import context

context.set_context(mode=context.GRAPH_MODE, device_target="GPU", device_id=0)
input_ms = ms.Tensor(input_np, ms.float32)
net_ms = UNet_ms(n_channels=3, n_classes=10)
net_ms.set_train(True)
# net_ms.set_train(False)
optimizer_ms = ms.nn.RMSProp(net_ms.trainable_params(), learning_rate=0.01, 
                decay=0.1, 
                epsilon=1e-5)
output_ms = net_ms(input_ms)
loss = ms.ops.ReduceSum(keep_dims=False)(output_ms)
sens = P.Fill()(P.DType()(loss), P.Shape()(loss), 1.0)
grads = C.GradOperation(get_by_list=True, sens_param=True)(net_ms, optimizer_ms.parameters)(input_ms, sens)
output_ms = F.depend(loss, optimizer_ms(grads))


np.testing.assert_allclose(output_torch.detach().numpy(), output_ms.asnumpy(), rtol=1e-3)



