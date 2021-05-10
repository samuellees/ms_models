
from mindspore.ops import operations as P
from mindspore import nn, context
import mindspore as ms
import numpy as np
import time
from mindspore import dtype as mstype


# context.set_context(mode=context.PYNATIVE_MODE)
context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

# input1 = ms.Tensor(np.ones((32, 3, 32, 32), np.float16))
# input2 = ms.Tensor(np.ones((32, 3, 32, 32), np.float16))

# output = input1 + input2
# _ = output.asnumpy()
# start = time.time()
# for _ in range(100):
#   output = input1 + input2
#   _ = output.asnumpy()
# end = time.time()
# print("ms Add: ", end-start)

# output = input1 * input2
# _ = output.asnumpy()
# start = time.time()
# for _ in range(100):
#   output = input1 * input2
#   _ = output.asnumpy()
# end = time.time()
# print("ms Mul: ", end-start)

# output = input1 - input2
# _ = output.asnumpy()
# start = time.time()
# for _ in range(100):
#   output = input1 - input2
#   _ = output.asnumpy()
# end = time.time()
# print("ms Sub: ", end-start)

# output = input1.mean()
# _ = output.asnumpy()
# start = time.time()
# for _ in range(100):
#   output = input1.mean()
#   _ = output.asnumpy()
# end = time.time()
# print("ms Mean: ", end-start)



class BatchNorm3d(nn.Cell):
    def __init__(self, num_features):
        super().__init__()
        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.bn2d = nn.BatchNorm2d(num_features, data_format="NCHW")

    def construct(self, x):
        x_shape = self.shape(x)
        x = self.reshape(x, (x_shape[0], x_shape[1], x_shape[2] * x_shape[3], x_shape[4]))
        bn2d_out = self.bn2d(x)
        bn3d_out = self.reshape(bn2d_out, x_shape)
        return bn3d_out

input1 = ms.Tensor(np.ones((1, 16, 112, 112, 48), dtype=np.float32))
input2 = ms.Tensor(np.ones((1, 16, 112, 112, 48), dtype=np.float32))
input3 = ms.Tensor(np.ones((1, 16, 112, 112, 48), dtype=np.float32))
input4 = ms.Tensor(np.ones((1, 16, 112, 112, 48), dtype=np.float32))
input5 = ms.Tensor(np.ones((1, 16, 112, 112, 48), dtype=np.float32))
input6 = ms.Tensor(np.ones((1, 16, 112, 112, 48), dtype=np.float32))
input7 = ms.Tensor(np.ones((1, 16, 112, 112, 48), dtype=np.float32))
input8 = ms.Tensor(np.ones((1, 16, 112, 112, 48), dtype=np.float32))
input9 = ms.Tensor(np.ones((1, 16, 112, 112, 48), dtype=np.float32))
input10 = ms.Tensor(np.ones((1, 16, 112, 112, 48), dtype=np.float32))
bn1 = nn.BatchNorm3d(1, momentum=0.9, use_batch_statistics=True)
bn2 = nn.BatchNorm3d(1, momentum=0.9, use_batch_statistics=True)
bn3 = nn.BatchNorm3d(1, momentum=0.9, use_batch_statistics=True)
bn4 = nn.BatchNorm3d(1, momentum=0.9, use_batch_statistics=True)
bn5 = nn.BatchNorm3d(1, momentum=0.9, use_batch_statistics=True)
bn6 = nn.BatchNorm3d(1, momentum=0.9, use_batch_statistics=True)
bn7 = nn.BatchNorm3d(1, momentum=0.9, use_batch_statistics=True)
bn8 = nn.BatchNorm3d(1, momentum=0.9, use_batch_statistics=True)
bn9 = nn.BatchNorm3d(1, momentum=0.9, use_batch_statistics=True)
bn10 = nn.BatchNorm3d(1, momentum=0.9, use_batch_statistics=True)

output = bn(inputs)
_ = output.asnumpy()
start = time.time()
for _ in range(100):
  output = bn(inputs)
  _ = output.asnumpy()
end = time.time()
print("ms BatchNorm3D: ", (end-start)/100)