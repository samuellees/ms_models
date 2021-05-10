import numpy as np
import time
import torch
import torch.nn.functional as F

# input1 = torch.Tensor(np.ones((32, 3, 32, 32), np.float16)).cuda()
# input2 = torch.Tensor(np.ones((32, 3, 32, 32), np.float16)).cuda()

# output = input1 + input2
# _ = output.cpu().numpy()
# start = time.time()
# for _ in range(100):
#   output = input1 + input2
#   _ = output.cpu().numpy()
# end = time.time()
# print("torch Add: ", end-start)

# output = input1 * input2
# _ = output.cpu().numpy()
# start = time.time()
# for _ in range(100):
#   output = input1 * input2
#   _ = output.cpu().numpy()
# end = time.time()
# print("torch Mul: ", end-start)

# output = input1 - input2
# _ = output.cpu().numpy()
# start = time.time()
# for _ in range(100):
#   output = input1 - input2
#   _ = output.cpu().numpy()
# end = time.time()
# print("torch Sub: ", end-start)

# output = input1.mean()
# _ = output.cpu().numpy()
# start = time.time()
# for _ in range(100):
#   output = input1.mean()
#   _ = output.cpu().numpy()
# end = time.time()
# print("torch Mean: ", end-start)




device = torch.device("cuda:0")
inputs = torch.ones((1, 1, 224, 224, 96)).type(torch.float32).to(device)
bn = torch.nn.BatchNorm3d(1, momentum=0.9, affine=True, track_running_stats=True).to(device)

output = bn(inputs)
_ = output.detach().cpu().numpy()
start = time.time()
for _ in range(100):
  output = bn(inputs)
  _ = output.detach().cpu().numpy()
end = time.time()
print("torch BatchNorm3D: ", (end-start)/100)