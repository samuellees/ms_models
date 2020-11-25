# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from xception import xception
import torch
#from model_ref import AlignedXception

input = torch.zeros(32, 3, 299, 299)

#model_ref = AlignedXception(8)
model = xception()
writer = SummaryWriter('./graph_torch')
writer.add_graph(model, input)
writer.close()

