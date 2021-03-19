# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from monai.networks.nets import UNet
from monai.networks.layers import Norm
import torch
#from model_ref import AlignedXception

input = torch.zeros(2, 3, 128, 128, 128)

model = UNet(
    dimensions=3,
    in_channels=3,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
)

writer = SummaryWriter('./graph_torch')
writer.add_graph(model, input)
writer.close()

