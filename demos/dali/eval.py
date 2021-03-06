import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from config import cfg
from dataset import create_dataset_pytorch
from xception import Xception


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pytorch on CUDA')
    parser.add_argument('--data_path', type=str, default="./data",
                        help='path where the dataset is saved')
    parser.add_argument('--ckpt_path', type=str, default="./checkpoint",
                        help='path where the checkpoint to be saved')
    parser.add_argument('--device_id', type=int, default=0, help='device id of GPU. (Default: 0)')
    args = parser.parse_args()

    device = torch.device('cuda:'+str(args.device_id))
#     network=nn.DataParallel(Xception(num_classes=cfg.num_classes))
    network = torch.load(args.ckpt_path)
    network = network.module
    network.to(device)

    dataloader = create_dataset_pytorch(args.data_path + "/val")
    with torch.no_grad():
        total_samples = 0.0
        correct_samples = 0.0
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = network(inputs)
            _, max_index = torch.max(outputs, dim=-1)
            total_samples += labels.size(0)
            correct_samples += (max_index == labels).sum()
        print('Accuracy: {}'.format(correct_samples / total_samples), flush=True)

    print('Finished Testing', flush=True)
