import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim

from config import cfg
from dataset import create_dataset_pytorch
from xception import Xception


def main_worker(local_rank, args):
    args.local_rank = local_rank
    dist.init_process_group(backend='nccl', rank=local_rank, world_size=args.world_size)
    torch.cuda.set_device(args.local_rank)
    
#     network=nn.DataParallel(Xception(num_classes=cfg.num_classes))
    if local_rank == 0:
        network = torch.load(args.ckpt_path)

        dataloader = create_dataset_pytorch(args.data_path)
        with torch.no_grad():
            total_samples = 0.0
            correct_samples = 0.0
            for inputs, labels in dataloader:
                inputs = inputs.cuda()
                labels = labels.cuda()
                outputs = network(inputs)
                _, max_index = torch.max(outputs, dim=-1)
                total_samples += labels.size(0)
                correct_samples += (max_index == labels).sum()
            print('Accuracy: {}'.format(correct_samples / total_samples), flush=True)

        print('Finished Testing', flush=True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pytorch on CUDA')
    parser.add_argument('--data_path', type=str, default="./data",
                        help='path where the dataset is saved')
    parser.add_argument('--ckpt_path', type=str, default="./checkpoint",
                        help='path where the checkpoint to be saved')
    main_args = parser.parse_args()
    main_args.world_size = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    mp.spawn(main_worker, nprocs=main_args.world_size, args=(main_args,))
