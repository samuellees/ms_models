import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import os

from dali_pipeline import HybridTrainPipe
from nvidia.dali.plugin.pytorch import DALIClassificationIterator

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        return output


def main_worker(local_rank, args):
    # Parameters
    input_size = 5
    output_size = 2
    batch_size_per_gpu = 8
    data_size = 1000
    world_size = args.world_size
    # prepare dist environment
    dist.init_process_group(backend='nccl', rank=local_rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    # model
    model = Model(input_size, output_size)
    model = model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank])
    # dataloader
    # train_dataset = RandomDataset(input_size, data_size)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, rank=local_rank)
    # rand_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_per_gpu, sampler=train_sampler, num_workers=4)
    # pipeline
    pipe = HybridTrainPipe(batch_size=cfg.batch_size,
                           num_threads=cfg.n_workers,
                           device_id=local_rank,
                           data_dir=args.data_path,
                           crop=cfg.image_size,
                           local_rank=args.local_rank,
                           world_size=args.world_size)
    pipe.build()
    dataloader = DALIClassificationIterator(pipe, reader_name="Reader")
    # run
    for data in rand_loader:
        input = data.cuda()
        output = model(input)
        # print("local_rank=", local_rank, ", Outside: input size", input.size(),
        #     "output_size", output.size())



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pytorch on CUDA')
    main_args = parser.parse_args()
    main_args.world_size = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    mp.spawn(main_worker, nprocs=main_args.world_size, args=(main_args,))
