import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import os
from queue import Queue
from torch.cuda.amp import autocast as autocast, GradScaler

from config import cfg
from dali_pipeline import HybridTrainPipe
from nvidia.dali.plugin.pytorch import DALIClassificationIterator
from xception import Xception


def main_worker(local_rank, args):
    args.local_rank = local_rank
    # prepare dist environment
    dist.init_process_group(backend='nccl', rank=args.local_rank, world_size=args.world_size)
    torch.cuda.set_device(args.local_rank)
    network = Xception(num_classes=cfg.num_classes)
    network = network.cuda()
    network = torch.nn.parallel.DistributedDataParallel(
        network, device_ids=[args.local_rank])
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(network.parameters(),
                          lr=cfg.lr_init, momentum=cfg.SGD_momentum)
#     optimizer = optim.RMSprop(network.parameters(),
#                                 lr=cfg.lr_init,
#                                 eps=cfg.rmsprop_epsilon,
#                                 momentum=cfg.rmsprop_momentum,
#                                 alpha=cfg.rmsprop_decay)

    # prepare data
    # pipe = HybridTrainPipe(batch_size=cfg.batch_size,
    #                        num_threads=cfg.n_workers,
    #                        device_id=args.local_rank,
    #                        data_dir=args.data_path,
    #                        crop=cfg.image_size,
    #                        local_rank=args.local_rank,
    #                        world_size=args.world_size)
    # pipe.build()
    # dataloader = DALIClassificationIterator(pipe, reader_name="Reader")
    # step_per_epoch = dataloader.size / cfg.batch_size

    transform = transforms.Compose([
        transforms.RandomResizedCrop(cfg.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    train_dataset = torchvision.datasets.ImageFolder(root=args.data_path, transform=transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, rank=local_rank)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, sampler=train_sampler, num_workers=cfg.n_workers)
    step_per_epoch = len(dataloader)

    print("step_per_epoch =", step_per_epoch)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        gamma=cfg.lr_decay_rate,
        step_size=cfg.lr_decay_epoch*step_per_epoch)
    
    scaler = GradScaler()

    if args.local_rank == 0:
        q_ckpt = Queue(maxsize=cfg.keep_checkpoint_max)
    global_step_id = 0
    for epoch in range(cfg.epoch_size):
        time_epoch = 0.0
        for i, data in enumerate(dataloader):
            time_start = time.time()
            # inputs = data[0]["data"].cuda(non_blocking=True)
            # labels = data[0]["label"].squeeze().long().cuda(non_blocking=True)
            inputs, labels = data[0].cuda(), data[1].cuda()
            # zeros the parameter gradients
            optimizer.zero_grad()
            
            with autocast():
                outputs = network(inputs)
                loss = criterion(outputs, labels)
#             outputs = network(inputs)
#             loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
#             loss.backward()
#             optimizer.step()
            # print statistics
            running_loss = loss.item()
            time_end = time.time()
            time_step = time_end - time_start
            time_epoch = time_epoch + time_step
            # print result and save model
            if args.local_rank == 0:
                print('Epoch: [%3d/%3d], step: [%5d/%5d], loss: [%6.4f], time: [%.4f]' %
                      (epoch + 1, cfg.epoch_size, i + 1, step_per_epoch,
                       running_loss, time_step), flush=True)
                global_step_id = global_step_id + 1
                if global_step_id % step_per_epoch == 0:
                    if q_ckpt.full():
                        last_file = q_ckpt.get()
                        os.remove(last_file)
                    ckpt_file = ('%s/%d-%d.ckpt' %
                                 (args.ckpt_path, epoch + 1, i + 1))
                    q_ckpt.put(ckpt_file)
                    torch.save(network, ckpt_file)
        # end loop data
#         dataloader.reset()
        if args.local_rank == 0:
            print('Epoch time: %10.4f, per step time: %7.4f' %
                  (time_epoch, time_epoch / step_per_epoch), flush=True)
    # end loop epoches
    if args.local_rank == 0:
        print('Finished Training', flush=True)


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