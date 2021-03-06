import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import os
from queue import Queue

from config import cfg
from dali_pipeline import HybridTrainPipe
from nvidia.dali.plugin.pytorch import DALIClassificationIterator
from dataset import create_dataset_pytorch
from xception import Xception

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pytorch on CUDA')
    parser.add_argument('--data_path', type=str, default="./data",
                        help='path where the dataset is saved')
    parser.add_argument('--ckpt_path', type=str, default="./checkpoint",
                        help='path where the checkpoint to be saved')
    parser.add_argument('--device_id', type=int, default=0,
                        help='device id of GPU. (Default: 0)')
    args = parser.parse_args()
    args.local_rank = 0
    args.world_size = 1

    network = Xception(num_classes=cfg.num_classes)
    # network = nn.DataParallel(network)
    network = network.cuda()
    criterion = nn.CrossEntropyLoss()
#     optimizer = optim.RMSprop(network.parameters(), 
#                                 lr=cfg.lr_init, 
#                                 eps=cfg.rmsprop_epsilon,
#                                 momentum=cfg.rmsprop_momentum, 
#                                 alpha=cfg.rmsprop_decay)
    optimizer = optim.SGD(network.parameters(), lr=cfg.lr_init, momentum=cfg.SGD_momentum)
    # prepare data
    # dataloader = create_dataset_pytorch(args.data_path + "/train")
    pipe = HybridTrainPipe(batch_size=cfg.batch_size,
                           num_threads=cfg.n_workers,
                           device_id=args.local_rank,
                           data_dir=args.data_path,
                           crop=cfg.image_size,
                           local_rank=args.local_rank,
                           world_size=args.world_size)
    pipe.build()
    dataloader = DALIClassificationIterator(pipe, reader_name="Reader")
    step_per_epoch = dataloader.size / cfg.batch_size
    print("step_per_epoch =", step_per_epoch)
    scheduler = optim.lr_scheduler.StepLR(
                                optimizer, 
                                gamma=cfg.lr_decay_rate, 
                                step_size=cfg.lr_decay_epoch*step_per_epoch)
    q_ckpt = Queue(maxsize=cfg.keep_checkpoint_max)

    global_step_id = 0
    for epoch in range(cfg.epoch_size):
        time_epoch = 0.0
        torch.cuda.synchronize()
        for i, data in enumerate(dataloader):
            time_start = time.time()
            # inputs, labels = data
            # inputs = inputs.cuda()
            # labels = labels.cuda()
            inputs = data[0]["data"].cuda(non_blocking=True)
            labels = data[0]["label"].squeeze().long().cuda(non_blocking=True)
            # zeros the parameter gradients
            optimizer.zero_grad()
            outputs = network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()  
            # print statistics
            running_loss = loss.item()
            torch.cuda.synchronize()
            time_end = time.time()
            time_step = time_end - time_start
            time_epoch = time_epoch + time_step
            print('Epoch: [%3d/%3d], step: [%5d/%5d], loss: [%6.4f], time: [%.4f]' %
                    (epoch + 1, cfg.epoch_size, i + 1, step_per_epoch, 
                    running_loss, time_step), flush=True)
            
            # save checkpoint every epoch
            global_step_id = global_step_id + 1
            if global_step_id % step_per_epoch == 0:
                if q_ckpt.full():
                    last_file = q_ckpt.get()
                    os.remove(last_file)
                ckpt_file = ('%s/%d-%d.ckpt' % 
                            (args.ckpt_path, epoch + 1, i + 1))
                q_ckpt.put(ckpt_file)
                torch.save(network, ckpt_file)
        dataloader.reset()
        print('Epoch time: %10.4f, per step time: %7.4f' %
            (time_epoch, time_epoch / step_per_epoch), flush=True)

    print('Finished Training', flush=True)
    
