import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import os
from queue import Queue

from config import cfg
from dataset import create_dataset_pytorch_cifar10
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

    device = torch.device('cuda:'+str(args.device_id))
    network = Xception(num_classes=cfg.num_classes)
    network.train()
    network.to(device)
    criterion = nn.CrossEntropyLoss()
#     optimizer = optim.RMSprop(network.parameters(), 
#                                 lr=cfg.lr_init, 
#                                 eps=cfg.rmsprop_epsilon,
#                                 momentum=cfg.rmsprop_momentum, 
#                                 alpha=cfg.rmsprop_decay)
    
    optimizer = optim.SGD(network.parameters(), lr=cfg.lr_init, momentum=cfg.SGD_momentum)
    dataloader = create_dataset_pytorch_cifar10(args.data_path)
    step_per_epoch = len(dataloader)
    scheduler = optim.lr_scheduler.StepLR(
                                optimizer, 
                                gamma=cfg.lr_decay_rate, 
                                step_size=cfg.lr_decay_epoch*step_per_epoch)
    q_ckpt = Queue(maxsize=cfg.keep_checkpoint_max)

    global_step_id = 0
#     for epoch in range(cfg.epoch_size):
    for epoch in range(1):
        time_epoch = 0.0
        for i, data in enumerate(dataloader, 0):
            time_start = time.time()
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zeros the parameter gradients
            optimizer.zero_grad()
            with torch.cuda.profiler.profile():
                network(inputs)
                with torch.autograd.profiler.emit_nvtx():
                    outputs = network(inputs)
            break
#             with torch.autograd.profiler.profile(use_cuda=True) as prof:
#                 outputs = network(inputs)
#             print(prof)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            # print statistics
            running_loss = loss.item()
            time_end = time.time()
            time_step = time_end - time_start
            time_epoch = time_epoch + time_step
            print('Epoch: [%3d/%3d], step: [%5d/%5d], loss: [%6.4f], time: [%.4f]' %
                        (epoch + 1, cfg.epoch_size, i + 1, step_per_epoch, 
                        running_loss, time_step), flush=True)

            # save checkpoint every epoch
            global_step_id = global_step_id + 1
            if global_step_id % cfg.save_checkpoint_steps == 0:
                if q_ckpt.full():
                    last_file = q_ckpt.get()
                    os.remove(last_file)
                ckpt_file = ('%s/%d-%d.ckpt' % 
                            (args.ckpt_path, epoch + 1, i + 1))
                q_ckpt.put(ckpt_file)
                torch.save(network, ckpt_file)
        
        print('Epoch time: %10.4f, per step time: %7.4f' %
            (time_epoch, time_epoch / step_per_epoch), flush=True)

    print('Finished Training', flush=True)
    
