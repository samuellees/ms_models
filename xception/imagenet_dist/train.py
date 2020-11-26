import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import os
from queue import Queue

from config import cfg
from dataset import create_dataset_pytorch_imagenet_dist
from dali_pipeline import HybridTrainPipe
from nvidia.dali.plugin.pytorch import DALIClassificationIterator
from xception import Xception

class Trainer:
    def __init__(self, network=None, criterion=None, optimizer=None, scheduler=None,
                 dataloader_train=None, dataloader_test=None, device=None, 
                 summary_writer=None, epoch_size=None, local_rank=None,
                 ckpt_path=None):
        self.network = network
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloader_train = dataloader_train
        self.dataloader_test = dataloader_test
        self.device = device
        self.summary_writer = summary_writer
        self.epoch_size = epoch_size
        self.local_rank = local_rank
        self.ckpt_path = ckpt_path
        self.step_per_epoch = len(dataloader_train)
        self.epoch_id = 1
        self.global_step_id = 1
        self.checkpoints = Queue(maxsize=1)
        self.best_acc = 0.0

    def train_epoch(self):
        self.network.train()
        time_epoch = 0.0
        for batch_idx, data in enumerate(self.dataloader_train, 0):
            time_start = time.time()
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            # zeros the parameter gradients
            self.optimizer.zero_grad()
            outputs = self.network(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            # print statistics
            running_loss = loss.item()
            time_end = time.time()
            time_step = time_end - time_start
            time_epoch = time_epoch + time_step
            if self.local_rank == 0:
                print('Epoch: [%3d/%3d], step: [%5d/%5d], loss: [%6.4f], time: [%.4f]' %
                            (self.epoch_id, self.epoch_size, batch_idx + 1, self.step_per_epoch, 
                            running_loss, time_step))
                self.summary_writer.add_scalar('Train/loss', running_loss, self.global_step_id)
            self.global_step_id += 1
        
        if self.local_rank == 0:
            print('Epoch time: %10.4f, per step time: %7.4f' % (time_epoch, time_epoch / self.step_per_epoch))
    
    def eval_training(self):
        self.network.eval()
        time_start = time.time()
        test_loss = 0.0 # cost function error
        total_samples = 0.0
        correct_samples = 0.0
        for (inputs, labels) in self.dataloader_test:
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = self.network(inputs)
            loss = self.criterion(outputs, labels)
            _, max_index = torch.max(outputs, dim=-1)
            test_loss += loss.item()
            total_samples += labels.size(0)
            correct_samples += (max_index == labels).sum()

        time_finish = time.time()
        accuracy = correct_samples / total_samples
        avg_loss = test_loss / len(self.dataloader_test)
        if self.local_rank == 0:
            print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
                avg_loss, accuracy, time_finish - time_start
            ))
            self.summary_writer.add_scalar('Test/Average loss', avg_loss, self.epoch_id)
            self.summary_writer.add_scalar('Test/Accuracy', accuracy, self.epoch_id)
        return accuracy

    def save_network(self):
        if self.local_rank == 0:
            # save best checkpoint
            if self.checkpoints.full():
                last_file = self.checkpoints.get()
                os.remove(last_file)
            ckpt_file = ('%s/%d.ckpt' % (self.ckpt_path, self.epoch_id))
            self.checkpoints.put(ckpt_file)
            torch.save(self.network, ckpt_file)

    def step(self):
        self.train_epoch()
        acc = self.eval_training()
        if acc > self.best_acc:
            self.best_acc = acc
            self.save_network()
        self.epoch_id += 1

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
    dataloader_train = create_dataset_pytorch_imagenet_dist(data_path=args.data_path+'train', local_rank=local_rank)
    dataloader_test = create_dataset_pytorch_imagenet_dist(data_path=args.data_path+'val', is_train=False)
    step_per_epoch = len(dataloader_train)
    scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=cfg.lr_decay_rate,
                                        step_size=cfg.lr_decay_epoch*step_per_epoch)
    summary_writer = None
    if local_rank == 0:
        summary_writer = SummaryWriter(log_dir='./summary')
    trainer = Trainer(network=network, criterion=criterion, optimizer=optimizer, scheduler=scheduler,
                      dataloader_train=dataloader_train, dataloader_test=dataloader_test,
                      summary_writer=summary_writer, epoch_size=cfg.epoch_size,
                      ckpt_path=args.ckpt_path, local_rank=local_rank)

    for epoch_id in range(cfg.epoch_size):
        trainer.step()


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
