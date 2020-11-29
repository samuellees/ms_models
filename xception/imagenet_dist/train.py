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
from dataset import create_dataset_pytorch_imagenet_dist_train, create_dataset_pytorch_imagenet_dist_train_2222
from dataset import create_dataset_pytorch_imagenet
from xception import Xception

class Trainer:
    def __init__(self, network=None, criterion=None, optimizer=None, scheduler=None,
                 dataloader_train=None, dataloader_test=None, device=None, 
                 summary_writer=None, epoch_size=None, local_rank=None,
                 ckpt_path=None, barrier=None):
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
        self.barrier = barrier
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
    
    # only called by process 0 
    def eval_training(self):
        if self.local_rank != 0: return
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
        print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
            avg_loss, accuracy, time_finish - time_start
        ))
        # save best checkpoint
        if accuracy > self.best_acc:
            self.best_acc = accuracy
            if self.checkpoints.full():
                last_file = self.checkpoints.get()
                os.remove(last_file)
            ckpt_file = ('%s/%d.ckpt' % (self.ckpt_path, self.epoch_id))
            self.checkpoints.put(ckpt_file)
            torch.save(self.network, ckpt_file)
        # summary
        self.summary_writer.add_scalar('Test/Average loss', avg_loss, self.epoch_id)
        self.summary_writer.add_scalar('Test/Accuracy', accuracy, self.epoch_id)
        self.summary_writer.add_scalar('Test/Best accuracy', self.best_acc, self.epoch_id)
        return accuracy

    def step(self):
        self.train_epoch()
        # self.eval_training()
        # barrier.wait()
        self.epoch_id += 1

def main_worker(local_rank, args, barrier):
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
    # dataloader_test = create_dataset_pytorch_imagenet(
    #         data_path=args.data_path+'val', is_train=False, n_workers=cfg.n_workers)

    # dataloader_train = create_dataset_pytorch_imagenet_dist_train(
    #         data_path=args.data_path, local_rank=local_rank, n_workers=cfg.n_workers)
    ########### unfold #####
    # transform = transforms.Compose([
    #     transforms.RandomCrop((32, 32), (4, 4, 4, 4)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.Resize((cfg.image_size, cfg.image_size)),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    # dataset = torchvision.datasets.ImageFolder(root=data_path, transform=transform)
    # sampler = torch.utils.data.distributed.DistributedSampler(dataset, rank=local_rank)
    # data_loader = DataLoader(dataset=dataset, batch_size=cfg.batch_size, drop_last=True, sampler=sampler, num_workers=n_workers)
    ########################



    dataloader_train = create_dataset_pytorch_imagenet_dist_train_2222(
            data_path=args.data_path, local_rank=local_rank, n_workers=cfg.n_workers)
    ### ok ####
    # transform = transforms.Compose([
    #     transforms.RandomResizedCrop(cfg.image_size),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    # train_dataset = torchvision.datasets.ImageFolder(root=args.data_path, transform=transform)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, rank=local_rank)
    # dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, sampler=train_sampler, num_workers=cfg.n_workers)
    ### ok ####




    step_per_epoch = len(dataloader_train)

    print("step_per_epoch =", step_per_epoch)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        gamma=cfg.lr_decay_rate,
        step_size=cfg.lr_decay_epoch*step_per_epoch)

    global_step_id = 0
    for epoch in range(cfg.epoch_size):
        time_epoch = 0.0
        for i, data in enumerate(dataloader_train):
            time_start = time.time()
            # inputs = data[0]["data"].cuda(non_blocking=True)
            # labels = data[0]["label"].squeeze().long().cuda(non_blocking=True)
            inputs, labels = data[0].cuda(), data[1].cuda()
            # zeros the parameter gradients
            optimizer.zero_grad()
            outputs = network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
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
        # end loop data
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
    barrier = mp.Barrier(parties=main_args.world_size)
    mp.spawn(main_worker, nprocs=main_args.world_size, args=(main_args, barrier,))
