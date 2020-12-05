import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time
import os
from queue import Queue
from torch.cuda.amp import autocast as autocast, GradScaler
import logging

from config import cfg
from utils import accuracy, CrossEntropyLabelSmooth
from dataset import create_dataset_pytorch_cifar10
from shuffleNet import ShuffleNetV1

logging.basicConfig(format='%(asctime)s %(filename)s:%(lineno)d '
                           '%(levelname)s: %(message)s',level=logging.INFO)

class Trainer:
    def __init__(self, network, criterion, optimizer, scheduler, dataloader_train, dataloader_test, device, summary_writer,
                        epoch_size, ckpt_path):
        self.network = network
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloader_train = dataloader_train
        self.dataloader_test = dataloader_test
        self.device = device
        self.summary_writer = summary_writer
        self.epoch_size = epoch_size
        self.ckpt_path = ckpt_path
        self.step_per_epoch = len(dataloader_train)
        self.epoch_id = 1
        self.global_step_id = 1
        self.checkpoints_1 = Queue(maxsize=1)
        self.scaler = GradScaler()
        self.best_acc = 0.0

    def train_epoch(self):
        self.network.train()
        time_epoch = 0.0
        total_samples = 0.0
        correct_samples = 0.0
        for batch_idx, data in enumerate(self.dataloader_train, 0):
            time_start = time.time()
            inputs, labels = data
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            # zeros the parameter gradients
            # self.optimizer.zero_grad()    // 好像不清零也没事？？？？？
            with autocast():
                outputs = self.network(inputs)
                loss = self.criterion(outputs, labels)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            # acc
            _, max_index = torch.max(outputs, dim=-1)
            total_samples += labels.size(0)
            correct_samples += (max_index == labels).sum()
            # print statistics
            running_loss = loss.item()
            time_end = time.time()
            time_step = time_end - time_start
            time_epoch = time_epoch + time_step
            logging.info('Epoch: [%3d/%3d], step: [%5d/%5d], loss: [%6.4f], time: [%.4f]' %
                        (self.epoch_id, self.epoch_size, batch_idx + 1, self.step_per_epoch, 
                        running_loss, time_step))
            self.summary_writer.add_scalar('Train/loss', running_loss, self.global_step_id)
            self.summary_writer.add_scalar('Train/lr', self.optimizer.state_dict()['param_groups'][0]['lr'], self.global_step_id)
            self.global_step_id += 1
        
        logging.info('Epoch time: %10.4f, per step time: %7.4f' % (time_epoch, time_epoch / self.step_per_epoch))
        self.summary_writer.add_scalar('Train/acc', correct_samples / total_samples, self.epoch_id)
    
    def eval_training(self):
        self.network.eval()
        time_start = time.time()
        test_loss = 0.0 # cost function error
        total_samples = 0.0
        correct_samples = 0.0
        with torch.no_grad():
            for (inputs, labels) in self.dataloader_test:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.network(inputs)
                loss = self.criterion(outputs, labels)
                _, max_index = torch.max(outputs, dim=-1)
                test_loss += loss.item()
                total_samples += labels.size(0)
                correct_samples += (max_index == labels).sum()

        time_finish = time.time()
        accuracy = correct_samples / total_samples
        avg_loss = test_loss / len(self.dataloader_test)
        logging.info('Test set: Average loss: {:.4f}, Accuracy1: {:.4f}, Time consumed:{:.2f}s'.format(
            avg_loss, accuracy, time_finish - time_start
        ))
        # save best checkpoint
        if accuracy > self.best_acc:
            self.best_acc = accuracy
            if self.checkpoints_1.full():
                last_file = self.checkpoints_1.get()
                os.remove(last_file)
            ckpt_file = ('%s/%d.ckpt_acc1' % (self.ckpt_path, self.epoch_id))
            self.checkpoints_1.put(ckpt_file)
            torch.save(self.network, ckpt_file)
        # summary
        self.summary_writer.add_scalar('Test/Average loss', avg_loss, self.epoch_id)
        self.summary_writer.add_scalar('Test/Accuracy', accuracy, self.epoch_id)
        self.summary_writer.add_scalar('Test/Best accuracy', self.best_acc, self.epoch_id)
        return accuracy

    def step(self):
        self.train_epoch()
        self.eval_training()
        self.epoch_id += 1

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
    network = ShuffleNetV1(input_size=cfg.image_size, n_class=cfg.num_classes, model_size='2.0x', group=3)
    network.to(device)
    criterion = CrossEntropyLabelSmooth(cfg.num_classes, 0.1)
    optimizer = optim.SGD(network.parameters(), lr=cfg.lr_init, momentum=cfg.SGD_momentum, weight_decay=cfg.SGD_weight_decay)
    dataloader_train = create_dataset_pytorch_cifar10(args.data_path)
    dataloader_test = create_dataset_pytorch_cifar10(args.data_path, is_train=False)
    
    total_iters = len(dataloader_train) * cfg.epoch_size
    scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                    lambda step : (1.0 - step * 1.0 / total_iters) if step <= total_iters else 0, 
                    last_epoch=-1)
    summary_writer = SummaryWriter(log_dir='./summary')
    trainer = Trainer(network=network, criterion=criterion, optimizer=optimizer, scheduler=scheduler,
                      dataloader_train=dataloader_train, dataloader_test=dataloader_test, device=device,
                      summary_writer=summary_writer, epoch_size=cfg.epoch_size,
                      ckpt_path=args.ckpt_path)

    for epoch_id in range(cfg.epoch_size):
        trainer.step()

    summary_writer.close()
    print('Finished Training')
    
