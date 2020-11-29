from config import cfg
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def create_dataset_pytorch_cifar10(data_path, is_train=True, n_workers=8):

  transform = transforms.Compose([
    transforms.RandomResizedCrop(cfg.image_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
  ds_cifar10 = torchvision.datasets.CIFAR10(root=data_path, train=is_train,
                                          download=False, transform=transform)
  data_loader = DataLoader(dataset=ds_cifar10, 
                          batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=n_workers)
  return data_loader


def create_dataset_pytorch_imagenet(data_path, is_train=True, n_workers=4):
  if is_train:
      transform = transforms.Compose([
        transforms.RandomResizedCrop(cfg.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
  else:
    transform = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

  dataset = torchvision.datasets.ImageFolder(root=data_path, transform=transform)
  data_loader = DataLoader(dataset=dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=n_workers)

  return data_loader


def create_dataset_pytorch_imagenet_dist_train(data_path, local_rank=0, n_workers=4):
  transform = transforms.Compose([
    transforms.RandomResizedCrop(cfg.image_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

  dataset = torchvision.datasets.ImageFolder(root=data_path, transform=transform)
  sampler = torch.utils.data.distributed.DistributedSampler(dataset, rank=local_rank)
  data_loader = DataLoader(dataset=dataset, batch_size=cfg.batch_size, drop_last=True, sampler=sampler, num_workers=n_workers)
  return data_loader