
import torchvision


data_path = '/userhome/datasets/ImageNet2012/mini_batch/'
dataset = torchvision.datasets.ImageFolder(root=data_path, transform=None)
train_dataset = torchvision.datasets.ImageFolder(root=data_path+'train', transform=None)
val_dataset = torchvision.datasets.ImageFolder(root=data_path+'val', transform=None)

print(len(dataset.imgs))
print(len(train_dataset.imgs))
print(len(val_dataset.imgs))

