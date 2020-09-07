import argparse


parser = argparse.ArgumentParser(description='Hyper-parameters management')

# Hardware options
parser.add_argument('--n_threads', type=int, default=6,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

# data in/out and dataset
parser.add_argument('--dataset_path',default = './fixed',
                    help='fixed trainset root path')
parser.add_argument('--save',default='test_model',
                    help='save path of trained model')
parser.add_argument('--resize_scale', type=float, default=0.5,
                    help='resize scale for input data')
parser.add_argument('--crop_size', type=list, default=[16, 96, 96],
                    help='patch size of train samples after resize')
parser.add_argument('--batch_size', type=list, default=4,
                    help='batch size of trainset')

# train
parser.add_argument('--epochs', type=int, default=3, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')

args = parser.parse_args()


