import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch


class SeparableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, relu_position='none', has_bn=False):
        super(SeparableConvBlock, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                   stride=1, padding=1, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1, bias=bias)
        # bn + relu
        self.has_bn = has_bn
        self.relu_position = relu_position
        if self.has_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        if self.relu_position == 'before' or self.relu_position == 'after':
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.relu_position == 'before':
            x = self.relu(x)
        x = self.depthwise(x)
        x = self.pointwise(x)
        if self.has_bn:
            x = self.bn(x)
        if self.relu_position == 'after':
            x = self.relu(x)
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, has_bn=False, has_relu=False, **kwargs):
        super(BasicConv2d, self).__init__()
        self.has_bn = has_bn
        self.has_relu = has_relu
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        if self.has_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        if self.has_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)
        return x


class EntryFlowBlockA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EntryFlowBlockA, self).__init__()
        self.sep_conv1 = SeparableConvBlock(in_channels, out_channels, relu_position='none', has_bn=True)
        self.sep_conv2 = SeparableConvBlock(out_channels, out_channels, relu_position='before', has_bn=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.shortcut = BasicConv2d(in_channels, out_channels, kernel_size=1, stride=2, has_bn=True, has_relu=False)

    def forward(self, x):
        main_branch = self.sep_conv1(x)
        main_branch = self.sep_conv2(main_branch)
        main_branch = self.max_pool(main_branch)
        main_branch += self.shortcut(x)
        return main_branch


class EntryFlowBlockB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EntryFlowBlockB, self).__init__()
        self.sep_conv1 = SeparableConvBlock(in_channels, out_channels, relu_position='before', has_bn=True)
        self.sep_conv2 = SeparableConvBlock(out_channels, out_channels, relu_position='before', has_bn=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.shortcut = BasicConv2d(in_channels, out_channels, kernel_size=1, stride=2, has_bn=True, has_relu=False)

    def forward(self, x):
        main_branch = self.sep_conv1(x)
        main_branch = self.sep_conv2(main_branch)
        main_branch = self.max_pool(main_branch)
        main_branch += self.shortcut(x)
        return main_branch


class MiddleFlowBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MiddleFlowBlock, self).__init__()
        self.sep_conv1 = SeparableConvBlock(in_channels, out_channels, relu_position='before', has_bn=True)
        self.sep_conv2 = SeparableConvBlock(out_channels, out_channels, relu_position='before', has_bn=True)
        self.sep_conv3 = SeparableConvBlock(out_channels, out_channels, relu_position='before', has_bn=True)
        
    def forward(self, x):
        main_branch = self.sep_conv1(x)
        main_branch = self.sep_conv2(main_branch)
        main_branch = self.sep_conv3(main_branch)
        main_branch += x
        return main_branch


class ExitFlowBlock(nn.Module):
    def __init__(self, in_channels, out_channels_1, out_channels_2):
        super(ExitFlowBlock, self).__init__()
        self.sep_conv1 = SeparableConvBlock(in_channels, out_channels_1, relu_position='before', has_bn=True)
        self.sep_conv2 = SeparableConvBlock(out_channels_1, out_channels_2, relu_position='before', has_bn=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.shortcut = BasicConv2d(in_channels, out_channels_2, kernel_size=1, stride=2, has_bn=True, has_relu=False)

    def forward(self, x):
        main_branch = self.sep_conv1(x)
        main_branch = self.sep_conv2(main_branch)
        main_branch = self.max_pool(main_branch)
        main_branch += self.shortcut(x)
        return main_branch


class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, num_classes=1000):
        super(Xception, self).__init__()
        self.num_classes = num_classes

        # entry flow
        self.entry_conv1 = BasicConv2d(3, 32, kernel_size=3, stride=2, padding=0, bias=False, has_bn=True, has_relu=True)
        self.entry_conv2 = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=0, bias=False, has_bn=True, has_relu=True)
    
        self.entry_block1 = EntryFlowBlockA(64, 128)
        self.entry_block2 = EntryFlowBlockB(128, 256)
        self.entry_block3 = EntryFlowBlockB(256, 728)

        # middle flow
        self.middle_block_list = nn.ModuleList([MiddleFlowBlock(728, 728) for i in range(8)])

        # exit flow
        self.exit_block = ExitFlowBlock(728, 728, 1024)
        self.exit_sep_conv1 = SeparableConvBlock(1024, 1536, has_bn=True, relu_position='after')
        self.exit_sep_conv2 = SeparableConvBlock(1536, 2048, has_bn=True, relu_position='after')
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(2048, num_classes)

        # ------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                init.xavier_uniform_(m.weight)
            # elif isinstance(m, nn.Linear):
            #     init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        # -----------------------------

    def forward(self, x):
        # entry flow
        x = self.entry_conv1(x)
        x = self.entry_conv2(x)
        x = self.entry_block1(x)
        x = self.entry_block2(x)
        x = self.entry_block3(x)

        # middle flow
        for middle_block in self.middle_block_list:
            x = middle_block(x)

        # exit flow
        x = self.exit_block(x)
        x = self.exit_sep_conv1(x)
        x = self.exit_sep_conv2(x)
        x = self.global_avg_pool(x)
        
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
