import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

__all__ = ['SqueezeNet', 'LabelSmoothingCrossEntropy']

def with_bias():
    return False

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = BasicConv2d(inplanes, squeeze_planes, kernel_size=1, bias=with_bias())
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = BasicConv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1, bias=with_bias())
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = BasicConv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1, bias=with_bias())
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class SqueezeNet(nn.Module):

    def __init__(self, num_classes=1000, version='1.0'):
        super(SqueezeNet, self).__init__()
        self.num_classes = num_classes
        if version == '1.0':
            self.features = nn.Sequential(
                BasicConv2d(3, 96, kernel_size=7, stride=2, bias=with_bias()),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False),
                # (32, 256, 54, 54)
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False),
                # (32, 256, 27, 27)
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False),
                Fire(512, 64, 256, 256)
                # (32, 256, 13, 13)
            )
        # elif version == '1.1':
        #     self.features = nn.Sequential(
        #         nn.Conv2d(3, 64, kernel_size=3, stride=2, bias=with_bias()),
        #         nn.ReLU(inplace=True),
        #         nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False),
        #         Fire(64, 16, 64, 64),
        #         Fire(128, 16, 64, 64),
        #         nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False),
        #         Fire(128, 32, 128, 128),
        #         Fire(256, 32, 128, 128),
        #         nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False),
        #         Fire(256, 48, 192, 192),
        #         Fire(384, 48, 192, 192),
        #         Fire(384, 64, 256, 256),
        #         Fire(512, 64, 256, 256)
        #     )
        else:
            # FIXME: Is this needed? SqueezeNet should only be called from the
            # FIXME: squeezenet1_x() functions
            # FIXME: This checking is not done for the other models
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1.0 or 1.1 expected".format(version=version))

        # Final convolution is initialized differently from the rest
        final_conv = BasicConv2d(512, self.num_classes, kernel_size=1, bias=with_bias())
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        for m in self.modules():
            if isinstance(m, BasicConv2d):
                if m is final_conv:
                    init.normal_(m.conv.weight, mean=0.0, std=0.01)
                else:
                    init.xavier_uniform_(m.conv.weight)
                if m.conv.bias is not None:
                    init.constant_(m.conv.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction=='mean' else loss.sum() if reduction=='sum' else loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon:float=0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
    
    def forward(self, outputs, labels):
        num_classes = outputs.size()[-1]
        log_outputs = F.log_softmax(outputs, dim=-1)
        labels = F.one_hot(labels, num_classes)
        labels = (1-self.epsilon)*labels + self.epsilon/num_classes
        loss = -log_outputs * labels
        loss = reduce_loss(loss.sum(dim=-1), self.reduction)
        return loss