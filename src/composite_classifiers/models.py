
"""
Model architectures adapted/copied from:

basic net -
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

small CNN -
https://www.kaggle.com/code/shadabhussain/cifar-10-cnn-using-pytorch

DLA net -
https://github.com/kuangliu/pytorch-cifar/blob/master/models/dla.py
Deep Layer Aggregation. https://arxiv.org/abs/1707.06484

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Basic_Net_512(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=(5,5), padding=(2,2))
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(5,5), padding=(2,2))
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 128 * 128, 1024)
        self.do1 = nn.Dropout(0.25)
        self.relu_fc1 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 32)
        self.do2 = nn.Dropout(0.25)
        self.relu_fc2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.fc1(x)
        x = self.do1(x)
        x = self.relu_fc1(x)
        x = self.fc2(x)
        x = self.do2(x)
        x = self.relu_fc2(x)
        x = self.fc3(x)
        return x


class Basic_Net(nn.Module):

    def __init__(self, input_size = [32,32]):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=(3,3), padding=(1,1))
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3,3), padding=(1,1))
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3,3), padding=(1,1))
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2, 2)

        self.num_out_features = 0
        if input_size[0] == 32 and input_size[1] == 32:
            self.num_out_features = 512
        elif input_size[0] == 540 and input_size[1] == 1024:
            self.num_out_features = 64 * 33 * 64
        elif input_size[0] == 428 and input_size[1] == 1024:
            self.num_out_features = 64 * 26 * 64

        self.fc1 = nn.Linear(self.num_out_features, 1024)
        self.do1 = nn.Dropout(0.25)
        self.relu_fc1 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 32)
        self.do2 = nn.Dropout(0.25)
        self.relu_fc2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool4(x)

        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.fc1(x)
        x = self.do1(x)
        x = self.relu_fc1(x)
        x = self.fc2(x)
        x = self.do2(x)
        x = self.relu_fc2(x)
        x = self.fc3(x)
        return x


class Small_CNN(nn.Module):

    def __init__(self, input_size = [32,32], do_bn = False):
        super().__init__()

        self.do_bn = do_bn

        self.conv1a = nn.Conv2d(3, 16, kernel_size=(3,3), padding=(1,1))
        self.relu1a = nn.ReLU()
        # self.conv1b = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1,1), padding=(1, 1))
        # self.relu1b = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2a = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=(1, 1))
        self.relu2a = nn.ReLU()
        # self.conv2b = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # self.relu2b = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3a = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
        self.relu3a = nn.ReLU()
        # self.conv3b = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # self.relu3b = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4a = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
        self.relu4a = nn.ReLU()
        # self.conv4b = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # self.relu4b = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2, 2)

        if self.do_bn == True:
            self.bn1 = nn.BatchNorm2d(16)
            self.bn2 = nn.BatchNorm2d(32)
            self.bn3 = nn.BatchNorm2d(64)
            self.bn4 = nn.BatchNorm2d(128)

        self.flatten = nn.Flatten()
        self.num_out_features = 0
        if input_size[0] == 32 and input_size[1] == 32:
            self.num_out_features = 1024
        elif input_size[0] == 540 and input_size[1] == 1024:
            self.num_out_features = 128 * 33 * 64
        elif input_size[0] == 428 and input_size[1] == 1024:
            self.num_out_features = 128 * 26 * 64

        self.fc1 = nn.Linear(self.num_out_features, 1024)
        self.do1 = nn.Dropout(0.25)
        self.relu_fc1 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 32)
        self.do2 = nn.Dropout(0.25)
        self.relu_fc2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.conv1a(x)
        if self.do_bn == True:
            x = self.bn1(x)
        x = self.relu1a(x)
        # x = self.conv1b(x)
        # x = self.relu1b(x)
        x = self.pool1(x)

        x = self.conv2a(x)
        if self.do_bn == True:
            x = self.bn2(x)
        x = self.relu2a(x)
        # x = self.conv2b(x)
        # x = self.relu2b(x)
        x = self.pool2(x)

        x = self.conv3a(x)
        if self.do_bn == True:
            x = self.bn3(x)
        x = self.relu3a(x)
        # x = self.conv3b(x)
        # x = self.relu3b(x)
        x = self.pool3(x)

        x = self.conv4a(x)
        if self.do_bn == True:
            x = self.bn4(x)
        x = self.relu4a(x)
        # x = self.conv4b(x)
        # x = self.relu4b(x)
        x = self.pool4(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.do1(x)
        x = self.relu_fc1(x)
        x = self.fc2(x)
        x = self.do2(x)
        x = self.relu_fc2(x)
        x = self.fc3(x)
        return x


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Root(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=1, padding=(kernel_size - 1) // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, xs):
        x = torch.cat(xs, 1)
        out = F.relu(self.bn(self.conv(x)))
        return out


class Tree(nn.Module):

    def __init__(self, block, in_channels, out_channels, level=1, stride=1):
        super(Tree, self).__init__()
        self.level = level
        if level == 1:
            self.root = Root(2*out_channels, out_channels)
            self.left_node = block(in_channels, out_channels, stride=stride)
            self.right_node = block(out_channels, out_channels, stride=1)
        else:
            self.root = Root((level+2)*out_channels, out_channels)
            for i in reversed(range(1, level)):
                subtree = Tree(block, in_channels, out_channels,
                               level=i, stride=stride)
                self.__setattr__('level_%d' % i, subtree)
            self.prev_root = block(in_channels, out_channels, stride=stride)
            self.left_node = block(out_channels, out_channels, stride=1)
            self.right_node = block(out_channels, out_channels, stride=1)

    def forward(self, x):
        xs = [self.prev_root(x)] if self.level > 1 else []
        for i in reversed(range(1, self.level)):
            level_i = self.__getattr__('level_%d' % i)
            x = level_i(x)
            xs.append(x)
        x = self.left_node(x)
        xs.append(x)
        x = self.right_node(x)
        xs.append(x)
        out = self.root(xs)
        return out


class DLA(nn.Module):

    def __init__(self, input_size=[32,32], block=BasicBlock, num_classes=2):

        super(DLA, self).__init__()
        self.base = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )

        self.layer1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )

        self.layer3 = Tree(block,  32,  64, level=1, stride=1)
        self.layer4 = Tree(block,  64, 128, level=2, stride=2)
        self.layer5 = Tree(block, 128, 256, level=2, stride=2)
        self.layer6 = Tree(block, 256, 512, level=1, stride=2)

        self.num_out_features = 0
        if input_size[0]==32 and input_size[1]==32:
            self.num_out_features = 512
        elif input_size[0]==256 and input_size[1]==256:
            self.num_out_features = 32768
        elif input_size[0] == 384 and input_size[1] == 384:
            self.num_out_features = 73728
        elif input_size[0] == 512 and input_size[1] == 512:
            self.num_out_features = 131072
        elif input_size[0]==270 and input_size[1]==512:
            self.num_out_features = 65536
        elif input_size[0]==214 and input_size[1]==512:
            self.num_out_features = 49152
        elif input_size[0] == 428 and input_size[1] == 1024:
            self.num_out_features = 212992
        self.do1 = nn.Dropout(0.25)

        # self.linear = nn.Linear(512, num_classes)
        # self.linear = nn.Linear(self.num_out_features, num_classes)
        self.linear1 = nn.Linear(self.num_out_features, 32)
        self.do2 = nn.Dropout(0.25)
        self.linear2 = nn.Linear(32, num_classes)


    def forward(self, x):
        out = self.base(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        # out = self.linear(out)
        out = self.do1(out)
        out = self.linear1(out)
        out = self.do2(out)
        out = self.linear2(out)
        return out



def get_net(cfg):
    model = cfg['model']['arch']
    do_resize = cfg['data']['do_resize']
    do_bn = cfg['model'].get('do_bn', False)
    net = None
    if model == 'basic_512':
        net = Basic_Net_512()
    elif model == 'basic':
        net = Basic_Net(input_size=do_resize)
    elif model == 'small_cnn':
        net = Small_CNN(input_size=do_resize, do_bn=do_bn)
    elif model == 'dla':
        net = DLA(input_size=do_resize)
    return net