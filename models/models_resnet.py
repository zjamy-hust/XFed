'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, input_channel):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(input_channel, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(num_classes, input_channel):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, input_channel)

class ResNet18_with_mask_head(nn.Module):
    def __init__(self, block, num_blocks, num_classes, input_channel, hard_mask=False):
        """_summary_

        Args:
            block (_type_): _description_
            num_blocks (_type_): _description_
            num_classes (_type_): _description_
            input_channel (_type_): _description_
            output_mask (bool, optional): 是否采用linear 2产生mask输出。默认为False
            hard_mask (bool, optional): 是否产生hard mask。默认为False
        """
        super(ResNet18_with_mask_head, self).__init__()
        self.in_planes = 64

        self.backbone_1 = nn.Sequential(
            nn.Conv2d(input_channel, 64, kernel_size=3,stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            self._make_layer(block, 64, num_blocks[0], stride=1),
            self._make_layer(block, 128, num_blocks[1], stride=2),
            self._make_layer(block, 256, num_blocks[2], stride=2),
            self._make_layer(block, 512, num_blocks[3], stride=2),
        )
        self.backbone_2 = copy.deepcopy(self.backbone_1)
        self.backbone_1 = self.backbone_1.requires_grad_(False) #取消梯度
        
        self.linear = nn.Linear(512*block.expansion, num_classes)
        
        self.hard_mask = hard_mask
        self.linear2 = nn.ConvTranspose2d(512*block.expansion, 2, stride=7, kernel_size=7,output_padding=0)
        

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def get_masks(self, x):
        out = self.backbone_1(x)
        
        mask = self.linear2(out.detach()).permute(0,2,3,1).unsqueeze(1)   #梯度不需要继续向前一个backbone传播
        if self.hard_mask == True:
            mask = F.gumbel_softmax(mask, tau=1, dim=-1, hard=True)
        else:
            mask = F.softmax(mask, dim=-1)
        
        return mask
        
    def forward(self, x):
        x_out_ = self.backbone_2(x)    
        
        #进行预测
        pred = F.avg_pool2d(x_out_, 4)
        pred = pred.view(pred.size(0), -1)
        pred = self.linear(pred)
        return pred
    
def ResNet18_with_mask(num_classes, input_channel,hard_mask):
    return ResNet18_with_mask_head(BasicBlock, [2, 2, 2, 2], num_classes, input_channel, hard_mask=hard_mask)

def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()
