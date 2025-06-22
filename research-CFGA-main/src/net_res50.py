#define network
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from cfga_core.bwconncomp import largestConnectComponent
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import numpy as np




def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out





class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.class_fc = nn.Linear(512 * block.expansion, 100,bias=False)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)
    def extract_conv_feature(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        out=[]
        # print(x.shape)
        x = self.extract_conv_feature(x)
        # print(x.shape)
        x_out=x.data+0
        out.append(x_out)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.class_fc(x)

        return out,x

    def forward(self, x):

        return self._forward_impl(x)




class ResNet_for_cam(nn.Module):

    def __init__(self, block, layers, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet_for_cam, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.class_fc = nn.Linear(512 * block.expansion, 100,bias=False)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)
    def extract_conv_feature(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def save_gradient(self, grad):
        self.gradients.append(grad)


    def _forward_impl(self, x):
        # See note [TorchScript super()]
        out=[]
        self.gradients = []
        # print(x.shape)
        x = self.extract_conv_feature(x)
        x.register_hook(self.save_gradient)
        # print(x.shape)
        x_out=x.data+0
        out.append(x_out)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.class_fc(x)

        return out,x

    def forward(self, x):

        return self._forward_impl(x)









class ResNet_circle_loss(nn.Module):

    def __init__(self, block, layers,mode=None,  zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet_circle_loss, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.class_fc = nn.Linear(512 * block.expansion, 512)

        self.class_fc_2 = nn.Parameter(torch.FloatTensor(100,512))
        # self.class_fc_2 =nn.Linear(512,100,bias=False)
        nn.init.xavier_uniform_(self.class_fc_2)
        self.mode=mode

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)
    def extract_conv_feature(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x



    def _forward_imp2(self, x):
        # See note [TorchScript super()]
        out=[]
        x = self.extract_conv_feature(x)
        out.append(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.class_fc(x)
        x = F.normalize(x, p=2, dim=1)
        similarity_matrix = nn.functional.linear(x, nn.functional.normalize(self.class_fc_2,p=2, dim=1, eps=1e-12))
        # self.class_fc_2.weight.data=nn.functional.normalize(self.class_fc_2.weight,p=2, dim=1, eps=1e-12)
        # similarity_matrix = self.class_fc_2(x)

        return out, x ,similarity_matrix,self.class_fc_2 # [batchsize,200]


    def forward(self, x):
        if self.mode=='circle_loss':
            return self._forward_imp2(x)







class ResNet_DGCRL(nn.Module):

    def __init__(self, block, layers,mode=None,  zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None,scale=128,num_trained_cls = 98):
        super(ResNet_DGCRL, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.class_fc = nn.Linear(512 * block.expansion, num_trained_cls,bias=False)
        self.class_fc_2 = nn.Linear(512 * block.expansion * 2, num_trained_cls,bias=False)
        self._scale=scale
        self.mode=mode

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)
    def extract_conv_feature(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x



    def _forward_imp2(self, x):
        # See note [TorchScript super()]
        out=[]
        x = self.extract_conv_feature(x)
        out.append(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = F.normalize(x, p=2, dim=1)
        x = x * self._scale
        x = self.class_fc(x)

        return out, x, self.class_fc.weight  # [batchsize,200]

    def _forward_imp3(self, x):
        # See note [TorchScript super()]
        out = []
        x = self.extract_conv_feature(x)
        out.append(x)
        avg_x = self.global_avg_pool(x)
        avg_x = avg_x.view(avg_x.size(0), -1)
        avg_x = F.normalize(avg_x, p=2, dim=1)
        max_x = self.global_max_pool(x)
        max_x = max_x.view(max_x.size(0), -1)
        max_x = F.normalize(max_x, p=2, dim=1)
        x = torch.cat((avg_x, max_x), dim=1)
        x = x * self._scale
        x = self.class_fc_2(x)

        return out, x, self.class_fc_2.weight  # [batchsize,200]


    def _forward_imp3_1(self, x):
        # See note [TorchScript super()]
        out = []
        x = self.extract_conv_feature(x)
        out.append(x)
        avg_x = self.global_avg_pool(x)
        avg_x = avg_x.view(avg_x.size(0), -1)
        # avg_x = F.normalize(avg_x, p=2, dim=1)
        max_x = self.global_max_pool(x)
        max_x = max_x.view(max_x.size(0), -1)
        # max_x = F.normalize(max_x, p=2, dim=1)
        x = torch.cat((avg_x, max_x), dim=1)
        x = F.normalize(x, p=2, dim=1)
        x = x * self._scale
        ##################################################
        #有问题就注释掉,这是对齐mr-zeng的实现的修改的第一处
        # self.class_fc_2.weight.data=nn.functional.normalize(self.class_fc_2.weight,p=2, dim=1, eps=1e-12)
        # print(self.class_fc_2.weight.size())
        ##################################################
        x = self.class_fc_2(x)

        return out, x, self.class_fc_2.weight  # [batchsize,200]

    def _forward_imp3_2(self, x):
        # See note [TorchScript super()]
        out = []
        x = self.extract_conv_feature(x)
        out.append(x)
        max_x = self.global_max_pool(x)
        max_x = max_x.view(max_x.size(0), -1)
        # max_x = F.normalize(max_x, p=2, dim=1)
        x = F.normalize(max_x, p=2, dim=1)
        x = x * self._scale
        x = self.class_fc(x)

        return out, x, self.class_fc.weight  # [batchsize,200]



    def forward(self, x):
        if self.mode=='DGCRL_avg':
            return self._forward_imp2(x)
        elif self.mode=='DGCRL_max_avg':
            return self._forward_imp3(x)
        elif self.mode=='DGCRL_max':
            return self._forward_imp3_2(x)
        elif self.mode == 'DGPCRL':
            return self._forward_imp3_1(x)


class ResNet_DGCRL_for_cam(nn.Module):

    def __init__(self, block, layers,mode=None,  zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None,scale=128):
        super(ResNet_DGCRL_for_cam, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.class_fc = nn.Linear(512 * block.expansion, 100,bias=False)
        self.class_fc_2 = nn.Linear(512 * block.expansion * 2, 100,bias=False)
        self._scale=scale
        self.mode=mode

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)
    def extract_conv_feature(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def save_gradient(self, grad):
        self.gradients.append(grad)



    def _forward_imp2(self, x):
        # See note [TorchScript super()]
        out=[]
        self.gradients = []
        x = self.extract_conv_feature(x)
        x.register_hook(self.save_gradient)
        out.append(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = F.normalize(x, p=2, dim=1)
        x = x * self._scale
        x = self.class_fc(x)

        return out, x, self.class_fc.weight  # [batchsize,200]

    def _forward_imp3(self, x):
        # See note [TorchScript super()]
        out = []
        self.gradients = []
        x = self.extract_conv_feature(x)
        x.register_hook(self.save_gradient)
        out.append(x)
        avg_x = self.global_avg_pool(x)
        avg_x = avg_x.view(avg_x.size(0), -1)
        avg_x = F.normalize(avg_x, p=2, dim=1)
        max_x = self.global_max_pool(x)
        max_x = max_x.view(max_x.size(0), -1)
        max_x = F.normalize(max_x, p=2, dim=1)
        x = torch.cat((avg_x, max_x), dim=1)
        x = x * self._scale
        x = self.class_fc_2(x)

        return out, x, self.class_fc_2.weight  # [batchsize,200]

    def _forward_imp3_1(self, x):
        # See note [TorchScript super()]
        out = []
        self.gradients = []
        x = self.extract_conv_feature(x)
        x.register_hook(self.save_gradient)
        out.append(x)
        avg_x = self.global_avg_pool(x)
        avg_x = avg_x.view(avg_x.size(0), -1)
        # avg_x = F.normalize(avg_x, p=2, dim=1)
        max_x = self.global_max_pool(x)
        max_x = max_x.view(max_x.size(0), -1)
        # max_x = F.normalize(max_x, p=2, dim=1)
        x = torch.cat((avg_x, max_x), dim=1)
        x = F.normalize(x, p=2, dim=1)
        x = x * self._scale
        x = self.class_fc_2(x)

        return out, x, self.class_fc_2.weight  # [batchsize,200]

    def _forward_imp3_2(self, x):
        # See note [TorchScript super()]
        out = []
        self.gradients = []
        x = self.extract_conv_feature(x)
        x.register_hook(self.save_gradient)
        out.append(x)
        max_x = self.global_max_pool(x)
        max_x = max_x.view(max_x.size(0), -1)
        # max_x = F.normalize(max_x, p=2, dim=1)
        x = F.normalize(max_x, p=2, dim=1)
        x = x * self._scale
        x = self.class_fc(x)

        return out, x, self.class_fc_2.weight  # [batchsize,200]


    def forward(self, x):
        if self.mode=='DGCRL_avg':
            return self._forward_imp2(x)
        elif self.mode=='DGCRL_max_avg':
            return self._forward_imp3(x)
        elif self.mode=='DGCRL_max':
            return self._forward_imp3_2(x)
        elif self.mode == 'DGPCRL':
            return self._forward_imp3_1(x)






class ResNet_DGCRL_scda(nn.Module):

    def __init__(self, block, layers,mode=None,  zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None,scale=128):
        super(ResNet_DGCRL_scda, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.class_fc = nn.Linear(512 * block.expansion, 100,bias=False)
        self.class_fc_2 = nn.Linear(512 * block.expansion * 2, 100,bias=False)
        self._scale=scale
        self.mode=mode

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)
    def extract_conv_feature(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x



    def _forward_imp2(self, x):
        # See note [TorchScript super()]
        out=[]
        x = self.extract_conv_feature(x)
        batch_size, c_L31, h_L31, w_L31 = x.data.size()
        feature_maps_L31_sum = torch.sum(x.data,
                                         1)  # 如果要将CAM或者gard_cam引入SCDA，需要更改的程序也就仅仅是这一步
        L31_sum_mean = torch.mean(feature_maps_L31_sum, dim=[1, 2])  # 返回的是scalar tensor ,这是与matlab的mean不同的地方
        # print(feature_maps_L31_sum.shape)
        # print(L31_sum_mean.shape)
        # print(feature_maps_L31_sum>L31_sum_mean)
        L31_sum_mean = L31_sum_mean.view([L31_sum_mean.size()[0], 1, 1])
        jj = (feature_maps_L31_sum > L31_sum_mean).cpu().data.numpy()
        highlight = torch.from_numpy(jj + 0).cuda().float()
        highlight_conn_L31 = highlight
        # highlight_conn_L31 = torch.tensor(largestConnectComponent(highlight.cpu().numpy()) + 0)#[7*7],我们需要将其变成[512,7,7]
        feature_maps_L31 = x.data.cpu().numpy()  # [1,512,7,7]==>[512,7,7]
        highlight_conn_L31 = highlight_conn_L31.cpu().data.numpy()
        highlight_conn_L31 = highlight_conn_L31.reshape(batch_size, 1, h_L31, w_L31) * np.ones_like(
            feature_maps_L31)  # [7,7]=>[1,7,7]=>[512,7,7]
        x = x * torch.from_numpy(highlight_conn_L31).cuda()
        # out.append(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = F.normalize(x, p=2, dim=1)
        out.append(x)
        x = x * self._scale
        x = self.class_fc(x)

        return out, x, self.class_fc.weight  # [batchsize,200]

    def _forward_imp3(self, x):
        # See note [TorchScript super()]
        out = []
        x = self.extract_conv_feature(x)
        batch_size, c_L31, h_L31, w_L31 = x.data.size()
        feature_maps_L31_sum = torch.sum(x.data,
                                         1)  # 如果要将CAM或者gard_cam引入SCDA，需要更改的程序也就仅仅是这一步
        L31_sum_mean = torch.mean(feature_maps_L31_sum, dim=[1, 2])  # 返回的是scalar tensor ,这是与matlab的mean不同的地方
        # print(feature_maps_L31_sum.shape)
        # print(L31_sum_mean.shape)
        # print(feature_maps_L31_sum>L31_sum_mean)
        L31_sum_mean = L31_sum_mean.view([L31_sum_mean.size()[0], 1, 1])
        jj = (feature_maps_L31_sum > L31_sum_mean).cpu().data.numpy()
        highlight = torch.from_numpy(jj + 0).cuda().float()
        highlight_conn_L31 = highlight
        # highlight_conn_L31 = torch.tensor(largestConnectComponent(highlight.cpu().numpy()) + 0)#[7*7],我们需要将其变成[512,7,7]
        feature_maps_L31 = x.data.cpu().numpy()  # [1,512,7,7]==>[512,7,7]
        highlight_conn_L31 = highlight_conn_L31.cpu().data.numpy()
        highlight_conn_L31 = highlight_conn_L31.reshape(batch_size, 1, h_L31, w_L31) * np.ones_like(
            feature_maps_L31)  # [7,7]=>[1,7,7]=>[512,7,7]
        x = x * torch.from_numpy(highlight_conn_L31).cuda()
        # out.append(x)
        avg_x = self.global_avg_pool(x)
        avg_x = avg_x.view(avg_x.size(0), -1)
        avg_x = F.normalize(avg_x, p=2, dim=1)
        max_x = self.global_max_pool(x)
        max_x = max_x.view(max_x.size(0), -1)
        max_x = F.normalize(max_x, p=2, dim=1)
        x = torch.cat((avg_x, max_x), dim=1)
        out.append(x)
        x = x * self._scale
        x = self.class_fc_2(x)

        return out, x, self.class_fc_2.weight  # [batchsize,200]

    def _forward_imp3_1(self, x):
        # See note [TorchScript super()]
        out = []
        x = self.extract_conv_feature(x)
        #
        batch_size, c_L31, h_L31, w_L31 = x.data.size()
        feature_maps_L31_sum = torch.sum(x.data,
                                         1)  # 如果要将CAM或者gard_cam引入SCDA，需要更改的程序也就仅仅是这一步
        L31_sum_mean = torch.mean(feature_maps_L31_sum, dim=[1, 2])  # 返回的是scalar tensor ,这是与matlab的mean不同的地方
        # print(feature_maps_L31_sum.shape)
        # print(L31_sum_mean.shape)
        # print(feature_maps_L31_sum>L31_sum_mean)
        L31_sum_mean = L31_sum_mean.view([L31_sum_mean.size()[0], 1, 1])
        jj = (feature_maps_L31_sum > L31_sum_mean).cpu().data.numpy()
        highlight = torch.from_numpy(jj + 0).cuda().float()
        # highlight_conn_L31 = highlight
        highlight_conn_L31 = torch.tensor(largestConnectComponent(highlight.cpu().numpy()) + 0)#[7*7],我们需要将其变成[512,7,7]
        feature_maps_L31 = x.data.cpu().numpy()  # [1,512,7,7]==>[512,7,7]
        highlight_conn_L31 = highlight_conn_L31.cpu().data.numpy()
        highlight_conn_L31 = highlight_conn_L31.reshape(batch_size, 1, h_L31, w_L31) * np.ones_like(
            feature_maps_L31)  # [7,7]=>[1,7,7]=>[512,7,7]
        x = x * torch.from_numpy(highlight_conn_L31).cuda().float()
        # # out.append(x)
        #
        avg_x = self.global_avg_pool(x)
        avg_x = avg_x.view(avg_x.size(0), -1)

        # avg_x = F.normalize(avg_x, p=2, dim=1)
        max_x = self.global_max_pool(x)
        max_x = max_x.view(max_x.size(0), -1)

        # max_x = F.normalize(max_x, p=2, dim=1)
        x = torch.cat((avg_x, max_x), dim=1)
        out.append(x)
        x = F.normalize(x, p=2, dim=1)
        x = x * self._scale
        x = self.class_fc_2(x)

        return out, x, self.class_fc_2.weight  # [batchsize,200]

    def _forward_imp3_2(self, x):
        # See note [TorchScript super()]
        out = []
        x = self.extract_conv_feature(x)
        batch_size, c_L31, h_L31, w_L31 = x.data.size()
        feature_maps_L31_sum = torch.sum(x.data,
                                         1)  # 如果要将CAM或者gard_cam引入SCDA，需要更改的程序也就仅仅是这一步
        L31_sum_mean = torch.mean(feature_maps_L31_sum, dim=[1, 2])  # 返回的是scalar tensor ,这是与matlab的mean不同的地方
        # print(feature_maps_L31_sum.shape)
        # print(L31_sum_mean.shape)
        # print(feature_maps_L31_sum>L31_sum_mean)
        L31_sum_mean = L31_sum_mean.view([L31_sum_mean.size()[0], 1, 1])
        jj = (feature_maps_L31_sum > L31_sum_mean).cpu().data.numpy()
        highlight = torch.from_numpy(jj + 0).cuda().float()
        highlight_conn_L31 = highlight
        # highlight_conn_L31 = torch.tensor(largestConnectComponent(highlight.cpu().numpy()) + 0)#[7*7],我们需要将其变成[512,7,7]
        feature_maps_L31 = x.data.cpu().numpy()  # [1,512,7,7]==>[512,7,7]
        highlight_conn_L31 = highlight_conn_L31.cpu().data.numpy()
        highlight_conn_L31 = highlight_conn_L31.reshape(batch_size, 1, h_L31, w_L31) * np.ones_like(
            feature_maps_L31)  # [7,7]=>[1,7,7]=>[512,7,7]
        x = x * torch.from_numpy(highlight_conn_L31).cuda()
        # out.append(x)
        max_x = self.global_max_pool(x)
        max_x = max_x.view(max_x.size(0), -1)
        # max_x = F.normalize(max_x, p=2, dim=1)
        x = F.normalize(max_x, p=2, dim=1)
        out.append(x)
        x = x * self._scale
        x = self.class_fc(x)

        return out, x, self.class_fc_2.weight  # [batchsize,200]


    def forward(self, x):
        if self.mode=='DGCRL_avg':
            return self._forward_imp2(x)
        elif self.mode=='DGCRL_max_avg':
            return self._forward_imp3(x)
        elif self.mode=='DGCRL_max':
            return self._forward_imp3_2(x)
        elif self.mode == 'DGPCRL':
            return self._forward_imp3_1(x)













class ResNet_AMsoftmax_for_cam(nn.Module):

    def __init__(self, block, layers,mode=None,  zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet_AMsoftmax_for_cam, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.class_fc = nn.Linear(512 * block.expansion, 100)
        self.class_fc_2 = nn.Linear(512 * block.expansion, 512)
        self.class_fc_3 = nn.Linear(512, 100)
        self.mode=mode

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)
    def extract_conv_feature(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def _forward_imp4(self, x):
        # See note [TorchScript super()]
        out=[]
        self.gradients = []
        x = self.extract_conv_feature(x)
        x.register_hook(self.save_gradient)
        out.append(x)
        img_f = self.avgpool(x)
        img_f = img_f.view(img_f.shape[0], -1)  # [batchsize,512,1,1]====>(batch_size,512)
        img_f = self.class_fc_2(img_f)
        img_f = F.normalize(img_f, p=2, dim=1)
        w_norm = torch.norm(self.class_fc_3.weight.data, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        self.class_fc_3.weight.data = torch.div(self.class_fc_3.weight.data, w_norm)
        score = self.class_fc_3(img_f)
        return out, img_f, score, self.class_fc_3.weight  # [batchsize,200]



    def forward(self, x):
        if self.mode=='AMsoftmax_embedding':
            return self._forward_imp4(x)









class ResNet_AMsoftmax(nn.Module):

    def __init__(self, block, layers,mode=None,  zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet_AMsoftmax, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))

        self.class_fc_3 = nn.Linear(512, 100,bias=False)
        self.mode=mode

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)
    def extract_conv_feature(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x



    def _forward_imp4(self, x):
        # See note [TorchScript super()]
        out=[]
        x = self.extract_conv_feature(x)
        out.append(x)
        img_f = self.avgpool(x)
        img_f = img_f.view(img_f.shape[0], -1)  # [batchsize,512,1,1]====>(batch_size,512)
        img_f = self.class_fc_2(img_f)
        img_f = F.normalize(img_f, p=2, dim=1)
        w_norm = torch.norm(self.class_fc_3.weight.data, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        self.class_fc_3.weight.data = torch.div(self.class_fc_3.weight.data, w_norm)
        score = self.class_fc_3(img_f)
        return out, img_f, score, self.class_fc_3.weight  # [batchsize,200]



    def forward(self, x):
        if self.mode=='AMsoftmax_embedding':
            return self._forward_imp4(x)





























def _resnet( block, layers, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model


def _resnet_DGCRL( block, layers,mode, **kwargs):
    model = ResNet_DGCRL(block, layers,mode, **kwargs)
    return model


def _resnet_circle_loss( block, layers,mode, **kwargs):
    model = ResNet_circle_loss(block, layers,mode, **kwargs)
    return model



def _resnet_AMsoftmax( block, layers,mode, **kwargs):
    model = ResNet_AMsoftmax(block, layers,mode, **kwargs)
    return model




def _resnet_for_cam( block, layers, **kwargs):
    model = ResNet_for_cam(block, layers, **kwargs)
    return model


def _resnet_DGCRL_for_cam( block, layers,mode, **kwargs):
    model = ResNet_DGCRL_for_cam(block, layers,mode, **kwargs)
    return model


def _resnet_AMsoftmax_for_cam( block, layers,mode, **kwargs):
    model = ResNet_AMsoftmax_for_cam(block, layers,mode, **kwargs)
    return model



def _resnet_DGCRL_scda( block, layers,mode, **kwargs):
    model = ResNet_DGCRL_scda(block, layers,mode, **kwargs)
    return model




def FGIAnet100(**kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(Bottleneck, [3, 4, 6, 3],
                   **kwargs)



def FGIAnet100_metric5(**kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_DGCRL(Bottleneck, [3, 4, 6, 3],'DGCRL_avg',
                   **kwargs)



def FGIAnet100_metric5_1(**kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_DGCRL(Bottleneck, [3, 4, 6, 3],'DGCRL_max_avg',
                   **kwargs)




def FGIAnet100_metric5_2(**kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_DGCRL(Bottleneck, [3, 4, 6, 3],'DGPCRL',
                   **kwargs)



def FGIAnet100_metric5_circle(**kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_circle_loss(Bottleneck, [3, 4, 6, 3],'circle_loss',
                   **kwargs)




def FGIAnet100_metric5_3(**kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_DGCRL(Bottleneck, [3, 4, 6, 3],'DGCRL_max',
                   **kwargs)






def FGIAnet100_metric6_1(**kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_AMsoftmax(Bottleneck, [3, 4, 6, 3],'AMsoftmax_embedding',
                   **kwargs)





def FGIAnet100_for_cam(**kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_for_cam(Bottleneck, [3, 4, 6, 3],
                   **kwargs)



def FGIAnet100_metric5_for_cam(**kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_DGCRL_for_cam(Bottleneck, [3, 4, 6, 3],'DGCRL_avg',
                   **kwargs)



def FGIAnet100_metric5_1_for_cam(**kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_DGCRL_for_cam(Bottleneck, [3, 4, 6, 3],'DGCRL_max_avg',
                   **kwargs)





def FGIAnet100_metric5_2_for_cam(**kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_DGCRL_for_cam(Bottleneck, [3, 4, 6, 3],'DGPCRL',
                   **kwargs)







def FGIAnet100_metric5_3_for_cam(**kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_DGCRL_for_cam(Bottleneck, [3, 4, 6, 3],'DGCRL_max',
                   **kwargs)








def FGIAnet100_metric6_1_for_cam(**kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_AMsoftmax_for_cam(Bottleneck, [3, 4, 6, 3],'AMsoftmax_embedding',
                   **kwargs)







def FGIAnet100_metric5_scda(**kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_DGCRL_scda(Bottleneck, [3, 4, 6, 3],'DGCRL_avg',
                   **kwargs)



def FGIAnet100_metric5_1_scda(**kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_DGCRL_scda(Bottleneck, [3, 4, 6, 3],'DGCRL_max_avg',
                   **kwargs)




def FGIAnet100_metric5_2_scda(**kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_DGCRL_scda(Bottleneck, [3, 4, 6, 3],'DGPCRL',
                   **kwargs)







def FGIAnet100_metric5_3_scda(**kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_DGCRL_scda(Bottleneck, [3, 4, 6, 3],'DGCRL_max',
                   **kwargs)






