# from net import FGIAnet
# from net_grad_cam import FGIAnet_GARD_CAM
import dataloader
from SCDA_cars_resnet50.bwconncomp import largestConnectComponent

import argparse
from os.path import join
import uuid
import time
import json
import pickle

import torch
from torch.nn.functional import interpolate
import torch.nn as nn
from torchvision import transforms
import numpy as np
from torch.nn import functional as F
import cv2



global_avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
global_max_pool = torch.nn.AdaptiveMaxPool2d((1, 1))



class PoolCam_yuhan_kernel_embedding():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, ii=None,flip=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别

        feature_maps,output= self.net(input.cuda())


        # print('yuhan')
        feature_maps_L31 = feature_maps[
            0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        batch_size, c_L31, h_L31, w_L31 = feature_maps_L31.size()

        # Pool5   #feature_maps_L31  [512,7,7]==[512,]
        feature_maps_L31_sum = torch.sum(feature_maps_L31[0],
                                         0)  # 如果要将CAM或者gard_cam引入SCDA，需要更改的程序也就仅仅是这一步
        L31_sum_mean = torch.mean(feature_maps_L31_sum)  # 返回的是scalar tensor ,这是与matlab的mean不同的地方

        # highlight_index = torch.nonzero(feature_maps_L31_sum >= L31_sum_mean)  # 一个K*2的tensor,每一行一个符合要求的点
        # a, b = highlight_index.size()
        # print('yu',a)
        # largestConnectComponent将最大连通区域所对应的像素点置为true
        jj = (feature_maps_L31_sum > L31_sum_mean).cpu().data.numpy()
        highlight = torch.from_numpy(jj + 0).cuda().float()
        # print(highlight)

        highlight_conn_L31 = highlight
        # print('yu',torch.sum(highlight_conn_L31))
        # highlight_conn_L31 = torch.tensor(largestConnectComponent(highlight.cpu().numpy()) + 0)#[7*7],我们需要将其变成[512,7,7]
        # print(highlight_conn_L31)
        a=torch.sum(highlight_conn_L31).item()

        feature_maps_L31 = feature_maps_L31[0].cpu().data.numpy()  # [1,512,7,7]==>[512,7,7]
        highlight_conn_L31 = highlight_conn_L31.cpu().data.numpy()
        # print(highlight_conn_L31.shape)
        highlight_conn_L31 = highlight_conn_L31.reshape(1, h_L31, w_L31) * np.ones_like(
            feature_maps_L31)  # [7,7]=>[1,7,7]=>[512,7,7]
        # print(highlight_conn_L31.shape)





        feature_maps_L31_mean=output[0].cpu().data.numpy()

        if np.linalg.norm(feature_maps_L31_mean) > 0:
            feature_maps_L31_mean_norm = feature_maps_L31_mean / np.linalg.norm(feature_maps_L31_mean)
        else:
            print("出现了")
            feature_maps_L31_mean_norm = np.zeros_like(feature_maps_L31_mean)
            pass



        output_mean=[]
        output_3 = []
        output_3.append(feature_maps_L31_mean_norm)


        feature_maps_L31 = feature_maps[
            0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        # feature_maps_L31 = feature_maps_L31[0]  # [1,512,7,7]==>[512,7,7]
        # print(feature_maps_L31.size())
        # print(torch.from_numpy(highlight_conn_L31).unsqueeze(0).cuda().float().size())
        feature_maps_L31 = feature_maps_L31 * torch.from_numpy(highlight_conn_L31).unsqueeze(0).cuda().float()
        feature_maps_L31_max = global_avg_pool(feature_maps_L31)
        coefficient=(h_L31*w_L31)/a
        feature_maps_L31_max=coefficient * feature_maps_L31_max
        feature_maps_L31_mean = global_max_pool(feature_maps_L31)
        feature_maps_L31_mean_max = torch.cat((feature_maps_L31_mean, feature_maps_L31_max), dim=1)

        loss2 = torch.sum(feature_maps_L31_mean_max)
        self.net.zero_grad()
        loss2.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚

        # output_max=[]
        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['inception_5b_3x3.weight','inception_5b_double_3x3_2.weight']:
                # print(name)
                tmp = (param.grad).data
                # print(tmp.size())
                # tmp=torch.abs(tmp)
                # print(tmp)
                # tmp_mean_reuse=tmp.view(tmp.size()[0],-1)
                # print(tmp_mean_reuse.shape)
                tmp_mean_reuse_std = torch.std(tmp, dim=(2, 3))

                tmp_mean_reuse = F.adaptive_max_pool2d(tmp, output_size=1).squeeze(-1).squeeze(
                    -1) + F.adaptive_max_pool2d(-1 * tmp, output_size=1).squeeze(-1).squeeze(-1)
                tmp_mean_reuse = tmp_mean_reuse + tmp_mean_reuse_std
                # print(tmp_mean_reuse.size())

                # tmp_mean_reuse=F.adaptive_max_pool2d(tmp,output_size=1).squeeze(-1).squeeze(-1)
                # tmp_mean_reuse=F.adaptive_max_pool2d(-1 * tmp,output_size=1).squeeze(-1).squeeze(-1)

                # tmp_mean_reuse_1 = torch.mean(tmp_1, axis=(2, 3))
                # tmp_mean_reuse = tmp_mean_reuse * tmp_mean_reuse_1
                # tmp_mean_reuse = torch.mean(tmp, dim=(2, 3))
                ############################################################
                #$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%%$%$%$$%$$%$%$%$%$%$%$
                #待会回复注释
                tmp_mean_reuse_bool = torch.mean(tmp_mean_reuse, dim=0).unsqueeze(0)
                tmp_mean_reuse_bool = tmp_mean_reuse > tmp_mean_reuse_bool
                tmp_mean_reuse_bool = tmp_mean_reuse_bool.float()

                tmp_mean_reuse_bool_2 = torch.mean(tmp_mean_reuse, dim=1).unsqueeze(0).T
                tmp_mean_reuse_bool_2 = tmp_mean_reuse > tmp_mean_reuse_bool_2
                tmp_mean_reuse_bool_2 = tmp_mean_reuse_bool_2.float()

                tmp_mean_reuse = tmp_mean_reuse * tmp_mean_reuse_bool * tmp_mean_reuse_bool_2
###############################################################################################
                #$$$$$$$$$$$$$$$$$$$$$$%%%%%%%%%%%%%%%%%%%%%%^^^^^^^^^^^^^^^^^^^$$$$$$$$$$$$$$$$$$$
                tmp_mean = torch.mean(tmp_mean_reuse, dim=0).cpu().data.numpy()
                # tmp_mean,_ = torch.max(tmp_mean_reuse, dim=0)
                # tmp_mean=tmp_mean.numpy()
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean.append(tmp_mean)
            elif name in ['inception_5b_1x1.weight','inception_5b_pool_proj.weight']:
                # print(name)
                tmp = (param.grad).data
                # print(tmp.size())

                # print(tmp)
                tmp=torch.abs(tmp)
                # print(tmp)
                tmp_mean_reuse=tmp.view(tmp.size()[0],-1)
                # print(tmp_mean_reuse.shape)
                # tmp_mean_reuse_std = torch.std(tmp, dim=(2, 3))
                #
                # tmp_mean_reuse = F.adaptive_max_pool2d(tmp, output_size=1).squeeze(-1).squeeze(
                #     -1) + F.adaptive_max_pool2d(-1 * tmp, output_size=1).squeeze(-1).squeeze(-1)
                # tmp_mean_reuse = tmp_mean_reuse + tmp_mean_reuse_std
                # tmp_mean_reuse=F.adaptive_max_pool2d(tmp,output_size=1).squeeze(-1).squeeze(-1)
                # tmp_mean_reuse=F.adaptive_max_pool2d(-1 * tmp,output_size=1).squeeze(-1).squeeze(-1)

                # tmp_mean_reuse_1 = torch.mean(tmp_1, axis=(2, 3))
                # tmp_mean_reuse = tmp_mean_reuse * tmp_mean_reuse_1
                # tmp_mean_reuse = torch.mean(tmp, dim=(2, 3))
                #待会恢复
                ########################################
                # print(tmp_mean_reuse.size())

                tmp_mean_reuse_bool = torch.mean(tmp_mean_reuse, dim=0).unsqueeze(0)
                tmp_mean_reuse_bool = tmp_mean_reuse > tmp_mean_reuse_bool
                tmp_mean_reuse_bool = tmp_mean_reuse_bool.float()

                tmp_mean_reuse_bool_2 = torch.mean(tmp_mean_reuse, dim=1).unsqueeze(0).T
                tmp_mean_reuse_bool_2 = tmp_mean_reuse > tmp_mean_reuse_bool_2
                tmp_mean_reuse_bool_2 = tmp_mean_reuse_bool_2.float()

                tmp_mean_reuse = tmp_mean_reuse * tmp_mean_reuse_bool * tmp_mean_reuse_bool_2
######################################################################
                tmp_mean = torch.mean(tmp_mean_reuse, dim=0).cpu().data.numpy()
                # tmp_mean,_ = torch.max(tmp_mean_reuse, dim=0)
                # tmp_mean=tmp_mean.numpy()
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean.append(tmp_mean)

        return output_mean,output_3








class PoolCam_yuhan_kernel_embedding_1():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, ii=None,flip=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别

        feature_maps,output= self.net(input.cuda())


        # print('yuhan')
        feature_maps_L31 = feature_maps[
            0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        batch_size, c_L31, h_L31, w_L31 = feature_maps_L31.size()

        # Pool5   #feature_maps_L31  [512,7,7]==[512,]
        feature_maps_L31_sum = torch.sum(feature_maps_L31[0],
                                         0)  # 如果要将CAM或者gard_cam引入SCDA，需要更改的程序也就仅仅是这一步
        L31_sum_mean = torch.mean(feature_maps_L31_sum)  # 返回的是scalar tensor ,这是与matlab的mean不同的地方

        # highlight_index = torch.nonzero(feature_maps_L31_sum >= L31_sum_mean)  # 一个K*2的tensor,每一行一个符合要求的点
        # a, b = highlight_index.size()
        # print('yu',a)
        # largestConnectComponent将最大连通区域所对应的像素点置为true
        jj = (feature_maps_L31_sum > L31_sum_mean).cpu().data.numpy()
        highlight = torch.from_numpy(jj + 0).cuda().float()
        # print(highlight)

        highlight_conn_L31 = highlight
        # print('yu',torch.sum(highlight_conn_L31))
        # highlight_conn_L31 = torch.tensor(largestConnectComponent(highlight.cpu().numpy()) + 0)#[7*7],我们需要将其变成[512,7,7]
        # print(highlight_conn_L31)
        a=torch.sum(highlight_conn_L31).item()

        feature_maps_L31 = feature_maps_L31[0].cpu().data.numpy()  # [1,512,7,7]==>[512,7,7]
        highlight_conn_L31 = highlight_conn_L31.cpu().data.numpy()
        # print(highlight_conn_L31.shape)
        highlight_conn_L31 = highlight_conn_L31.reshape(1, h_L31, w_L31) * np.ones_like(
            feature_maps_L31)  # [7,7]=>[1,7,7]=>[512,7,7]
        # print(highlight_conn_L31.shape)





        feature_maps_L31_mean=output[0].cpu().data.numpy()

        if np.linalg.norm(feature_maps_L31_mean) > 0:
            feature_maps_L31_mean_norm = feature_maps_L31_mean / np.linalg.norm(feature_maps_L31_mean)
        else:
            print("出现了")
            feature_maps_L31_mean_norm = np.zeros_like(feature_maps_L31_mean)
            pass



        output_mean=[]
        output_3 = []
        output_3.append(feature_maps_L31_mean_norm)


        feature_maps_L31 = feature_maps[
            0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        # feature_maps_L31 = feature_maps_L31[0]  # [1,512,7,7]==>[512,7,7]
        # print(feature_maps_L31.size())
        # print(torch.from_numpy(highlight_conn_L31).unsqueeze(0).cuda().float().size())
        feature_maps_L31 = feature_maps_L31 * torch.from_numpy(highlight_conn_L31).unsqueeze(0).cuda().float()
        feature_maps_L31_max = global_avg_pool(feature_maps_L31)
        coefficient=(h_L31*w_L31)/a
        feature_maps_L31_max=coefficient * feature_maps_L31_max
        feature_maps_L31_mean = global_max_pool(feature_maps_L31)
        feature_maps_L31_mean_max = torch.cat((feature_maps_L31_mean, feature_maps_L31_max), dim=1)

        loss2 = torch.sum(feature_maps_L31_mean_max)
        self.net.zero_grad()
        loss2.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚

        output_mean_no_norm=[]
        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['inception_5b_3x3.weight','inception_5b_double_3x3_2.weight']:
                # print(name)
                tmp = (param.grad).data
                # print(tmp.size())
                # tmp=torch.abs(tmp)
                # print(tmp)
                # tmp_mean_reuse=tmp.view(tmp.size()[0],-1)
                # print(tmp_mean_reuse.shape)
                tmp_mean_reuse_std = torch.std(tmp, dim=(2, 3))

                tmp_mean_reuse = F.adaptive_max_pool2d(tmp, output_size=1).squeeze(-1).squeeze(
                    -1) + F.adaptive_max_pool2d(-1 * tmp, output_size=1).squeeze(-1).squeeze(-1)
                tmp_mean_reuse = tmp_mean_reuse + tmp_mean_reuse_std
                # print(tmp_mean_reuse.size())

                # tmp_mean_reuse=F.adaptive_max_pool2d(tmp,output_size=1).squeeze(-1).squeeze(-1)
                # tmp_mean_reuse=F.adaptive_max_pool2d(-1 * tmp,output_size=1).squeeze(-1).squeeze(-1)

                # tmp_mean_reuse_1 = torch.mean(tmp_1, axis=(2, 3))
                # tmp_mean_reuse = tmp_mean_reuse * tmp_mean_reuse_1
                # tmp_mean_reuse = torch.mean(tmp, dim=(2, 3))
                ############################################################
                #$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%%$%$%$$%$$%$%$%$%$%$%$
                #待会回复注释
                tmp_mean_reuse_bool = torch.mean(tmp_mean_reuse, dim=0).unsqueeze(0)
                tmp_mean_reuse_bool = tmp_mean_reuse > tmp_mean_reuse_bool
                tmp_mean_reuse_bool = tmp_mean_reuse_bool.float()

                tmp_mean_reuse_bool_2 = torch.mean(tmp_mean_reuse, dim=1).unsqueeze(0).T
                tmp_mean_reuse_bool_2 = tmp_mean_reuse > tmp_mean_reuse_bool_2
                tmp_mean_reuse_bool_2 = tmp_mean_reuse_bool_2.float()

                tmp_mean_reuse = tmp_mean_reuse * tmp_mean_reuse_bool * tmp_mean_reuse_bool_2
###############################################################################################
                #$$$$$$$$$$$$$$$$$$$$$$%%%%%%%%%%%%%%%%%%%%%%^^^^^^^^^^^^^^^^^^^$$$$$$$$$$$$$$$$$$$
                tmp_mean = torch.mean(tmp_mean_reuse, dim=0).cpu().data.numpy()
                # tmp_mean,_ = torch.max(tmp_mean_reuse, dim=0)
                # tmp_mean=tmp_mean.numpy()
                output_mean_no_norm.append(tmp_mean)
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean.append(tmp_mean)
            elif name in ['inception_5b_1x1.weight','inception_5b_pool_proj.weight']:
                # print(name)
                tmp = (param.grad).data
                # print(tmp.size())

                # print(tmp)
                tmp=torch.abs(tmp)
                # print(tmp)
                tmp_mean_reuse=tmp.view(tmp.size()[0],-1)
                # print(tmp_mean_reuse.shape)
                # tmp_mean_reuse_std = torch.std(tmp, dim=(2, 3))
                #
                # tmp_mean_reuse = F.adaptive_max_pool2d(tmp, output_size=1).squeeze(-1).squeeze(
                #     -1) + F.adaptive_max_pool2d(-1 * tmp, output_size=1).squeeze(-1).squeeze(-1)
                # tmp_mean_reuse = tmp_mean_reuse + tmp_mean_reuse_std
                # tmp_mean_reuse=F.adaptive_max_pool2d(tmp,output_size=1).squeeze(-1).squeeze(-1)
                # tmp_mean_reuse=F.adaptive_max_pool2d(-1 * tmp,output_size=1).squeeze(-1).squeeze(-1)

                # tmp_mean_reuse_1 = torch.mean(tmp_1, axis=(2, 3))
                # tmp_mean_reuse = tmp_mean_reuse * tmp_mean_reuse_1
                # tmp_mean_reuse = torch.mean(tmp, dim=(2, 3))
                #待会恢复
                ########################################
                # print(tmp_mean_reuse.size())

                tmp_mean_reuse_bool = torch.mean(tmp_mean_reuse, dim=0).unsqueeze(0)
                tmp_mean_reuse_bool = tmp_mean_reuse > tmp_mean_reuse_bool
                tmp_mean_reuse_bool = tmp_mean_reuse_bool.float()

                tmp_mean_reuse_bool_2 = torch.mean(tmp_mean_reuse, dim=1).unsqueeze(0).T
                tmp_mean_reuse_bool_2 = tmp_mean_reuse > tmp_mean_reuse_bool_2
                tmp_mean_reuse_bool_2 = tmp_mean_reuse_bool_2.float()

                tmp_mean_reuse = tmp_mean_reuse * tmp_mean_reuse_bool * tmp_mean_reuse_bool_2
######################################################################
                tmp_mean = torch.mean(tmp_mean_reuse, dim=0).cpu().data.numpy()
                # tmp_mean,_ = torch.max(tmp_mean_reuse, dim=0)
                # tmp_mean=tmp_mean.numpy()
                output_mean_no_norm.append(tmp_mean)
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean.append(tmp_mean)

        return output_mean,output_3,output_mean_no_norm
