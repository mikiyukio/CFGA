# from net import FGIAnet
# from net_grad_cam import FGIAnet_GARD_CAM
import dataloader
from SCDA_cub_resnet50.bwconncomp import largestConnectComponent

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




class PoolCam_yuhan_kernel_scda_origin():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, ii=None,flip=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别

        feature_maps,_,_ = self.net(input.cuda())

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

        # highlight_conn_L31 = highlight
        # print('yu',torch.sum(highlight_conn_L31))
        highlight_conn_L31 = torch.tensor(largestConnectComponent(highlight.cpu().numpy()) + 0)#[7*7],我们需要将其变成[512,7,7]
        # print(highlight_conn_L31)
        a=torch.sum(highlight_conn_L31).item()
        # print(a)
        # print(a)
        # print(220/43)
#       yu待会注释
#         highlight_index = torch.nonzero(highlight_conn_L31)  # 一个K*2的tensor,每一行一个符合要求的点
#         a, b = highlight_index.size()
        # print('yu',a)

        feature_maps_L31 = feature_maps_L31[0].cpu().data.numpy()  # [1,512,7,7]==>[512,7,7]
        highlight_conn_L31 = highlight_conn_L31.cpu().data.numpy()
        # print(highlight_conn_L31.shape)
        highlight_conn_L31 = highlight_conn_L31.reshape(1, h_L31, w_L31) * np.ones_like(
            feature_maps_L31)  # [7,7]=>[1,7,7]=>[512,7,7]
        # print(highlight_conn_L31.shape)

        feature_maps_L31 = feature_maps_L31 * highlight_conn_L31


        feature_maps_L31_max = np.max(feature_maps_L31, axis=(1, 2))
        if np.linalg.norm(feature_maps_L31_max) > 0:
            feature_maps_L31_max_norm = feature_maps_L31_max / np.linalg.norm(feature_maps_L31_max)
        else:
            print("出现了")
            feature_maps_L31_max_norm = np.zeros_like(feature_maps_L31_max)
            pass
        feature_maps_L31_mean = np.sum(feature_maps_L31, axis=(1, 2))/a
        if np.linalg.norm(feature_maps_L31_mean) > 0:
            feature_maps_L31_mean_norm = feature_maps_L31_mean / np.linalg.norm(feature_maps_L31_mean)
        else:
            print("出现了")
            feature_maps_L31_mean_norm = np.zeros_like(feature_maps_L31_mean)
            pass

        feature_maps_L31_mean_max = np.hstack([feature_maps_L31_mean, feature_maps_L31_max])

        if np.linalg.norm(feature_maps_L31_mean_max) > 0:
            feature_maps_L31_mean_max_norm = feature_maps_L31_mean_max / np.linalg.norm(feature_maps_L31_mean_max)
        else:
            print("出现了")
            feature_maps_L31_mean_max_norm = np.zeros_like(feature_maps_L31_mean_max)
            pass


        output_mean = []
        output_2=[]
        output_2.append(np.hstack([feature_maps_L31_mean_norm,feature_maps_L31_max_norm]))#先norm再 拼接
        output_2.append(feature_maps_L31_mean_max_norm)  # 先拼接再norm
        # output_2.append(feature_maps_L31_mean_max_norm)  # 先拼接再norm

        feature_maps_L31 = feature_maps[
            0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        feature_maps_L31 = feature_maps_L31[0].cpu().data.numpy()  # [1,512,7,7]==>[512,7,7]
        feature_maps_L31_max = np.max(feature_maps_L31, axis=(1, 2))
        feature_maps_L31_mean = np.mean(feature_maps_L31, axis=(1, 2))
        feature_maps_L31_mean_max = np.hstack([feature_maps_L31_mean, feature_maps_L31_max])

        if np.linalg.norm(feature_maps_L31_max) > 0:
            feature_maps_L31_max_norm = feature_maps_L31_max / np.linalg.norm(feature_maps_L31_max)
        else:
            print("出现了")
            feature_maps_L31_max_norm = np.zeros_like(feature_maps_L31_max)
            pass

        if np.linalg.norm(feature_maps_L31_mean) > 0:
            feature_maps_L31_mean_norm = feature_maps_L31_mean / np.linalg.norm(feature_maps_L31_mean)
        else:
            print("出现了")
            feature_maps_L31_mean_norm = np.zeros_like(feature_maps_L31_mean)
            pass

        feature_maps_L31_norm_mean_max = np.hstack([feature_maps_L31_mean_norm, feature_maps_L31_max_norm])

        if np.linalg.norm(feature_maps_L31_mean_max) > 0:
            feature_maps_L31_mean_max_norm = feature_maps_L31_mean_max / np.linalg.norm(feature_maps_L31_mean_max)
        else:
            print("出现了")
            feature_maps_L31_mean_max_norm = np.zeros_like(feature_maps_L31_mean_max)
            pass
        output_3 = []
        # output_3.append(feature_maps_L31_mean_norm)
        ###############################################################
        #等待被恢复注释
        output_3.append(feature_maps_L31_mean_max_norm)
        output_3.append(feature_maps_L31_norm_mean_max)

        # output_3.append(feature_maps_L31_mean_norm)
        # output_3.append(feature_maps_L31_max_norm)
        #############################################################
        # output_3.append(feature_maps_L31_norm_mean_max)

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
            if name in ['layer4.2.conv2.weight','layer4.1.conv2.weight']:
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
            elif name in ['layer4.2.conv1.weight','layer4.1.conv1.weight','layer4.2.conv3.weight','layer4.1.conv3.weight']:
                # print(name)
                tmp = (param.grad).data
                # print(tmp.size())

                # print(tmp)
                tmp=torch.abs(tmp)
                # print(tmp)
                tmp_mean_reuse=tmp.view(tmp.size()[0],-1)

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

        return output_mean,output_2,output_3



class PoolCam_yuhan_kernel_embedding_proxy_anchor():
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

        # highlight_conn_L31 = highlight
        # print('yu',torch.sum(highlight_conn_L31))
        highlight_conn_L31 = torch.tensor(largestConnectComponent(highlight.cpu().numpy()) + 0)#[7*7],我们需要将其变成[512,7,7]
        # print(highlight_conn_L31)
        a=torch.sum(highlight_conn_L31).item()

        feature_maps_L31 = feature_maps_L31[0].cpu().data.numpy()  # [1,512,7,7]==>[512,7,7]
        highlight_conn_L31 = highlight_conn_L31.cpu().data.numpy()
        # print(highlight_conn_L31.shape)
        highlight_conn_L31 = highlight_conn_L31.reshape(1, h_L31, w_L31) * np.ones_like(
            feature_maps_L31)  # [7,7]=>[1,7,7]=>[512,7,7]
        # print(highlight_conn_L31.shape)



        # feature_maps_L31 = feature_maps[
        #     0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        # # print(feature_maps_L31.size())
        # feature_maps_L31 = feature_maps_L31[0].cpu().data.numpy()  # [1,512,7,7]==>[512,7,7]
        # feature_maps_L31_max = np.max(feature_maps_L31, axis=(1, 2))
        # feature_maps_L31_mean = np.mean(feature_maps_L31, axis=(1, 2))
        # feature_maps_L31_mean_max = np.hstack([feature_maps_L31_mean, feature_maps_L31_max])
        #
        # if np.linalg.norm(feature_maps_L31_mean_max) > 0:
        #     feature_maps_L31_mean_max_norm = feature_maps_L31_mean_max / np.linalg.norm(feature_maps_L31_mean_max)
        # else:
        #     print("出现了")
        #     feature_maps_L31_mean_max_norm = np.zeros_like(feature_maps_L31_mean_max)
        #     pass
        #
        # # if np.linalg.norm(feature_maps_L31_max) > 0:
        # #     feature_maps_L31_max_norm = feature_maps_L31_max / np.linalg.norm(feature_maps_L31_max)
        # # else:
        # #     print("出现了")
        # #     feature_maps_L31_max_norm = np.zeros_like(feature_maps_L31_max)
        # #     pass
        #
        #
        # output_3 = []
        # # output_3.append(feature_maps_L31_mean_norm)
        # ###############################################################
        # # 等待被恢复注释
        # output_3.append(feature_maps_L31_mean_max_norm)
        # #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        # # output_3.append(feature_maps_L31_norm_mean_max)
        #
        # # output_3.append(feature_maps_L31_mean_norm)
        # # output_3.append(feature_maps_L31_max_norm)
        # #############################################################
        # # output_3.append(feature_maps_L31_norm_mean_max)





        feature_maps_L31_mean = output[0].cpu().data.numpy()

        if np.linalg.norm(feature_maps_L31_mean) > 0:
            feature_maps_L31_mean_norm = feature_maps_L31_mean / np.linalg.norm(feature_maps_L31_mean)
        else:
            print("出现了")
            feature_maps_L31_mean_norm = np.zeros_like(feature_maps_L31_mean)
            pass

        # output_mean = []
        output_3 = []
        output_3.append(feature_maps_L31_mean_norm)





        output_mean=[]
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
            if name in ['model.layer4.2.conv2.weight','model.layer4.1.conv2.weight']:
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
            elif name in ['model.layer4.2.conv1.weight','model.layer4.1.conv1.weight','model.layer4.2.conv3.weight','model.layer4.1.conv3.weight']:
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






class PoolCam_yuhan_kernel_scda_origin_eq5():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, ii=None,flip=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别

        feature_maps,_,_ = self.net(input.cuda())

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

        # highlight_conn_L31 = highlight
        # print('yu',torch.sum(highlight_conn_L31))
        highlight_conn_L31 = torch.tensor(largestConnectComponent(highlight.cpu().numpy()) + 0)#[7*7],我们需要将其变成[512,7,7]
        # print(highlight_conn_L31)
        a=torch.sum(highlight_conn_L31).item()
        # print(a)
        # print(a)
        # print(220/43)
#       yu待会注释
#         highlight_index = torch.nonzero(highlight_conn_L31)  # 一个K*2的tensor,每一行一个符合要求的点
#         a, b = highlight_index.size()
        # print('yu',a)

        feature_maps_L31 = feature_maps_L31[0].cpu().data.numpy()  # [1,512,7,7]==>[512,7,7]
        highlight_conn_L31 = highlight_conn_L31.cpu().data.numpy()
        # print(highlight_conn_L31.shape)
        highlight_conn_L31 = highlight_conn_L31.reshape(1, h_L31, w_L31) * np.ones_like(
            feature_maps_L31)  # [7,7]=>[1,7,7]=>[512,7,7]
        # print(highlight_conn_L31.shape)

        feature_maps_L31 = feature_maps_L31 * highlight_conn_L31


        feature_maps_L31_max = np.max(feature_maps_L31, axis=(1, 2))
        if np.linalg.norm(feature_maps_L31_max) > 0:
            feature_maps_L31_max_norm = feature_maps_L31_max / np.linalg.norm(feature_maps_L31_max)
        else:
            print("出现了")
            feature_maps_L31_max_norm = np.zeros_like(feature_maps_L31_max)
            pass
        feature_maps_L31_mean = np.sum(feature_maps_L31, axis=(1, 2))/a
        if np.linalg.norm(feature_maps_L31_mean) > 0:
            feature_maps_L31_mean_norm = feature_maps_L31_mean / np.linalg.norm(feature_maps_L31_mean)
        else:
            print("出现了")
            feature_maps_L31_mean_norm = np.zeros_like(feature_maps_L31_mean)
            pass

        feature_maps_L31_mean_max = np.hstack([feature_maps_L31_mean, feature_maps_L31_max])

        if np.linalg.norm(feature_maps_L31_mean_max) > 0:
            feature_maps_L31_mean_max_norm = feature_maps_L31_mean_max / np.linalg.norm(feature_maps_L31_mean_max)
        else:
            print("出现了")
            feature_maps_L31_mean_max_norm = np.zeros_like(feature_maps_L31_mean_max)
            pass


        output_mean = []
        output_2=[]
        output_2.append(np.hstack([feature_maps_L31_mean_norm,feature_maps_L31_max_norm]))#先norm再 拼接
        output_2.append(feature_maps_L31_mean_max_norm)  # 先拼接再norm
        # output_2.append(feature_maps_L31_mean_max_norm)  # 先拼接再norm

        feature_maps_L31 = feature_maps[
            0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        feature_maps_L31 = feature_maps_L31[0].cpu().data.numpy()  # [1,512,7,7]==>[512,7,7]
        feature_maps_L31_max = np.max(feature_maps_L31, axis=(1, 2))
        feature_maps_L31_mean = np.mean(feature_maps_L31, axis=(1, 2))
        feature_maps_L31_mean_max = np.hstack([feature_maps_L31_mean, feature_maps_L31_max])

        if np.linalg.norm(feature_maps_L31_max) > 0:
            feature_maps_L31_max_norm = feature_maps_L31_max / np.linalg.norm(feature_maps_L31_max)
        else:
            print("出现了")
            feature_maps_L31_max_norm = np.zeros_like(feature_maps_L31_max)
            pass

        if np.linalg.norm(feature_maps_L31_mean) > 0:
            feature_maps_L31_mean_norm = feature_maps_L31_mean / np.linalg.norm(feature_maps_L31_mean)
        else:
            print("出现了")
            feature_maps_L31_mean_norm = np.zeros_like(feature_maps_L31_mean)
            pass

        feature_maps_L31_norm_mean_max = np.hstack([feature_maps_L31_mean_norm, feature_maps_L31_max_norm])

        if np.linalg.norm(feature_maps_L31_mean_max) > 0:
            feature_maps_L31_mean_max_norm = feature_maps_L31_mean_max / np.linalg.norm(feature_maps_L31_mean_max)
        else:
            print("出现了")
            feature_maps_L31_mean_max_norm = np.zeros_like(feature_maps_L31_mean_max)
            pass
        output_3 = []
        # output_3.append(feature_maps_L31_mean_norm)
        output_3.append(feature_maps_L31_mean_max_norm)
        output_3.append(feature_maps_L31_norm_mean_max)
        # output_3.append(feature_maps_L31_norm_mean_max)

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
            if name in ['layer4.2.conv2.weight','layer4.1.conv2.weight']:
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

                # tmp_mean_reuse_bool = torch.mean(tmp_mean_reuse, dim=0).unsqueeze(0)
                # tmp_mean_reuse_bool = tmp_mean_reuse > tmp_mean_reuse_bool
                # tmp_mean_reuse_bool = tmp_mean_reuse_bool.float()
                #
                # tmp_mean_reuse_bool_2 = torch.mean(tmp_mean_reuse, dim=1).unsqueeze(0).T
                # tmp_mean_reuse_bool_2 = tmp_mean_reuse > tmp_mean_reuse_bool_2
                # tmp_mean_reuse_bool_2 = tmp_mean_reuse_bool_2.float()
                #
                # tmp_mean_reuse = tmp_mean_reuse * tmp_mean_reuse_bool * tmp_mean_reuse_bool_2
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
            elif name in ['layer4.2.conv1.weight','layer4.1.conv1.weight','layer4.2.conv3.weight','layer4.1.conv3.weight']:
                # print(name)
                tmp = (param.grad).data
                # print(tmp.size())

                # print(tmp)
                tmp=torch.abs(tmp)
                # print(tmp)
                tmp_mean_reuse=tmp.view(tmp.size()[0],-1)

                # tmp_mean_reuse_bool = torch.mean(tmp_mean_reuse, dim=0).unsqueeze(0)
                # tmp_mean_reuse_bool = tmp_mean_reuse > tmp_mean_reuse_bool
                # tmp_mean_reuse_bool = tmp_mean_reuse_bool.float()
                #
                # tmp_mean_reuse_bool_2 = torch.mean(tmp_mean_reuse, dim=1).unsqueeze(0).T
                # tmp_mean_reuse_bool_2 = tmp_mean_reuse > tmp_mean_reuse_bool_2
                # tmp_mean_reuse_bool_2 = tmp_mean_reuse_bool_2.float()
                #
                # tmp_mean_reuse = tmp_mean_reuse * tmp_mean_reuse_bool * tmp_mean_reuse_bool_2
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

        return output_mean,output_2,output_3















class PoolCam_yuhan_kernel_scda_origin_eq6():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, ii=None,flip=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别

        feature_maps,_,_ = self.net(input.cuda())

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

        # highlight_conn_L31 = highlight
        # print('yu',torch.sum(highlight_conn_L31))
        highlight_conn_L31 = torch.tensor(largestConnectComponent(highlight.cpu().numpy()) + 0)#[7*7],我们需要将其变成[512,7,7]
        # print(highlight_conn_L31)
        a=torch.sum(highlight_conn_L31).item()
        # print(a)
        # print(a)
        # print(220/43)
#       yu待会注释
#         highlight_index = torch.nonzero(highlight_conn_L31)  # 一个K*2的tensor,每一行一个符合要求的点
#         a, b = highlight_index.size()
        # print('yu',a)

        feature_maps_L31 = feature_maps_L31[0].cpu().data.numpy()  # [1,512,7,7]==>[512,7,7]
        highlight_conn_L31 = highlight_conn_L31.cpu().data.numpy()
        # print(highlight_conn_L31.shape)
        highlight_conn_L31 = highlight_conn_L31.reshape(1, h_L31, w_L31) * np.ones_like(
            feature_maps_L31)  # [7,7]=>[1,7,7]=>[512,7,7]
        # print(highlight_conn_L31.shape)

        feature_maps_L31 = feature_maps_L31 * highlight_conn_L31


        feature_maps_L31_max = np.max(feature_maps_L31, axis=(1, 2))
        if np.linalg.norm(feature_maps_L31_max) > 0:
            feature_maps_L31_max_norm = feature_maps_L31_max / np.linalg.norm(feature_maps_L31_max)
        else:
            print("出现了")
            feature_maps_L31_max_norm = np.zeros_like(feature_maps_L31_max)
            pass
        feature_maps_L31_mean = np.sum(feature_maps_L31, axis=(1, 2))/a
        if np.linalg.norm(feature_maps_L31_mean) > 0:
            feature_maps_L31_mean_norm = feature_maps_L31_mean / np.linalg.norm(feature_maps_L31_mean)
        else:
            print("出现了")
            feature_maps_L31_mean_norm = np.zeros_like(feature_maps_L31_mean)
            pass

        feature_maps_L31_mean_max = np.hstack([feature_maps_L31_mean, feature_maps_L31_max])

        if np.linalg.norm(feature_maps_L31_mean_max) > 0:
            feature_maps_L31_mean_max_norm = feature_maps_L31_mean_max / np.linalg.norm(feature_maps_L31_mean_max)
        else:
            print("出现了")
            feature_maps_L31_mean_max_norm = np.zeros_like(feature_maps_L31_mean_max)
            pass


        output_mean = []
        output_2=[]
        output_2.append(np.hstack([feature_maps_L31_mean_norm,feature_maps_L31_max_norm]))#先norm再 拼接
        output_2.append(feature_maps_L31_mean_max_norm)  # 先拼接再norm
        # output_2.append(feature_maps_L31_mean_max_norm)  # 先拼接再norm

        feature_maps_L31 = feature_maps[
            0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        feature_maps_L31 = feature_maps_L31[0].cpu().data.numpy()  # [1,512,7,7]==>[512,7,7]
        feature_maps_L31_max = np.max(feature_maps_L31, axis=(1, 2))
        feature_maps_L31_mean = np.mean(feature_maps_L31, axis=(1, 2))
        feature_maps_L31_mean_max = np.hstack([feature_maps_L31_mean, feature_maps_L31_max])

        if np.linalg.norm(feature_maps_L31_max) > 0:
            feature_maps_L31_max_norm = feature_maps_L31_max / np.linalg.norm(feature_maps_L31_max)
        else:
            print("出现了")
            feature_maps_L31_max_norm = np.zeros_like(feature_maps_L31_max)
            pass

        if np.linalg.norm(feature_maps_L31_mean) > 0:
            feature_maps_L31_mean_norm = feature_maps_L31_mean / np.linalg.norm(feature_maps_L31_mean)
        else:
            print("出现了")
            feature_maps_L31_mean_norm = np.zeros_like(feature_maps_L31_mean)
            pass

        feature_maps_L31_norm_mean_max = np.hstack([feature_maps_L31_mean_norm, feature_maps_L31_max_norm])

        if np.linalg.norm(feature_maps_L31_mean_max) > 0:
            feature_maps_L31_mean_max_norm = feature_maps_L31_mean_max / np.linalg.norm(feature_maps_L31_mean_max)
        else:
            print("出现了")
            feature_maps_L31_mean_max_norm = np.zeros_like(feature_maps_L31_mean_max)
            pass
        output_3 = []
        # output_3.append(feature_maps_L31_mean_norm)
        output_3.append(feature_maps_L31_mean_max_norm)
        output_3.append(feature_maps_L31_norm_mean_max)
        # output_3.append(feature_maps_L31_norm_mean_max)

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
            if name in ['layer4.2.conv2.weight','layer4.1.conv2.weight']:
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

                tmp_mean_reuse_bool = torch.mean(tmp_mean_reuse, dim=0).unsqueeze(0)
                tmp_mean_reuse_bool = tmp_mean_reuse > tmp_mean_reuse_bool
                tmp_mean_reuse_bool = tmp_mean_reuse_bool.float()

                tmp_mean_reuse_bool_2 = torch.mean(tmp_mean_reuse, dim=1).unsqueeze(0).T
                tmp_mean_reuse_bool_2 = tmp_mean_reuse > tmp_mean_reuse_bool_2
                tmp_mean_reuse_bool_2 = tmp_mean_reuse_bool_2.float()

                tmp_mean_reuse = tmp_mean_reuse * tmp_mean_reuse_bool * tmp_mean_reuse_bool_2
###############################################################################################
                #$$$$$$$$$$$$$$$$$$$$$$%%%%%%%%%%%%%%%%%%%%%%^^^^^^^^^^^^^^^^^^^$$$$$$$$$$$$$$$$$$$
                tmp_mean = torch.mean(tmp_mean_reuse, dim=1).cpu().data.numpy()
                # tmp_mean,_ = torch.max(tmp_mean_reuse, dim=0)
                # tmp_mean=tmp_mean.numpy()
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean.append(tmp_mean)
            elif name in ['layer4.2.conv1.weight','layer4.1.conv1.weight','layer4.2.conv3.weight','layer4.1.conv3.weight']:
                # print(name)
                tmp = (param.grad).data
                # print(tmp.size())

                # print(tmp)
                tmp=torch.abs(tmp)
                # print(tmp)
                tmp_mean_reuse=tmp.view(tmp.size()[0],-1)

                tmp_mean_reuse_bool = torch.mean(tmp_mean_reuse, dim=0).unsqueeze(0)
                tmp_mean_reuse_bool = tmp_mean_reuse > tmp_mean_reuse_bool
                tmp_mean_reuse_bool = tmp_mean_reuse_bool.float()

                tmp_mean_reuse_bool_2 = torch.mean(tmp_mean_reuse, dim=1).unsqueeze(0).T
                tmp_mean_reuse_bool_2 = tmp_mean_reuse > tmp_mean_reuse_bool_2
                tmp_mean_reuse_bool_2 = tmp_mean_reuse_bool_2.float()

                tmp_mean_reuse = tmp_mean_reuse * tmp_mean_reuse_bool * tmp_mean_reuse_bool_2
######################################################################
                tmp_mean = torch.mean(tmp_mean_reuse, dim=1).cpu().data.numpy()
                # tmp_mean,_ = torch.max(tmp_mean_reuse, dim=0)
                # tmp_mean=tmp_mean.numpy()
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean.append(tmp_mean)

        return output_mean,output_2,output_3





class PoolCam_yuhan_kernel_scda_origin_R18():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, ii=None, flip=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别

        feature_maps, _, _ = self.net(input.cuda())

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

        # highlight_conn_L31 = highlight
        # print('yu',torch.sum(highlight_conn_L31))
        highlight_conn_L31 = torch.tensor(
            largestConnectComponent(highlight.cpu().numpy()) + 0)  # [7*7],我们需要将其变成[512,7,7]
        # print(highlight_conn_L31)
        a = torch.sum(highlight_conn_L31).item()
        # print(a)
        # print(a)
        # print(220/43)
        #       yu待会注释
        #         highlight_index = torch.nonzero(highlight_conn_L31)  # 一个K*2的tensor,每一行一个符合要求的点
        #         a, b = highlight_index.size()
        # print('yu',a)

        feature_maps_L31 = feature_maps_L31[0].cpu().data.numpy()  # [1,512,7,7]==>[512,7,7]
        highlight_conn_L31 = highlight_conn_L31.cpu().data.numpy()
        # print(highlight_conn_L31.shape)
        highlight_conn_L31 = highlight_conn_L31.reshape(1, h_L31, w_L31) * np.ones_like(
            feature_maps_L31)  # [7,7]=>[1,7,7]=>[512,7,7]
        # print(highlight_conn_L31.shape)

        feature_maps_L31 = feature_maps_L31 * highlight_conn_L31

        feature_maps_L31_max = np.max(feature_maps_L31, axis=(1, 2))
        if np.linalg.norm(feature_maps_L31_max) > 0:
            feature_maps_L31_max_norm = feature_maps_L31_max / np.linalg.norm(feature_maps_L31_max)
        else:
            print("出现了")
            feature_maps_L31_max_norm = np.zeros_like(feature_maps_L31_max)
            pass
        feature_maps_L31_mean = np.sum(feature_maps_L31, axis=(1, 2)) / a
        if np.linalg.norm(feature_maps_L31_mean) > 0:
            feature_maps_L31_mean_norm = feature_maps_L31_mean / np.linalg.norm(feature_maps_L31_mean)
        else:
            print("出现了")
            feature_maps_L31_mean_norm = np.zeros_like(feature_maps_L31_mean)
            pass

        feature_maps_L31_mean_max = np.hstack([feature_maps_L31_mean, feature_maps_L31_max])

        if np.linalg.norm(feature_maps_L31_mean_max) > 0:
            feature_maps_L31_mean_max_norm = feature_maps_L31_mean_max / np.linalg.norm(feature_maps_L31_mean_max)
        else:
            print("出现了")
            feature_maps_L31_mean_max_norm = np.zeros_like(feature_maps_L31_mean_max)
            pass

        output_mean = []
        output_2 = []
        output_2.append(np.hstack([feature_maps_L31_mean_norm, feature_maps_L31_max_norm]))  # 先norm再 拼接
        output_2.append(feature_maps_L31_mean_max_norm)  # 先拼接再norm
        # output_2.append(feature_maps_L31_mean_max_norm)  # 先拼接再norm

        feature_maps_L31 = feature_maps[
            0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        feature_maps_L31 = feature_maps_L31[0].cpu().data.numpy()  # [1,512,7,7]==>[512,7,7]
        feature_maps_L31_max = np.max(feature_maps_L31, axis=(1, 2))
        feature_maps_L31_mean = np.mean(feature_maps_L31, axis=(1, 2))
        feature_maps_L31_mean_max = np.hstack([feature_maps_L31_mean, feature_maps_L31_max])

        if np.linalg.norm(feature_maps_L31_max) > 0:
            feature_maps_L31_max_norm = feature_maps_L31_max / np.linalg.norm(feature_maps_L31_max)
        else:
            print("出现了")
            feature_maps_L31_max_norm = np.zeros_like(feature_maps_L31_max)
            pass

        if np.linalg.norm(feature_maps_L31_mean) > 0:
            feature_maps_L31_mean_norm = feature_maps_L31_mean / np.linalg.norm(feature_maps_L31_mean)
        else:
            print("出现了")
            feature_maps_L31_mean_norm = np.zeros_like(feature_maps_L31_mean)
            pass

        feature_maps_L31_norm_mean_max = np.hstack([feature_maps_L31_mean_norm, feature_maps_L31_max_norm])

        if np.linalg.norm(feature_maps_L31_mean_max) > 0:
            feature_maps_L31_mean_max_norm = feature_maps_L31_mean_max / np.linalg.norm(feature_maps_L31_mean_max)
        else:
            print("出现了")
            feature_maps_L31_mean_max_norm = np.zeros_like(feature_maps_L31_mean_max)
            pass
        output_3 = []
        # output_3.append(feature_maps_L31_mean_norm)
        output_3.append(feature_maps_L31_mean_max_norm)
        output_3.append(feature_maps_L31_norm_mean_max)
        # output_3.append(feature_maps_L31_norm_mean_max)

        feature_maps_L31 = feature_maps[
            0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        # feature_maps_L31 = feature_maps_L31[0]  # [1,512,7,7]==>[512,7,7]
        # print(feature_maps_L31.size())
        # print(torch.from_numpy(highlight_conn_L31).unsqueeze(0).cuda().float().size())
        feature_maps_L31 = feature_maps_L31 * torch.from_numpy(highlight_conn_L31).unsqueeze(0).cuda().float()
        feature_maps_L31_max = global_avg_pool(feature_maps_L31)
        coefficient = (h_L31 * w_L31) / a
        feature_maps_L31_max = coefficient * feature_maps_L31_max
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
            if name in ['layer4.0.conv1.weight', 'layer4.0.conv2.weight','layer4.1.conv1.weight','layer4.1.conv2.weight']:
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

                tmp_mean_reuse_bool = torch.mean(tmp_mean_reuse, dim=0).unsqueeze(0)
                tmp_mean_reuse_bool = tmp_mean_reuse > tmp_mean_reuse_bool
                tmp_mean_reuse_bool = tmp_mean_reuse_bool.float()

                tmp_mean_reuse_bool_2 = torch.mean(tmp_mean_reuse, dim=1).unsqueeze(0).T
                tmp_mean_reuse_bool_2 = tmp_mean_reuse > tmp_mean_reuse_bool_2
                tmp_mean_reuse_bool_2 = tmp_mean_reuse_bool_2.float()

                tmp_mean_reuse = tmp_mean_reuse * tmp_mean_reuse_bool * tmp_mean_reuse_bool_2
                ###############################################################################################
                # $$$$$$$$$$$$$$$$$$$$$$%%%%%%%%%%%%%%%%%%%%%%^^^^^^^^^^^^^^^^^^^$$$$$$$$$$$$$$$$$$$
                tmp_mean = torch.mean(tmp_mean_reuse, dim=0).cpu().data.numpy()
                # tmp_mean,_ = torch.max(tmp_mean_reuse, dim=0)
                # tmp_mean=tmp_mean.numpy()
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean.append(tmp_mean)


        return output_mean, output_2, output_3


class PoolCam_yuhan_kernel_scda_origin_R34():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, ii=None, flip=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别

        feature_maps, _, _ = self.net(input.cuda())

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

        # highlight_conn_L31 = highlight
        # print('yu',torch.sum(highlight_conn_L31))
        highlight_conn_L31 = torch.tensor(
            largestConnectComponent(highlight.cpu().numpy()) + 0)  # [7*7],我们需要将其变成[512,7,7]
        # print(highlight_conn_L31)
        a = torch.sum(highlight_conn_L31).item()
        # print(a)
        # print(a)
        # print(220/43)
        #       yu待会注释
        #         highlight_index = torch.nonzero(highlight_conn_L31)  # 一个K*2的tensor,每一行一个符合要求的点
        #         a, b = highlight_index.size()
        # print('yu',a)

        feature_maps_L31 = feature_maps_L31[0].cpu().data.numpy()  # [1,512,7,7]==>[512,7,7]
        highlight_conn_L31 = highlight_conn_L31.cpu().data.numpy()
        # print(highlight_conn_L31.shape)
        highlight_conn_L31 = highlight_conn_L31.reshape(1, h_L31, w_L31) * np.ones_like(
            feature_maps_L31)  # [7,7]=>[1,7,7]=>[512,7,7]
        # print(highlight_conn_L31.shape)

        feature_maps_L31 = feature_maps_L31 * highlight_conn_L31

        feature_maps_L31_max = np.max(feature_maps_L31, axis=(1, 2))
        if np.linalg.norm(feature_maps_L31_max) > 0:
            feature_maps_L31_max_norm = feature_maps_L31_max / np.linalg.norm(feature_maps_L31_max)
        else:
            print("出现了")
            feature_maps_L31_max_norm = np.zeros_like(feature_maps_L31_max)
            pass
        feature_maps_L31_mean = np.sum(feature_maps_L31, axis=(1, 2)) / a
        if np.linalg.norm(feature_maps_L31_mean) > 0:
            feature_maps_L31_mean_norm = feature_maps_L31_mean / np.linalg.norm(feature_maps_L31_mean)
        else:
            print("出现了")
            feature_maps_L31_mean_norm = np.zeros_like(feature_maps_L31_mean)
            pass

        feature_maps_L31_mean_max = np.hstack([feature_maps_L31_mean, feature_maps_L31_max])

        if np.linalg.norm(feature_maps_L31_mean_max) > 0:
            feature_maps_L31_mean_max_norm = feature_maps_L31_mean_max / np.linalg.norm(feature_maps_L31_mean_max)
        else:
            print("出现了")
            feature_maps_L31_mean_max_norm = np.zeros_like(feature_maps_L31_mean_max)
            pass

        output_mean = []
        output_2 = []
        output_2.append(np.hstack([feature_maps_L31_mean_norm, feature_maps_L31_max_norm]))  # 先norm再 拼接
        output_2.append(feature_maps_L31_mean_max_norm)  # 先拼接再norm
        # output_2.append(feature_maps_L31_mean_max_norm)  # 先拼接再norm

        feature_maps_L31 = feature_maps[
            0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        feature_maps_L31 = feature_maps_L31[0].cpu().data.numpy()  # [1,512,7,7]==>[512,7,7]
        feature_maps_L31_max = np.max(feature_maps_L31, axis=(1, 2))
        feature_maps_L31_mean = np.mean(feature_maps_L31, axis=(1, 2))
        feature_maps_L31_mean_max = np.hstack([feature_maps_L31_mean, feature_maps_L31_max])

        if np.linalg.norm(feature_maps_L31_max) > 0:
            feature_maps_L31_max_norm = feature_maps_L31_max / np.linalg.norm(feature_maps_L31_max)
        else:
            print("出现了")
            feature_maps_L31_max_norm = np.zeros_like(feature_maps_L31_max)
            pass

        if np.linalg.norm(feature_maps_L31_mean) > 0:
            feature_maps_L31_mean_norm = feature_maps_L31_mean / np.linalg.norm(feature_maps_L31_mean)
        else:
            print("出现了")
            feature_maps_L31_mean_norm = np.zeros_like(feature_maps_L31_mean)
            pass

        feature_maps_L31_norm_mean_max = np.hstack([feature_maps_L31_mean_norm, feature_maps_L31_max_norm])

        if np.linalg.norm(feature_maps_L31_mean_max) > 0:
            feature_maps_L31_mean_max_norm = feature_maps_L31_mean_max / np.linalg.norm(feature_maps_L31_mean_max)
        else:
            print("出现了")
            feature_maps_L31_mean_max_norm = np.zeros_like(feature_maps_L31_mean_max)
            pass
        output_3 = []
        # output_3.append(feature_maps_L31_mean_norm)
        output_3.append(feature_maps_L31_mean_max_norm)
        output_3.append(feature_maps_L31_norm_mean_max)
        # output_3.append(feature_maps_L31_norm_mean_max)

        feature_maps_L31 = feature_maps[
            0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        # feature_maps_L31 = feature_maps_L31[0]  # [1,512,7,7]==>[512,7,7]
        # print(feature_maps_L31.size())
        # print(torch.from_numpy(highlight_conn_L31).unsqueeze(0).cuda().float().size())
        feature_maps_L31 = feature_maps_L31 * torch.from_numpy(highlight_conn_L31).unsqueeze(0).cuda().float()
        feature_maps_L31_max = global_avg_pool(feature_maps_L31)
        coefficient = (h_L31 * w_L31) / a
        feature_maps_L31_max = coefficient * feature_maps_L31_max
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
            if name in ['layer4.1.conv1.weight', 'layer4.1.conv2.weight','layer4.2.conv1.weight','layer4.2.conv2.weight']:
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

                tmp_mean_reuse_bool = torch.mean(tmp_mean_reuse, dim=0).unsqueeze(0)
                tmp_mean_reuse_bool = tmp_mean_reuse > tmp_mean_reuse_bool
                tmp_mean_reuse_bool = tmp_mean_reuse_bool.float()

                tmp_mean_reuse_bool_2 = torch.mean(tmp_mean_reuse, dim=1).unsqueeze(0).T
                tmp_mean_reuse_bool_2 = tmp_mean_reuse > tmp_mean_reuse_bool_2
                tmp_mean_reuse_bool_2 = tmp_mean_reuse_bool_2.float()

                tmp_mean_reuse = tmp_mean_reuse * tmp_mean_reuse_bool * tmp_mean_reuse_bool_2
                ###############################################################################################
                # $$$$$$$$$$$$$$$$$$$$$$%%%%%%%%%%%%%%%%%%%%%%^^^^^^^^^^^^^^^^^^^$$$$$$$$$$$$$$$$$$$
                tmp_mean = torch.mean(tmp_mean_reuse, dim=0).cpu().data.numpy()
                # tmp_mean,_ = torch.max(tmp_mean_reuse, dim=0)
                # tmp_mean=tmp_mean.numpy()
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean.append(tmp_mean)


        return output_mean, output_2, output_3


class PoolCam_yuhan_kernel_scda_origin_R101():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, ii=None, flip=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别

        feature_maps, _, _ = self.net(input.cuda())

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

        # highlight_conn_L31 = highlight
        # print('yu',torch.sum(highlight_conn_L31))
        highlight_conn_L31 = torch.tensor(
            largestConnectComponent(highlight.cpu().numpy()) + 0)  # [7*7],我们需要将其变成[512,7,7]
        # print(highlight_conn_L31)
        a = torch.sum(highlight_conn_L31).item()
        # print(a)
        # print(a)
        # print(220/43)
        #       yu待会注释
        #         highlight_index = torch.nonzero(highlight_conn_L31)  # 一个K*2的tensor,每一行一个符合要求的点
        #         a, b = highlight_index.size()
        # print('yu',a)

        feature_maps_L31 = feature_maps_L31[0].cpu().data.numpy()  # [1,512,7,7]==>[512,7,7]
        highlight_conn_L31 = highlight_conn_L31.cpu().data.numpy()
        # print(highlight_conn_L31.shape)
        highlight_conn_L31 = highlight_conn_L31.reshape(1, h_L31, w_L31) * np.ones_like(
            feature_maps_L31)  # [7,7]=>[1,7,7]=>[512,7,7]
        # print(highlight_conn_L31.shape)

        feature_maps_L31 = feature_maps_L31 * highlight_conn_L31

        feature_maps_L31_max = np.max(feature_maps_L31, axis=(1, 2))
        if np.linalg.norm(feature_maps_L31_max) > 0:
            feature_maps_L31_max_norm = feature_maps_L31_max / np.linalg.norm(feature_maps_L31_max)
        else:
            print("出现了")
            feature_maps_L31_max_norm = np.zeros_like(feature_maps_L31_max)
            pass
        feature_maps_L31_mean = np.sum(feature_maps_L31, axis=(1, 2)) / a
        if np.linalg.norm(feature_maps_L31_mean) > 0:
            feature_maps_L31_mean_norm = feature_maps_L31_mean / np.linalg.norm(feature_maps_L31_mean)
        else:
            print("出现了")
            feature_maps_L31_mean_norm = np.zeros_like(feature_maps_L31_mean)
            pass

        feature_maps_L31_mean_max = np.hstack([feature_maps_L31_mean, feature_maps_L31_max])

        if np.linalg.norm(feature_maps_L31_mean_max) > 0:
            feature_maps_L31_mean_max_norm = feature_maps_L31_mean_max / np.linalg.norm(feature_maps_L31_mean_max)
        else:
            print("出现了")
            feature_maps_L31_mean_max_norm = np.zeros_like(feature_maps_L31_mean_max)
            pass

        output_mean = []
        output_2 = []
        output_2.append(np.hstack([feature_maps_L31_mean_norm, feature_maps_L31_max_norm]))  # 先norm再 拼接
        output_2.append(feature_maps_L31_mean_max_norm)  # 先拼接再norm
        # output_2.append(feature_maps_L31_mean_max_norm)  # 先拼接再norm

        feature_maps_L31 = feature_maps[
            0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        feature_maps_L31 = feature_maps_L31[0].cpu().data.numpy()  # [1,512,7,7]==>[512,7,7]
        feature_maps_L31_max = np.max(feature_maps_L31, axis=(1, 2))
        feature_maps_L31_mean = np.mean(feature_maps_L31, axis=(1, 2))
        feature_maps_L31_mean_max = np.hstack([feature_maps_L31_mean, feature_maps_L31_max])

        if np.linalg.norm(feature_maps_L31_max) > 0:
            feature_maps_L31_max_norm = feature_maps_L31_max / np.linalg.norm(feature_maps_L31_max)
        else:
            print("出现了")
            feature_maps_L31_max_norm = np.zeros_like(feature_maps_L31_max)
            pass

        if np.linalg.norm(feature_maps_L31_mean) > 0:
            feature_maps_L31_mean_norm = feature_maps_L31_mean / np.linalg.norm(feature_maps_L31_mean)
        else:
            print("出现了")
            feature_maps_L31_mean_norm = np.zeros_like(feature_maps_L31_mean)
            pass

        feature_maps_L31_norm_mean_max = np.hstack([feature_maps_L31_mean_norm, feature_maps_L31_max_norm])

        if np.linalg.norm(feature_maps_L31_mean_max) > 0:
            feature_maps_L31_mean_max_norm = feature_maps_L31_mean_max / np.linalg.norm(feature_maps_L31_mean_max)
        else:
            print("出现了")
            feature_maps_L31_mean_max_norm = np.zeros_like(feature_maps_L31_mean_max)
            pass
        output_3 = []
        # output_3.append(feature_maps_L31_mean_norm)
        output_3.append(feature_maps_L31_mean_max_norm)
        output_3.append(feature_maps_L31_norm_mean_max)
        # output_3.append(feature_maps_L31_norm_mean_max)

        feature_maps_L31 = feature_maps[
            0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        # feature_maps_L31 = feature_maps_L31[0]  # [1,512,7,7]==>[512,7,7]
        # print(feature_maps_L31.size())
        # print(torch.from_numpy(highlight_conn_L31).unsqueeze(0).cuda().float().size())
        feature_maps_L31 = feature_maps_L31 * torch.from_numpy(highlight_conn_L31).unsqueeze(0).cuda().float()
        feature_maps_L31_max = global_avg_pool(feature_maps_L31)
        coefficient = (h_L31 * w_L31) / a
        feature_maps_L31_max = coefficient * feature_maps_L31_max
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
            if name in ['layer4.2.conv2.weight', 'layer4.1.conv2.weight']:
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

                tmp_mean_reuse_bool = torch.mean(tmp_mean_reuse, dim=0).unsqueeze(0)
                tmp_mean_reuse_bool = tmp_mean_reuse > tmp_mean_reuse_bool
                tmp_mean_reuse_bool = tmp_mean_reuse_bool.float()

                tmp_mean_reuse_bool_2 = torch.mean(tmp_mean_reuse, dim=1).unsqueeze(0).T
                tmp_mean_reuse_bool_2 = tmp_mean_reuse > tmp_mean_reuse_bool_2
                tmp_mean_reuse_bool_2 = tmp_mean_reuse_bool_2.float()

                tmp_mean_reuse = tmp_mean_reuse * tmp_mean_reuse_bool * tmp_mean_reuse_bool_2
                ###############################################################################################
                # $$$$$$$$$$$$$$$$$$$$$$%%%%%%%%%%%%%%%%%%%%%%^^^^^^^^^^^^^^^^^^^$$$$$$$$$$$$$$$$$$$
                tmp_mean = torch.mean(tmp_mean_reuse, dim=0).cpu().data.numpy()
                # tmp_mean,_ = torch.max(tmp_mean_reuse, dim=0)
                # tmp_mean=tmp_mean.numpy()
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean.append(tmp_mean)
            elif name in ['layer4.2.conv1.weight', 'layer4.1.conv1.weight', 'layer4.2.conv3.weight',
                          'layer4.1.conv3.weight']:
                # print(name)
                tmp = (param.grad).data
                # print(tmp.size())

                # print(tmp)
                tmp = torch.abs(tmp)
                # print(tmp)
                tmp_mean_reuse = tmp.view(tmp.size()[0], -1)

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

        return output_mean, output_2, output_3









class PoolCam_yuhan_kernel_scda_origin_R101_eq5():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, ii=None, flip=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别

        feature_maps, _, _ = self.net(input.cuda())

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

        # highlight_conn_L31 = highlight
        # print('yu',torch.sum(highlight_conn_L31))
        highlight_conn_L31 = torch.tensor(
            largestConnectComponent(highlight.cpu().numpy()) + 0)  # [7*7],我们需要将其变成[512,7,7]
        # print(highlight_conn_L31)
        a = torch.sum(highlight_conn_L31).item()
        # print(a)
        # print(a)
        # print(220/43)
        #       yu待会注释
        #         highlight_index = torch.nonzero(highlight_conn_L31)  # 一个K*2的tensor,每一行一个符合要求的点
        #         a, b = highlight_index.size()
        # print('yu',a)

        feature_maps_L31 = feature_maps_L31[0].cpu().data.numpy()  # [1,512,7,7]==>[512,7,7]
        highlight_conn_L31 = highlight_conn_L31.cpu().data.numpy()
        # print(highlight_conn_L31.shape)
        highlight_conn_L31 = highlight_conn_L31.reshape(1, h_L31, w_L31) * np.ones_like(
            feature_maps_L31)  # [7,7]=>[1,7,7]=>[512,7,7]
        # print(highlight_conn_L31.shape)

        feature_maps_L31 = feature_maps_L31 * highlight_conn_L31

        feature_maps_L31_max = np.max(feature_maps_L31, axis=(1, 2))
        if np.linalg.norm(feature_maps_L31_max) > 0:
            feature_maps_L31_max_norm = feature_maps_L31_max / np.linalg.norm(feature_maps_L31_max)
        else:
            print("出现了")
            feature_maps_L31_max_norm = np.zeros_like(feature_maps_L31_max)
            pass
        feature_maps_L31_mean = np.sum(feature_maps_L31, axis=(1, 2)) / a
        if np.linalg.norm(feature_maps_L31_mean) > 0:
            feature_maps_L31_mean_norm = feature_maps_L31_mean / np.linalg.norm(feature_maps_L31_mean)
        else:
            print("出现了")
            feature_maps_L31_mean_norm = np.zeros_like(feature_maps_L31_mean)
            pass

        feature_maps_L31_mean_max = np.hstack([feature_maps_L31_mean, feature_maps_L31_max])

        if np.linalg.norm(feature_maps_L31_mean_max) > 0:
            feature_maps_L31_mean_max_norm = feature_maps_L31_mean_max / np.linalg.norm(feature_maps_L31_mean_max)
        else:
            print("出现了")
            feature_maps_L31_mean_max_norm = np.zeros_like(feature_maps_L31_mean_max)
            pass

        output_mean = []
        output_2 = []
        output_2.append(np.hstack([feature_maps_L31_mean_norm, feature_maps_L31_max_norm]))  # 先norm再 拼接
        output_2.append(feature_maps_L31_mean_max_norm)  # 先拼接再norm
        # output_2.append(feature_maps_L31_mean_max_norm)  # 先拼接再norm

        feature_maps_L31 = feature_maps[
            0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        feature_maps_L31 = feature_maps_L31[0].cpu().data.numpy()  # [1,512,7,7]==>[512,7,7]
        feature_maps_L31_max = np.max(feature_maps_L31, axis=(1, 2))
        feature_maps_L31_mean = np.mean(feature_maps_L31, axis=(1, 2))
        feature_maps_L31_mean_max = np.hstack([feature_maps_L31_mean, feature_maps_L31_max])

        if np.linalg.norm(feature_maps_L31_max) > 0:
            feature_maps_L31_max_norm = feature_maps_L31_max / np.linalg.norm(feature_maps_L31_max)
        else:
            print("出现了")
            feature_maps_L31_max_norm = np.zeros_like(feature_maps_L31_max)
            pass

        if np.linalg.norm(feature_maps_L31_mean) > 0:
            feature_maps_L31_mean_norm = feature_maps_L31_mean / np.linalg.norm(feature_maps_L31_mean)
        else:
            print("出现了")
            feature_maps_L31_mean_norm = np.zeros_like(feature_maps_L31_mean)
            pass

        feature_maps_L31_norm_mean_max = np.hstack([feature_maps_L31_mean_norm, feature_maps_L31_max_norm])

        if np.linalg.norm(feature_maps_L31_mean_max) > 0:
            feature_maps_L31_mean_max_norm = feature_maps_L31_mean_max / np.linalg.norm(feature_maps_L31_mean_max)
        else:
            print("出现了")
            feature_maps_L31_mean_max_norm = np.zeros_like(feature_maps_L31_mean_max)
            pass
        output_3 = []
        # output_3.append(feature_maps_L31_mean_norm)
        output_3.append(feature_maps_L31_mean_max_norm)
        output_3.append(feature_maps_L31_norm_mean_max)
        # output_3.append(feature_maps_L31_norm_mean_max)

        feature_maps_L31 = feature_maps[
            0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        # feature_maps_L31 = feature_maps_L31[0]  # [1,512,7,7]==>[512,7,7]
        # print(feature_maps_L31.size())
        # print(torch.from_numpy(highlight_conn_L31).unsqueeze(0).cuda().float().size())
        feature_maps_L31 = feature_maps_L31 * torch.from_numpy(highlight_conn_L31).unsqueeze(0).cuda().float()
        feature_maps_L31_max = global_avg_pool(feature_maps_L31)
        coefficient = (h_L31 * w_L31) / a
        feature_maps_L31_max = coefficient * feature_maps_L31_max
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
            if name in ['layer4.2.conv2.weight', 'layer4.1.conv2.weight']:
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

                # tmp_mean_reuse_bool = torch.mean(tmp_mean_reuse, dim=0).unsqueeze(0)
                # tmp_mean_reuse_bool = tmp_mean_reuse > tmp_mean_reuse_bool
                # tmp_mean_reuse_bool = tmp_mean_reuse_bool.float()
                #
                # tmp_mean_reuse_bool_2 = torch.mean(tmp_mean_reuse, dim=1).unsqueeze(0).T
                # tmp_mean_reuse_bool_2 = tmp_mean_reuse > tmp_mean_reuse_bool_2
                # tmp_mean_reuse_bool_2 = tmp_mean_reuse_bool_2.float()
                #
                # tmp_mean_reuse = tmp_mean_reuse * tmp_mean_reuse_bool * tmp_mean_reuse_bool_2
                ###############################################################################################
                # $$$$$$$$$$$$$$$$$$$$$$%%%%%%%%%%%%%%%%%%%%%%^^^^^^^^^^^^^^^^^^^$$$$$$$$$$$$$$$$$$$
                tmp_mean = torch.mean(tmp_mean_reuse, dim=0).cpu().data.numpy()
                # tmp_mean,_ = torch.max(tmp_mean_reuse, dim=0)
                # tmp_mean=tmp_mean.numpy()
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean.append(tmp_mean)
            elif name in ['layer4.2.conv1.weight', 'layer4.1.conv1.weight', 'layer4.2.conv3.weight',
                          'layer4.1.conv3.weight']:
                # print(name)
                tmp = (param.grad).data
                # print(tmp.size())

                # print(tmp)
                tmp = torch.abs(tmp)
                # print(tmp)
                tmp_mean_reuse = tmp.view(tmp.size()[0], -1)

                # tmp_mean_reuse_bool = torch.mean(tmp_mean_reuse, dim=0).unsqueeze(0)
                # tmp_mean_reuse_bool = tmp_mean_reuse > tmp_mean_reuse_bool
                # tmp_mean_reuse_bool = tmp_mean_reuse_bool.float()
                #
                # tmp_mean_reuse_bool_2 = torch.mean(tmp_mean_reuse, dim=1).unsqueeze(0).T
                # tmp_mean_reuse_bool_2 = tmp_mean_reuse > tmp_mean_reuse_bool_2
                # tmp_mean_reuse_bool_2 = tmp_mean_reuse_bool_2.float()
                #
                # tmp_mean_reuse = tmp_mean_reuse * tmp_mean_reuse_bool * tmp_mean_reuse_bool_2
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

        return output_mean, output_2, output_3













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
            if name in ['layer4.2.conv2.weight','layer4.1.conv2.weight']:
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
            elif name in ['layer4.2.conv1.weight','layer4.1.conv1.weight','layer4.2.conv3.weight','layer4.1.conv3.weight']:
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








class PoolCam_yuhan_kernel_scda_origin_no_lcc():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, ii=None,flip=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别

        feature_maps,output,_ = self.net(input.cuda())
        # output_max = torch.max(output)
        # output_min = torch.min(output)
        # output_norm = torch.div((output - output_min).float(), (output_max - output_min))
        # ################################################################3
        # output_norm = output_norm .cpu().tolist()
        # output_norm = torch.tensor(output_norm).float().cuda()
        # ###################################################################
        # one_hot = torch.sum(output * output_norm)
        # self.net.zero_grad()
        # one_hot.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚
        # # params = list(self.net.parameters())
        # # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #

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
        # print(a)
        # print(a)
        # print(220/43)
#       yu待会注释
#         highlight_index = torch.nonzero(highlight_conn_L31)  # 一个K*2的tensor,每一行一个符合要求的点
#         a, b = highlight_index.size()
        # print('yu',a)

        feature_maps_L31 = feature_maps_L31[0].cpu().data.numpy()  # [1,512,7,7]==>[512,7,7]
        highlight_conn_L31 = highlight_conn_L31.cpu().data.numpy()
        # print(highlight_conn_L31.shape)
        highlight_conn_L31 = highlight_conn_L31.reshape(1, h_L31, w_L31) * np.ones_like(
            feature_maps_L31)  # [7,7]=>[1,7,7]=>[512,7,7]
        # print(highlight_conn_L31.shape)

        feature_maps_L31 = feature_maps_L31 * highlight_conn_L31


        feature_maps_L31_max = np.max(feature_maps_L31, axis=(1, 2))
        if np.linalg.norm(feature_maps_L31_max) > 0:
            feature_maps_L31_max_norm = feature_maps_L31_max / np.linalg.norm(feature_maps_L31_max)
        else:
            print("出现了")
            feature_maps_L31_max_norm = np.zeros_like(feature_maps_L31_max)
            pass
        feature_maps_L31_mean = np.sum(feature_maps_L31, axis=(1, 2))/a
        if np.linalg.norm(feature_maps_L31_mean) > 0:
            feature_maps_L31_mean_norm = feature_maps_L31_mean / np.linalg.norm(feature_maps_L31_mean)
        else:
            print("出现了")
            feature_maps_L31_mean_norm = np.zeros_like(feature_maps_L31_mean)
            pass

        feature_maps_L31_mean_max = np.hstack([feature_maps_L31_mean, feature_maps_L31_max])

        if np.linalg.norm(feature_maps_L31_mean_max) > 0:
            feature_maps_L31_mean_max_norm = feature_maps_L31_mean_max / np.linalg.norm(feature_maps_L31_mean_max)
        else:
            print("出现了")
            feature_maps_L31_mean_max_norm = np.zeros_like(feature_maps_L31_mean_max)
            pass


        output_mean = []
        output_2=[]
        output_2.append(np.hstack([feature_maps_L31_mean_norm,feature_maps_L31_max_norm]))#先norm再 拼接
        output_2.append(feature_maps_L31_mean_max_norm)  # 先拼接再norm
        # output_2.append(feature_maps_L31_mean_max_norm)  # 先拼接再norm

        feature_maps_L31 = feature_maps[
            0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        feature_maps_L31 = feature_maps_L31[0].cpu().data.numpy()  # [1,512,7,7]==>[512,7,7]
        feature_maps_L31_max = np.max(feature_maps_L31, axis=(1, 2))
        feature_maps_L31_mean = np.mean(feature_maps_L31, axis=(1, 2))
        feature_maps_L31_mean_max = np.hstack([feature_maps_L31_mean, feature_maps_L31_max])

        if np.linalg.norm(feature_maps_L31_max) > 0:
            feature_maps_L31_max_norm = feature_maps_L31_max / np.linalg.norm(feature_maps_L31_max)
        else:
            print("出现了")
            feature_maps_L31_max_norm = np.zeros_like(feature_maps_L31_max)
            pass

        if np.linalg.norm(feature_maps_L31_mean) > 0:
            feature_maps_L31_mean_norm = feature_maps_L31_mean / np.linalg.norm(feature_maps_L31_mean)
        else:
            print("出现了")
            feature_maps_L31_mean_norm = np.zeros_like(feature_maps_L31_mean)
            pass

        feature_maps_L31_norm_mean_max = np.hstack([feature_maps_L31_mean_norm, feature_maps_L31_max_norm])

        if np.linalg.norm(feature_maps_L31_mean_max) > 0:
            feature_maps_L31_mean_max_norm = feature_maps_L31_mean_max / np.linalg.norm(feature_maps_L31_mean_max)
        else:
            print("出现了")
            feature_maps_L31_mean_max_norm = np.zeros_like(feature_maps_L31_mean_max)
            pass
        output_3 = []
        # output_3.append(feature_maps_L31_mean_norm)
        output_3.append(feature_maps_L31_mean_max_norm)
        output_3.append(feature_maps_L31_norm_mean_max)
        # output_3.append(feature_maps_L31_norm_mean_max)

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
            if name in ['layer4.2.conv2.weight','layer4.1.conv2.weight']:
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
            elif name in ['layer4.2.conv1.weight','layer4.1.conv1.weight','layer4.2.conv3.weight','layer4.1.conv3.weight']:
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

        return output_mean,output_2,output_3








class PoolCam_yuhan_kernel_scda_weight():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, ii=None,flip=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别

        feature_maps,output,_ = self.net(input.cuda())
        # output_max = torch.max(output)
        # output_min = torch.min(output)
        # output_norm = torch.div((output - output_min).float(), (output_max - output_min))
        # ################################################################3
        # output_norm = output_norm .cpu().tolist()
        # output_norm = torch.tensor(output_norm).float().cuda()
        # ###################################################################
        # one_hot = torch.sum(output * output_norm)
        # self.net.zero_grad()
        # one_hot.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚
        # # params = list(self.net.parameters())
        # # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #


        feature_maps_L31 = feature_maps[
            0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        batch_size, c_L31, h_L31, w_L31 = feature_maps_L31.size()

        # Pool5   #feature_maps_L31  [512,7,7]==[512,]
        feature_maps_L31_sum = torch.sum(feature_maps_L31[0],
                                         0)  # 如果要将CAM或者gard_cam引入SCDA，需要更改的程序也就仅仅是这一步
        L31_sum_mean = torch.mean(feature_maps_L31_sum)  # 返回的是scalar tensor ,这是与matlab的mean不同的地方
        # largestConnectComponent将最大连通区域所对应的像素点置为true
        jj = (feature_maps_L31_sum > L31_sum_mean).cpu().data.numpy()
        highlight = torch.from_numpy(jj + 0).cuda().float()

        highlight_conn_L31 = highlight
        # highlight_conn_L31 = torch.tensor(largestConnectComponent(highlight.cpu().numpy()) + 0)#[7*7],我们需要将其变成[512,7,7]

        feature_maps_L31 = feature_maps_L31[0].cpu().data.numpy()  # [1,512,7,7]==>[512,7,7]
        highlight_conn_L31 = highlight_conn_L31.cpu().data.numpy()
        highlight_conn_L31 = highlight_conn_L31.reshape(1, h_L31, w_L31) * np.ones_like(
            feature_maps_L31)  # [7,7]=>[1,7,7]=>[512,7,7]

        feature_maps_L31 = feature_maps_L31 * highlight_conn_L31


        feature_maps_L31_max = np.max(feature_maps_L31, axis=(1, 2))
        if np.linalg.norm(feature_maps_L31_max) > 0:
            feature_maps_L31_max_norm = feature_maps_L31_max / np.linalg.norm(feature_maps_L31_max)
        else:
            print("出现了")
            feature_maps_L31_max_norm = np.zeros_like(feature_maps_L31_max)
            pass
        feature_maps_L31_mean = np.mean(feature_maps_L31, axis=(1, 2))
        if np.linalg.norm(feature_maps_L31_mean) > 0:
            feature_maps_L31_mean_norm = feature_maps_L31_mean / np.linalg.norm(feature_maps_L31_mean)
        else:
            print("出现了")
            feature_maps_L31_mean_norm = np.zeros_like(feature_maps_L31_mean)
            pass

        feature_maps_L31_mean_max = np.hstack([feature_maps_L31_mean, feature_maps_L31_max])

        if np.linalg.norm(feature_maps_L31_mean_max) > 0:
            feature_maps_L31_mean_max_norm = feature_maps_L31_mean_max / np.linalg.norm(feature_maps_L31_mean_max)
        else:
            print("出现了")
            feature_maps_L31_mean_max_norm = np.zeros_like(feature_maps_L31_mean_max)
            pass


        output_mean = []
        output_2=[]
        output_2.append(np.hstack([feature_maps_L31_mean_norm,feature_maps_L31_max_norm]))#先norm再 拼接
        output_2.append(feature_maps_L31_mean_max_norm)  # 先拼接再norm

        feature_maps_L31 = feature_maps[
            0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        feature_maps_L31 = feature_maps_L31[0].cpu().data.numpy()  # [1,512,7,7]==>[512,7,7]
        feature_maps_L31_max = np.max(feature_maps_L31, axis=(1, 2))
        feature_maps_L31_mean = np.mean(feature_maps_L31, axis=(1, 2))
        feature_maps_L31_mean_max = np.hstack([feature_maps_L31_mean, feature_maps_L31_max])

        if np.linalg.norm(feature_maps_L31_max) > 0:
            feature_maps_L31_max_norm = feature_maps_L31_max / np.linalg.norm(feature_maps_L31_max)
        else:
            print("出现了")
            feature_maps_L31_max_norm = np.zeros_like(feature_maps_L31_max)
            pass

        if np.linalg.norm(feature_maps_L31_mean) > 0:
            feature_maps_L31_mean_norm = feature_maps_L31_mean / np.linalg.norm(feature_maps_L31_mean)
        else:
            print("出现了")
            feature_maps_L31_mean_norm = np.zeros_like(feature_maps_L31_mean)
            pass

        feature_maps_L31_norm_mean_max = np.hstack([feature_maps_L31_mean_norm, feature_maps_L31_max_norm])

        if np.linalg.norm(feature_maps_L31_mean_max) > 0:
            feature_maps_L31_mean_max_norm = feature_maps_L31_mean_max / np.linalg.norm(feature_maps_L31_mean_max)
        else:
            print("出现了")
            feature_maps_L31_mean_max_norm = np.zeros_like(feature_maps_L31_mean_max)
            pass
        output_3 = []
        # output_3.append(feature_maps_L31_mean_norm)
        output_3.append(feature_maps_L31_mean_max_norm)
        output_3.append(feature_maps_L31_norm_mean_max)

        feature_maps_L31 = feature_maps[
            0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        # feature_maps_L31 = feature_maps_L31[0]  # [1,512,7,7]==>[512,7,7]
        # print(feature_maps_L31.size())
        # print(torch.from_numpy(highlight_conn_L31).unsqueeze(0).cuda().float().size())
        feature_maps_L31 = feature_maps_L31 * torch.from_numpy(highlight_conn_L31).unsqueeze(0).cuda().float()
        feature_maps_L31_max = global_avg_pool(feature_maps_L31)
        feature_maps_L31_mean = global_max_pool(feature_maps_L31)
        feature_maps_L31_mean_max = torch.cat((feature_maps_L31_mean, feature_maps_L31_max), dim=1)
        feature_maps_L31_mean_max_weight=torch.from_numpy(feature_maps_L31_mean_max.cpu().data.numpy()).cuda()
        feature_maps_L31_mean_max=feature_maps_L31_mean_max * feature_maps_L31_mean_max_weight

        loss2 = torch.sum(feature_maps_L31_mean_max)
        self.net.zero_grad()
        loss2.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚

        # output_max=[]
        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['layer4.2.conv2.weight','layer4.1.conv2.weight']:
                # print(name)
                tmp = (param.grad).data
                # print(tmp)
                # tmp=torch.abs(tmp)
                # print(tmp)
                # tmp_mean_reuse=tmp.view(tmp.size()[0],-1)
                # print(tmp_mean_reuse.shape)
                tmp_mean_reuse_std = torch.std(tmp, dim=(2, 3))

                tmp_mean_reuse = F.adaptive_max_pool2d(tmp, output_size=1).squeeze(-1).squeeze(
                    -1) + F.adaptive_max_pool2d(-1 * tmp, output_size=1).squeeze(-1).squeeze(-1)
                tmp_mean_reuse = tmp_mean_reuse + tmp_mean_reuse_std
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
            elif name in ['layer4.2.conv1.weight','layer4.1.conv1.weight','layer4.2.conv3.weight','layer4.1.conv3.weight']:
                # print(name)
                tmp = (param.grad).data
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

        return output_mean,output_2,output_3







class GradCam_yuhan_kernel_version30_DGPCRL_scda_many_4():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, ii=None,flip=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别

        feature_maps,output,_ = self.net(input.cuda())
        output_max = torch.max(output)
        output_min = torch.min(output)
        output_norm = torch.div((output - output_min).float(), (output_max - output_min))
        ################################################################3
        output_norm = output_norm .cpu().tolist()
        output_norm = torch.tensor(output_norm).float().cuda()
        ###################################################################
        one_hot = torch.sum(output * output_norm)
        self.net.zero_grad()
        one_hot.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #


        feature_maps_L31 = feature_maps[
            0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        batch_size, c_L31, h_L31, w_L31 = feature_maps_L31.size()

        # Pool5   #feature_maps_L31  [512,7,7]==[512,]
        feature_maps_L31_sum = torch.sum(feature_maps_L31[0],
                                         0)  # 如果要将CAM或者gard_cam引入SCDA，需要更改的程序也就仅仅是这一步
        L31_sum_mean = torch.mean(feature_maps_L31_sum)  # 返回的是scalar tensor ,这是与matlab的mean不同的地方
        # largestConnectComponent将最大连通区域所对应的像素点置为true
        jj = (feature_maps_L31_sum > L31_sum_mean).cpu().data.numpy()
        highlight = torch.from_numpy(jj + 0).cuda().float()

        highlight_conn_L31 = highlight
        # highlight_conn_L31 = torch.tensor(largestConnectComponent(highlight.cpu().numpy()) + 0)#[7*7],我们需要将其变成[512,7,7]

        feature_maps_L31 = feature_maps_L31[0].cpu().data.numpy()  # [1,512,7,7]==>[512,7,7]
        highlight_conn_L31 = highlight_conn_L31.cpu().data.numpy()
        highlight_conn_L31 = highlight_conn_L31.reshape(1, h_L31, w_L31) * np.ones_like(
            feature_maps_L31)  # [7,7]=>[1,7,7]=>[512,7,7]

        feature_maps_L31 = feature_maps_L31 * highlight_conn_L31


        feature_maps_L31_max = np.max(feature_maps_L31, axis=(1, 2))
        if np.linalg.norm(feature_maps_L31_max) > 0:
            feature_maps_L31_max_norm = feature_maps_L31_max / np.linalg.norm(feature_maps_L31_max)
        else:
            print("出现了")
            feature_maps_L31_max_norm = np.zeros_like(feature_maps_L31_max)
            pass
        feature_maps_L31_mean = np.mean(feature_maps_L31, axis=(1, 2))
        if np.linalg.norm(feature_maps_L31_mean) > 0:
            feature_maps_L31_mean_norm = feature_maps_L31_mean / np.linalg.norm(feature_maps_L31_mean)
        else:
            print("出现了")
            feature_maps_L31_mean_norm = np.zeros_like(feature_maps_L31_mean)
            pass

        feature_maps_L31_mean_max = np.hstack([feature_maps_L31_mean, feature_maps_L31_max])

        if np.linalg.norm(feature_maps_L31_mean_max) > 0:
            feature_maps_L31_mean_max_norm = feature_maps_L31_mean_max / np.linalg.norm(feature_maps_L31_mean_max)
        else:
            print("出现了")
            feature_maps_L31_mean_max_norm = np.zeros_like(feature_maps_L31_mean_max)
            pass


        output_mean = []
        output_2=[]
        output_2.append(np.hstack([feature_maps_L31_mean_norm,feature_maps_L31_max_norm]))#先norm再 拼接
        output_2.append(feature_maps_L31_mean_max_norm)  # 先拼接再norm


        feature_maps_L31 = feature_maps[
            0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        feature_maps_L31 = feature_maps_L31[0].cpu().data.numpy()  # [1,512,7,7]==>[512,7,7]
        feature_maps_L31_max = np.max(feature_maps_L31, axis=(1, 2))
        feature_maps_L31_mean = np.mean(feature_maps_L31, axis=(1, 2))
        feature_maps_L31_mean_max=np.hstack([feature_maps_L31_mean,feature_maps_L31_max])

        if np.linalg.norm(feature_maps_L31_max) > 0:
            feature_maps_L31_max_norm = feature_maps_L31_max / np.linalg.norm(feature_maps_L31_max)
        else:
            print("出现了")
            feature_maps_L31_max_norm = np.zeros_like(feature_maps_L31_max)
            pass

        if np.linalg.norm(feature_maps_L31_mean) > 0:
            feature_maps_L31_mean_norm = feature_maps_L31_mean / np.linalg.norm(feature_maps_L31_mean)
        else:
            print("出现了")
            feature_maps_L31_mean_norm = np.zeros_like(feature_maps_L31_mean)
            pass

        feature_maps_L31_norm_mean_max = np.hstack([feature_maps_L31_mean_norm, feature_maps_L31_max_norm])

        if np.linalg.norm(feature_maps_L31_mean_max) > 0:
            feature_maps_L31_mean_max_norm = feature_maps_L31_mean_max / np.linalg.norm(feature_maps_L31_mean_max)
        else:
            print("出现了")
            feature_maps_L31_mean_max_norm = np.zeros_like(feature_maps_L31_mean_max)
            pass
        output_3 = []
        # output_3.append(feature_maps_L31_mean_norm)
        output_3.append(feature_maps_L31_mean_max_norm)
        output_3.append(feature_maps_L31_norm_mean_max)

        # output_max=[]
        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['layer4.2.conv2.weight','layer4.1.conv2.weight']:
                # print(name)
                tmp = (param.grad).data
                # print(tmp)
                # tmp=torch.abs(tmp)
                # print(tmp)
                # tmp_mean_reuse=tmp.view(tmp.size()[0],-1)
                # print(tmp_mean_reuse.shape)
                tmp_mean_reuse_std = torch.std(tmp, dim=(2, 3))

                tmp_mean_reuse = F.adaptive_max_pool2d(tmp, output_size=1).squeeze(-1).squeeze(
                    -1) + F.adaptive_max_pool2d(-1 * tmp, output_size=1).squeeze(-1).squeeze(-1)
                tmp_mean_reuse = tmp_mean_reuse + tmp_mean_reuse_std
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
            elif name in ['layer4.2.conv1.weight','layer4.1.conv1.weight','layer4.2.conv3.weight','layer4.1.conv3.weight']:
                # print(name)
                tmp = (param.grad).data
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

        return output_mean,output_2,output_3













class GradCam_yuhan_kernel_version30_DGCRL_scda_many_4():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, ii=None,flip=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别

        feature_maps,output,_ = self.net(input.cuda())
        output_max = torch.max(output)
        output_min = torch.min(output)
        output_norm = torch.div((output - output_min).float(), (output_max - output_min))
        ################################################################3
        output_norm = output_norm .cpu().tolist()
        output_norm = torch.tensor(output_norm).float().cuda()
        ###################################################################
        one_hot = torch.sum(output * output_norm)
        self.net.zero_grad()
        one_hot.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #


        feature_maps_L31 = feature_maps[
            0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        batch_size, c_L31, h_L31, w_L31 = feature_maps_L31.size()

        # Pool5   #feature_maps_L31  [512,7,7]==[512,]
        feature_maps_L31_sum = torch.sum(feature_maps_L31[0],
                                         0)  # 如果要将CAM或者gard_cam引入SCDA，需要更改的程序也就仅仅是这一步
        L31_sum_mean = torch.mean(feature_maps_L31_sum)  # 返回的是scalar tensor ,这是与matlab的mean不同的地方
        # largestConnectComponent将最大连通区域所对应的像素点置为true
        jj = (feature_maps_L31_sum > L31_sum_mean).cpu().data.numpy()
        highlight = torch.from_numpy(jj + 0).cuda().float()

        highlight_conn_L31 = highlight
        # highlight_conn_L31 = torch.tensor(largestConnectComponent(highlight.cpu().numpy()) + 0)#[7*7],我们需要将其变成[512,7,7]

        feature_maps_L31 = feature_maps_L31[0].cpu().data.numpy()  # [1,512,7,7]==>[512,7,7]
        highlight_conn_L31 = highlight_conn_L31.cpu().data.numpy()
        highlight_conn_L31 = highlight_conn_L31.reshape(1, h_L31, w_L31) * np.ones_like(
            feature_maps_L31)  # [7,7]=>[1,7,7]=>[512,7,7]

        feature_maps_L31 = feature_maps_L31 * highlight_conn_L31


        feature_maps_L31_max = np.max(feature_maps_L31, axis=(1, 2))
        if np.linalg.norm(feature_maps_L31_max) > 0:
            feature_maps_L31_max_norm = feature_maps_L31_max / np.linalg.norm(feature_maps_L31_max)
        else:
            print("出现了")
            feature_maps_L31_max_norm = np.zeros_like(feature_maps_L31_max)
            pass

        feature_maps_L31_mean = np.mean(feature_maps_L31, axis=(1, 2))
        if np.linalg.norm(feature_maps_L31_mean) > 0:
            feature_maps_L31_mean_norm = feature_maps_L31_mean / np.linalg.norm(feature_maps_L31_mean)
        else:
            print("出现了")
            feature_maps_L31_mean_norm = np.zeros_like(feature_maps_L31_mean)
            pass

        output_mean = []
        output_2=[]
        output_2.append(feature_maps_L31_mean_norm)
        output_2.append(feature_maps_L31_max_norm)


        feature_maps_L31 = feature_maps[
            0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        feature_maps_L31 = feature_maps_L31[0].cpu().data.numpy()  # [1,512,7,7]==>[512,7,7]
        feature_maps_L31_max = np.max(feature_maps_L31, axis=(1, 2))
        if np.linalg.norm(feature_maps_L31_max) > 0:
            feature_maps_L31_max_norm = feature_maps_L31_max / np.linalg.norm(feature_maps_L31_max)
        else:
            print("出现了")
            feature_maps_L31_max_norm = np.zeros_like(feature_maps_L31_max)
            pass
        feature_maps_L31_mean = np.mean(feature_maps_L31, axis=(1, 2))
        if np.linalg.norm(feature_maps_L31_mean) > 0:
            feature_maps_L31_mean_norm = feature_maps_L31_mean / np.linalg.norm(feature_maps_L31_mean)
        else:
            print("出现了")
            feature_maps_L31_mean_norm = np.zeros_like(feature_maps_L31_mean)
            pass
        output_3 = []
        output_3.append(feature_maps_L31_mean_norm)
        output_3.append(feature_maps_L31_max_norm)

        # output_max=[]
        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['layer4.2.conv2.weight','layer4.1.conv2.weight']:
                # print(name)
                tmp = (param.grad).data
                # print(tmp)
                # tmp=torch.abs(tmp)
                # print(tmp)
                # tmp_mean_reuse=tmp.view(tmp.size()[0],-1)
                # print(tmp_mean_reuse.shape)
                tmp_mean_reuse_std = torch.std(tmp, dim=(2, 3))

                tmp_mean_reuse = F.adaptive_max_pool2d(tmp, output_size=1).squeeze(-1).squeeze(
                    -1) + F.adaptive_max_pool2d(-1 * tmp, output_size=1).squeeze(-1).squeeze(-1)
                tmp_mean_reuse = tmp_mean_reuse + tmp_mean_reuse_std
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
            elif name in ['layer4.2.conv1.weight','layer4.1.conv1.weight','layer4.2.conv3.weight','layer4.1.conv3.weight']:
                # print(name)
                tmp = (param.grad).data
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

        return output_mean,output_2,output_3









class GradCam_yuhan_kernel_version30_DGCRL_scda_many_5():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, ii=None,flip=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别

        feature_maps,output,_ = self.net(input.cuda())
        output_max = torch.max(output)
        output_min = torch.min(output)
        output_norm = torch.div((output - output_min).float(), (output_max - output_min))
        ################################################################3
        output_norm = output_norm .cpu().tolist()
        output_norm = torch.tensor(output_norm).float().cuda()
        ###################################################################
        one_hot = torch.sum(output * output_norm)
        self.net.zero_grad()
        one_hot.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #


        feature_maps_L31 = feature_maps[
            0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        batch_size, c_L31, h_L31, w_L31 = feature_maps_L31.size()

        # Pool5   #feature_maps_L31  [512,7,7]==[512,]
        feature_maps_L31_sum = torch.sum(feature_maps_L31[0],
                                         0)  # 如果要将CAM或者gard_cam引入SCDA，需要更改的程序也就仅仅是这一步
        L31_sum_mean = torch.mean(feature_maps_L31_sum)  # 返回的是scalar tensor ,这是与matlab的mean不同的地方
        # largestConnectComponent将最大连通区域所对应的像素点置为true
        jj = (feature_maps_L31_sum > L31_sum_mean).cpu().data.numpy()
        highlight = torch.from_numpy(jj + 0).cuda().float()

        # highlight_conn_L31 = highlight
        highlight_conn_L31 = torch.tensor(largestConnectComponent(highlight.cpu().numpy()) + 0)#[7*7],我们需要将其变成[512,7,7]

        feature_maps_L31 = feature_maps_L31[0].cpu().data.numpy()  # [1,512,7,7]==>[512,7,7]
        highlight_conn_L31 = highlight_conn_L31.cpu().data.numpy()
        highlight_conn_L31 = highlight_conn_L31.reshape(1, h_L31, w_L31) * np.ones_like(
            feature_maps_L31)  # [7,7]=>[1,7,7]=>[512,7,7]

        feature_maps_L31 = feature_maps_L31 * highlight_conn_L31


        feature_maps_L31_max = np.max(feature_maps_L31, axis=(1, 2))
        if np.linalg.norm(feature_maps_L31_max) > 0:
            feature_maps_L31_max_norm = feature_maps_L31_max / np.linalg.norm(feature_maps_L31_max)
        else:
            print("出现了")
            feature_maps_L31_max_norm = np.zeros_like(feature_maps_L31_max)
            pass

        feature_maps_L31_mean = np.mean(feature_maps_L31, axis=(1, 2))
        if np.linalg.norm(feature_maps_L31_mean) > 0:
            feature_maps_L31_mean_norm = feature_maps_L31_mean / np.linalg.norm(feature_maps_L31_mean)
        else:
            print("出现了")
            feature_maps_L31_mean_norm = np.zeros_like(feature_maps_L31_mean)
            pass

        output_mean = []
        output_2=[]
        output_2.append(feature_maps_L31_mean_norm)
        output_2.append(feature_maps_L31_max_norm)


        feature_maps_L31 = feature_maps[
            0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        feature_maps_L31 = feature_maps_L31[0].cpu().data.numpy()  # [1,512,7,7]==>[512,7,7]
        feature_maps_L31_max = np.max(feature_maps_L31, axis=(1, 2))
        if np.linalg.norm(feature_maps_L31_max) > 0:
            feature_maps_L31_max_norm = feature_maps_L31_max / np.linalg.norm(feature_maps_L31_max)
        else:
            print("出现了")
            feature_maps_L31_max_norm = np.zeros_like(feature_maps_L31_max)
            pass
        feature_maps_L31_mean = np.mean(feature_maps_L31, axis=(1, 2))
        if np.linalg.norm(feature_maps_L31_mean) > 0:
            feature_maps_L31_mean_norm = feature_maps_L31_mean / np.linalg.norm(feature_maps_L31_mean)
        else:
            print("出现了")
            feature_maps_L31_mean_norm = np.zeros_like(feature_maps_L31_mean)
            pass
        output_3 = []
        output_3.append(feature_maps_L31_mean_norm)
        output_3.append(feature_maps_L31_max_norm)

        # output_max=[]
        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['layer4.2.conv2.weight','layer4.1.conv2.weight']:
                # print(name)
                tmp = (param.grad).data
                # print(tmp)
                # tmp=torch.abs(tmp)
                # print(tmp)
                # tmp_mean_reuse=tmp.view(tmp.size()[0],-1)
                # print(tmp_mean_reuse.shape)
                tmp_mean_reuse_std = torch.std(tmp, dim=(2, 3))

                tmp_mean_reuse = F.adaptive_max_pool2d(tmp, output_size=1).squeeze(-1).squeeze(
                    -1) + F.adaptive_max_pool2d(-1 * tmp, output_size=1).squeeze(-1).squeeze(-1)
                tmp_mean_reuse = tmp_mean_reuse + tmp_mean_reuse_std
                # tmp_mean_reuse=F.adaptive_max_pool2d(tmp,output_size=1).squeeze(-1).squeeze(-1)
                # tmp_mean_reuse=F.adaptive_max_pool2d(-1 * tmp,output_size=1).squeeze(-1).squeeze(-1)

                # tmp_mean_reuse_1 = torch.mean(tmp_1, axis=(2, 3))
                # tmp_mean_reuse = tmp_mean_reuse * tmp_mean_reuse_1
                # tmp_mean_reuse = torch.mean(tmp, dim=(2, 3))
                ############################################################
                #$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%%$%$%$$%$$%$%$%$%$%$%$
                #待会回复注释
                # tmp_mean_reuse_bool = torch.mean(tmp_mean_reuse, dim=0).unsqueeze(0)
                # tmp_mean_reuse_bool = tmp_mean_reuse > tmp_mean_reuse_bool
                # tmp_mean_reuse_bool = tmp_mean_reuse_bool.float()
                #
                # tmp_mean_reuse_bool_2 = torch.mean(tmp_mean_reuse, dim=1).unsqueeze(0).T
                # tmp_mean_reuse_bool_2 = tmp_mean_reuse > tmp_mean_reuse_bool_2
                # tmp_mean_reuse_bool_2 = tmp_mean_reuse_bool_2.float()
                #
                # tmp_mean_reuse = tmp_mean_reuse * tmp_mean_reuse_bool * tmp_mean_reuse_bool_2
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
            elif name in ['layer4.2.conv1.weight','layer4.1.conv1.weight','layer4.2.conv3.weight','layer4.1.conv3.weight']:
                # print(name)
                tmp = (param.grad).data
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
                # tmp_mean_reuse_bool = torch.mean(tmp_mean_reuse, dim=0).unsqueeze(0)
                # tmp_mean_reuse_bool = tmp_mean_reuse > tmp_mean_reuse_bool
                # tmp_mean_reuse_bool = tmp_mean_reuse_bool.float()
                #
                # tmp_mean_reuse_bool_2 = torch.mean(tmp_mean_reuse, dim=1).unsqueeze(0).T
                # tmp_mean_reuse_bool_2 = tmp_mean_reuse > tmp_mean_reuse_bool_2
                # tmp_mean_reuse_bool_2 = tmp_mean_reuse_bool_2.float()
                #
                # tmp_mean_reuse = tmp_mean_reuse * tmp_mean_reuse_bool * tmp_mean_reuse_bool_2
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

        return output_mean,output_2,output_3





class GradCam_yuhan_kernel_version30_DGCRL_scda_many_6():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, ii=None,flip=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别

        feature_maps,output,_ = self.net(input.cuda())
        # output_max = torch.max(output)
        # output_min = torch.min(output)
        # output_norm = torch.div((output - output_min).float(), (output_max - output_min))
        # ################################################################3
        # output_norm = output_norm .cpu().tolist()
        # output_norm = torch.tensor(output_norm).float().cuda()
        # ###################################################################
        # one_hot = torch.sum(output * output_norm)
        # self.net.zero_grad()
        # one_hot.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #
        feature_maps_L31 = feature_maps[
            0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        feature_maps_L31_max = feature_maps_L31.cpu().data.numpy()
        if np.linalg.norm(feature_maps_L31_max) > 0:
            feature_maps_L31_max_norm = feature_maps_L31_max / np.linalg.norm(feature_maps_L31_max)
        else:
            print("出现了")
            feature_maps_L31_max_norm = np.zeros_like(feature_maps_L31_max)
            pass
        output_3 = []
        output_3.append(feature_maps_L31_max_norm)
        output_mean=[]


        ################################################################3
        ###################################################################
        one_hot = torch.sum(feature_maps_L31* torch.from_numpy(feature_maps_L31.cpu().data.numpy()).cuda())
        self.net.zero_grad()
        one_hot.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚


        # output_max=[]
        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['layer4.2.conv2.weight','layer4.1.conv2.weight']:
                tmp = (param.grad).data
                tmp_mean_reuse_std = torch.std(tmp, dim=(2, 3))
                tmp_mean_reuse = F.adaptive_max_pool2d(tmp, output_size=1).squeeze(-1).squeeze(
                    -1) + F.adaptive_max_pool2d(-1 * tmp, output_size=1).squeeze(-1).squeeze(-1)
                tmp_mean_reuse = tmp_mean_reuse + tmp_mean_reuse_std

                ############################################################
                #$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%%$%$%$$%$$%$%$%$%$%$%$
                #待会回复注释
                # tmp_mean_reuse_bool = torch.mean(tmp_mean_reuse, dim=0).unsqueeze(0)
                # tmp_mean_reuse_bool = tmp_mean_reuse > tmp_mean_reuse_bool
                # tmp_mean_reuse_bool = tmp_mean_reuse_bool.float()
                #
                # tmp_mean_reuse_bool_2 = torch.mean(tmp_mean_reuse, dim=1).unsqueeze(0).T
                # tmp_mean_reuse_bool_2 = tmp_mean_reuse > tmp_mean_reuse_bool_2
                # tmp_mean_reuse_bool_2 = tmp_mean_reuse_bool_2.float()
                #
                # tmp_mean_reuse = tmp_mean_reuse * tmp_mean_reuse_bool * tmp_mean_reuse_bool_2
###############################################################################################
                #$$$$$$$$$$$$$$$$$$$$$$%%%%%%%%%%%%%%%%%%%%%%^^^^^^^^^^^^^^^^^^^$$$$$$$$$$$$$$$$$$$
                tmp_mean = torch.mean(tmp_mean_reuse, dim=0).cpu().data.numpy()
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean.append(tmp_mean)
            elif name in ['layer4.2.conv1.weight','layer4.1.conv1.weight','layer4.2.conv3.weight','layer4.1.conv3.weight']:
                # print(name)
                tmp = (param.grad).data
                # print(tmp)
                tmp=torch.abs(tmp)
                # print(tmp)
                tmp_mean_reuse=tmp.view(tmp.size()[0],-1)

                ########################################
                # tmp_mean_reuse_bool = torch.mean(tmp_mean_reuse, dim=0).unsqueeze(0)
                # tmp_mean_reuse_bool = tmp_mean_reuse > tmp_mean_reuse_bool
                # tmp_mean_reuse_bool = tmp_mean_reuse_bool.float()
                #
                # tmp_mean_reuse_bool_2 = torch.mean(tmp_mean_reuse, dim=1).unsqueeze(0).T
                # tmp_mean_reuse_bool_2 = tmp_mean_reuse > tmp_mean_reuse_bool_2
                # tmp_mean_reuse_bool_2 = tmp_mean_reuse_bool_2.float()
                #
                # tmp_mean_reuse = tmp_mean_reuse * tmp_mean_reuse_bool * tmp_mean_reuse_bool_2
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













class GradCam_yuhan_kernel_version30_DGCRL_scda_many_2():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, ii=None,flip=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别

        feature_maps,output,_ = self.net(input.cuda())
        output_max = torch.max(output)
        output_min = torch.min(output)
        output_norm = torch.div((output - output_min).float(), (output_max - output_min))
        ################################################################3
        output_norm = output_norm .cpu().tolist()
        output_norm = torch.tensor(output_norm).float().cuda()
        output_norm = torch.exp(output_norm)

        ###################################################################
        one_hot = torch.sum(output * output_norm)
        # one_hot = torch.sum(output )

        self.net.zero_grad()
        one_hot.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #


        feature_maps_L31 = feature_maps[
            0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        batch_size, c_L31, h_L31, w_L31 = feature_maps_L31.size()

        # Pool5   #feature_maps_L31  [512,7,7]==[512,]
        feature_maps_L31_sum = torch.sum(feature_maps_L31[0],
                                         0)  # 如果要将CAM或者gard_cam引入SCDA，需要更改的程序也就仅仅是这一步
        L31_sum_mean = torch.mean(feature_maps_L31_sum)  # 返回的是scalar tensor ,这是与matlab的mean不同的地方
        # largestConnectComponent将最大连通区域所对应的像素点置为true
        jj = (feature_maps_L31_sum > L31_sum_mean).cpu().data.numpy()
        highlight = torch.from_numpy(jj + 0).cuda().float()

        # highlight_conn_L31 = highlight
        highlight_conn_L31 = torch.tensor(largestConnectComponent(highlight.cpu().numpy()) + 0)#[7*7],我们需要将其变成[512,7,7]

        feature_maps_L31 = feature_maps_L31[0].cpu().data.numpy()  # [1,512,7,7]==>[512,7,7]
        highlight_conn_L31 = highlight_conn_L31.cpu().data.numpy()
        highlight_conn_L31 = highlight_conn_L31.reshape(1, h_L31, w_L31) * np.ones_like(
            feature_maps_L31)  # [7,7]=>[1,7,7]=>[512,7,7]

        feature_maps_L31 = feature_maps_L31 * highlight_conn_L31


        feature_maps_L31_max = np.max(feature_maps_L31, axis=(1, 2))
        if np.linalg.norm(feature_maps_L31_max) > 0:
            feature_maps_L31_max_norm = feature_maps_L31_max / np.linalg.norm(feature_maps_L31_max)
        else:
            print("出现了")
            feature_maps_L31_max_norm = np.zeros_like(feature_maps_L31_max)
            pass

        feature_maps_L31_mean = np.mean(feature_maps_L31, axis=(1, 2))
        if np.linalg.norm(feature_maps_L31_mean) > 0:
            feature_maps_L31_mean_norm = feature_maps_L31_mean / np.linalg.norm(feature_maps_L31_mean)
        else:
            print("出现了")
            feature_maps_L31_mean_norm = np.zeros_like(feature_maps_L31_mean)
            pass

        output_mean = []
        output_2=[]



        output_2.append(feature_maps_L31_mean_norm)
        output_2.append(feature_maps_L31_max_norm)


        # output_max=[]
        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['layer4.2.conv2.weight','layer4.1.conv2.weight']:
                # print(name)
                tmp = (param.grad).data
                # print(tmp)
                # tmp=torch.abs(tmp)
                # print(tmp)
                # tmp_mean_reuse=tmp.view(tmp.size()[0],-1)
                # print(tmp_mean_reuse.shape)
                tmp_mean_reuse_std = torch.std(tmp, dim=(2, 3))

                tmp_mean_reuse = F.adaptive_max_pool2d(tmp, output_size=1).squeeze(-1).squeeze(
                    -1) + F.adaptive_max_pool2d(-1 * tmp, output_size=1).squeeze(-1).squeeze(-1)
                tmp_mean_reuse = tmp_mean_reuse + tmp_mean_reuse_std
                # tmp_mean_reuse=F.adaptive_max_pool2d(tmp,output_size=1).squeeze(-1).squeeze(-1)
                # tmp_mean_reuse=F.adaptive_max_pool2d(-1 * tmp,output_size=1).squeeze(-1).squeeze(-1)

                # tmp_mean_reuse_1 = torch.mean(tmp_1, axis=(2, 3))
                # tmp_mean_reuse = tmp_mean_reuse * tmp_mean_reuse_1
                # tmp_mean_reuse = torch.mean(tmp, dim=(2, 3))
                ############################################################
                #$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%%$%$%$$%$$%$%$%$%$%$%$
                #待会回复注释
                # tmp_mean_reuse_bool = torch.mean(tmp_mean_reuse, dim=0).unsqueeze(0)
                # tmp_mean_reuse_bool = tmp_mean_reuse > tmp_mean_reuse_bool
                # tmp_mean_reuse_bool = tmp_mean_reuse_bool.float()
                #
                # tmp_mean_reuse_bool_2 = torch.mean(tmp_mean_reuse, dim=1).unsqueeze(0).T
                # tmp_mean_reuse_bool_2 = tmp_mean_reuse > tmp_mean_reuse_bool_2
                # tmp_mean_reuse_bool_2 = tmp_mean_reuse_bool_2.float()
                #
                # tmp_mean_reuse = tmp_mean_reuse * tmp_mean_reuse_bool * tmp_mean_reuse_bool_2
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
            elif name in ['layer4.2.conv1.weight','layer4.1.conv1.weight','layer4.2.conv3.weight','layer4.1.conv3.weight']:
                # print(name)
                tmp = (param.grad).data
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

        return output_mean,output_2








class GradCam_yuhan_kernel_version30_DGCRL_scda_many_3():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, ii=None,flip=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别

        feature_maps,output,_ = self.net(input.cuda())
        output_max = torch.max(output)
        output_min = torch.min(output)
        output_norm = torch.div((output - output_min).float(), (output_max - output_min))
        ################################################################3
        output_norm = output_norm .cpu().tolist()
        output_norm = torch.tensor(output_norm).float().cuda()
        # output_norm = torch.exp(output_norm)

        ###################################################################
        one_hot = torch.sum(output * output_norm)
        # one_hot = torch.sum(output )

        self.net.zero_grad()
        one_hot.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #


        feature_maps_L31 = feature_maps[
            0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        batch_size, c_L31, h_L31, w_L31 = feature_maps_L31.size()
        #
        # # Pool5   #feature_maps_L31  [512,7,7]==[512,]
        # feature_maps_L31_sum = torch.sum(feature_maps_L31[0],
        #                                  0)  # 如果要将CAM或者gard_cam引入SCDA，需要更改的程序也就仅仅是这一步
        # L31_sum_mean = torch.mean(feature_maps_L31_sum)  # 返回的是scalar tensor ,这是与matlab的mean不同的地方
        # # largestConnectComponent将最大连通区域所对应的像素点置为true
        # jj = (feature_maps_L31_sum > L31_sum_mean).cpu().data.numpy()
        # highlight = torch.from_numpy(jj + 0).cuda().float()
        #
        # # highlight_conn_L31 = highlight
        # highlight_conn_L31 = torch.tensor(largestConnectComponent(highlight.cpu().numpy()) + 0)#[7*7],我们需要将其变成[512,7,7]

        feature_maps_L31 = feature_maps_L31[0].cpu().data.numpy()  # [1,512,7,7]==>[512,7,7]
        # highlight_conn_L31 = highlight_conn_L31.cpu().data.numpy()
        # highlight_conn_L31 = highlight_conn_L31.reshape(1, h_L31, w_L31) * np.ones_like(
        #     feature_maps_L31)  # [7,7]=>[1,7,7]=>[512,7,7]

        # feature_maps_L31 = feature_maps_L31 * highlight_conn_L31


        feature_maps_L31_max = np.max(feature_maps_L31, axis=(1, 2))
        if np.linalg.norm(feature_maps_L31_max) > 0:
            feature_maps_L31_max_norm = feature_maps_L31_max / np.linalg.norm(feature_maps_L31_max)
        else:
            print("出现了")
            feature_maps_L31_max_norm = np.zeros_like(feature_maps_L31_max)
            pass

        feature_maps_L31_mean = np.mean(feature_maps_L31, axis=(1, 2))
        if np.linalg.norm(feature_maps_L31_mean) > 0:
            feature_maps_L31_mean_norm = feature_maps_L31_mean / np.linalg.norm(feature_maps_L31_mean)
        else:
            print("出现了")
            feature_maps_L31_mean_norm = np.zeros_like(feature_maps_L31_mean)
            pass

        output_mean = []
        output_2=[]



        output_2.append(feature_maps_L31_mean_norm)
        output_2.append(feature_maps_L31_max_norm)


        # output_max=[]
        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['layer4.2.conv2.weight','layer4.1.conv2.weight']:
                # print(name)
                tmp = (param.grad).data
                # print(tmp)
                # tmp=torch.abs(tmp)
                # print(tmp)
                # tmp_mean_reuse=tmp.view(tmp.size()[0],-1)
                # print(tmp_mean_reuse.shape)
                tmp_mean_reuse_std = torch.std(tmp, dim=(2, 3))

                tmp_mean_reuse = F.adaptive_max_pool2d(tmp, output_size=1).squeeze(-1).squeeze(
                    -1) + F.adaptive_max_pool2d(-1 * tmp, output_size=1).squeeze(-1).squeeze(-1)
                tmp_mean_reuse = tmp_mean_reuse + tmp_mean_reuse_std
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
            elif name in ['layer4.2.conv1.weight','layer4.1.conv1.weight','layer4.2.conv3.weight','layer4.1.conv3.weight']:
                # print(name)
                tmp = (param.grad).data
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

        return output_mean,output_2


























class GradCam_yuhan_kernel_version30_DGCRL_scda_pro_many():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, ii=None,flip=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别

        feature_maps,output,_ = self.net(input.cuda())
        output_max = torch.max(output)
        output_min = torch.min(output)
        output_norm = torch.div((output - output_min).float(), (output_max - output_min))
        ################################################################3
        output_norm = (output_norm ).cpu().tolist()
        output_norm = torch.tensor(output_norm).float().cuda()
        ###################################################################
        one_hot = torch.sum(output * output_norm)
        self.net.zero_grad()
        one_hot.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #


        #$%
        features_maps_mul_grad = []
        # hata=lisalisa.grad * lisalisa
        # hata = hata.cpu().data.numpy()[0, :]
        # features_maps.append(hata)
        for i in range(len(feature_maps)):
            miki = feature_maps[i] * self.net.gradients[-1 * (i + 1)]
            # miki = miki.cpu().data.numpy()[0, :]
            # miki = np.maximum(miki, 0)

            features_maps_mul_grad.append(miki)

        # feature_maps_L31 = features_maps_mul_grad[
        #     0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        # batch_size, c_L31, h_L31, w_L31 = feature_maps_L31.size()
        #
        # # Pool5   #feature_maps_L31  [512,7,7]==[512,]
        # feature_maps_L31_sum = torch.sum(feature_maps_L31[0],
        #                                  0)  # 如果要将CAM或者gard_cam引入SCDA，需要更改的程序也就仅仅是这一步
        # feature_maps_L31_sum=torch.nn.functional.relu(feature_maps_L31_sum)
        # # L31_sum_mean = torch.mean(feature_maps_L31_sum)  # 返回的是scalar tensor ,这是与matlab的mean不同的地方
        # # jj = (feature_maps_L31_sum > L31_sum_mean).cpu().data.numpy()
        # feature_maps_L31_sum = (feature_maps_L31_sum - torch.min(feature_maps_L31_sum)) / (torch.max(feature_maps_L31_sum) - torch.min(feature_maps_L31_sum))
        # # highlight = torch.from_numpy(jj + 0).cuda().float()
        # # highlight_conn_L31 = highlight+0
        # highlight_conn_L31 = feature_maps_L31_sum

        feature_maps_L31 = features_maps_mul_grad[
            0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        batch_size, c_L31, h_L31, w_L31 = feature_maps_L31.size()

        # Pool5   #feature_maps_L31  [512,7,7]==[512,]
        feature_maps_L31_sum = torch.sum(feature_maps_L31[0],
                                         0)  # 如果要将CAM或者gard_cam引入SCDA，需要更改的程序也就仅仅是这一步
        feature_maps_L31_sum = torch.nn.functional.relu(feature_maps_L31_sum)
        L31_sum_mean = torch.mean(feature_maps_L31_sum)  # 返回的是scalar tensor ,这是与matlab的mean不同的地方
        jj = (feature_maps_L31_sum > L31_sum_mean).cpu().data.numpy()
        # feature_maps_L31_sum = (feature_maps_L31_sum - torch.min(feature_maps_L31_sum)) / (
        #             torch.max(feature_maps_L31_sum) - torch.min(feature_maps_L31_sum))
        highlight = torch.from_numpy(jj + 0).cuda().float()
        highlight_conn_L31 = highlight
        # highlight_conn_L31 = feature_maps_L31_sum

        feature_maps_L31 = feature_maps[
            0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试
        feature_maps_L31 = feature_maps_L31[0].cpu().data.numpy()  # [1,512,7,7]==>[512,7,7]
        # highlight_conn_L31 = highlight_conn_L31*highlight
        # print(highlight_conn_L31==highlight)
        highlight_conn_L31 = highlight_conn_L31.cpu().data.numpy()
        highlight_conn_L31 = highlight_conn_L31.reshape(1, h_L31, w_L31) * np.ones_like(
            feature_maps_L31)  # [7,7]=>[1,7,7]=>[512,7,7]

        feature_maps_L31 = feature_maps_L31 * highlight_conn_L31


        feature_maps_L31_max = np.max(feature_maps_L31, axis=(1, 2))
        if np.linalg.norm(feature_maps_L31_max) > 0:
            feature_maps_L31_max_norm = feature_maps_L31_max / np.linalg.norm(feature_maps_L31_max)
        else:
            print("出现了")
            feature_maps_L31_max_norm = np.zeros_like(feature_maps_L31_max)
            pass

        feature_maps_L31_mean = np.mean(feature_maps_L31, axis=(1, 2))
        if np.linalg.norm(feature_maps_L31_mean) > 0:
            feature_maps_L31_mean_norm = feature_maps_L31_mean / np.linalg.norm(feature_maps_L31_mean)
        else:
            print("出现了")
            feature_maps_L31_mean_norm = np.zeros_like(feature_maps_L31_mean)
            pass

        output_mean = []
        output_2=[]



        output_2.append(feature_maps_L31_mean_norm)
        output_2.append(feature_maps_L31_max_norm)


        # output_max=[]
        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['layer4.2.conv2.weight','layer4.1.conv2.weight']:
                # print(name)
                tmp = (param.grad).data
                # print(tmp)
                # tmp=torch.abs(tmp)
                # print(tmp)
                # tmp_mean_reuse=tmp.view(tmp.size()[0],-1)
                # print(tmp_mean_reuse.shape)
                tmp_mean_reuse_std = torch.std(tmp, dim=(2, 3))

                tmp_mean_reuse = F.adaptive_max_pool2d(tmp, output_size=1).squeeze(-1).squeeze(
                    -1) + F.adaptive_max_pool2d(-1 * tmp, output_size=1).squeeze(-1).squeeze(-1)
                tmp_mean_reuse = tmp_mean_reuse + tmp_mean_reuse_std
                # tmp_mean_reuse=F.adaptive_max_pool2d(tmp,output_size=1).squeeze(-1).squeeze(-1)
                # tmp_mean_reuse=F.adaptive_max_pool2d(-1 * tmp,output_size=1).squeeze(-1).squeeze(-1)

                # tmp_mean_reuse_1 = torch.mean(tmp_1, axis=(2, 3))
                # tmp_mean_reuse = tmp_mean_reuse * tmp_mean_reuse_1
                # tmp_mean_reuse = torch.mean(tmp, dim=(2, 3))
                ############################################################
                #$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%$%%$%$%$$%$$%$%$%$%$%$%$
                #待会回复注释
                # tmp_mean_reuse_bool = torch.mean(tmp_mean_reuse, dim=0).unsqueeze(0)
                # tmp_mean_reuse_bool = tmp_mean_reuse > tmp_mean_reuse_bool
                # tmp_mean_reuse_bool = tmp_mean_reuse_bool.float()
                #
                # tmp_mean_reuse_bool_2 = torch.mean(tmp_mean_reuse, dim=1).unsqueeze(0).T
                # tmp_mean_reuse_bool_2 = tmp_mean_reuse > tmp_mean_reuse_bool_2
                # tmp_mean_reuse_bool_2 = tmp_mean_reuse_bool_2.float()
                #
                # tmp_mean_reuse = tmp_mean_reuse * tmp_mean_reuse_bool * tmp_mean_reuse_bool_2
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
            elif name in ['layer4.2.conv1.weight','layer4.1.conv1.weight','layer4.2.conv3.weight','layer4.1.conv3.weight']:
                # print(name)
                tmp = (param.grad).data
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
                # tmp_mean_reuse_bool = torch.mean(tmp_mean_reuse, dim=0).unsqueeze(0)
                # tmp_mean_reuse_bool = tmp_mean_reuse > tmp_mean_reuse_bool
                # tmp_mean_reuse_bool = tmp_mean_reuse_bool.float()
                #
                # tmp_mean_reuse_bool_2 = torch.mean(tmp_mean_reuse, dim=1).unsqueeze(0).T
                # tmp_mean_reuse_bool_2 = tmp_mean_reuse > tmp_mean_reuse_bool_2
                # tmp_mean_reuse_bool_2 = tmp_mean_reuse_bool_2.float()
                #
                # tmp_mean_reuse = tmp_mean_reuse * tmp_mean_reuse_bool * tmp_mean_reuse_bool_2
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

        return output_mean,output_2








class PoolCam_yuhan_kernel_for_retrieval():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, ii=None,flip=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别

        feature_maps,output,_ = self.net(input.cuda())
        # output_max = torch.max(output)
        # output_min = torch.min(output)
        # output_norm = torch.div((output - output_min).float(), (output_max - output_min))
        # ################################################################3
        # output_norm = output_norm .cpu().tolist()
        # output_norm = torch.tensor(output_norm).float().cuda()
        # ###################################################################
        # one_hot = torch.sum(output * output_norm)
        # self.net.zero_grad()
        # one_hot.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚
        # # params = list(self.net.parameters())
        # # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #


        feature_maps_L31 = feature_maps[
            0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        batch_size, c_L31, h_L31, w_L31 = feature_maps_L31.size()

        # Pool5   #feature_maps_L31  [512,7,7]==[512,]
        feature_maps_L31_sum = torch.sum(feature_maps_L31[0],
                                         0)  # 如果要将CAM或者gard_cam引入SCDA，需要更改的程序也就仅仅是这一步
        L31_sum_mean = torch.mean(feature_maps_L31_sum)  # 返回的是scalar tensor ,这是与matlab的mean不同的地方
        # largestConnectComponent将最大连通区域所对应的像素点置为true
        jj = (feature_maps_L31_sum > L31_sum_mean).cpu().data.numpy()
        highlight = torch.from_numpy(jj + 0).cuda().float()

        highlight_conn_L31 = highlight
        # highlight_conn_L31 = torch.tensor(largestConnectComponent(highlight.cpu().numpy()) + 0)#[7*7],我们需要将其变成[512,7,7]

        feature_maps_L31 = feature_maps_L31[0].cpu().data.numpy()  # [1,512,7,7]==>[512,7,7]
        highlight_conn_L31 = highlight_conn_L31.cpu().data.numpy()
        highlight_conn_L31 = highlight_conn_L31.reshape(1, h_L31, w_L31) * np.ones_like(
            feature_maps_L31)  # [7,7]=>[1,7,7]=>[512,7,7]

        feature_maps_L31 = feature_maps_L31 * highlight_conn_L31


        feature_maps_L31_max = np.max(feature_maps_L31, axis=(1, 2))
        if np.linalg.norm(feature_maps_L31_max) > 0:
            feature_maps_L31_max_norm = feature_maps_L31_max / np.linalg.norm(feature_maps_L31_max)
        else:
            print("出现了")
            feature_maps_L31_max_norm = np.zeros_like(feature_maps_L31_max)
            pass
        feature_maps_L31_mean = np.mean(feature_maps_L31, axis=(1, 2))
        if np.linalg.norm(feature_maps_L31_mean) > 0:
            feature_maps_L31_mean_norm = feature_maps_L31_mean / np.linalg.norm(feature_maps_L31_mean)
        else:
            print("出现了")
            feature_maps_L31_mean_norm = np.zeros_like(feature_maps_L31_mean)
            pass

        feature_maps_L31_mean_max = np.hstack([feature_maps_L31_mean, feature_maps_L31_max])

        if np.linalg.norm(feature_maps_L31_mean_max) > 0:
            feature_maps_L31_mean_max_norm = feature_maps_L31_mean_max / np.linalg.norm(feature_maps_L31_mean_max)
        else:
            print("出现了")
            feature_maps_L31_mean_max_norm = np.zeros_like(feature_maps_L31_mean_max)
            pass


        output_mean = []
        output_2=[]
        output_2.append(np.hstack([feature_maps_L31_mean_norm,feature_maps_L31_max_norm]))#先norm再 拼接
        output_2.append(feature_maps_L31_mean_max_norm)  # 先拼接再norm

        feature_maps_L31 = feature_maps[
            0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        feature_maps_L31 = feature_maps_L31[0].cpu().data.numpy()  # [1,512,7,7]==>[512,7,7]
        feature_maps_L31_max = np.max(feature_maps_L31, axis=(1, 2))
        feature_maps_L31_mean = np.mean(feature_maps_L31, axis=(1, 2))
        feature_maps_L31_mean_max = np.hstack([feature_maps_L31_mean, feature_maps_L31_max])

        if np.linalg.norm(feature_maps_L31_max) > 0:
            feature_maps_L31_max_norm = feature_maps_L31_max / np.linalg.norm(feature_maps_L31_max)
        else:
            print("出现了")
            feature_maps_L31_max_norm = np.zeros_like(feature_maps_L31_max)
            pass

        if np.linalg.norm(feature_maps_L31_mean) > 0:
            feature_maps_L31_mean_norm = feature_maps_L31_mean / np.linalg.norm(feature_maps_L31_mean)
        else:
            print("出现了")
            feature_maps_L31_mean_norm = np.zeros_like(feature_maps_L31_mean)
            pass

        feature_maps_L31_norm_mean_max = np.hstack([feature_maps_L31_mean_norm, feature_maps_L31_max_norm])

        if np.linalg.norm(feature_maps_L31_mean_max) > 0:
            feature_maps_L31_mean_max_norm = feature_maps_L31_mean_max / np.linalg.norm(feature_maps_L31_mean_max)
        else:
            print("出现了")
            feature_maps_L31_mean_max_norm = np.zeros_like(feature_maps_L31_mean_max)
            pass
        output_3 = []
        # output_3.append(feature_maps_L31_mean_norm)
        output_3.append(feature_maps_L31_mean_max_norm)
        output_3.append(feature_maps_L31_norm_mean_max)

        out_for_visualize=[feature_maps,jj]

        feature_maps_L31 = feature_maps[
            0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        # feature_maps_L31 = feature_maps_L31[0]  # [1,512,7,7]==>[512,7,7]

        feature_maps_L31_max = global_avg_pool(feature_maps_L31)
        feature_maps_L31_mean = global_max_pool(feature_maps_L31)
        # print(feature_maps_L31_max.size())
        # feature_maps_L31_max = feature_maps_L31_max.view( -1)
        # feature_maps_L31_mean = feature_maps_L31_mean.view(-1)
        feature_maps_L31_mean_max = torch.cat((feature_maps_L31_mean, feature_maps_L31_max), dim=1)
        # print(feature_maps_L31_mean_max.size())
        loss2 = torch.sum(feature_maps_L31_mean_max)
        # print(loss2)
        self.net.zero_grad()
        loss2.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚

        # output_max=[]
        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['layer4.2.conv2.weight','layer4.1.conv2.weight']:
                # print(name)
                tmp = (param.grad).data
                # print(tmp)
                # tmp=torch.abs(tmp)
                # print(tmp)
                # tmp_mean_reuse=tmp.view(tmp.size()[0],-1)
                # print(tmp_mean_reuse.shape)
                tmp_mean_reuse_std = torch.std(tmp, dim=(2, 3))

                tmp_mean_reuse = F.adaptive_max_pool2d(tmp, output_size=1).squeeze(-1).squeeze(
                    -1) + F.adaptive_max_pool2d(-1 * tmp, output_size=1).squeeze(-1).squeeze(-1)
                tmp_mean_reuse = tmp_mean_reuse + tmp_mean_reuse_std
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
            elif name in ['layer4.2.conv1.weight','layer4.1.conv1.weight','layer4.2.conv3.weight','layer4.1.conv3.weight']:
                # print(name)
                tmp = (param.grad).data
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

        return output_mean,output_2,output_3,out_for_visualize





class PoolCam_yuhan_kernel_saliency_map_for_retrieval():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, ii=None,flip=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别

        feature_maps,output,_ = self.net(input.cuda())
        # output_max = torch.max(output)
        # output_min = torch.min(output)
        # output_norm = torch.div((output - output_min).float(), (output_max - output_min))
        # ################################################################3
        # output_norm = output_norm .cpu().tolist()
        # output_norm = torch.tensor(output_norm).float().cuda()
        # ###################################################################
        # one_hot = torch.sum(output * output_norm)
        # self.net.zero_grad()
        # one_hot.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚
        # # params = list(self.net.parameters())
        # # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #


        feature_maps_L31 = feature_maps[
            0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        batch_size, c_L31, h_L31, w_L31 = feature_maps_L31.size()

        # Pool5   #feature_maps_L31  [512,7,7]==[512,]
        feature_maps_L31_sum = torch.sum(feature_maps_L31[0],
                                         0)  # 如果要将CAM或者gard_cam引入SCDA，需要更改的程序也就仅仅是这一步
        L31_sum_mean = torch.mean(feature_maps_L31_sum)  # 返回的是scalar tensor ,这是与matlab的mean不同的地方
        # largestConnectComponent将最大连通区域所对应的像素点置为true
        jj = (feature_maps_L31_sum > L31_sum_mean).cpu().data.numpy()
        highlight = torch.from_numpy(jj + 0).cuda().float()

        highlight_conn_L31 = highlight
        # highlight_conn_L31 = torch.tensor(largestConnectComponent(highlight.cpu().numpy()) + 0)#[7*7],我们需要将其变成[512,7,7]

        feature_maps_L31 = feature_maps_L31[0].cpu().data.numpy()  # [1,512,7,7]==>[512,7,7]
        highlight_conn_L31 = highlight_conn_L31.cpu().data.numpy()
        highlight_conn_L31 = highlight_conn_L31.reshape(1, h_L31, w_L31) * np.ones_like(
            feature_maps_L31)  # [7,7]=>[1,7,7]=>[512,7,7]

        feature_maps_L31 = feature_maps_L31 * highlight_conn_L31


        feature_maps_L31_max = np.max(feature_maps_L31, axis=(1, 2))
        if np.linalg.norm(feature_maps_L31_max) > 0:
            feature_maps_L31_max_norm = feature_maps_L31_max / np.linalg.norm(feature_maps_L31_max)
        else:
            print("出现了")
            feature_maps_L31_max_norm = np.zeros_like(feature_maps_L31_max)
            pass
        feature_maps_L31_mean = np.mean(feature_maps_L31, axis=(1, 2))
        if np.linalg.norm(feature_maps_L31_mean) > 0:
            feature_maps_L31_mean_norm = feature_maps_L31_mean / np.linalg.norm(feature_maps_L31_mean)
        else:
            print("出现了")
            feature_maps_L31_mean_norm = np.zeros_like(feature_maps_L31_mean)
            pass

        feature_maps_L31_mean_max = np.hstack([feature_maps_L31_mean, feature_maps_L31_max])

        if np.linalg.norm(feature_maps_L31_mean_max) > 0:
            feature_maps_L31_mean_max_norm = feature_maps_L31_mean_max / np.linalg.norm(feature_maps_L31_mean_max)
        else:
            print("出现了")
            feature_maps_L31_mean_max_norm = np.zeros_like(feature_maps_L31_mean_max)
            pass


        output_mean = []
        output_2=[]
        output_2.append(np.hstack([feature_maps_L31_mean_norm,feature_maps_L31_max_norm]))#先norm再 拼接
        output_2.append(feature_maps_L31_mean_max_norm)  # 先拼接再norm

        feature_maps_L31 = feature_maps[
            0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        feature_maps_L31 = feature_maps_L31[0].cpu().data.numpy()  # [1,512,7,7]==>[512,7,7]
        feature_maps_L31_max = np.max(feature_maps_L31, axis=(1, 2))
        feature_maps_L31_mean = np.mean(feature_maps_L31, axis=(1, 2))
        feature_maps_L31_mean_max = np.hstack([feature_maps_L31_mean, feature_maps_L31_max])

        if np.linalg.norm(feature_maps_L31_max) > 0:
            feature_maps_L31_max_norm = feature_maps_L31_max / np.linalg.norm(feature_maps_L31_max)
        else:
            print("出现了")
            feature_maps_L31_max_norm = np.zeros_like(feature_maps_L31_max)
            pass

        if np.linalg.norm(feature_maps_L31_mean) > 0:
            feature_maps_L31_mean_norm = feature_maps_L31_mean / np.linalg.norm(feature_maps_L31_mean)
        else:
            print("出现了")
            feature_maps_L31_mean_norm = np.zeros_like(feature_maps_L31_mean)
            pass

        feature_maps_L31_norm_mean_max = np.hstack([feature_maps_L31_mean_norm, feature_maps_L31_max_norm])

        if np.linalg.norm(feature_maps_L31_mean_max) > 0:
            feature_maps_L31_mean_max_norm = feature_maps_L31_mean_max / np.linalg.norm(feature_maps_L31_mean_max)
        else:
            print("出现了")
            feature_maps_L31_mean_max_norm = np.zeros_like(feature_maps_L31_mean_max)
            pass
        output_3 = []
        # output_3.append(feature_maps_L31_mean_norm)
        output_3.append(feature_maps_L31_mean_max_norm)
        output_3.append(feature_maps_L31_norm_mean_max)

        out_for_visualize=[feature_maps,jj]

        feature_maps_L31 = feature_maps[
            0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        # feature_maps_L31 = feature_maps_L31[0]  # [1,512,7,7]==>[512,7,7]
        feature_maps_L31 = feature_maps_L31 * torch.from_numpy(highlight_conn_L31).unsqueeze(0).cuda().float()
        feature_maps_L31_max = global_avg_pool(feature_maps_L31)
        feature_maps_L31_mean = global_max_pool(feature_maps_L31)
        # print(feature_maps_L31_max.size())
        # feature_maps_L31_max = feature_maps_L31_max.view( -1)
        # feature_maps_L31_mean = feature_maps_L31_mean.view(-1)
        feature_maps_L31_mean_max = torch.cat((feature_maps_L31_mean, feature_maps_L31_max), dim=1)
        # print(feature_maps_L31_mean_max.size())
        loss2 = torch.sum(feature_maps_L31_mean_max)
        # print(loss2)
        self.net.zero_grad()
        loss2.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚

        # output_max=[]
        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['layer4.2.conv2.weight','layer4.1.conv2.weight']:
                # print(name)
                tmp = (param.grad).data
                # print(tmp)
                # tmp=torch.abs(tmp)
                # print(tmp)
                # tmp_mean_reuse=tmp.view(tmp.size()[0],-1)
                # print(tmp_mean_reuse.shape)
                tmp_mean_reuse_std = torch.std(tmp, dim=(2, 3))

                tmp_mean_reuse = F.adaptive_max_pool2d(tmp, output_size=1).squeeze(-1).squeeze(
                    -1) + F.adaptive_max_pool2d(-1 * tmp, output_size=1).squeeze(-1).squeeze(-1)
                tmp_mean_reuse = tmp_mean_reuse + tmp_mean_reuse_std
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
            elif name in ['layer4.2.conv1.weight','layer4.1.conv1.weight','layer4.2.conv3.weight','layer4.1.conv3.weight']:
                # print(name)
                tmp = (param.grad).data
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

        return output_mean,output_2,output_3,out_for_visualize






class GradCam_yuhan_kernel_version30_DGPCRL_scda_many_4_for_retrieval():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, ii=None,flip=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别

        feature_maps,output,_ = self.net(input.cuda())
        output_max = torch.max(output)
        output_min = torch.min(output)
        output_norm = torch.div((output - output_min).float(), (output_max - output_min))
        ################################################################3
        output_norm = output_norm .cpu().tolist()
        output_norm = torch.tensor(output_norm).float().cuda()
        ###################################################################
        one_hot = torch.sum(output * output_norm)
        self.net.zero_grad()
        one_hot.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #


        feature_maps_L31 = feature_maps[
            0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        batch_size, c_L31, h_L31, w_L31 = feature_maps_L31.size()

        # Pool5   #feature_maps_L31  [512,7,7]==[512,]
        feature_maps_L31_sum = torch.sum(feature_maps_L31[0],
                                         0)  # 如果要将CAM或者gard_cam引入SCDA，需要更改的程序也就仅仅是这一步
        L31_sum_mean = torch.mean(feature_maps_L31_sum)  # 返回的是scalar tensor ,这是与matlab的mean不同的地方
        # largestConnectComponent将最大连通区域所对应的像素点置为true
        jj = (feature_maps_L31_sum > L31_sum_mean).cpu().data.numpy()
        highlight = torch.from_numpy(jj + 0).cuda().float()

        highlight_conn_L31 = highlight
        # highlight_conn_L31 = torch.tensor(largestConnectComponent(highlight.cpu().numpy()) + 0)#[7*7],我们需要将其变成[512,7,7]

        feature_maps_L31 = feature_maps_L31[0].cpu().data.numpy()  # [1,512,7,7]==>[512,7,7]
        highlight_conn_L31 = highlight_conn_L31.cpu().data.numpy()
        highlight_conn_L31 = highlight_conn_L31.reshape(1, h_L31, w_L31) * np.ones_like(
            feature_maps_L31)  # [7,7]=>[1,7,7]=>[512,7,7]

        feature_maps_L31 = feature_maps_L31 * highlight_conn_L31


        feature_maps_L31_max = np.max(feature_maps_L31, axis=(1, 2))
        if np.linalg.norm(feature_maps_L31_max) > 0:
            feature_maps_L31_max_norm = feature_maps_L31_max / np.linalg.norm(feature_maps_L31_max)
        else:
            print("出现了")
            feature_maps_L31_max_norm = np.zeros_like(feature_maps_L31_max)
            pass
        feature_maps_L31_mean = np.mean(feature_maps_L31, axis=(1, 2))
        if np.linalg.norm(feature_maps_L31_mean) > 0:
            feature_maps_L31_mean_norm = feature_maps_L31_mean / np.linalg.norm(feature_maps_L31_mean)
        else:
            print("出现了")
            feature_maps_L31_mean_norm = np.zeros_like(feature_maps_L31_mean)
            pass

        feature_maps_L31_mean_max = np.hstack([feature_maps_L31_mean, feature_maps_L31_max])

        if np.linalg.norm(feature_maps_L31_mean_max) > 0:
            feature_maps_L31_mean_max_norm = feature_maps_L31_mean_max / np.linalg.norm(feature_maps_L31_mean_max)
        else:
            print("出现了")
            feature_maps_L31_mean_max_norm = np.zeros_like(feature_maps_L31_mean_max)
            pass


        output_mean = []
        output_2=[]
        output_2.append(np.hstack([feature_maps_L31_mean_norm,feature_maps_L31_max_norm]))#先norm再 拼接
        output_2.append(feature_maps_L31_mean_max_norm)  # 先拼接再norm

        feature_maps_L31 = feature_maps[
            0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        feature_maps_L31 = feature_maps_L31[0].cpu().data.numpy()  # [1,512,7,7]==>[512,7,7]
        feature_maps_L31_max = np.max(feature_maps_L31, axis=(1, 2))
        feature_maps_L31_mean = np.mean(feature_maps_L31, axis=(1, 2))
        feature_maps_L31_mean_max = np.hstack([feature_maps_L31_mean, feature_maps_L31_max])

        if np.linalg.norm(feature_maps_L31_max) > 0:
            feature_maps_L31_max_norm = feature_maps_L31_max / np.linalg.norm(feature_maps_L31_max)
        else:
            print("出现了")
            feature_maps_L31_max_norm = np.zeros_like(feature_maps_L31_max)
            pass

        if np.linalg.norm(feature_maps_L31_mean) > 0:
            feature_maps_L31_mean_norm = feature_maps_L31_mean / np.linalg.norm(feature_maps_L31_mean)
        else:
            print("出现了")
            feature_maps_L31_mean_norm = np.zeros_like(feature_maps_L31_mean)
            pass

        feature_maps_L31_norm_mean_max = np.hstack([feature_maps_L31_mean_norm, feature_maps_L31_max_norm])

        if np.linalg.norm(feature_maps_L31_mean_max) > 0:
            feature_maps_L31_mean_max_norm = feature_maps_L31_mean_max / np.linalg.norm(feature_maps_L31_mean_max)
        else:
            print("出现了")
            feature_maps_L31_mean_max_norm = np.zeros_like(feature_maps_L31_mean_max)
            pass
        output_3 = []
        # output_3.append(feature_maps_L31_mean_norm)
        output_3.append(feature_maps_L31_mean_max_norm)
        output_3.append(feature_maps_L31_norm_mean_max)

        out_for_visualize=[feature_maps,jj]

        # output_max=[]
        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['layer4.2.conv2.weight','layer4.1.conv2.weight']:
                # print(name)
                tmp = (param.grad).data
                # print(tmp)
                # tmp=torch.abs(tmp)
                # print(tmp)
                # tmp_mean_reuse=tmp.view(tmp.size()[0],-1)
                # print(tmp_mean_reuse.shape)
                tmp_mean_reuse_std = torch.std(tmp, dim=(2, 3))

                tmp_mean_reuse = F.adaptive_max_pool2d(tmp, output_size=1).squeeze(-1).squeeze(
                    -1) + F.adaptive_max_pool2d(-1 * tmp, output_size=1).squeeze(-1).squeeze(-1)
                tmp_mean_reuse = tmp_mean_reuse + tmp_mean_reuse_std
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
            elif name in ['layer4.2.conv1.weight','layer4.1.conv1.weight','layer4.2.conv3.weight','layer4.1.conv3.weight']:
                # print(name)
                tmp = (param.grad).data
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

        return output_mean,output_2,output_3,out_for_visualize








class GradCam_yuhan_kernel_version30_DGCRL_scda_many_for_retrivial():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, ii=None,flip=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别

        feature_maps,output,_ = self.net(input.cuda())
        output_max = torch.max(output)
        output_min = torch.min(output)
        output_norm = torch.div((output - output_min).float(), (output_max - output_min))
        ################################################################3
        output_norm = (output_norm).cpu().tolist()
        output_norm = torch.tensor(output_norm).float().cuda()
        ###################################################################
        one_hot = torch.sum(output * output_norm)
        self.net.zero_grad()
        one_hot.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #


        feature_maps_L31 = feature_maps[
            0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        batch_size, c_L31, h_L31, w_L31 = feature_maps_L31.size()

        # Pool5   #feature_maps_L31  [512,7,7]==[512,]
        feature_maps_L31_sum = torch.sum(feature_maps_L31[0],
                                         0)  # 如果要将CAM或者gard_cam引入SCDA，需要更改的程序也就仅仅是这一步
        L31_sum_mean = torch.mean(feature_maps_L31_sum)  # 返回的是scalar tensor ,这是与matlab的mean不同的地方
        # largestConnectComponent将最大连通区域所对应的像素点置为true
        jj = (feature_maps_L31_sum > L31_sum_mean).cpu().data.numpy()
        highlight = torch.from_numpy(jj + 0).cuda().float()

        highlight_conn_L31 = highlight
        # highlight_conn_L31 = torch.tensor(largestConnectComponent(highlight.cpu().numpy()) + 0)#[7*7],我们需要将其变成[512,7,7]

        feature_maps_L31 = feature_maps_L31[0].cpu().data.numpy()  # [1,512,7,7]==>[512,7,7]
        highlight_conn_L31 = highlight_conn_L31.cpu().data.numpy()
        highlight_conn_L31 = highlight_conn_L31.reshape(1, h_L31, w_L31) * np.ones_like(
            feature_maps_L31)  # [7,7]=>[1,7,7]=>[512,7,7]

        feature_maps_L31 = feature_maps_L31 * highlight_conn_L31


        feature_maps_L31_max = np.max(feature_maps_L31, axis=(1, 2))
        if np.linalg.norm(feature_maps_L31_max) > 0:
            feature_maps_L31_max_norm = feature_maps_L31_max / np.linalg.norm(feature_maps_L31_max)
        else:
            print("出现了")
            feature_maps_L31_max_norm = np.zeros_like(feature_maps_L31_max)
            pass

        feature_maps_L31_mean = np.mean(feature_maps_L31, axis=(1, 2))
        if np.linalg.norm(feature_maps_L31_mean) > 0:
            feature_maps_L31_mean_norm = feature_maps_L31_mean / np.linalg.norm(feature_maps_L31_mean)
        else:
            print("出现了")
            feature_maps_L31_mean_norm = np.zeros_like(feature_maps_L31_mean)
            pass

        output_mean = []
        output_2=[]
        out_for_visualize=[feature_maps,jj]




        output_2.append(feature_maps_L31_mean_norm)
        output_2.append(feature_maps_L31_max_norm)


        # output_max=[]
        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['layer4.2.conv2.weight','layer4.1.conv2.weight']:
                # print(name)
                tmp = (param.grad).data
                # print(tmp)
                # tmp=torch.abs(tmp)
                # print(tmp)
                # tmp_mean_reuse=tmp.view(tmp.size()[0],-1)
                # print(tmp_mean_reuse.shape)
                tmp_mean_reuse_std = torch.std(tmp, dim=(2, 3))

                tmp_mean_reuse = F.adaptive_max_pool2d(tmp, output_size=1).squeeze(-1).squeeze(
                    -1) + F.adaptive_max_pool2d(-1 * tmp, output_size=1).squeeze(-1).squeeze(-1)
                tmp_mean_reuse = tmp_mean_reuse + tmp_mean_reuse_std
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
            elif name in ['layer4.2.conv1.weight','layer4.1.conv1.weight','layer4.2.conv3.weight','layer4.1.conv3.weight']:
                # print(name)
                tmp = (param.grad).data
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

        return output_mean,output_2,out_for_visualize

















class GradCam_yuhan_kernel_version30_softmax_scda():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, ii=None,flip=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别

        feature_maps,output= self.net(input.cuda())
        output_max = torch.max(output)
        output_min = torch.min(output)
        output_norm = torch.div((output - output_min).float(), (output_max - output_min))
        ################################################################3
        output_norm = (output_norm * -1).cpu().tolist()
        output_norm = torch.tensor(output_norm).float().cuda()
        ###################################################################
        one_hot = torch.sum(output * output_norm)
        self.net.zero_grad()
        one_hot.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #

        feature_maps_L28 = feature_maps[
            0]  # 如果输入图像的shape为[batch_size,3,224,224]的tensor的话，关于输出的feature map,shape为  #[batch_size,512,14,14]左右
        batch_size, c_L28, h_L28, w_L28 = feature_maps_L28.size()
        feature_maps_L31 = feature_maps[
            1]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        batch_size, c_L31, h_L31, w_L31 = feature_maps_L31.size()

        # Pool5   #feature_maps_L31  [512,7,7]==[512,]
        feature_maps_L31_sum = torch.sum(feature_maps_L31[0],
                                         0)  # 如果要将CAM或者gard_cam引入SCDA，需要更改的程序也就仅仅是这一步
        L31_sum_mean = torch.mean(feature_maps_L31_sum)  # 返回的是scalar tensor ,这是与matlab的mean不同的地方
        highlight = torch.zeros(feature_maps_L31_sum.size())  # 生成了一个h*w的全零矩阵
        # print(highlight)
        highlight_index = torch.nonzero(feature_maps_L31_sum >= L31_sum_mean)  # 一个K*2的tensor,每一行一个符合要求的点
        a, b = highlight_index.size()
        # print(highlight_index.size())

        # for i in range(a):  # a就是被选择出来的点的个数，这些点我们需要将其标记为1
        #     highlight[highlight_index[i][0], highlight_index[i][1]] = 1
        # 以上，我们便初步获取得到了聚合特征图掩码矩阵
        # 然后，我们需要对掩码矩阵进行进一步的处理，获得最大连通区域
        # largestConnectComponent将最大连通区域所对应的像素点置为true
        jj = (feature_maps_L31_sum > L31_sum_mean).cpu().data.numpy()
        highlight = torch.from_numpy(jj + 0).cuda().float()

        highlight_conn_L31 = highlight
        # highlight_conn_L31 = torch.tensor(largestConnectComponent(highlight.cpu().numpy()) + 0)#[7*7],我们需要将其变成[512,7,7]

        feature_maps_L31 = feature_maps_L31[0].cpu().data.numpy()  # [1,512,7,7]==>[512,7,7]
        highlight_conn_L31_beifen = highlight_conn_L31
        highlight_conn_L31 = highlight_conn_L31.cpu().data.numpy()
        highlight_conn_L31 = highlight_conn_L31.reshape(1, h_L31, w_L31) * np.ones_like(
            feature_maps_L31)  # [7,7]=>[1,7,7]=>[512,7,7]

        feature_maps_L31 = feature_maps_L31 * highlight_conn_L31


        feature_maps_L31_max = np.max(feature_maps_L31, axis=(1, 2))
        if np.linalg.norm(feature_maps_L31_max) > 0:
            feature_maps_L31_max_norm = feature_maps_L31_max / np.linalg.norm(feature_maps_L31_max)
        else:
            print("出现了")
            feature_maps_L31_max_norm = np.zeros_like(feature_maps_L31_max)
            pass
        # % Relu5_2
        feature_maps_L28_sum = torch.sum(feature_maps_L28[0],
                                         0)  # 如果要将CAM或者gard_cam引入SCDA，需要更改的程序也就仅仅是这一步
        L28_sum_mean = torch.mean(feature_maps_L28_sum)  # 返回的是scalar tensor ,这是与matlab的mean不同的地方
        highlight = torch.zeros(feature_maps_L28_sum.size())  # 生成了一个h*w的全零矩阵
        highlight_index = torch.nonzero(feature_maps_L28_sum >= L28_sum_mean)  # 一个K*2的tensor,每一行一个符合要求的点
        a, b = highlight_index.size()
        # for i in range(a):  # a就是被选择出来的点的个数，这些点我们需要将其标记为1
        #     highlight[highlight_index[i][0], highlight_index[i][1]] = 1

        jj = (feature_maps_L28_sum > L28_sum_mean).cpu().data.numpy()
        highlight = torch.from_numpy(jj + 0).cuda().float()

        highlight_conn_L31 = highlight_conn_L31_beifen
        highlight_conn_L31_to_L28 = interpolate(highlight_conn_L31.view(1, 1, h_L31, w_L31).float(),
                                                size=[h_L28, w_L28], mode="nearest").view(h_L28, w_L28)
        highlight_conn_L28 = highlight.mul(highlight_conn_L31_to_L28.cuda())  # 逐点按元素想乘，两个都是二值矩阵，可不就是按位与嘛
        feature_maps_L28 = feature_maps_L28[0].cpu().data.numpy()  # [1,512,7,7]==>[512,7,7]
        highlight_conn_L28 = highlight_conn_L28.cpu().data.numpy()
        highlight_conn_L28 = highlight_conn_L28.reshape(1, h_L28, w_L28) * np.ones_like(
            feature_maps_L28)  # [7,7]=>[1,7,7]=>[512,7,7]

        feature_maps_L28 = feature_maps_L28 * highlight_conn_L28

        feature_maps_L28_max = np.max(feature_maps_L28, axis=(1, 2))
        if np.linalg.norm(feature_maps_L28_max) > 0:
            feature_maps_L28_max_norm = feature_maps_L28_max / np.linalg.norm(feature_maps_L28_max)
        else:
            print("出现了")
            feature_maps_L28_max_norm = np.zeros_like(feature_maps_L28_max)
            pass

        output_mean = []
        output_2=[]



        output_2.append(feature_maps_L28_max_norm)
        output_2.append(feature_maps_L31_max_norm)


        # output_max=[]
        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['features.26.weight', 'features.28.weight']:
                tmp = (param.grad).data
                # print(tmp)
                # tmp=torch.abs(tmp)
                # print(tmp)
                # tmp_mean_reuse=tmp.view(tmp.size()[0],-1)
                # print(tmp_mean_reuse.shape)
                tmp_mean_reuse_std = torch.std(tmp, axis=(2, 3))

                tmp_mean_reuse = F.adaptive_max_pool2d(tmp, output_size=1).squeeze(-1).squeeze(
                    -1) + F.adaptive_max_pool2d(-1 * tmp, output_size=1).squeeze(-1).squeeze(-1)
                tmp_mean_reuse = tmp_mean_reuse + tmp_mean_reuse_std
                # tmp_mean_reuse=F.adaptive_max_pool2d(tmp,output_size=1).squeeze(-1).squeeze(-1)
                # tmp_mean_reuse=F.adaptive_max_pool2d(-1 * tmp,output_size=1).squeeze(-1).squeeze(-1)

                # tmp_mean_reuse_1 = torch.mean(tmp_1, axis=(2, 3))
                # tmp_mean_reuse = tmp_mean_reuse * tmp_mean_reuse_1
                # tmp_mean_reuse = torch.mean(tmp, dim=(2, 3))
                tmp_mean_reuse_bool = torch.mean(tmp_mean_reuse, dim=0).unsqueeze(0)
                tmp_mean_reuse_bool = tmp_mean_reuse > tmp_mean_reuse_bool
                tmp_mean_reuse_bool = tmp_mean_reuse_bool.float()

                tmp_mean_reuse_bool_2 = torch.mean(tmp_mean_reuse, dim=1).unsqueeze(0).T
                tmp_mean_reuse_bool_2 = tmp_mean_reuse > tmp_mean_reuse_bool_2
                tmp_mean_reuse_bool_2 = tmp_mean_reuse_bool_2.float()

                tmp_mean_reuse = tmp_mean_reuse * tmp_mean_reuse_bool * tmp_mean_reuse_bool_2

                tmp_mean = torch.mean(tmp_mean_reuse, dim=0).cpu().data.numpy()
                # tmp_mean,_ = torch.max(tmp_mean_reuse, dim=0)
                # tmp_mean=tmp_mean.numpy()
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean.append(tmp_mean)


        return output_mean,output_2






class GradCam_yuhan_kernel_version30_softmax_scda_simple():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, ii=None,flip=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别

        feature_maps,output= self.net(input.cuda())
        output_max = torch.max(output)
        output_min = torch.min(output)
        output_norm = torch.div((output - output_min).float(), (output_max - output_min))
        ################################################################3
        output_norm = (output_norm * -1).cpu().tolist()
        output_norm = torch.tensor(output_norm).float().cuda()
        ###################################################################
        one_hot = torch.sum(output * output_norm)
        self.net.zero_grad()
        one_hot.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #


        feature_maps_L31 = feature_maps[
            0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        batch_size, c_L31, h_L31, w_L31 = feature_maps_L31.size()
        # print(feature_maps_L31.size())

        # Pool5   #feature_maps_L31  [512,7,7]==[512,]
        feature_maps_L31_sum = torch.sum(feature_maps_L31[0],
                                         0)  # 如果要将CAM或者gard_cam引入SCDA，需要更改的程序也就仅仅是这一步
        L31_sum_mean = torch.mean(feature_maps_L31_sum)  # 返回的是scalar tensor ,这是与matlab的mean不同的地方
        highlight = torch.zeros(feature_maps_L31_sum.size())  # 生成了一个h*w的全零矩阵
        # print(highlight)
        highlight_index = torch.nonzero(feature_maps_L31_sum >= L31_sum_mean)  # 一个K*2的tensor,每一行一个符合要求的点
        a, b = highlight_index.size()
        # print(highlight_index.size())

        # for i in range(a):  # a就是被选择出来的点的个数，这些点我们需要将其标记为1
        #     highlight[highlight_index[i][0], highlight_index[i][1]] = 1
        # 以上，我们便初步获取得到了聚合特征图掩码矩阵
        # 然后，我们需要对掩码矩阵进行进一步的处理，获得最大连通区域
        # largestConnectComponent将最大连通区域所对应的像素点置为true
        jj = (feature_maps_L31_sum > L31_sum_mean).cpu().data.numpy()
        highlight = torch.from_numpy(jj + 0).cuda().float()

        highlight_conn_L31 = highlight
        # highlight_conn_L31 = torch.tensor(largestConnectComponent(highlight.cpu().numpy()) + 0)#[7*7],我们需要将其变成[512,7,7]

        feature_maps_L31 = feature_maps_L31[0].cpu().data.numpy()  # [1,512,7,7]==>[512,7,7]
        highlight_conn_L31_beifen = highlight_conn_L31
        highlight_conn_L31 = highlight_conn_L31.cpu().data.numpy()
        highlight_conn_L31 = highlight_conn_L31.reshape(1, h_L31, w_L31) * np.ones_like(
            feature_maps_L31)  # [7,7]=>[1,7,7]=>[512,7,7]

        feature_maps_L31 = feature_maps_L31 * highlight_conn_L31


        feature_maps_L31_max = np.max(feature_maps_L31, axis=(1, 2))
        if np.linalg.norm(feature_maps_L31_max) > 0:
            feature_maps_L31_max_norm = feature_maps_L31_max / np.linalg.norm(feature_maps_L31_max)
        else:
            print("出现了")
            feature_maps_L31_max_norm = np.zeros_like(feature_maps_L31_max)
            pass

        output_mean = []
        output_2=[]

        output_2.append(feature_maps_L31_max_norm)
        # print(feature_maps_L31_max_norm.shape)#2048



        # output_max=[]
        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['layer4.0.conv3.weight']:
                tmp = (param.grad).data
                # print(tmp)
                # tmp=torch.abs(tmp)
                # print(tmp)
                # tmp_mean_reuse=tmp.view(tmp.size()[0],-1)
                # print(tmp_mean_reuse.shape)
                # print(tmp.shape)#[2048,512,1,1]
                tmp_mean_reuse_std = torch.std(tmp, axis=(2, 3))

                tmp_mean_reuse = F.adaptive_max_pool2d(tmp, output_size=1).squeeze(-1).squeeze(
                    -1) + F.adaptive_max_pool2d(-1 * tmp, output_size=1).squeeze(-1).squeeze(-1)
                tmp_mean_reuse = tmp_mean_reuse + tmp_mean_reuse_std
                # tmp_mean_reuse=F.adaptive_max_pool2d(tmp,output_size=1).squeeze(-1).squeeze(-1)
                # tmp_mean_reuse=F.adaptive_max_pool2d(-1 * tmp,output_size=1).squeeze(-1).squeeze(-1)

                # tmp_mean_reuse_1 = torch.mean(tmp_1, axis=(2, 3))
                # tmp_mean_reuse = tmp_mean_reuse * tmp_mean_reuse_1
                # tmp_mean_reuse = torch.mean(tmp, dim=(2, 3))
                tmp_mean_reuse_bool = torch.mean(tmp_mean_reuse, dim=0).unsqueeze(0)
                tmp_mean_reuse_bool = tmp_mean_reuse > tmp_mean_reuse_bool
                tmp_mean_reuse_bool = tmp_mean_reuse_bool.float()

                tmp_mean_reuse_bool_2 = torch.mean(tmp_mean_reuse, dim=1).unsqueeze(0).T
                tmp_mean_reuse_bool_2 = tmp_mean_reuse > tmp_mean_reuse_bool_2
                tmp_mean_reuse_bool_2 = tmp_mean_reuse_bool_2.float()

                tmp_mean_reuse = tmp_mean_reuse * tmp_mean_reuse_bool * tmp_mean_reuse_bool_2

                tmp_mean = torch.mean(tmp_mean_reuse, dim=0).cpu().data.numpy()
                # tmp_mean,_ = torch.max(tmp_mean_reuse, dim=0)
                # tmp_mean=tmp_mean.numpy()
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean.append(tmp_mean)
                # print(tmp_mean.shape)


        return output_mean,output_2



class GradCam_yuhan_kernel_version30_softmax_scda_simple_test():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, ii=None,flip=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别

        feature_maps,output= self.net(input.cuda())
        output_yu=output+0
        output_yu = output_yu * -1
        output_max = torch.max(output_yu)
        output_min = torch.min(output_yu)
        output_norm = torch.div((output_yu - output_min).float(), (output_max - output_min))
        # ################################################################3
        output_norm = (output_norm * -1).cpu().tolist()
        output_norm = torch.tensor(output_norm).float().cuda()
        # ###################################################################
        one_hot = torch.sum(output * output_norm)
        # one_hot = torch.sum(output)

        self.net.zero_grad()
        one_hot.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #


        feature_maps_L31 = feature_maps[
            0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        batch_size, c_L31, h_L31, w_L31 = feature_maps_L31.size()

        # Pool5   #feature_maps_L31  [512,7,7]==[512,]
        feature_maps_L31_sum = torch.sum(feature_maps_L31[0],
                                         0)  # 如果要将CAM或者gard_cam引入SCDA，需要更改的程序也就仅仅是这一步
        L31_sum_mean = torch.mean(feature_maps_L31_sum)  # 返回的是scalar tensor ,这是与matlab的mean不同的地方
        highlight = torch.zeros(feature_maps_L31_sum.size())  # 生成了一个h*w的全零矩阵
        # print(highlight)
        highlight_index = torch.nonzero(feature_maps_L31_sum >= L31_sum_mean)  # 一个K*2的tensor,每一行一个符合要求的点
        a, b = highlight_index.size()
        # print(highlight_index.size())

        # for i in range(a):  # a就是被选择出来的点的个数，这些点我们需要将其标记为1
        #     highlight[highlight_index[i][0], highlight_index[i][1]] = 1
        # 以上，我们便初步获取得到了聚合特征图掩码矩阵
        # 然后，我们需要对掩码矩阵进行进一步的处理，获得最大连通区域
        # largestConnectComponent将最大连通区域所对应的像素点置为true
        jj = (feature_maps_L31_sum > L31_sum_mean).cpu().data.numpy()
        highlight = torch.from_numpy(jj + 0).cuda().float()

        highlight_conn_L31 = highlight
        # highlight_conn_L31 = torch.tensor(largestConnectComponent(highlight.cpu().numpy()) + 0)#[7*7],我们需要将其变成[512,7,7]

        feature_maps_L31 = feature_maps_L31[0].cpu().data.numpy()  # [1,512,7,7]==>[512,7,7]
        highlight_conn_L31_beifen = highlight_conn_L31
        highlight_conn_L31 = highlight_conn_L31.cpu().data.numpy()
        highlight_conn_L31 = highlight_conn_L31.reshape(1, h_L31, w_L31) * np.ones_like(
            feature_maps_L31)  # [7,7]=>[1,7,7]=>[512,7,7]

        feature_maps_L31 = feature_maps_L31 * highlight_conn_L31


        feature_maps_L31_max = np.max(feature_maps_L31, axis=(1, 2))
        if np.linalg.norm(feature_maps_L31_max) > 0:
            feature_maps_L31_max_norm = feature_maps_L31_max / np.linalg.norm(feature_maps_L31_max)
        else:
            print("出现了")
            feature_maps_L31_max_norm = np.zeros_like(feature_maps_L31_max)
            pass

        output_mean = []
        output_2=[]

        output_2.append(feature_maps_L31_max_norm)


        # output_max=[]
        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['features.28.weight']:
                tmp = (param.grad).data
                # print(tmp)
                # tmp=torch.abs(tmp)
                # print(tmp)
                # tmp_mean_reuse=tmp.view(tmp.size()[0],-1)
                # print(tmp_mean_reuse.shape)
                tmp_mean_reuse_std = torch.std(tmp, axis=(2, 3))

                tmp_mean_reuse = F.adaptive_max_pool2d(tmp, output_size=1).squeeze(-1).squeeze(
                    -1) + F.adaptive_max_pool2d(-1 * tmp, output_size=1).squeeze(-1).squeeze(-1)
                tmp_mean_reuse = tmp_mean_reuse + tmp_mean_reuse_std
                # tmp_mean_reuse=F.adaptive_max_pool2d(tmp,output_size=1).squeeze(-1).squeeze(-1)
                # tmp_mean_reuse=F.adaptive_max_pool2d(-1 * tmp,output_size=1).squeeze(-1).squeeze(-1)

                # tmp_mean_reuse_1 = torch.mean(tmp_1, axis=(2, 3))
                # tmp_mean_reuse = tmp_mean_reuse * tmp_mean_reuse_1
                # tmp_mean_reuse = torch.mean(tmp, dim=(2, 3))
                tmp_mean_reuse_bool = torch.mean(tmp_mean_reuse, dim=0).unsqueeze(0)
                tmp_mean_reuse_bool = tmp_mean_reuse > tmp_mean_reuse_bool
                tmp_mean_reuse_bool = tmp_mean_reuse_bool.float()

                tmp_mean_reuse_bool_2 = torch.mean(tmp_mean_reuse, dim=1).unsqueeze(0).T
                tmp_mean_reuse_bool_2 = tmp_mean_reuse > tmp_mean_reuse_bool_2
                tmp_mean_reuse_bool_2 = tmp_mean_reuse_bool_2.float()

                tmp_mean_reuse = tmp_mean_reuse * tmp_mean_reuse_bool * tmp_mean_reuse_bool_2

                tmp_mean = torch.mean(tmp_mean_reuse, dim=0).cpu().data.numpy()
                # tmp_mean,_ = torch.max(tmp_mean_reuse, dim=0)
                # tmp_mean=tmp_mean.numpy()
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean.append(tmp_mean)


        return output_mean,output_2










class GradCam_yuhan_kernel_version30_amsoftmax_embedding():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, ii=None,flip=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别

        _,embedding,output,_ = self.net(input.cuda())
        output_max = torch.max(output)
        output_min = torch.min(output)
        output_norm = torch.div((output - output_min).float(), (output_max - output_min))
        ################################################################3
        output_norm = (output_norm * -1).cpu().tolist()
        # output_norm = output_norm .cpu().tolist()

        output_norm = torch.tensor(output_norm).float().cuda()
        ###################################################################
        one_hot = torch.sum(output * output_norm)
        self.net.zero_grad()
        one_hot.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚

        output_mean = []
        output_2=[]


        embedding_yu=embedding[0].cpu().data.numpy()#batch_size=1
        if np.linalg.norm(embedding_yu) > 0:
            feature_maps_L28_max_norm = embedding_yu / np.linalg.norm(embedding_yu)
        else:
            print("出现了")
            feature_maps_L28_max_norm = np.zeros_like(embedding_yu)
            pass
        output_2.append(feature_maps_L28_max_norm)
        # output_2.append(feature_maps_L31_max_norm)
        # print(feature_maps_L28_max_norm.shape)

        # output_max=[]
        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['layer4.2.conv3.weight']:
                tmp = (param.grad).data
                # print(tmp)
                # tmp=torch.abs(tmp)
                # print(tmp)
                # tmp_mean_reuse=tmp.view(tmp.size()[0],-1)
                # print(tmp_mean_reuse.shape)
                tmp_mean_reuse_std = torch.std(tmp, axis=(2, 3))

                tmp_mean_reuse = F.adaptive_max_pool2d(tmp, output_size=1).squeeze(-1).squeeze(
                    -1) + F.adaptive_max_pool2d(-1 * tmp, output_size=1).squeeze(-1).squeeze(-1)
                tmp_mean_reuse = tmp_mean_reuse + tmp_mean_reuse_std
                # tmp_mean_reuse=F.adaptive_max_pool2d(tmp,output_size=1).squeeze(-1).squeeze(-1)
                # tmp_mean_reuse=F.adaptive_max_pool2d(-1 * tmp,output_size=1).squeeze(-1).squeeze(-1)

                # tmp_mean_reuse_1 = torch.mean(tmp_1, axis=(2, 3))
                # tmp_mean_reuse = tmp_mean_reuse * tmp_mean_reuse_1
                # tmp_mean_reuse = torch.mean(tmp, dim=(2, 3))
                tmp_mean_reuse_bool = torch.mean(tmp_mean_reuse, dim=0).unsqueeze(0)
                tmp_mean_reuse_bool = tmp_mean_reuse > tmp_mean_reuse_bool
                tmp_mean_reuse_bool = tmp_mean_reuse_bool.float()

                tmp_mean_reuse_bool_2 = torch.mean(tmp_mean_reuse, dim=1).unsqueeze(0).T
                tmp_mean_reuse_bool_2 = tmp_mean_reuse > tmp_mean_reuse_bool_2
                tmp_mean_reuse_bool_2 = tmp_mean_reuse_bool_2.float()

                tmp_mean_reuse = tmp_mean_reuse * tmp_mean_reuse_bool * tmp_mean_reuse_bool_2

                tmp_mean = torch.mean(tmp_mean_reuse, dim=0).cpu().data.numpy()
                # tmp_mean,_ = torch.max(tmp_mean_reuse, dim=0)
                # tmp_mean=tmp_mean.numpy()
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean.append(tmp_mean)


        return output_mean,output_2







class GradCam_yuhan_kernel_version30_amsoftmax_scda_simple():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, ii=None,flip=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别

        feature_maps,output,_ = self.net(input.cuda())
        output_max = torch.max(output)
        output_min = torch.min(output)
        output_norm = torch.div((output - output_min).float(), (output_max - output_min))
        ################################################################3
        output_norm = (output_norm).cpu().tolist()
        # output_norm = output_norm .cpu().tolist()

        output_norm = torch.tensor(output_norm).float().cuda()
        ###################################################################
        one_hot = torch.sum(output * output_norm)
        self.net.zero_grad()
        one_hot.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #


        feature_maps_L31 = feature_maps[
            0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        batch_size, c_L31, h_L31, w_L31 = feature_maps_L31.size()

        # Pool5   #feature_maps_L31  [512,7,7]==[512,]
        feature_maps_L31_sum = torch.sum(feature_maps_L31[0],
                                         0)  # 如果要将CAM或者gard_cam引入SCDA，需要更改的程序也就仅仅是这一步
        L31_sum_mean = torch.mean(feature_maps_L31_sum)  # 返回的是scalar tensor ,这是与matlab的mean不同的地方
        highlight = torch.zeros(feature_maps_L31_sum.size())  # 生成了一个h*w的全零矩阵
        # print(highlight)
        highlight_index = torch.nonzero(feature_maps_L31_sum >= L31_sum_mean)  # 一个K*2的tensor,每一行一个符合要求的点
        a, b = highlight_index.size()
        # print(highlight_index.size())

        # for i in range(a):  # a就是被选择出来的点的个数，这些点我们需要将其标记为1
        #     highlight[highlight_index[i][0], highlight_index[i][1]] = 1
        # 以上，我们便初步获取得到了聚合特征图掩码矩阵
        # 然后，我们需要对掩码矩阵进行进一步的处理，获得最大连通区域
        # largestConnectComponent将最大连通区域所对应的像素点置为true
        jj = (feature_maps_L31_sum > L31_sum_mean).cpu().data.numpy()
        highlight = torch.from_numpy(jj + 0).cuda().float()

        highlight_conn_L31 = highlight
        # highlight_conn_L31 = torch.tensor(largestConnectComponent(highlight.cpu().numpy()) + 0)#[7*7],我们需要将其变成[512,7,7]

        feature_maps_L31 = feature_maps_L31[0].cpu().data.numpy()  # [1,512,7,7]==>[512,7,7]
        highlight_conn_L31_beifen = highlight_conn_L31
        highlight_conn_L31 = highlight_conn_L31.cpu().data.numpy()
        highlight_conn_L31 = highlight_conn_L31.reshape(1, h_L31, w_L31) * np.ones_like(
            feature_maps_L31)  # [7,7]=>[1,7,7]=>[512,7,7]

        feature_maps_L31 = feature_maps_L31 * highlight_conn_L31


        feature_maps_L31_max = np.max(feature_maps_L31, axis=(1, 2))
        if np.linalg.norm(feature_maps_L31_max) > 0:
            feature_maps_L31_max_norm = feature_maps_L31_max / np.linalg.norm(feature_maps_L31_max)
        else:
            print("出现了")
            feature_maps_L31_max_norm = np.zeros_like(feature_maps_L31_max)
            pass


        output_mean = []
        output_2=[]

        output_2.append(feature_maps_L31_max_norm)
        # print(feature_maps_L31_max_norm.shape)

        # output_max=[]
        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['layer4.1.conv2.weight']:
                tmp = (param.grad).data
                # print(tmp)
                # print(tmp.shape)
                # tmp=torch.abs(tmp)
                # print(tmp.shape)
                # tmp_mean_reuse=tmp.squeeze(-1).squeeze(-1)
                # print(tmp_mean_reuse.shape)

                # tmp_mean_reuse=tmp.view(tmp.size()[0],-1)
                # print(tmp_mean_reuse.shape)
                #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
                tmp_mean_reuse_std = torch.std(tmp, axis=(2, 3))

                tmp_mean_reuse = F.adaptive_max_pool2d(tmp, output_size=1).squeeze(-1).squeeze(
                    -1) + F.adaptive_max_pool2d(-1 * tmp, output_size=1).squeeze(-1).squeeze(-1)
                tmp_mean_reuse = tmp_mean_reuse + tmp_mean_reuse_std
                #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
                # tmp_mean_reuse=F.adaptive_max_pool2d(tmp,output_size=1).squeeze(-1).squeeze(-1)
                # tmp_mean_reuse=F.adaptive_max_pool2d(-1 * tmp,output_size=1).squeeze(-1).squeeze(-1)

                # tmp_mean_reuse_1 = torch.mean(tmp_1, axis=(2, 3))
                # tmp_mean_reuse = tmp_mean_reuse * tmp_mean_reuse_1
                # tmp_mean_reuse = torch.mean(tmp, dim=(2, 3))
                tmp_mean_reuse_bool = torch.mean(tmp_mean_reuse, dim=0).unsqueeze(0)
                tmp_mean_reuse_bool = tmp_mean_reuse > tmp_mean_reuse_bool
                tmp_mean_reuse_bool = tmp_mean_reuse_bool.float()

                tmp_mean_reuse_bool_2 = torch.mean(tmp_mean_reuse, dim=1).unsqueeze(0).T
                tmp_mean_reuse_bool_2 = tmp_mean_reuse > tmp_mean_reuse_bool_2
                tmp_mean_reuse_bool_2 = tmp_mean_reuse_bool_2.float()

                tmp_mean_reuse = tmp_mean_reuse * tmp_mean_reuse_bool * tmp_mean_reuse_bool_2

                tmp_mean = torch.mean(tmp_mean_reuse, dim=0).cpu().data.numpy()
                # print(tmp_mean.shape)
                # tmp_mean,_ = torch.max(tmp_mean_reuse, dim=0)
                # tmp_mean=tmp_mean.numpy()
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean.append(tmp_mean)


        return output_mean,output_2







class GradCam_yuhan_kernel_version30_amsoftmax_scda_simple_2():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, ii=None,flip=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别

        feature_maps,output,_ = self.net(input.cuda())
        output_max = torch.max(output)
        output_min = torch.min(output)
        output_norm = torch.div((output - output_min).float(), (output_max - output_min))
        ################################################################3
        output_norm = (output_norm).cpu().tolist()
        # output_norm = output_norm .cpu().tolist()

        output_norm = torch.tensor(output_norm).float().cuda()
        ###################################################################
        one_hot = torch.sum(output * output_norm)
        self.net.zero_grad()
        one_hot.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #


        feature_maps_L31 = feature_maps[
            0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        batch_size, c_L31, h_L31, w_L31 = feature_maps_L31.size()

        # Pool5   #feature_maps_L31  [512,7,7]==[512,]
        feature_maps_L31_sum = torch.sum(feature_maps_L31[0],
                                         0)  # 如果要将CAM或者gard_cam引入SCDA，需要更改的程序也就仅仅是这一步
        L31_sum_mean = torch.mean(feature_maps_L31_sum)  # 返回的是scalar tensor ,这是与matlab的mean不同的地方
        highlight = torch.zeros(feature_maps_L31_sum.size())  # 生成了一个h*w的全零矩阵
        # print(highlight)
        highlight_index = torch.nonzero(feature_maps_L31_sum >= L31_sum_mean)  # 一个K*2的tensor,每一行一个符合要求的点
        a, b = highlight_index.size()
        # print(highlight_index.size())

        # for i in range(a):  # a就是被选择出来的点的个数，这些点我们需要将其标记为1
        #     highlight[highlight_index[i][0], highlight_index[i][1]] = 1
        # 以上，我们便初步获取得到了聚合特征图掩码矩阵
        # 然后，我们需要对掩码矩阵进行进一步的处理，获得最大连通区域
        # largestConnectComponent将最大连通区域所对应的像素点置为true
        jj = (feature_maps_L31_sum > L31_sum_mean).cpu().data.numpy()
        highlight = torch.from_numpy(jj + 0).cuda().float()

        highlight_conn_L31 = highlight
        # highlight_conn_L31 = torch.tensor(largestConnectComponent(highlight.cpu().numpy()) + 0)#[7*7],我们需要将其变成[512,7,7]

        feature_maps_L31 = feature_maps_L31[0].cpu().data.numpy()  # [1,512,7,7]==>[512,7,7]
        highlight_conn_L31_beifen = highlight_conn_L31
        highlight_conn_L31 = highlight_conn_L31.cpu().data.numpy()
        highlight_conn_L31 = highlight_conn_L31.reshape(1, h_L31, w_L31) * np.ones_like(
            feature_maps_L31)  # [7,7]=>[1,7,7]=>[512,7,7]

        feature_maps_L31 = feature_maps_L31 * highlight_conn_L31


        feature_maps_L31_max = np.max(feature_maps_L31, axis=(1, 2))
        if np.linalg.norm(feature_maps_L31_max) > 0:
            feature_maps_L31_max_norm = feature_maps_L31_max / np.linalg.norm(feature_maps_L31_max)
        else:
            print("出现了")
            feature_maps_L31_max_norm = np.zeros_like(feature_maps_L31_max)
            pass


        output_mean = []
        output_2=[]

        output_2.append(feature_maps_L31_max_norm)
        # print(feature_maps_L31_max_norm.shape)

        # output_max=[]
        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['layer4.2.bn2.weight']:
                tmp = (param.grad).data
                tmp_mean=tmp.cpu().data.numpy()
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean.append(tmp_mean)


        return output_mean,output_2























class GradCam_yuhan_kernel_version30_amsoftmax_scda_2():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, ii=None,flip=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别

        feature_maps,output,_ ,_= self.net(input.cuda())
        output_max = torch.max(output)
        output_min = torch.min(output)
        output_norm = torch.div((output - output_min).float(), (output_max - output_min))
        ################################################################3
        output_norm = (output_norm * -1).cpu().tolist()
        output_norm = torch.tensor(output_norm).float().cuda()
        ###################################################################
        one_hot = torch.sum(output * output_norm)
        self.net.zero_grad()
        one_hot.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #

        feature_maps_L28 = feature_maps[
            0]  # 如果输入图像的shape为[batch_size,3,224,224]的tensor的话，关于输出的feature map,shape为  #[batch_size,512,14,14]左右
        batch_size, c_L28, h_L28, w_L28 = feature_maps_L28.size()
        feature_maps_L31 = feature_maps[
            1]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        batch_size, c_L31, h_L31, w_L31 = feature_maps_L31.size()

        # Pool5   #feature_maps_L31  [512,7,7]==[512,]
        feature_maps_L31_sum = torch.sum(feature_maps_L31[0],
                                         0)  # 如果要将CAM或者gard_cam引入SCDA，需要更改的程序也就仅仅是这一步
        L31_sum_mean = torch.mean(feature_maps_L31_sum)  # 返回的是scalar tensor ,这是与matlab的mean不同的地方
        highlight = torch.zeros(feature_maps_L31_sum.size())  # 生成了一个h*w的全零矩阵
        # print(highlight)
        highlight_index = torch.nonzero(feature_maps_L31_sum >= L31_sum_mean)  # 一个K*2的tensor,每一行一个符合要求的点
        a, b = highlight_index.size()
        # print(highlight_index.size())

        # for i in range(a):  # a就是被选择出来的点的个数，这些点我们需要将其标记为1
        #     highlight[highlight_index[i][0], highlight_index[i][1]] = 1
        # 以上，我们便初步获取得到了聚合特征图掩码矩阵
        # 然后，我们需要对掩码矩阵进行进一步的处理，获得最大连通区域
        # largestConnectComponent将最大连通区域所对应的像素点置为true
        jj = (feature_maps_L31_sum > L31_sum_mean).cpu().data.numpy()
        highlight = torch.from_numpy(jj + 0).cuda().float()

        highlight_conn_L31 = highlight
        # highlight_conn_L31 = torch.tensor(largestConnectComponent(highlight.cpu().numpy()) + 0)#[7*7],我们需要将其变成[512,7,7]

        feature_maps_L31 = feature_maps_L31[0].cpu().data.numpy()  # [1,512,7,7]==>[512,7,7]
        highlight_conn_L31_beifen = highlight_conn_L31
        highlight_conn_L31 = highlight_conn_L31.cpu().data.numpy()
        highlight_conn_L31 = highlight_conn_L31.reshape(1, h_L31, w_L31) * np.ones_like(
            feature_maps_L31)  # [7,7]=>[1,7,7]=>[512,7,7]

        feature_maps_L31 = feature_maps_L31 * highlight_conn_L31


        feature_maps_L31_max = np.max(feature_maps_L31, axis=(1, 2))
        if np.linalg.norm(feature_maps_L31_max) > 0:
            feature_maps_L31_max_norm = feature_maps_L31_max / np.linalg.norm(feature_maps_L31_max)
        else:
            print("出现了")
            feature_maps_L31_max_norm = np.zeros_like(feature_maps_L31_max)
            pass
        # % Relu5_2
        feature_maps_L28_sum = torch.sum(feature_maps_L28[0],
                                         0)  # 如果要将CAM或者gard_cam引入SCDA，需要更改的程序也就仅仅是这一步
        L28_sum_mean = torch.mean(feature_maps_L28_sum)  # 返回的是scalar tensor ,这是与matlab的mean不同的地方
        highlight = torch.zeros(feature_maps_L28_sum.size())  # 生成了一个h*w的全零矩阵
        highlight_index = torch.nonzero(feature_maps_L28_sum >= L28_sum_mean)  # 一个K*2的tensor,每一行一个符合要求的点
        a, b = highlight_index.size()
        # for i in range(a):  # a就是被选择出来的点的个数，这些点我们需要将其标记为1
        #     highlight[highlight_index[i][0], highlight_index[i][1]] = 1

        jj = (feature_maps_L28_sum > L28_sum_mean).cpu().data.numpy()
        highlight = torch.from_numpy(jj + 0).cuda().float()

        highlight_conn_L31 = highlight_conn_L31_beifen
        highlight_conn_L31_to_L28 = interpolate(highlight_conn_L31.view(1, 1, h_L31, w_L31).float(),
                                                size=[h_L28, w_L28], mode="nearest").view(h_L28, w_L28)
        highlight_conn_L28 = highlight.mul(highlight_conn_L31_to_L28.cuda())  # 逐点按元素想乘，两个都是二值矩阵，可不就是按位与嘛
        feature_maps_L28 = feature_maps_L28[0].cpu().data.numpy()  # [1,512,7,7]==>[512,7,7]
        highlight_conn_L28 = highlight_conn_L28.cpu().data.numpy()
        highlight_conn_L28 = highlight_conn_L28.reshape(1, h_L28, w_L28) * np.ones_like(
            feature_maps_L28)  # [7,7]=>[1,7,7]=>[512,7,7]

        feature_maps_L28 = feature_maps_L28 * highlight_conn_L28

        feature_maps_L28_max = np.max(feature_maps_L28, axis=(1, 2))
        if np.linalg.norm(feature_maps_L28_max) > 0:
            feature_maps_L28_max_norm = feature_maps_L28_max / np.linalg.norm(feature_maps_L28_max)
        else:
            print("出现了")
            feature_maps_L28_max_norm = np.zeros_like(feature_maps_L28_max)
            pass

        output_mean = []
        output_2=[]



        output_2.append(feature_maps_L28_max_norm)
        output_2.append(feature_maps_L31_max_norm)


        # output_max=[]
        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['features.26.weight', 'features.28.weight']:
                tmp = (param.grad).data
                # print(tmp)
                # tmp=torch.abs(tmp)
                # print(tmp)
                # tmp_mean_reuse=tmp.view(tmp.size()[0],-1)
                # print(tmp_mean_reuse.shape)
                tmp_mean_reuse_std = torch.std(tmp, axis=(2, 3))

                tmp_mean_reuse = F.adaptive_max_pool2d(tmp, output_size=1).squeeze(-1).squeeze(
                    -1) + F.adaptive_max_pool2d(-1 * tmp, output_size=1).squeeze(-1).squeeze(-1)
                tmp_mean_reuse = tmp_mean_reuse + tmp_mean_reuse_std
                # tmp_mean_reuse=F.adaptive_max_pool2d(tmp,output_size=1).squeeze(-1).squeeze(-1)
                # tmp_mean_reuse=F.adaptive_max_pool2d(-1 * tmp,output_size=1).squeeze(-1).squeeze(-1)

                # tmp_mean_reuse_1 = torch.mean(tmp_1, axis=(2, 3))
                # tmp_mean_reuse = tmp_mean_reuse * tmp_mean_reuse_1
                # tmp_mean_reuse = torch.mean(tmp, dim=(2, 3))
                tmp_mean_reuse_bool = torch.mean(tmp_mean_reuse, dim=0).unsqueeze(0)
                tmp_mean_reuse_bool = tmp_mean_reuse > tmp_mean_reuse_bool
                tmp_mean_reuse_bool = tmp_mean_reuse_bool.float()

                tmp_mean_reuse_bool_2 = torch.mean(tmp_mean_reuse, dim=1).unsqueeze(0).T
                tmp_mean_reuse_bool_2 = tmp_mean_reuse > tmp_mean_reuse_bool_2
                tmp_mean_reuse_bool_2 = tmp_mean_reuse_bool_2.float()

                tmp_mean_reuse = tmp_mean_reuse * tmp_mean_reuse_bool * tmp_mean_reuse_bool_2

                tmp_mean = torch.mean(tmp_mean_reuse, dim=0).cpu().data.numpy()
                # tmp_mean,_ = torch.max(tmp_mean_reuse, dim=0)
                # tmp_mean=tmp_mean.numpy()
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean.append(tmp_mean)


        return output_mean,output_2











class GradCam_yuhan_kernel_version30_amsoftmax_scda_for_retrivial():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, ii=None,flip=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别

        feature_maps,output,_ = self.net(input.cuda())
        output_max = torch.max(output)
        output_min = torch.min(output)
        output_norm = torch.div((output - output_min).float(), (output_max - output_min))
        ################################################################3
        output_norm = (output_norm * -1).cpu().tolist()
        output_norm = torch.tensor(output_norm).float().cuda()
        ###################################################################
        one_hot = torch.sum(output * output_norm)
        self.net.zero_grad()
        one_hot.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #

        feature_maps_L28 = feature_maps[
            0]  # 如果输入图像的shape为[batch_size,3,224,224]的tensor的话，关于输出的feature map,shape为  #[batch_size,512,14,14]左右
        batch_size, c_L28, h_L28, w_L28 = feature_maps_L28.size()
        feature_maps_L31 = feature_maps[
            1]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        batch_size, c_L31, h_L31, w_L31 = feature_maps_L31.size()

        # Pool5   #feature_maps_L31  [512,7,7]==[512,]
        feature_maps_L31_sum = torch.sum(feature_maps_L31[0],
                                         0)  # 如果要将CAM或者gard_cam引入SCDA，需要更改的程序也就仅仅是这一步
        L31_sum_mean = torch.mean(feature_maps_L31_sum)  # 返回的是scalar tensor ,这是与matlab的mean不同的地方
        highlight = torch.zeros(feature_maps_L31_sum.size())  # 生成了一个h*w的全零矩阵
        # print(highlight)
        highlight_index = torch.nonzero(feature_maps_L31_sum >= L31_sum_mean)  # 一个K*2的tensor,每一行一个符合要求的点
        a, b = highlight_index.size()
        # print(highlight_index.size())

        # for i in range(a):  # a就是被选择出来的点的个数，这些点我们需要将其标记为1
        #     highlight[highlight_index[i][0], highlight_index[i][1]] = 1
        # 以上，我们便初步获取得到了聚合特征图掩码矩阵
        # 然后，我们需要对掩码矩阵进行进一步的处理，获得最大连通区域
        # largestConnectComponent将最大连通区域所对应的像素点置为true
        jj = (feature_maps_L31_sum > L31_sum_mean).cpu().data.numpy()
        highlight = torch.from_numpy(jj + 0).cuda().float()

        highlight_conn_L31 = highlight
        # highlight_conn_L31 = torch.tensor(largestConnectComponent(highlight.cpu().numpy()) + 0)#[7*7],我们需要将其变成[512,7,7]

        feature_maps_L31 = feature_maps_L31[0].cpu().data.numpy()  # [1,512,7,7]==>[512,7,7]
        highlight_conn_L31_beifen = highlight_conn_L31
        highlight_conn_L31 = highlight_conn_L31.cpu().data.numpy()
        highlight_conn_L31 = highlight_conn_L31.reshape(1, h_L31, w_L31) * np.ones_like(
            feature_maps_L31)  # [7,7]=>[1,7,7]=>[512,7,7]

        feature_maps_L31 = feature_maps_L31 * highlight_conn_L31


        feature_maps_L31_max = np.max(feature_maps_L31, axis=(1, 2))
        if np.linalg.norm(feature_maps_L31_max) > 0:
            feature_maps_L31_max_norm = feature_maps_L31_max / np.linalg.norm(feature_maps_L31_max)
        else:
            print("出现了")
            feature_maps_L31_max_norm = np.zeros_like(feature_maps_L31_max)
            pass
        # % Relu5_2
        feature_maps_L28_sum = torch.sum(feature_maps_L28[0],
                                         0)  # 如果要将CAM或者gard_cam引入SCDA，需要更改的程序也就仅仅是这一步
        L28_sum_mean = torch.mean(feature_maps_L28_sum)  # 返回的是scalar tensor ,这是与matlab的mean不同的地方
        highlight = torch.zeros(feature_maps_L28_sum.size())  # 生成了一个h*w的全零矩阵
        highlight_index = torch.nonzero(feature_maps_L28_sum >= L28_sum_mean)  # 一个K*2的tensor,每一行一个符合要求的点
        a, b = highlight_index.size()
        # for i in range(a):  # a就是被选择出来的点的个数，这些点我们需要将其标记为1
        #     highlight[highlight_index[i][0], highlight_index[i][1]] = 1

        jj = (feature_maps_L28_sum > L28_sum_mean).cpu().data.numpy()
        highlight = torch.from_numpy(jj + 0).cuda().float()

        highlight_conn_L31 = highlight_conn_L31_beifen
        highlight_conn_L31_to_L28 = interpolate(highlight_conn_L31.view(1, 1, h_L31, w_L31).float(),
                                                size=[h_L28, w_L28], mode="nearest").view(h_L28, w_L28)
        highlight_conn_L28 = highlight.mul(highlight_conn_L31_to_L28.cuda())  # 逐点按元素想乘，两个都是二值矩阵，可不就是按位与嘛
        feature_maps_L28 = feature_maps_L28[0].cpu().data.numpy()  # [1,512,7,7]==>[512,7,7]
        highlight_conn_L28 = highlight_conn_L28.cpu().data.numpy()
        highlight_conn_L28_beifen = highlight_conn_L28

        highlight_conn_L28 = highlight_conn_L28.reshape(1, h_L28, w_L28) * np.ones_like(
            feature_maps_L28)  # [7,7]=>[1,7,7]=>[512,7,7]

        feature_maps_L28 = feature_maps_L28 * highlight_conn_L28

        feature_maps_L28_max = np.max(feature_maps_L28, axis=(1, 2))
        if np.linalg.norm(feature_maps_L28_max) > 0:
            feature_maps_L28_max_norm = feature_maps_L28_max / np.linalg.norm(feature_maps_L28_max)
        else:
            print("出现了")
            feature_maps_L28_max_norm = np.zeros_like(feature_maps_L28_max)
            pass

        output_mean = []
        output_2=[]
        out_for_visualize=[feature_maps,highlight_conn_L31_beifen.cpu().numpy(),highlight_conn_L28_beifen]



        output_2.append(feature_maps_L28_max_norm)
        output_2.append(feature_maps_L31_max_norm)


        # output_max=[]
        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['features.26.weight', 'features.28.weight']:
                tmp = (param.grad).data
                # print(tmp)
                # tmp=torch.abs(tmp)
                # print(tmp)
                # tmp_mean_reuse=tmp.view(tmp.size()[0],-1)
                # print(tmp_mean_reuse.shape)
                tmp_mean_reuse_std = torch.std(tmp, axis=(2, 3))

                tmp_mean_reuse = F.adaptive_max_pool2d(tmp, output_size=1).squeeze(-1).squeeze(
                    -1) + F.adaptive_max_pool2d(-1 * tmp, output_size=1).squeeze(-1).squeeze(-1)
                tmp_mean_reuse = tmp_mean_reuse + tmp_mean_reuse_std
                # tmp_mean_reuse=F.adaptive_max_pool2d(tmp,output_size=1).squeeze(-1).squeeze(-1)
                # tmp_mean_reuse=F.adaptive_max_pool2d(-1 * tmp,output_size=1).squeeze(-1).squeeze(-1)

                # tmp_mean_reuse_1 = torch.mean(tmp_1, axis=(2, 3))
                # tmp_mean_reuse = tmp_mean_reuse * tmp_mean_reuse_1
                # tmp_mean_reuse = torch.mean(tmp, dim=(2, 3))
                tmp_mean_reuse_bool = torch.mean(tmp_mean_reuse, dim=0).unsqueeze(0)
                tmp_mean_reuse_bool = tmp_mean_reuse > tmp_mean_reuse_bool
                tmp_mean_reuse_bool = tmp_mean_reuse_bool.float()

                tmp_mean_reuse_bool_2 = torch.mean(tmp_mean_reuse, dim=1).unsqueeze(0).T
                tmp_mean_reuse_bool_2 = tmp_mean_reuse > tmp_mean_reuse_bool_2
                tmp_mean_reuse_bool_2 = tmp_mean_reuse_bool_2.float()

                tmp_mean_reuse = tmp_mean_reuse * tmp_mean_reuse_bool * tmp_mean_reuse_bool_2

                tmp_mean = torch.mean(tmp_mean_reuse, dim=0).cpu().data.numpy()
                # tmp_mean,_ = torch.max(tmp_mean_reuse, dim=0)
                # tmp_mean=tmp_mean.numpy()
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean.append(tmp_mean)


        return output_mean,output_2,out_for_visualize


class GradCam_yuhan_kernel_version30_amsoftmax_scda_for_retrivial_2():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, ii=None,flip=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别

        feature_maps,output = self.net(input.cuda())
        output_max = torch.max(output)
        output_min = torch.min(output)
        output_norm = torch.div((output - output_min).float(), (output_max - output_min))
        ################################################################3
        output_norm = (output_norm * -1).cpu().tolist()
        output_norm = torch.tensor(output_norm).float().cuda()
        ###################################################################
        one_hot = torch.sum(output * output_norm)
        self.net.zero_grad()
        one_hot.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #

        feature_maps_L28 = feature_maps[
            0]  # 如果输入图像的shape为[batch_size,3,224,224]的tensor的话，关于输出的feature map,shape为  #[batch_size,512,14,14]左右
        batch_size, c_L28, h_L28, w_L28 = feature_maps_L28.size()
        feature_maps_L31 = feature_maps[
            1]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        batch_size, c_L31, h_L31, w_L31 = feature_maps_L31.size()

        # Pool5   #feature_maps_L31  [512,7,7]==[512,]
        feature_maps_L31_sum = torch.sum(feature_maps_L31[0],
                                         0)  # 如果要将CAM或者gard_cam引入SCDA，需要更改的程序也就仅仅是这一步
        L31_sum_mean = torch.mean(feature_maps_L31_sum)  # 返回的是scalar tensor ,这是与matlab的mean不同的地方
        highlight = torch.zeros(feature_maps_L31_sum.size())  # 生成了一个h*w的全零矩阵
        # print(highlight)
        highlight_index = torch.nonzero(feature_maps_L31_sum >= L31_sum_mean)  # 一个K*2的tensor,每一行一个符合要求的点
        a, b = highlight_index.size()
        # print(highlight_index.size())

        # for i in range(a):  # a就是被选择出来的点的个数，这些点我们需要将其标记为1
        #     highlight[highlight_index[i][0], highlight_index[i][1]] = 1
        # 以上，我们便初步获取得到了聚合特征图掩码矩阵
        # 然后，我们需要对掩码矩阵进行进一步的处理，获得最大连通区域
        # largestConnectComponent将最大连通区域所对应的像素点置为true
        jj = (feature_maps_L31_sum > L31_sum_mean).cpu().data.numpy()
        highlight = torch.from_numpy(jj + 0).cuda().float()

        highlight_conn_L31 = highlight
        # highlight_conn_L31 = torch.tensor(largestConnectComponent(highlight.cpu().numpy()) + 0)#[7*7],我们需要将其变成[512,7,7]

        feature_maps_L31 = feature_maps_L31[0].cpu().data.numpy()  # [1,512,7,7]==>[512,7,7]
        highlight_conn_L31_beifen = highlight_conn_L31
        highlight_conn_L31 = highlight_conn_L31.cpu().data.numpy()
        highlight_conn_L31 = highlight_conn_L31.reshape(1, h_L31, w_L31) * np.ones_like(
            feature_maps_L31)  # [7,7]=>[1,7,7]=>[512,7,7]

        feature_maps_L31 = feature_maps_L31 * highlight_conn_L31


        feature_maps_L31_max = np.max(feature_maps_L31, axis=(1, 2))
        if np.linalg.norm(feature_maps_L31_max) > 0:
            feature_maps_L31_max_norm = feature_maps_L31_max / np.linalg.norm(feature_maps_L31_max)
        else:
            print("出现了")
            feature_maps_L31_max_norm = np.zeros_like(feature_maps_L31_max)
            pass
        # % Relu5_2
        feature_maps_L28_sum = torch.sum(feature_maps_L28[0],
                                         0)  # 如果要将CAM或者gard_cam引入SCDA，需要更改的程序也就仅仅是这一步
        L28_sum_mean = torch.mean(feature_maps_L28_sum)  # 返回的是scalar tensor ,这是与matlab的mean不同的地方
        highlight = torch.zeros(feature_maps_L28_sum.size())  # 生成了一个h*w的全零矩阵
        highlight_index = torch.nonzero(feature_maps_L28_sum >= L28_sum_mean)  # 一个K*2的tensor,每一行一个符合要求的点
        a, b = highlight_index.size()
        # for i in range(a):  # a就是被选择出来的点的个数，这些点我们需要将其标记为1
        #     highlight[highlight_index[i][0], highlight_index[i][1]] = 1

        jj = (feature_maps_L28_sum > L28_sum_mean).cpu().data.numpy()
        highlight = torch.from_numpy(jj + 0).cuda().float()

        highlight_conn_L31 = highlight_conn_L31_beifen
        highlight_conn_L31_to_L28 = interpolate(highlight_conn_L31.view(1, 1, h_L31, w_L31).float(),
                                                size=[h_L28, w_L28], mode="nearest").view(h_L28, w_L28)
        highlight_conn_L28 = highlight.mul(highlight_conn_L31_to_L28.cuda())  # 逐点按元素想乘，两个都是二值矩阵，可不就是按位与嘛
        feature_maps_L28 = feature_maps_L28[0].cpu().data.numpy()  # [1,512,7,7]==>[512,7,7]
        highlight_conn_L28 = highlight_conn_L28.cpu().data.numpy()
        highlight_conn_L28_beifen = highlight_conn_L28

        highlight_conn_L28 = highlight_conn_L28.reshape(1, h_L28, w_L28) * np.ones_like(
            feature_maps_L28)  # [7,7]=>[1,7,7]=>[512,7,7]

        feature_maps_L28 = feature_maps_L28 * highlight_conn_L28

        feature_maps_L28_max = np.max(feature_maps_L28, axis=(1, 2))
        if np.linalg.norm(feature_maps_L28_max) > 0:
            feature_maps_L28_max_norm = feature_maps_L28_max / np.linalg.norm(feature_maps_L28_max)
        else:
            print("出现了")
            feature_maps_L28_max_norm = np.zeros_like(feature_maps_L28_max)
            pass

        output_mean = []
        output_2=[]
        out_for_visualize=[feature_maps,highlight_conn_L31_beifen.cpu().numpy(),highlight_conn_L28_beifen]



        output_2.append(feature_maps_L28_max_norm)
        output_2.append(feature_maps_L31_max_norm)


        # output_max=[]
        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['features.26.weight', 'features.28.weight']:
                tmp = (param.grad).data
                # print(tmp)
                # tmp=torch.abs(tmp)
                # print(tmp)
                # tmp_mean_reuse=tmp.view(tmp.size()[0],-1)
                # print(tmp_mean_reuse.shape)
                tmp_mean_reuse_std = torch.std(tmp, axis=(2, 3))

                tmp_mean_reuse = F.adaptive_max_pool2d(tmp, output_size=1).squeeze(-1).squeeze(
                    -1) + F.adaptive_max_pool2d(-1 * tmp, output_size=1).squeeze(-1).squeeze(-1)
                tmp_mean_reuse = tmp_mean_reuse + tmp_mean_reuse_std
                # tmp_mean_reuse=F.adaptive_max_pool2d(tmp,output_size=1).squeeze(-1).squeeze(-1)
                # tmp_mean_reuse=F.adaptive_max_pool2d(-1 * tmp,output_size=1).squeeze(-1).squeeze(-1)

                # tmp_mean_reuse_1 = torch.mean(tmp_1, axis=(2, 3))
                # tmp_mean_reuse = tmp_mean_reuse * tmp_mean_reuse_1
                # tmp_mean_reuse = torch.mean(tmp, dim=(2, 3))
                tmp_mean_reuse_bool = torch.mean(tmp_mean_reuse, dim=0).unsqueeze(0)
                tmp_mean_reuse_bool = tmp_mean_reuse > tmp_mean_reuse_bool
                tmp_mean_reuse_bool = tmp_mean_reuse_bool.float()

                tmp_mean_reuse_bool_2 = torch.mean(tmp_mean_reuse, dim=1).unsqueeze(0).T
                tmp_mean_reuse_bool_2 = tmp_mean_reuse > tmp_mean_reuse_bool_2
                tmp_mean_reuse_bool_2 = tmp_mean_reuse_bool_2.float()

                tmp_mean_reuse = tmp_mean_reuse * tmp_mean_reuse_bool * tmp_mean_reuse_bool_2

                tmp_mean = torch.mean(tmp_mean_reuse, dim=0).cpu().data.numpy()
                # tmp_mean,_ = torch.max(tmp_mean_reuse, dim=0)
                # tmp_mean=tmp_mean.numpy()
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean.append(tmp_mean)


        return output_mean,output_2,out_for_visualize












class GradCam_yuhan_kernel_version30_10():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, ii=None,flip=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别

        feature_maps,output = self.net(input.cuda())
        output_max = torch.max(output)
        output_min = torch.min(output)
        output_norm = torch.div((output - output_min).float(), (output_max - output_min))
        ################################################################3
        output_norm = (output_norm * -1).cpu().tolist()
        output_norm = torch.tensor(output_norm).float().cuda()
        ###################################################################
        one_hot = torch.sum(output * output_norm)
        self.net.zero_grad()
        one_hot.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #




        feature_maps_L28 = feature_maps[
            0][0]  # 如果输入图像的shape为[batch_size,3,224,224]的tensor的话，关于输出的feature map,shape为  #[batch_size,512,14,14]左右
        # print(feature_maps_L28.shape)
        # print(feature_maps_L28)
        feature_maps_L28_channelwise_mean = torch.mean(feature_maps_L28, dim=(1, 2))
        # print(feature_maps_L28_channelwise_mean.shape)
        highlight_chanelwise_L28 = (feature_maps_L28 > feature_maps_L28_channelwise_mean.view(-1, 1, 1)).float()
        # print(highlight_chanelwise_L28.shape)#为这512通道的每一个通道生成了一个（由该通道均值决定的）专属于该通道的二值掩码矩阵，将该通道的显著性区域标注了出来
        feature_maps_L28_sum = torch.sum(feature_maps_L28, dim=0)
        # print(feature_maps_L28_sum.shape)
        highlight_L28 = (feature_maps_L28_sum > torch.mean(feature_maps_L28_sum)).float()
        # print(highlight_L28.shape)
        h28 = highlight_chanelwise_L28 * highlight_L28.unsqueeze(
            0)  # h28是一个【512，h,w】的二值矩阵，这是与总体二值矩阵交集得到的，可能在某一个通道上是一个全零矩阵
        # （一个完全对背景噪声响应的通道）
        # print(h28.shape)
        feature_maps_L28 = feature_maps_L28 * h28  # 同为shape为【512,7,7】的feature_map 与通道级二值矩阵（总体显著性区域与通道级显著性区域的交集）进行点乘，
        # chanelweight_l28 = torch.nn.functional.adaptive_max_pool2d(feature_maps_L28, output_size=1).squeeze(-1).squeeze(-1)
        chanelweight_l28 = torch.sum(feature_maps_L28, dim=(1, 2)) #/ torch.sum(h28, dim=(1, 2))
        # chanelweight_l28=F.adaptive_max_pool2d(feature_maps_L28
        #                                        ,output_size=1).squeeze(-1).squeeze(-1)
        # print(chanelweight_l28)
        # print(chanelweight_l28.shape)
        # 我们用0代替了nan值
        chanelweight_l28 = torch.where(torch.isnan(chanelweight_l28), torch.full_like(chanelweight_l28, 0),
                                       chanelweight_l28)
        # chanelweight_l28=chanelweight_l28/torch.sum(highlight_L28)
        # print(chanelweight_l28)
        # chanelweight_l28 = (chanelweight_l28 - torch.min(chanelweight_l28)) / (
        #         torch.max(chanelweight_l28) - torch.min(chanelweight_l28))
        # chanelweight_l28=(chanelweight_l28>torch.mean(chanelweight_l28)).float()
        # chanelweight_l28_bool = (chanelweight_l28 > torch.mean(chanelweight_l28)).float()
        # chanelweight_l28_bool_2 = torch.ones_like(chanelweight_l28_bool) - chanelweight_l28_bool
        # chanelweight_l28 = chanelweight_l28_bool * chanelweight_l28
        # chanelweight_l28 = chanelweight_l28 + chanelweight_l28_bool
        # print(chanelweight_l28)
        chanelweight_l28 = chanelweight_l28.cpu().data.numpy()
        aa = np.linalg.norm(chanelweight_l28)
        if aa != 0:
            chanelweight_l28 = chanelweight_l28 / aa
        else:
            chanelweight_l28 = np.zeros_like(chanelweight_l28)




        feature_maps_L31 = feature_maps[
            1][
            0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        feature_maps_L31_channelwise_mean = torch.mean(feature_maps_L31, dim=(1, 2))
        highlight_chanelwise_L31 = (feature_maps_L31 > feature_maps_L31_channelwise_mean.view(-1, 1,
                                                                                              1)).float()  # 得到的结果依然是【512,7,7】的一个二值矩阵
        feature_maps_L31_sum = torch.sum(feature_maps_L31, dim=0)
        highlight_L31 = (feature_maps_L31_sum > torch.mean(feature_maps_L31_sum)).float()
        h31 = highlight_chanelwise_L31 * highlight_L31.unsqueeze(0)
        feature_maps_L31 = feature_maps_L31 * h31
        # chanelweight_l31 = torch.nn.functional.adaptive_max_pool2d(feature_maps_L31, output_size=1).squeeze(-1).squeeze(
        #     -1)
        chanelweight_l31 = torch.sum(feature_maps_L31, dim=(1, 2)) #/ torch.sum(h31, dim=(1, 2))
        # chanelweight_l31 = F.adaptive_max_pool2d(feature_maps_L31
        #                                          , output_size=1).squeeze(-1).squeeze(-1)
        chanelweight_l31 = torch.where(torch.isnan(chanelweight_l31), torch.full_like(chanelweight_l31, 0),
                                       chanelweight_l31)
        chanelweight_l31=chanelweight_l31.cpu().data.numpy()

        # chanelweight_l31=torch.sum(h31,dim=(1,2))
        # chanelweight_l31=chanelweight_l31/torch.sum(highlight_L31)
        # chanelweight_l31 = (chanelweight_l31 - torch.min(chanelweight_l31)) / (
        #         torch.max(chanelweight_l31) - torch.min(chanelweight_l31))

        aa = np.linalg.norm(chanelweight_l31)
        if aa != 0:
            chanelweight_l31 = chanelweight_l31 / aa
        else:
            chanelweight_l31 = np.zeros_like(chanelweight_l31)

        # chanelweight_l31=(chanelweight_l31>torch.mean(chanelweight_l31)).float()

        #
        # chanelweight_l31_bool=(chanelweight_l31>torch.mean(chanelweight_l31)).float()
        # chanelweight_l31_bool_2=torch.ones_like(chanelweight_l31_bool)-chanelweight_l31_bool
        # chanelweight_l31=chanelweight_l31_bool * chanelweight_l31
        # chanelweight_l31=chanelweight_l31 + chanelweight_l31_bool
        # print(chanelweight_l31)
        # print(torch.sum((chanelweight_l31/np.linalg.norm(chanelweight_l31.cpu().data.numpy()))*(chanelweight_l28/np.linalg.norm(chanelweight_l28.cpu().data.numpy()))))

        ###************
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #

        output_mean = []
        output_2=[]



        output_2.append(chanelweight_l28)
        output_2.append(chanelweight_l31)


        # output_max=[]
        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['features.26.weight', 'features.28.weight']:
                tmp = (param.grad).data
                # print(tmp)
                # tmp=torch.abs(tmp)
                # print(tmp)
                # tmp_mean_reuse=tmp.view(tmp.size()[0],-1)
                # print(tmp_mean_reuse.shape)
                tmp_mean_reuse_std = torch.std(tmp, axis=(2, 3))

                tmp_mean_reuse = F.adaptive_max_pool2d(tmp, output_size=1).squeeze(-1).squeeze(
                    -1) + F.adaptive_max_pool2d(-1 * tmp, output_size=1).squeeze(-1).squeeze(-1)
                tmp_mean_reuse = tmp_mean_reuse + tmp_mean_reuse_std
                # tmp_mean_reuse=F.adaptive_max_pool2d(tmp,output_size=1).squeeze(-1).squeeze(-1)
                # tmp_mean_reuse=F.adaptive_max_pool2d(-1 * tmp,output_size=1).squeeze(-1).squeeze(-1)

                # tmp_mean_reuse_1 = torch.mean(tmp_1, axis=(2, 3))
                # tmp_mean_reuse = tmp_mean_reuse * tmp_mean_reuse_1
                # tmp_mean_reuse = torch.mean(tmp, dim=(2, 3))
                tmp_mean_reuse_bool = torch.mean(tmp_mean_reuse, dim=0).unsqueeze(0)
                tmp_mean_reuse_bool = tmp_mean_reuse > tmp_mean_reuse_bool
                tmp_mean_reuse_bool = tmp_mean_reuse_bool.float()

                tmp_mean_reuse_bool_2 = torch.mean(tmp_mean_reuse, dim=1).unsqueeze(0).T
                tmp_mean_reuse_bool_2 = tmp_mean_reuse > tmp_mean_reuse_bool_2
                tmp_mean_reuse_bool_2 = tmp_mean_reuse_bool_2.float()

                tmp_mean_reuse = tmp_mean_reuse * tmp_mean_reuse_bool * tmp_mean_reuse_bool_2

                tmp_mean = torch.mean(tmp_mean_reuse, dim=0).cpu().data.numpy()
                # tmp_mean,_ = torch.max(tmp_mean_reuse, dim=0)
                # tmp_mean=tmp_mean.numpy()
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean.append(tmp_mean)


        return output_mean,output_2











class GradCam_yuhan_kernel_version30_0_0():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, ii=None,flip=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别

        feature_maps,output = self.net(input.cuda())
        output_max = torch.max(output)
        output_min = torch.min(output)
        output_norm = torch.div((output - output_min).float(), (output_max - output_min))
        ################################################################3
        output_norm = (output_norm * -1).cpu().tolist()
        output_norm = torch.tensor(output_norm).float().cuda()
        ###################################################################
        one_hot = torch.sum(output * output_norm)
        self.net.zero_grad()
        one_hot.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #




        feature_maps_L28 = feature_maps[
            0][0]  # 如果输入图像的shape为[batch_size,3,224,224]的tensor的话，关于输出的feature map,shape为  #[batch_size,512,14,14]左右
        # print(feature_maps_L28.shape)
        # print(feature_maps_L28)
        feature_maps_L28_channelwise_mean = torch.mean(feature_maps_L28, dim=(1, 2))
        # print(feature_maps_L28_channelwise_mean.shape)
        highlight_chanelwise_L28 = (feature_maps_L28 > feature_maps_L28_channelwise_mean.view(-1, 1, 1)).float()
        # print(highlight_chanelwise_L28.shape)#为这512通道的每一个通道生成了一个（由该通道均值决定的）专属于该通道的二值掩码矩阵，将该通道的显著性区域标注了出来
        feature_maps_L28_sum = torch.sum(feature_maps_L28, dim=0)
        # print(feature_maps_L28_sum.shape)
        highlight_L28 = (feature_maps_L28_sum > torch.mean(feature_maps_L28_sum)).float()
        # print(highlight_L28.shape)
        h28 = highlight_chanelwise_L28 * highlight_L28.unsqueeze(
            0)  # h28是一个【512，h,w】的二值矩阵，这是与总体二值矩阵交集得到的，可能在某一个通道上是一个全零矩阵
        # （一个完全对背景噪声响应的通道）
        # print(h28.shape)
        feature_maps_L28 = feature_maps_L28 * h28  # 同为shape为【512,7,7】的feature_map 与通道级二值矩阵（总体显著性区域与通道级显著性区域的交集）进行点乘，
        # chanelweight_l28 = torch.nn.functional.adaptive_max_pool2d(feature_maps_L28, output_size=1).squeeze(-1).squeeze(-1)

        # feature_maps_L28_channelwise_min = F.adaptive_max_pool2d(-1 * feature_maps_L28, output_size=1).squeeze(-1).squeeze(-1)
        # print(feature_maps_L28_channelwise_min)#都是0,所以没用
        # feature_maps_L28_min=torch.ones_like(feature_maps_L28)*feature_maps_L28_channelwise_min.unsqueeze(0).unsqueeze(0).T
        # feature_maps_L28 = feature_maps_L28 +feature_maps_L28_min
        # feature_maps_L28 = feature_maps_L28 * h28
        chanelweight_l28 = F.adaptive_max_pool2d(feature_maps_L28, output_size=1).squeeze(
            -1).squeeze(-1)

        # chanelweight_l28 = torch.sum(feature_maps_L28, dim=(1, 2)) / torch.sum(h28, dim=(1, 2))
        # chanelweight_l28=F.adaptive_max_pool2d(feature_maps_L28
        #                                        ,output_size=1).squeeze(-1).squeeze(-1)
        # print(chanelweight_l28)
        # print(chanelweight_l28.shape)
        # 我们用0代替了nan值
        chanelweight_l28 = torch.where(torch.isnan(chanelweight_l28), torch.full_like(chanelweight_l28, 0),
                                       chanelweight_l28)
        # chanelweight_l28=chanelweight_l28/torch.sum(highlight_L28)
        # print(chanelweight_l28)
        # chanelweight_l28 = (chanelweight_l28 - torch.min(chanelweight_l28)) / (
        #         torch.max(chanelweight_l28) - torch.min(chanelweight_l28))
        # chanelweight_l28=(chanelweight_l28>torch.mean(chanelweight_l28)).float()
        # chanelweight_l28_bool = (chanelweight_l28 > torch.mean(chanelweight_l28)).float()
        # chanelweight_l28_bool_2 = torch.ones_like(chanelweight_l28_bool) - chanelweight_l28_bool
        # chanelweight_l28 = chanelweight_l28_bool * chanelweight_l28
        # chanelweight_l28 = chanelweight_l28 + chanelweight_l28_bool
        # print(chanelweight_l28)
        chanelweight_l28 = chanelweight_l28.cpu().data.numpy()
        aa = np.linalg.norm(chanelweight_l28)
        if aa != 0:
            chanelweight_l28 = chanelweight_l28 / aa
        else:
            chanelweight_l28 = np.zeros_like(chanelweight_l28)




        feature_maps_L31 = feature_maps[
            1][
            0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        feature_maps_L31_channelwise_mean = torch.mean(feature_maps_L31, dim=(1, 2))
        highlight_chanelwise_L31 = (feature_maps_L31 > feature_maps_L31_channelwise_mean.view(-1, 1,
                                                                                              1)).float()  # 得到的结果依然是【512,7,7】的一个二值矩阵
        feature_maps_L31_sum = torch.sum(feature_maps_L31, dim=0)
        highlight_L31 = (feature_maps_L31_sum > torch.mean(feature_maps_L31_sum)).float()
        h31 = highlight_chanelwise_L31 * highlight_L31.unsqueeze(0)
        feature_maps_L31 = feature_maps_L31 * h31
        # chanelweight_l31 = torch.nn.functional.adaptive_max_pool2d(feature_maps_L31, output_size=1).squeeze(-1).squeeze(
        #     -1)

        # feature_maps_L31_channelwise_min = F.adaptive_max_pool2d(-1 * feature_maps_L31, output_size=1).squeeze(
        #     -1).squeeze(-1)
        # feature_maps_L31_min = torch.ones_like(feature_maps_L31) * feature_maps_L31_channelwise_min.unsqueeze(0).unsqueeze(
        #     0).T
        # feature_maps_L31 = feature_maps_L31 +feature_maps_L31_min
        # feature_maps_L31 = feature_maps_L31 * h31
        chanelweight_l31 = F.adaptive_max_pool2d(feature_maps_L31, output_size=1).squeeze(
                -1).squeeze(-1)
        # chanelweight_l31 = torch.sum(feature_maps_L31, dim=(1, 2)) / torch.sum(h31, dim=(1, 2))
        # chanelweight_l31 = F.adaptive_max_pool2d(feature_maps_L31
        #                                          , output_size=1).squeeze(-1).squeeze(-1)
        chanelweight_l31 = torch.where(torch.isnan(chanelweight_l31), torch.full_like(chanelweight_l31, 0),
                                       chanelweight_l31)
        chanelweight_l31=chanelweight_l31.cpu().data.numpy()

        # chanelweight_l31=torch.sum(h31,dim=(1,2))
        # chanelweight_l31=chanelweight_l31/torch.sum(highlight_L31)
        # chanelweight_l31 = (chanelweight_l31 - torch.min(chanelweight_l31)) / (
        #         torch.max(chanelweight_l31) - torch.min(chanelweight_l31))

        aa = np.linalg.norm(chanelweight_l31)
        if aa != 0:
            chanelweight_l31 = chanelweight_l31 / aa
        else:
            chanelweight_l31 = np.zeros_like(chanelweight_l31)

        # chanelweight_l31=(chanelweight_l31>torch.mean(chanelweight_l31)).float()

        #
        # chanelweight_l31_bool=(chanelweight_l31>torch.mean(chanelweight_l31)).float()
        # chanelweight_l31_bool_2=torch.ones_like(chanelweight_l31_bool)-chanelweight_l31_bool
        # chanelweight_l31=chanelweight_l31_bool * chanelweight_l31
        # chanelweight_l31=chanelweight_l31 + chanelweight_l31_bool
        # print(chanelweight_l31)
        # print(torch.sum((chanelweight_l31/np.linalg.norm(chanelweight_l31.cpu().data.numpy()))*(chanelweight_l28/np.linalg.norm(chanelweight_l28.cpu().data.numpy()))))

        ###************
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #

        output_mean = []
        output_2=[]



        output_2.append(chanelweight_l28)
        output_2.append(chanelweight_l31)


        # output_max=[]
        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['features.26.weight', 'features.28.weight']:
                tmp = (param.grad).data
                # print(tmp)
                # tmp=torch.abs(tmp)
                # print(tmp)
                # tmp_mean_reuse=tmp.view(tmp.size()[0],-1)
                # print(tmp_mean_reuse.shape)
                tmp_mean_reuse_std = torch.std(tmp, axis=(2, 3))

                tmp_mean_reuse = F.adaptive_max_pool2d(tmp, output_size=1).squeeze(-1).squeeze(
                    -1) + F.adaptive_max_pool2d(-1 * tmp, output_size=1).squeeze(-1).squeeze(-1)
                tmp_mean_reuse = tmp_mean_reuse + tmp_mean_reuse_std
                # tmp_mean_reuse=F.adaptive_max_pool2d(tmp,output_size=1).squeeze(-1).squeeze(-1)
                # tmp_mean_reuse=F.adaptive_max_pool2d(-1 * tmp,output_size=1).squeeze(-1).squeeze(-1)

                # tmp_mean_reuse_1 = torch.mean(tmp_1, axis=(2, 3))
                # tmp_mean_reuse = tmp_mean_reuse * tmp_mean_reuse_1
                # tmp_mean_reuse = torch.mean(tmp, dim=(2, 3))
                tmp_mean_reuse_bool = torch.mean(tmp_mean_reuse, dim=0).unsqueeze(0)
                tmp_mean_reuse_bool = tmp_mean_reuse > tmp_mean_reuse_bool
                tmp_mean_reuse_bool = tmp_mean_reuse_bool.float()

                tmp_mean_reuse_bool_2 = torch.mean(tmp_mean_reuse, dim=1).unsqueeze(0).T
                tmp_mean_reuse_bool_2 = tmp_mean_reuse > tmp_mean_reuse_bool_2
                tmp_mean_reuse_bool_2 = tmp_mean_reuse_bool_2.float()

                tmp_mean_reuse = tmp_mean_reuse * tmp_mean_reuse_bool * tmp_mean_reuse_bool_2

                tmp_mean = torch.mean(tmp_mean_reuse, dim=0).cpu().data.numpy()
                # tmp_mean,_ = torch.max(tmp_mean_reuse, dim=0)
                # tmp_mean=tmp_mean.numpy()
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean.append(tmp_mean)


        return output_mean,output_2












class GradCam_yuhan_kernel_version30_0_1():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, ii=None,flip=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别

        feature_maps,output = self.net(input.cuda())
        output_max = torch.max(output)
        output_min = torch.min(output)
        output_norm = torch.div((output - output_min).float(), (output_max - output_min))
        ################################################################3
        output_norm = (output_norm * -1).cpu().tolist()
        output_norm = torch.tensor(output_norm).float().cuda()
        ###################################################################
        one_hot = torch.sum(output * output_norm)
        self.net.zero_grad()
        one_hot.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #




        feature_maps_L28 = feature_maps[
            1][0]  # 如果输入图像的shape为[batch_size,3,224,224]的tensor的话，关于输出的feature map,shape为  #[batch_size,512,14,14]左右
        # print(feature_maps_L28.shape)
        # print(feature_maps_L28)
        feature_maps_L28_channelwise_mean = torch.mean(feature_maps_L28, dim=(1, 2))
        # print(feature_maps_L28_channelwise_mean.shape)
        highlight_chanelwise_L28 = (feature_maps_L28 > feature_maps_L28_channelwise_mean.view(-1, 1, 1)).float()
        # print(highlight_chanelwise_L28.shape)#为这512通道的每一个通道生成了一个（由该通道均值决定的）专属于该通道的二值掩码矩阵，将该通道的显著性区域标注了出来
        feature_maps_L28_sum = torch.sum(feature_maps_L28, dim=0)
        # print(feature_maps_L28_sum.shape)
        highlight_L28 = (feature_maps_L28_sum > torch.mean(feature_maps_L28_sum)).float()
        # print(highlight_L28.shape)
        h28 = highlight_chanelwise_L28 * highlight_L28.unsqueeze(
            0)  # h28是一个【512，h,w】的二值矩阵，这是与总体二值矩阵交集得到的，可能在某一个通道上是一个全零矩阵
        # （一个完全对背景噪声响应的通道）
        # print(h28.shape)
        feature_maps_L28 = feature_maps[0][0] * h28  # 同为shape为【512,7,7】的feature_map 与通道级二值矩阵（总体显著性区域与通道级显著性区域的交集）进行点乘，
        # chanelweight_l28 = torch.nn.functional.adaptive_max_pool2d(feature_maps_L28, output_size=1).squeeze(-1).squeeze(-1)

        # feature_maps_L28_channelwise_min = F.adaptive_max_pool2d(-1 * feature_maps_L28, output_size=1).squeeze(-1).squeeze(-1)
        # print(feature_maps_L28_channelwise_min)#都是0,所以没用
        # feature_maps_L28_min=torch.ones_like(feature_maps_L28)*feature_maps_L28_channelwise_min.unsqueeze(0).unsqueeze(0).T
        # feature_maps_L28 = feature_maps_L28 +feature_maps_L28_min
        # feature_maps_L28 = feature_maps_L28 * h28
        chanelweight_l28 = F.adaptive_max_pool2d(feature_maps_L28, output_size=1).squeeze(-1).squeeze(-1)\
                           + F.adaptive_max_pool2d(-1 * feature_maps_L28, output_size=1).squeeze(-1).squeeze(-1)

        # chanelweight_l28 = torch.sum(feature_maps_L28, dim=(1, 2)) / torch.sum(h28, dim=(1, 2))
        # chanelweight_l28=F.adaptive_max_pool2d(feature_maps_L28
        #                                        ,output_size=1).squeeze(-1).squeeze(-1)
        # print(chanelweight_l28)
        # print(chanelweight_l28.shape)
        # 我们用0代替了nan值
        chanelweight_l28 = torch.where(torch.isnan(chanelweight_l28), torch.full_like(chanelweight_l28, 0),
                                       chanelweight_l28)
        # chanelweight_l28=chanelweight_l28/torch.sum(highlight_L28)
        # print(chanelweight_l28)
        # chanelweight_l28 = (chanelweight_l28 - torch.min(chanelweight_l28)) / (
        #         torch.max(chanelweight_l28) - torch.min(chanelweight_l28))
        # chanelweight_l28=(chanelweight_l28>torch.mean(chanelweight_l28)).float()
        # chanelweight_l28_bool = (chanelweight_l28 > torch.mean(chanelweight_l28)).float()
        # chanelweight_l28_bool_2 = torch.ones_like(chanelweight_l28_bool) - chanelweight_l28_bool
        # chanelweight_l28 = chanelweight_l28_bool * chanelweight_l28
        # chanelweight_l28 = chanelweight_l28 + chanelweight_l28_bool
        # print(chanelweight_l28)
        chanelweight_l28 = chanelweight_l28.cpu().data.numpy()
        aa = np.linalg.norm(chanelweight_l28)
        if aa != 0:
            chanelweight_l28 = chanelweight_l28 / aa
        else:
            chanelweight_l28 = np.zeros_like(chanelweight_l28)




        feature_maps_L31 = feature_maps[
            3][
            0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        feature_maps_L31_channelwise_mean = torch.mean(feature_maps_L31, dim=(1, 2))
        highlight_chanelwise_L31 = (feature_maps_L31 > feature_maps_L31_channelwise_mean.view(-1, 1,
                                                                                              1)).float()  # 得到的结果依然是【512,7,7】的一个二值矩阵
        feature_maps_L31_sum = torch.sum(feature_maps_L31, dim=0)
        highlight_L31 = (feature_maps_L31_sum > torch.mean(feature_maps_L31_sum)).float()
        h31 = highlight_chanelwise_L31 * highlight_L31.unsqueeze(0)
        feature_maps_L31 = feature_maps[2][0] * h31
        # chanelweight_l31 = torch.nn.functional.adaptive_max_pool2d(feature_maps_L31, output_size=1).squeeze(-1).squeeze(
        #     -1)

        # feature_maps_L31_channelwise_min = F.adaptive_max_pool2d(-1 * feature_maps_L31, output_size=1).squeeze(
        #     -1).squeeze(-1)
        # feature_maps_L31_min = torch.ones_like(feature_maps_L31) * feature_maps_L31_channelwise_min.unsqueeze(0).unsqueeze(
        #     0).T
        # feature_maps_L31 = feature_maps_L31 +feature_maps_L31_min
        # feature_maps_L31 = feature_maps_L31 * h31
        chanelweight_l31 = F.adaptive_max_pool2d(feature_maps_L31, output_size=1).squeeze(-1).squeeze(-1) \
                           + F.adaptive_max_pool2d(-1 * feature_maps_L31, output_size=1).squeeze(-1).squeeze(-1)
        # chanelweight_l31 = torch.sum(feature_maps_L31, dim=(1, 2)) / torch.sum(h31, dim=(1, 2))
        # chanelweight_l31 = F.adaptive_max_pool2d(feature_maps_L31
        #                                          , output_size=1).squeeze(-1).squeeze(-1)
        chanelweight_l31 = torch.where(torch.isnan(chanelweight_l31), torch.full_like(chanelweight_l31, 0),
                                       chanelweight_l31)
        chanelweight_l31=chanelweight_l31.cpu().data.numpy()

        # chanelweight_l31=torch.sum(h31,dim=(1,2))
        # chanelweight_l31=chanelweight_l31/torch.sum(highlight_L31)
        # chanelweight_l31 = (chanelweight_l31 - torch.min(chanelweight_l31)) / (
        #         torch.max(chanelweight_l31) - torch.min(chanelweight_l31))

        aa = np.linalg.norm(chanelweight_l31)
        if aa != 0:
            chanelweight_l31 = chanelweight_l31 / aa
        else:
            chanelweight_l31 = np.zeros_like(chanelweight_l31)

        # chanelweight_l31=(chanelweight_l31>torch.mean(chanelweight_l31)).float()

        #
        # chanelweight_l31_bool=(chanelweight_l31>torch.mean(chanelweight_l31)).float()
        # chanelweight_l31_bool_2=torch.ones_like(chanelweight_l31_bool)-chanelweight_l31_bool
        # chanelweight_l31=chanelweight_l31_bool * chanelweight_l31
        # chanelweight_l31=chanelweight_l31 + chanelweight_l31_bool
        # print(chanelweight_l31)
        # print(torch.sum((chanelweight_l31/np.linalg.norm(chanelweight_l31.cpu().data.numpy()))*(chanelweight_l28/np.linalg.norm(chanelweight_l28.cpu().data.numpy()))))

        ###************
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #

        output_mean = []
        output_2=[]



        output_2.append(chanelweight_l28)
        output_2.append(chanelweight_l31)


        # output_max=[]
        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['features.26.weight', 'features.28.weight']:
                tmp = (param.grad).data
                # print(tmp)
                # tmp=torch.abs(tmp)
                # print(tmp)
                # tmp_mean_reuse=tmp.view(tmp.size()[0],-1)
                # print(tmp_mean_reuse.shape)
                tmp_mean_reuse_std = torch.std(tmp, axis=(2, 3))

                tmp_mean_reuse = F.adaptive_max_pool2d(tmp, output_size=1).squeeze(-1).squeeze(
                    -1) + F.adaptive_max_pool2d(-1 * tmp, output_size=1).squeeze(-1).squeeze(-1)
                tmp_mean_reuse = tmp_mean_reuse + tmp_mean_reuse_std
                # tmp_mean_reuse=F.adaptive_max_pool2d(tmp,output_size=1).squeeze(-1).squeeze(-1)
                # tmp_mean_reuse=F.adaptive_max_pool2d(-1 * tmp,output_size=1).squeeze(-1).squeeze(-1)

                # tmp_mean_reuse_1 = torch.mean(tmp_1, axis=(2, 3))
                # tmp_mean_reuse = tmp_mean_reuse * tmp_mean_reuse_1
                # tmp_mean_reuse = torch.mean(tmp, dim=(2, 3))
                tmp_mean_reuse_bool = torch.mean(tmp_mean_reuse, dim=0).unsqueeze(0)
                tmp_mean_reuse_bool = tmp_mean_reuse > tmp_mean_reuse_bool
                tmp_mean_reuse_bool = tmp_mean_reuse_bool.float()

                tmp_mean_reuse_bool_2 = torch.mean(tmp_mean_reuse, dim=1).unsqueeze(0).T
                tmp_mean_reuse_bool_2 = tmp_mean_reuse > tmp_mean_reuse_bool_2
                tmp_mean_reuse_bool_2 = tmp_mean_reuse_bool_2.float()

                tmp_mean_reuse = tmp_mean_reuse * tmp_mean_reuse_bool * tmp_mean_reuse_bool_2

                tmp_mean = torch.mean(tmp_mean_reuse, dim=0).cpu().data.numpy()
                # tmp_mean,_ = torch.max(tmp_mean_reuse, dim=0)
                # tmp_mean=tmp_mean.numpy()
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean.append(tmp_mean)


        return output_mean,output_2

























# 层: features.0.weight torch.Size([64, 3, 3, 3])
# 层: features.0.bias torch.Size([64])
# 层: features.2.weight torch.Size([64, 64, 3, 3])
# 层: features.2.bias torch.Size([64])
# 层: features.5.weight torch.Size([128, 64, 3, 3])
# 层: features.5.bias torch.Size([128])
# 层: features.7.weight torch.Size([128, 128, 3, 3])
# 层: features.7.bias torch.Size([128])
# 层: features.10.weight torch.Size([256, 128, 3, 3])
# 层: features.10.bias torch.Size([256])
# 层: features.12.weight torch.Size([256, 256, 3, 3])
# 层: features.12.bias torch.Size([256])
# 层: features.14.weight torch.Size([256, 256, 3, 3])
# 层: features.14.bias torch.Size([256])
# 层: features.17.weight torch.Size([512, 256, 3, 3])
# 层: features.17.bias torch.Size([512])
# 层: features.19.weight torch.Size([512, 512, 3, 3])
# 层: features.19.bias torch.Size([512])
# 层: features.21.weight torch.Size([512, 512, 3, 3])
# 层: features.21.bias torch.Size([512])
# 层: features.24.weight torch.Size([512, 512, 3, 3])
# 层: features.24.bias torch.Size([512])
# 层: features.26.weight torch.Size([512, 512, 3, 3])
# 层: features.26.bias torch.Size([512])
# 层: features.28.weight torch.Size([512, 512, 3, 3])
# 层: features.28.bias torch.Size([512])
# 层: img_classifier.weight torch.Size([100, 512])
# 层: img_classifier.bias torch.Size([100])
# 层: features.0.weight torch.Size([64, 3, 3, 3])
# 层: features.0.bias torch.Size([64])
# 层: features.2.weight torch.Size([64, 64, 3, 3])
# 层: features.2.bias torch.Size([64])
# 层: features.5.weight torch.Size([128, 64, 3, 3])
# 层: features.5.bias torch.Size([128])
# 层: features.7.weight torch.Size([128, 128, 3, 3])
# 层: features.7.bias torch.Size([128])
# 层: features.10.weight torch.Size([256, 128, 3, 3])
# 层: features.10.bias torch.Size([256])
# 层: features.12.weight torch.Size([256, 256, 3, 3])
# 层: features.12.bias torch.Size([256])
# 层: features.14.weight torch.Size([256, 256, 3, 3])
# 层: features.14.bias torch.Size([256])
# 层: features.17.weight torch.Size([512, 256, 3, 3])
# 层: features.17.bias torch.Size([512])
# 层: features.19.weight torch.Size([512, 512, 3, 3])
# 层: features.19.bias torch.Size([512])
# 层: features.21.weight torch.Size([512, 512, 3, 3])
# 层: features.21.bias torch.Size([512])
# 层: features.24.weight torch.Size([512, 512, 3, 3])
# 层: features.24.bias torch.Size([512])
# 层: features.26.weight torch.Size([512, 512, 3, 3])
# 层: features.26.bias torch.Size([512])
# 层: features.28.weight torch.Size([512, 512, 3, 3])
# 层: features.28.bias torch.Size([512])
# 层: img_classifier.weight torch.Size([100, 512])
# 层: img_classifier.bias torch.Size([100])




class GradCam_yuhan_kernel_version28_6_3():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, index=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别
        output = self.net(input.cuda())
        output_max = torch.max(output)
        output_min = torch.min(output)
        output_norm = torch.div((output - output_min).float(), (output_max - output_min))
        ################################################################3
        output_norm = (output_norm * -1).cpu().tolist()
        output_norm = torch.tensor(output_norm).float().cuda()
        ###################################################################
        one_hot = torch.sum(output * output_norm)
        self.net.zero_grad()
        one_hot.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #
        output_mean = []


        # output_max=[]
        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['img_classifier.weight']:
                tmp = (param.grad).data
                # print(tmp)
                # tmp=torch.abs(tmp)
                # print(tmp)
                # tmp_mean_reuse=tmp.view(tmp.size()[0],-1)
                # print(tmp_mean_reuse.shape)
                tmp_mean_reuse=tmp
                # tmp_mean_reuse=F.adaptive_max_pool2d(tmp,output_size=1).squeeze(-1).squeeze(-1) + F.adaptive_max_pool2d(-1 * tmp,output_size=1).squeeze(-1).squeeze(-1)
                # tmp_mean_reuse=F.adaptive_max_pool2d(tmp,output_size=1).squeeze(-1).squeeze(-1)
                # tmp_mean_reuse=F.adaptive_max_pool2d(-1 * tmp,output_size=1).squeeze(-1).squeeze(-1)
                tmp_mean_reuse_bool=torch.mean(tmp_mean_reuse,dim=0).unsqueeze(0)
                tmp_mean_reuse_bool=tmp_mean_reuse>tmp_mean_reuse_bool
                tmp_mean_reuse_bool=tmp_mean_reuse_bool.float()

                tmp_mean_reuse_bool_2 = torch.mean(tmp_mean_reuse, dim=1).unsqueeze(0).T
                tmp_mean_reuse_bool_2 = tmp_mean_reuse > tmp_mean_reuse_bool_2
                tmp_mean_reuse_bool_2 = tmp_mean_reuse_bool_2.float()

                tmp_mean_reuse=tmp_mean_reuse * tmp_mean_reuse_bool * tmp_mean_reuse_bool_2
                # tmp_mean_reuse_1 = torch.mean(tmp_1, axis=(2, 3))
                # tmp_mean_reuse = tmp_mean_reuse * tmp_mean_reuse_1
                # tmp_mean_reuse = torch.mean(tmp, dim=(2, 3))

                tmp_mean = torch.mean(tmp_mean_reuse, dim=0).cpu().data.numpy()
                # tmp_mean,_ = torch.max(tmp_mean_reuse, dim=0)
                # tmp_mean=tmp_mean.numpy()
                # tmp_mean = tmp_mean.cpu().data.numpy()

                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean.append(tmp_mean)
                output_mean.append(tmp_mean)


        return output_mean








 # tmp=tmp.view(tmp.size()[0],tmp.size()[1],-1)
 #                tmp,_=torch.sort(tmp,dim=2)
 #                # print(tmp[0][0])
 #                tmp=tmp * torch.tensor([-1.,-1.,0.,0.,0.,0.,0.,1.,1.]).unsqueeze(0).unsqueeze(0)
 #                # print(tmp[0][0])
 #                # tmp=tmp[:][:][-1]+tmp[:][:][-2]-tmp[:][:][0]-tmp[:][:][1]
 #                # print(tmp.size())
 #                # print(tmp.size())
 #                # tmp_mean_reuse=F.adaptive_max_pool1d(tmp,output_size=2).squeeze(-1)# + F.adaptive_max_pool1d(-1 * tmp,output_size=1).squeeze(-1)
 #                tmp_mean_reuse=torch.sum(tmp,dim=2)
 #






class GradCam_yuhan_kernel_version28_7():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, index=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别
        output = self.net(input.cuda())
        # print(output)
        output_max = torch.max(output)
        output_min = torch.min(output)
        output_norm = torch.div((output - output_min).float(), (output_max - output_min))
        ################################################################3
        output_norm = (output_norm * -1).cpu().tolist()
        output_norm = torch.tensor(output_norm).float().cuda()
        ###################################################################
        one_hot = torch.sum(output * output_norm)
        self.net.zero_grad()
        one_hot.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #
        output_mean = []


        # output_max=[]
        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['features.26.weight', 'features.28.weight']:
                tmp = (param.grad).data
                # print(tmp)
                # tmp=torch.abs(tmp)
                # print(tmp)
                # tmp_mean_reuse=tmp.view(tmp.size()[0],-1)
                # print(tmp_mean_reuse.shape)
                # tmp_mean_reuse_std = torch.std(tmp, axis=(2, 3))
                # tmp_mean_reuse_std = tmp_mean_reuse_std ** 2
                # tmp_mean_reuse=F.adaptive_max_pool2d(tmp,output_size=1).squeeze(-1).squeeze(-1) + F.adaptive_max_pool2d(-1 * tmp,output_size=1).squeeze(-1).squeeze(-1)
                tmp_mean_channelwise = F.adaptive_max_pool2d(-1 * tmp, output_size=1)
                tmp = tmp + tmp_mean_channelwise
                tmp=tmp.view(tmp.size()[0],tmp.size()[1],-1)
                tmp,_=torch.sort(tmp,dim=2)
                tmp=tmp * torch.tensor([0.,0.,0.,0.,0.,0.,0.,1.,1.]).cuda().unsqueeze(0).unsqueeze(0)
                tmp_mean_reuse=torch.sum(tmp,dim=2)


                # tmp_mean_reuse=torch.mean(tmp,axis=( 2, 3))

                # tmp_mean_reuse = tmp_mean_reuse_std
                # tmp_mean_reuse=F.adaptive_max_pool2d(tmp,output_size=1).squeeze(-1).squeeze(-1)
                # tmp_mean_reuse=F.adaptive_max_pool2d(-1 * tmp,output_size=1).squeeze(-1).squeeze(-1)
                tmp_mean_reuse_bool=torch.mean(tmp_mean_reuse,dim=0).unsqueeze(0)
                tmp_mean_reuse_bool=tmp_mean_reuse>tmp_mean_reuse_bool
                tmp_mean_reuse_bool=tmp_mean_reuse_bool.float()

                tmp_mean_reuse_bool_2 = torch.mean(tmp_mean_reuse, dim=1).unsqueeze(0).T
                tmp_mean_reuse_bool_2 = tmp_mean_reuse > tmp_mean_reuse_bool_2
                tmp_mean_reuse_bool_2 = tmp_mean_reuse_bool_2.float()

                tmp_mean_reuse=tmp_mean_reuse * tmp_mean_reuse_bool * tmp_mean_reuse_bool_2
                # tmp_mean_reuse_1 = torch.mean(tmp_1, axis=(2, 3))
                # tmp_mean_reuse = tmp_mean_reuse * tmp_mean_reuse_1
                # tmp_mean_reuse = torch.mean(tmp, dim=(2, 3))

                tmp_mean = torch.mean(tmp_mean_reuse, dim=0).cpu().data.numpy()
                # tmp_mean,_ = torch.max(tmp_mean_reuse, dim=0)
                # tmp_mean=tmp_mean.numpy()
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean.append(tmp_mean)


        return output_mean













class GradCam_yuhan_kernel_version28_4():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, index=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别
        output = self.net(input.cuda())
        output_max = torch.max(output)
        output_min = torch.min(output)
        output_norm = torch.div((output - output_min).float(), (output_max - output_min))
        ################################################################3
        output_norm = (output_norm * -1).cpu().tolist()
        output_norm = torch.tensor(output_norm).float().cuda()
        ###################################################################
        one_hot = torch.sum(output * output_norm)
        self.net.zero_grad()
        one_hot.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #
        output_mean = []


        # output_max=[]
        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['features.26.weight', 'features.28.weight']:
                tmp = (param.grad).cpu().data
                # print(tmp)
                # tmp=torch.abs(tmp)
                # print(tmp)
                # tmp_mean_reuse=tmp.view(tmp.size()[0],-1)
                # print(tmp_mean_reuse.shape)
                tmp_mean_reuse_std = torch.std(tmp, axis=(2, 3))

                tmp_mean_reuse=F.adaptive_max_pool2d(tmp,output_size=1).squeeze(-1).squeeze(-1) + F.adaptive_max_pool2d(-1 * tmp,output_size=1).squeeze(-1).squeeze(-1)
                tmp_mean_reuse=tmp_mean_reuse + tmp_mean_reuse_std
                # tmp_mean_reuse=F.adaptive_max_pool2d(tmp,output_size=1).squeeze(-1).squeeze(-1)
                # tmp_mean_reuse=F.adaptive_max_pool2d(-1 * tmp,output_size=1).squeeze(-1).squeeze(-1)
                tmp_mean_reuse_bool=torch.mean(tmp_mean_reuse,dim=0).unsqueeze(0)
                tmp_mean_reuse_bool=tmp_mean_reuse>tmp_mean_reuse_bool
                tmp_mean_reuse_bool=tmp_mean_reuse_bool.float()

                tmp_mean_reuse_bool_2 = torch.mean(tmp_mean_reuse, dim=1).unsqueeze(0).T
                tmp_mean_reuse_bool_2 = tmp_mean_reuse > tmp_mean_reuse_bool_2
                tmp_mean_reuse_bool_2 = tmp_mean_reuse_bool_2.float()

                tmp_mean_reuse=tmp_mean_reuse * tmp_mean_reuse_bool * tmp_mean_reuse_bool_2
                # tmp_mean_reuse_1 = torch.mean(tmp_1, axis=(2, 3))
                # tmp_mean_reuse = tmp_mean_reuse * tmp_mean_reuse_1
                # tmp_mean_reuse = torch.mean(tmp, dim=(2, 3))

                tmp_mean = torch.mean(tmp_mean_reuse, dim=1).numpy()
                # tmp_mean,_ = torch.max(tmp_mean_reuse, dim=0)
                # tmp_mean=tmp_mean.numpy()
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean.append(tmp_mean)


        return output_mean


















class GradCam_yuhan_kernel_version28_2():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, index=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别
        output = self.net(input.cuda())
        output_max = torch.max(output)
        output_min = torch.min(output)
        output_norm = torch.div((output - output_min).float(), (output_max - output_min))
        ################################################################3
        output_norm = (output_norm * -1).cpu().tolist()
        output_norm = torch.tensor(output_norm).float().cuda()
        ###################################################################
        one_hot = torch.sum(output * output_norm)
        self.net.zero_grad()
        one_hot.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #
        output_mean = []


        # output_max=[]
        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['features.26.weight', 'features.28.weight']:
                tmp = (param.grad).cpu().data
                # print(tmp)
                # tmp=torch.abs(tmp)
                # print(tmp)
                # tmp_mean_reuse=tmp.view(tmp.size()[0],-1)
                # print(tmp_mean_reuse.shape)
                tmp_mean_reuse=F.adaptive_max_pool2d(tmp,output_size=1).squeeze(-1).squeeze(-1) + F.adaptive_max_pool2d(-1 * tmp,output_size=1).squeeze(-1).squeeze(-1)
                # tmp_mean_reuse=F.adaptive_max_pool2d(tmp,output_size=1).squeeze(-1).squeeze(-1)
                # tmp_mean_reuse=F.adaptive_max_pool2d(-1 * tmp,output_size=1).squeeze(-1).squeeze(-1)
                tmp_mean_reuse_bool=torch.mean(tmp_mean_reuse,dim=0).unsqueeze(0)
                tmp_mean_reuse_bool=tmp_mean_reuse>tmp_mean_reuse_bool
                tmp_mean_reuse_bool=tmp_mean_reuse_bool.float()

                tmp_mean_reuse_bool_2 = torch.mean(tmp_mean_reuse, dim=1).unsqueeze(0).T
                tmp_mean_reuse_bool_2 = tmp_mean_reuse > tmp_mean_reuse_bool_2
                tmp_mean_reuse_bool_2 = tmp_mean_reuse_bool_2.float()

                tmp_mean_reuse=tmp_mean_reuse * tmp_mean_reuse_bool * tmp_mean_reuse_bool_2
                # tmp_mean_reuse_1 = torch.mean(tmp_1, axis=(2, 3))
                # tmp_mean_reuse = tmp_mean_reuse * tmp_mean_reuse_1
                # tmp_mean_reuse = torch.mean(tmp, dim=(2, 3))

                tmp_mean = torch.mean(tmp_mean_reuse, dim=0).numpy()
                # tmp_mean,_ = torch.max(tmp_mean_reuse, dim=0)
                # tmp_mean=tmp_mean.numpy()
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean.append(tmp_mean)
        # print()
        output = self.net(input.cuda())
        output_max = torch.max(output)
        output_min = torch.min(output)
        output_norm = torch.div((output - output_min).float(), (output_max - output_min))
        output_norm = torch.exp(output_norm)
        ################################################################3
        output_norm = (output_norm * -1).cpu().tolist()
        output_norm = torch.tensor(output_norm).float().cuda()
        ###################################################################
        one_hot = torch.sum(output * output_norm)
        self.net.zero_grad()
        one_hot.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #
        # output_mean = []

        # output_max=[]
        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['features.26.weight', 'features.28.weight']:
                tmp = (param.grad).cpu().data
                # print(tmp)
                # tmp=torch.abs(tmp)
                # print(tmp)
                # tmp_mean_reuse=tmp.view(tmp.size()[0],-1)
                # print(tmp_mean_reuse.shape)
                tmp_mean_reuse = F.adaptive_max_pool2d(tmp, output_size=1).squeeze(-1).squeeze(
                    -1) + F.adaptive_max_pool2d(-1 * tmp, output_size=1).squeeze(-1).squeeze(-1)
                # tmp_mean_reuse=F.adaptive_max_pool2d(tmp,output_size=1).squeeze(-1).squeeze(-1)
                # tmp_mean_reuse=F.adaptive_max_pool2d(-1 * tmp,output_size=1).squeeze(-1).squeeze(-1)
                tmp_mean_reuse_bool = torch.mean(tmp_mean_reuse, dim=0).unsqueeze(0)
                tmp_mean_reuse_bool = tmp_mean_reuse > tmp_mean_reuse_bool
                tmp_mean_reuse_bool = tmp_mean_reuse_bool.float()

                tmp_mean_reuse_bool_2 = torch.mean(tmp_mean_reuse, dim=1).unsqueeze(0).T
                tmp_mean_reuse_bool_2 = tmp_mean_reuse > tmp_mean_reuse_bool_2
                tmp_mean_reuse_bool_2 = tmp_mean_reuse_bool_2.float()

                tmp_mean_reuse = tmp_mean_reuse * tmp_mean_reuse_bool * tmp_mean_reuse_bool_2
                # tmp_mean_reuse_1 = torch.mean(tmp_1, axis=(2, 3))
                # tmp_mean_reuse = tmp_mean_reuse * tmp_mean_reuse_1
                # tmp_mean_reuse = torch.mean(tmp, dim=(2, 3))

                tmp_mean = torch.mean(tmp_mean_reuse, dim=0).numpy()
                # tmp_mean,_ = torch.max(tmp_mean_reuse, dim=0)
                # tmp_mean=tmp_mean.numpy()
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean.append(tmp_mean)
        output_mean_last=[]
        output_mean_last.append(np.hstack((output_mean[0],output_mean[2])))
        output_mean_last.append(np.hstack((output_mean[1],output_mean[3])))




        return output_mean_last










class GradCam_yuhan_kernel_version28_0():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, ii=None,flip=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别
        feature_maps,output = self.net(input.cuda())
        output_max = torch.max(output)
        output_min = torch.min(output)
        output_norm = torch.div((output - output_min).float(), (output_max - output_min))
        ################################################################3
        output_norm = (output_norm * -1).cpu().tolist()
        output_norm = torch.tensor(output_norm).float().cuda()
        ###################################################################
        one_hot = torch.sum(output * output_norm)
        self.net.zero_grad()
        one_hot.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #

        feature_maps_L28 = feature_maps[
            0][0]  # 如果输入图像的shape为[batch_size,3,224,224]的tensor的话，关于输出的feature map,shape为  #[batch_size,512,14,14]左右
        feature_maps_L28_channelwise_mean = torch.mean(feature_maps_L28, dim=(1, 2))
        # print(feature_maps_L28_channelwise_mean.shape)
        highlight_chanelwise_L28 = (feature_maps_L28 > feature_maps_L28_channelwise_mean.view(-1, 1, 1)).float()
        # print(highlight_chanelwise_L28.shape)#为这512通道的每一个通道生成了一个（由该通道均值决定的）专属于该通道的二值掩码矩阵，将该通道的显著性区域标注了出来
        feature_maps_L28_sum = torch.sum(feature_maps_L28, dim=0)
        # print(feature_maps_L28_sum.shape)
        highlight_L28 = (feature_maps_L28_sum > torch.mean(feature_maps_L28_sum)).float()
        # print(highlight_L28.shape)
        h28 = highlight_chanelwise_L28 * highlight_L28.unsqueeze(
            0)  # h28是一个【512，h,w】的二值矩阵，这是与总体二值矩阵交集得到的，可能在某一个通道上是一个全零矩阵
        feature_maps_L28 = feature_maps_L28 * h28  # 同为shape为【512,7,7】的feature_map 与通道级二值矩阵（总体显著性区域与通道级显著性区域的交集）进行点乘，
        # chanelweight_l28 = torch.nn.functional.adaptive_max_pool2d(feature_maps_L28, output_size=1).squeeze(-1).squeeze(-1)
        chanelweight_l28 = torch.sum(feature_maps_L28, dim=(1, 2)) / torch.sum(h28, dim=(1, 2))
        chanelweight_l28 = torch.where(torch.isnan(chanelweight_l28), torch.full_like(chanelweight_l28, 0),
                                       chanelweight_l28)
        # chanelweight_l28=chanelweight_l28/torch.sum(highlight_L28)
        # print(chanelweight_l28)
        # chanelweight_l28 = (chanelweight_l28 - torch.min(chanelweight_l28)) / (
        #         torch.max(chanelweight_l28) - torch.min(chanelweight_l28))
        chanelweight_l28=(chanelweight_l28>torch.mean(chanelweight_l28)).float()


        feature_maps_L31 = feature_maps[
            1][
            0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        feature_maps_L31_channelwise_mean = torch.mean(feature_maps_L31, dim=(1, 2))
        highlight_chanelwise_L31 = (feature_maps_L31 > feature_maps_L31_channelwise_mean.view(-1, 1,
                                                                                               1)).float()  # 得到的结果依然是【512,7,7】的一个二值矩阵
        feature_maps_L31_sum = torch.sum(feature_maps_L31, dim=0)
        highlight_L31 = (feature_maps_L31_sum > torch.mean(feature_maps_L31_sum)).float()
        h31 = highlight_chanelwise_L31 * highlight_L31.unsqueeze(0)
        feature_maps_L31 = feature_maps_L31 * h31
        chanelweight_l31 = torch.sum(feature_maps_L31, dim=(1, 2)) / torch.sum(h31, dim=(1, 2))
        chanelweight_l31 = torch.where(torch.isnan(chanelweight_l31), torch.full_like(chanelweight_l31, 0),
                                       chanelweight_l31)
        chanelweight_l31=(chanelweight_l31>torch.mean(chanelweight_l31)).float()

        output_mean = []


        # output_max=[]
        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['features.26.weight', 'features.28.weight']:
                tmp = (param.grad).data
                # print(tmp)
                # tmp=torch.abs(tmp)
                # print(tmp)
                # tmp_mean_reuse=tmp.view(tmp.size()[0],-1)
                # print(tmp_mean_reuse.shape)
                tmp_mean_reuse=F.adaptive_max_pool2d(tmp,output_size=1).squeeze(-1).squeeze(-1) + F.adaptive_max_pool2d(-1 * tmp,output_size=1).squeeze(-1).squeeze(-1)
                # tmp_mean_reuse=F.adaptive_max_pool2d(tmp,output_size=1).squeeze(-1).squeeze(-1)
                # tmp_mean_reuse=F.adaptive_max_pool2d(-1 * tmp,output_size=1).squeeze(-1).squeeze(-1)
                if name == 'features.26.weight':
                    tmp_mean_reuse=tmp_mean_reuse*chanelweight_l28.unsqueeze(0).T
                else:
                    tmp_mean_reuse=tmp_mean_reuse*chanelweight_l31.unsqueeze(0).T

                # tmp_mean_reuse_bool=torch.mean(tmp_mean_reuse,dim=0).unsqueeze(0)
                # tmp_mean_reuse_bool=tmp_mean_reuse>tmp_mean_reuse_bool
                # tmp_mean_reuse_bool=tmp_mean_reuse_bool.float()
                #
                # tmp_mean_reuse_bool_2 = torch.mean(tmp_mean_reuse, dim=1).unsqueeze(0).T
                # tmp_mean_reuse_bool_2 = tmp_mean_reuse > tmp_mean_reuse_bool_2
                # tmp_mean_reuse_bool_2 = tmp_mean_reuse_bool_2.float()

                tmp_mean_reuse=tmp_mean_reuse# * tmp_mean_reuse_bool * tmp_mean_reuse_bool_2
                # tmp_mean_reuse_1 = torch.mean(tmp_1, axis=(2, 3))
                # tmp_mean_reuse = tmp_mean_reuse * tmp_mean_reuse_1
                # tmp_mean_reuse = torch.mean(tmp, dim=(2, 3))

                tmp_mean = torch.mean(tmp_mean_reuse, dim=0).cpu().data.numpy()
                # tmp_mean,_ = torch.max(tmp_mean_reuse, dim=0)
                # tmp_mean=tmp_mean.numpy()
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean.append(tmp_mean)


        return output_mean













class GradCam_yuhan_kernel_version28_1():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, ii=None,flip=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别
        feature_maps,output = self.net(input.cuda())
        output_max = torch.max(output)
        output_min = torch.min(output)
        output_norm = torch.div((output - output_min).float(), (output_max - output_min))
        ################################################################3
        output_norm = (output_norm * -1).cpu().tolist()
        output_norm = torch.tensor(output_norm).float().cuda()
        ###################################################################
        one_hot = torch.sum(output * output_norm)
        self.net.zero_grad()
        one_hot.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #

        feature_maps_L28 = feature_maps[
            0][0]  # 如果输入图像的shape为[batch_size,3,224,224]的tensor的话，关于输出的feature map,shape为  #[batch_size,512,14,14]左右
        feature_maps_L28_channelwise_mean = torch.mean(feature_maps_L28, dim=(1, 2))
        # print(feature_maps_L28_channelwise_mean.shape)
        highlight_chanelwise_L28 = (feature_maps_L28 > feature_maps_L28_channelwise_mean.view(-1, 1, 1)).float()
        # print(highlight_chanelwise_L28.shape)#为这512通道的每一个通道生成了一个（由该通道均值决定的）专属于该通道的二值掩码矩阵，将该通道的显著性区域标注了出来
        feature_maps_L28_sum = torch.sum(feature_maps_L28, dim=0)
        # print(feature_maps_L28_sum.shape)
        highlight_L28 = (feature_maps_L28_sum > torch.mean(feature_maps_L28_sum)).float()
        # print(highlight_L28.shape)
        h28 = highlight_chanelwise_L28 * highlight_L28.unsqueeze(
            0)  # h28是一个【512，h,w】的二值矩阵，这是与总体二值矩阵交集得到的，可能在某一个通道上是一个全零矩阵
        feature_maps_L28 = feature_maps_L28 * h28  # 同为shape为【512,7,7】的feature_map 与通道级二值矩阵（总体显著性区域与通道级显著性区域的交集）进行点乘，
        # chanelweight_l28 = torch.nn.functional.adaptive_max_pool2d(feature_maps_L28, output_size=1).squeeze(-1).squeeze(-1)
        chanelweight_l28 = torch.sum(feature_maps_L28, dim=(1, 2)) / torch.sum(h28, dim=(1, 2))
        chanelweight_l28 = torch.where(torch.isnan(chanelweight_l28), torch.full_like(chanelweight_l28, 0),
                                       chanelweight_l28)
        # chanelweight_l28=chanelweight_l28/torch.sum(highlight_L28)
        # print(chanelweight_l28)
        # chanelweight_l28 = (chanelweight_l28 - torch.min(chanelweight_l28)) / (
        #         torch.max(chanelweight_l28) - torch.min(chanelweight_l28))
        chanelweight_l28=(chanelweight_l28>torch.mean(chanelweight_l28)).float()
        chanelweight_l28_zero=torch.ones_like(chanelweight_l28).cuda().float() - chanelweight_l28


        feature_maps_L31 = feature_maps[
            1][
            0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        feature_maps_L31_channelwise_mean = torch.mean(feature_maps_L31, dim=(1, 2))
        highlight_chanelwise_L31 = (feature_maps_L31 > feature_maps_L31_channelwise_mean.view(-1, 1,
                                                                                              1)).float()  # 得到的结果依然是【512,7,7】的一个二值矩阵
        feature_maps_L31_sum = torch.sum(feature_maps_L31, dim=0)
        highlight_L31 = (feature_maps_L31_sum > torch.mean(feature_maps_L31_sum)).float()
        h31 = highlight_chanelwise_L31 * highlight_L31.unsqueeze(0)
        feature_maps_L31 = feature_maps_L31 * h31
        chanelweight_l31 = torch.sum(feature_maps_L31, dim=(1, 2)) / torch.sum(h31, dim=(1, 2))
        chanelweight_l31 = torch.where(torch.isnan(chanelweight_l31), torch.full_like(chanelweight_l31, 0),
                                       chanelweight_l31)
        chanelweight_l31=(chanelweight_l31>torch.mean(chanelweight_l31)).float()
        chanelweight_l31_zero=torch.ones_like(chanelweight_l31).cuda().float() - chanelweight_l31

        output_mean = []


        # output_max=[]
        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['features.26.weight', 'features.28.weight']:
                tmp = (param.grad).data
                # print(tmp)
                # tmp=torch.abs(tmp)
                # print(tmp)
                # tmp_mean_reuse=tmp.view(tmp.size()[0],-1)
                # print(tmp_mean_reuse.shape)
                tmp_mean_reuse=F.adaptive_max_pool2d(tmp,output_size=1).squeeze(-1).squeeze(-1) + F.adaptive_max_pool2d(-1 * tmp,output_size=1).squeeze(-1).squeeze(-1)
                # tmp_mean_reuse=F.adaptive_max_pool2d(tmp,output_size=1).squeeze(-1).squeeze(-1)
                # tmp_mean_reuse=F.adaptive_max_pool2d(-1 * tmp,output_size=1).squeeze(-1).squeeze(-1)
                if name == 'features.26.weight':
                    tmp_mean_reuse=tmp_mean_reuse*chanelweight_l28.unsqueeze(0).T
                    tmp_mean_reuse_bg=tmp_mean_reuse*chanelweight_l28_zero.unsqueeze(0).T

                else:
                    tmp_mean_reuse=tmp_mean_reuse*chanelweight_l31.unsqueeze(0).T
                    tmp_mean_reuse_bg=tmp_mean_reuse*chanelweight_l31_zero.unsqueeze(0).T

                # tmp_mean_reuse_bool=torch.mean(tmp_mean_reuse,dim=0).unsqueeze(0)
                # tmp_mean_reuse_bool=tmp_mean_reuse>tmp_mean_reuse_bool
                # tmp_mean_reuse_bool=tmp_mean_reuse_bool.float()
                #
                # tmp_mean_reuse_bool_2 = torch.mean(tmp_mean_reuse, dim=1).unsqueeze(0).T
                # tmp_mean_reuse_bool_2 = tmp_mean_reuse > tmp_mean_reuse_bool_2
                # tmp_mean_reuse_bool_2 = tmp_mean_reuse_bool_2.float()

                tmp_mean_reuse=tmp_mean_reuse# * tmp_mean_reuse_bool * tmp_mean_reuse_bool_2
                # tmp_mean_reuse_1 = torch.mean(tmp_1, axis=(2, 3))
                # tmp_mean_reuse = tmp_mean_reuse * tmp_mean_reuse_1
                # tmp_mean_reuse = torch.mean(tmp, dim=(2, 3))

                tmp_mean = torch.mean(tmp_mean_reuse, dim=0).cpu().data.numpy()
                # tmp_mean,_ = torch.max(tmp_mean_reuse, dim=0)
                # tmp_mean=tmp_mean.numpy()
                #############################################################################
                #############################################################################
                #############################################################################

                # tmp_mean_reuse_bool_bg = torch.mean(tmp_mean_reuse_bg, dim=0).unsqueeze(0)
                # tmp_mean_reuse_bool_bg = tmp_mean_reuse_bg > tmp_mean_reuse_bool_bg
                # tmp_mean_reuse_bool_bg = tmp_mean_reuse_bool_bg.float()
                #
                # tmp_mean_reuse_bool_2_bg = torch.mean(tmp_mean_reuse_bg, dim=1).unsqueeze(0).T
                # tmp_mean_reuse_bool_2_bg = tmp_mean_reuse_bg > tmp_mean_reuse_bool_2_bg
                # tmp_mean_reuse_bool_2_bg = tmp_mean_reuse_bool_2_bg.float()

                tmp_mean_reuse_bg = tmp_mean_reuse_bg# * tmp_mean_reuse_bool_bg * tmp_mean_reuse_bool_2_bg
                # tmp_mean_reuse_1 = torch.mean(tmp_1, axis=(2, 3))
                # tmp_mean_reuse = tmp_mean_reuse * tmp_mean_reuse_1
                # tmp_mean_reuse = torch.mean(tmp, dim=(2, 3))

                tmp_mean_bg = torch.mean(tmp_mean_reuse_bg, dim=0).cpu().data.numpy()
                # tmp_mean,_ = torch.max(tmp_mean_reuse, dim=0)
                # tmp_mean=tmp_mean.numpy()



                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                aa_bg = np.linalg.norm(tmp_mean_bg)
                if aa_bg != 0:
                    tmp_mean_bg = tmp_mean_bg / aa_bg
                else:
                    tmp_mean_bg = np.zeros_like(tmp_mean_bg)

                tmp_mean_stack=np.hstack((tmp_mean,tmp_mean_bg))
                # print(tmp_mean_stack.shape)
                output_mean.append(tmp_mean_stack)


        return output_mean



































class GradCam_yuhan_kernel_version23():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, index=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别
        output = self.net(input.cuda())
        output_max = torch.max(output)
        output_min = torch.min(output)
        output_norm = torch.div((output - output_min).float(), (output_max - output_min))
        ################################################################3
        output_norm = (output_norm * -1).cpu().tolist()
        output_norm = torch.tensor(output_norm).float().cuda()
        ###################################################################
        one_hot = torch.sum(output * output_norm)
        self.net.zero_grad()
        one_hot.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #
        output_mean = []


        # output_max=[]
        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['features.26.weight', 'features.28.weight']:
                tmp = (param.grad).cpu().data
                # print(tmp)
                # tmp=torch.abs(tmp)
                # print(tmp)
                # tmp_mean_reuse=tmp.view(tmp.size()[0],-1)
                # print(tmp_mean_reuse.shape)
                tmp_mean_reuse=F.adaptive_max_pool2d(tmp,output_size=1).squeeze(-1).squeeze(-1) + F.adaptive_max_pool2d(-1 * tmp,output_size=1).squeeze(-1).squeeze(-1)
                # tmp_mean_reuse=F.adaptive_max_pool2d(tmp,output_size=1).squeeze(-1).squeeze(-1)
                # tmp_mean_reuse=F.adaptive_max_pool2d(-1 * tmp,output_size=1).squeeze(-1).squeeze(-1)
                tmp_mean_reuse_bool=torch.mean(tmp_mean_reuse,dim=1).unsqueeze(0).T
                tmp_mean_reuse_bool=tmp_mean_reuse>tmp_mean_reuse_bool
                tmp_mean_reuse_bool=tmp_mean_reuse_bool.float()
                tmp_mean_reuse=tmp_mean_reuse * tmp_mean_reuse_bool
                # tmp_mean_reuse_1 = torch.mean(tmp_1, axis=(2, 3))
                # tmp_mean_reuse = tmp_mean_reuse * tmp_mean_reuse_1
                # tmp_mean_reuse = torch.mean(tmp, dim=(2, 3))

                tmp_mean = torch.sum(tmp_mean_reuse, dim=0)/torch.sum(tmp_mean_reuse_bool, dim=0)
                tmp_mean = torch.where(torch.isnan(tmp_mean), torch.full_like(tmp_mean, 0),
                                               tmp_mean)
                tmp_mean = tmp_mean.numpy()
                # tmp_mean,_ = torch.max(tmp_mean_reuse, dim=0)
                # tmp_mean=tmp_mean.numpy()
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean.append(tmp_mean)


        return output_mean
















class GradCam_yuhan_kernel_version21():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, index=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别
        features,output = self.net(input.cuda())
        output_max = torch.max(output)
        output_min = torch.min(output)
        output_norm = torch.div((output - output_min).float(), (output_max - output_min))
        ################################################################3
        output_norm = (output_norm * -1).cpu().tolist()
        output_norm = torch.tensor(output_norm).float().cuda()
        ###################################################################
        one_hot = torch.sum(output * output_norm)
        self.net.zero_grad()
        one_hot.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #


        feature_maps_L28 = features[
            0][0]  # 如果输入图像的shape为[batch_size,3,224,224]的tensor的话，关于输出的feature map,shape为  #[batch_size,512,14,14]左右
        # print(feature_maps_L28.shape)
        # print(feature_maps_L28)

        feature_maps_L28_channelwise_mean = torch.mean(feature_maps_L28, dim=(1, 2))
        # print(feature_maps_L28_channelwise_mean.shape)
        highlight_chanelwise_L28 = (feature_maps_L28 > feature_maps_L28_channelwise_mean.view(-1, 1, 1)).float()
        # print(highlight_chanelwise_L28.shape)#为这512通道的每一个通道生成了一个（由该通道均值决定的）专属于该通道的二值掩码矩阵，将该通道的显著性区域标注了出来
        feature_maps_L28_sum = torch.sum(feature_maps_L28, dim=0)
        # print(feature_maps_L28_sum.shape)
        highlight_L28 = (feature_maps_L28_sum > torch.mean(feature_maps_L28_sum)).float()
        # print(highlight_L28.shape)
        h28 = highlight_chanelwise_L28 * highlight_L28.unsqueeze(
            0)  # h28是一个【512，h,w】的二值矩阵，这是与总体二值矩阵交集得到的，可能在某一个通道上是一个全零矩阵
        # （一个完全对背景噪声响应的通道）
        # print(h28.shape)
        feature_maps_L28 = feature_maps_L28 * h28  # 同为shape为【512,7,7】的feature_map 与通道级二值矩阵（总体显著性区域与通道级显著性区域的交集）进行点乘，
        # chanelweight_l28 = torch.nn.functional.adaptive_max_pool2d(feature_maps_L28, output_size=1).squeeze(-1).squeeze(-1)
        chanelweight_l28 = torch.sum(feature_maps_L28, dim=(1, 2)) / torch.sum(h28, dim=(1, 2))
        # chanelweight_l28=F.adaptive_max_pool2d(feature_maps_L28
        #                                        ,output_size=1).squeeze(-1).squeeze(-1)

        # print(chanelweight_l28)
        # print(chanelweight_l28.shape)
        # 我们用0代替了nan值
        chanelweight_l28 = torch.where(torch.isnan(chanelweight_l28), torch.full_like(chanelweight_l28, 0),
                                       chanelweight_l28)
        # chanelweight_l28=chanelweight_l28/torch.sum(highlight_L28)
        # print(chanelweight_l28)
        chanelweight_l28 = (chanelweight_l28 - torch.min(chanelweight_l28)) / (
                    torch.max(chanelweight_l28) - torch.min(chanelweight_l28))
        # chanelweight_l28=(chanelweight_l28>torch.mean(chanelweight_l28)).float()
        # chanelweight_l28_bool = (chanelweight_l28 > torch.mean(chanelweight_l28)).float()
        # chanelweight_l28_bool_2 = torch.ones_like(chanelweight_l28_bool) - chanelweight_l28_bool
        # chanelweight_l28 = chanelweight_l28_bool * chanelweight_l28
        # chanelweight_l28 = chanelweight_l28 + chanelweight_l28_bool
        # print(chanelweight_l28)

        feature_maps_L31 = features[
            1][
            0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        feature_maps_L31_channelwise_mean = torch.mean(feature_maps_L31, dim=(1, 2))
        highlight_chanelwise_L31 = (feature_maps_L31 > feature_maps_L31_channelwise_mean.view(-1, 1,
                                                                                              1)).float()  # 得到的结果依然是【512,7,7】的一个二值矩阵
        feature_maps_L31_sum = torch.sum(feature_maps_L31, dim=0)
        highlight_L31 = (feature_maps_L31_sum > torch.mean(feature_maps_L31_sum)).float()
        h31 = highlight_chanelwise_L31 * highlight_L31.unsqueeze(0)
        feature_maps_L31 = feature_maps_L31 * h31
        # chanelweight_l31 = torch.nn.functional.adaptive_max_pool2d(feature_maps_L31, output_size=1).squeeze(-1).squeeze(
        #     -1)
        chanelweight_l31 = torch.sum(feature_maps_L31, dim=(1, 2)) / torch.sum(h31, dim=(1, 2))
        # chanelweight_l31 = F.adaptive_max_pool2d(feature_maps_L31
        #                                          , output_size=1).squeeze(-1).squeeze(-1)
        chanelweight_l31 = torch.where(torch.isnan(chanelweight_l31), torch.full_like(chanelweight_l31, 0),
                                       chanelweight_l31)

        # chanelweight_l31=torch.sum(h31,dim=(1,2))
        # chanelweight_l31=chanelweight_l31/torch.sum(highlight_L31)
        chanelweight_l31 = (chanelweight_l31 - torch.min(chanelweight_l31)) / (
                    torch.max(chanelweight_l31) - torch.min(chanelweight_l31))
        # chanelweight_l31=(chanelweight_l31>torch.mean(chanelweight_l31)).float()

        #
        # chanelweight_l31_bool=(chanelweight_l31>torch.mean(chanelweight_l31)).float()
        # chanelweight_l31_bool_2=torch.ones_like(chanelweight_l31_bool)-chanelweight_l31_bool
        # chanelweight_l31=chanelweight_l31_bool * chanelweight_l31
        # chanelweight_l31=chanelweight_l31 + chanelweight_l31_bool
        # print(chanelweight_l31)
        # print(torch.sum((chanelweight_l31/np.linalg.norm(chanelweight_l31.cpu().data.numpy()))*(chanelweight_l28/np.linalg.norm(chanelweight_l28.cpu().data.numpy()))))

        ###************
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #







        output_mean = []


        # output_max=[]
        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['features.26.weight', 'features.28.weight']:
                tmp = (param.grad).cpu().data
                # print(tmp)
                # tmp_1=torch.abs(tmp)
                # print(tmp)
                # tmp_mean_reuse=tmp.view(tmp.size()[0],-1)
                # print(tmp_mean_reuse.shape)
                tmp_mean_reuse=F.adaptive_max_pool2d(tmp,output_size=1).squeeze(-1).squeeze(-1) + F.adaptive_max_pool2d(-1 * tmp,output_size=1).squeeze(-1).squeeze(-1)
                # tmp_mean_reuse=F.adaptive_max_pool2d(tmp,output_size=1).squeeze(-1).squeeze(-1)
                # tmp_mean_reuse=F.adaptive_max_pool2d(-1 * tmp,output_size=1).squeeze(-1).squeeze(-1)

                # tmp_mean_reuse_1 = torch.mean(tmp_1, axis=(2, 3))
                # tmp_mean_reuse = tmp_mean_reuse * tmp_mean_reuse_1
                # tmp_mean_reuse = torch.mean(tmp, dim=(2, 3))


                if name=='features.26.weight':
                    # tmp_mean_reuse_bool = torch.mean(tmp_mean_reuse, dim=1).unsqueeze(0).T
                    # tmp_mean_reuse_bool = tmp_mean_reuse > tmp_mean_reuse_bool
                    # tmp_mean_reuse_bool = tmp_mean_reuse_bool.float()
                    # tmp_mean_reuse = tmp_mean_reuse * tmp_mean_reuse_bool
                    tmp_mean_reuse = tmp_mean_reuse * chanelweight_l28.cpu().unsqueeze(0).T
                    # tmp_mean_reuse_direction2 = tmp_mean_reuse

                    #【512,512】的矩阵的每一行乘以一个固定的权重，也就是一组卷积核乘以一组固定的权重；这个权重是如何得到的呢?是这样的，该层的一组卷积核一组决定了该层输出特征图的某一个通道
                    #如果能够通过某种方式得到该通道在特征图整体的512个通道中的重要性的话，也就是说，我们得到了该通道的权重，也就是该组卷积核的权重，也就是tmp_mean_reuse 这个矩阵的某一行的权重。
                    #那么这个权重是如何得到的呢？这样的，scda不是得到了一个【h，w】的聚合特征图掩码矩阵嘛；其实我们完全可以得到一个通道级的而非整体的聚合特征图掩码矩阵，也就是一个【512，h，w】
                    #的掩码矩阵，其实就是生成一个通道级均值，特征图的每一个通道，高于该均值的位置掩码为1，否则为0；这就得到了【512，h,w】的掩码矩阵
                    #接下来呢，就是通道级掩码矩阵的每一个通道与整体的[h,w]的掩码进行按位与；最终得到了一个【512，h，w】的处理之后的掩码矩阵；
                    #这就决定了通道级掩码矩阵的每一个通道的每一个为1的位置，都是与图像的主要目标而非背景有关的；我们可以据此进一步的获知输出特征图的每一个通道激活的都是目标还是背景，也就是该通道
                    #进而就是该通道对应的该组滤波器的重要程度
                    #接下来就是最后的一步了，我们将处理之后的【512，h,w】的通道级掩码矩阵与输出特征图进行点乘
                    #进而对于处理后的特征图的每一个通道的不为零的元素们，假设该通道共N个非零值，对这N个值求和在除以N；进而得到了该通道的一个权重
                    #如果某通道所有值都为0的话，就单纯的将该通道的权重设置为0便好啦
                    #最终得到了512个权重（当然啦，还得进行一步0~1的归一化啦），它既是每一个通道的权重，也是每一组卷积核的权重啦
                    #当然这最后一步还有很多可以尝试的空间啦
                else:
                    # tmp_mean_reuse_bool = torch.mean(tmp_mean_reuse, dim=1).unsqueeze(0).T
                    # tmp_mean_reuse_bool = tmp_mean_reuse > tmp_mean_reuse_bool
                    # tmp_mean_reuse_bool = tmp_mean_reuse_bool.float()
                    # tmp_mean_reuse = tmp_mean_reuse * tmp_mean_reuse_bool
                    tmp_mean_reuse = tmp_mean_reuse * chanelweight_l31.cpu().unsqueeze(0).T


                tmp_mean = torch.mean(tmp_mean_reuse, dim=0).numpy()
                # tmp_mean,_ = torch.max(tmp_mean_reuse, dim=0)
                # tmp_mean=tmp_mean.numpy()
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean.append(tmp_mean)


        return output_mean









class GradCam_yuhan_kernel_version5_pro():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, index=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别
        output = self.net(input.cuda())
        output_max=torch.max(output)
        output_min=torch.min(output)
        output_norm=torch.exp(torch.div((output-output_min).float(),(output_max-output_min)))
        ################################################################3
        output_norm=(output_norm * -1).cpu().tolist()
        output_norm=torch.tensor(output_norm).float().cuda()
        ###################################################################
        one_hot = torch.sum(output * output_norm)
        self.net.zero_grad()
        one_hot.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #
        output_mean=[]
        output_mean_direction2=[]

        # output_max=[]
        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['features.26.weight','features.28.weight']:
                tmp=(param * param.grad).cpu().data
                # tmp_mean_reuse=torch.mean(tmp,axis=( 2, 3))
                # tmp_mean_reuse=F.adaptive_max_pool2d(tmp,output_size=1).squeeze(-1).squeeze(-1) + F.adaptive_max_pool2d(-1 * tmp,output_size=1).squeeze(-1).squeeze(-1)
                tmp_mean_reuse = torch.std(tmp, axis=(2, 3))

                tmp_mean=torch.mean(tmp_mean_reuse,axis=0).numpy()
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                # tmp_max = torch.max(tmp, axis=( 2, 3))
                # tmp_max = torch.nn.functional.adaptive_max_pool2d(tmp, output_size=1).squeeze(-1).squeeze(-1)
                # tmp_max,_=torch.max(tmp_mean_reuse,axis=1)
                # tmp_max=tmp_max.numpy()
                #
                #
                # # tmp_max = tmp_max.reshape(1, -1).numpy()
                # aa = np.linalg.norm(tmp_max)
                # if aa != 0:
                #     tmp_max = tmp_max / aa
                # else:
                #     tmp_max = np.zeros_like(tmp_max)
                output_mean.append(tmp_mean)
                # output_max.append(tmp_max)
                tmp_mean = torch.mean(tmp_mean_reuse, axis=0).numpy()
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                # tmp_max = torch.max(tmp, axis=( 2, 3))
                # tmp_max = torch.nn.functional.adaptive_max_pool2d(tmp, output_size=1).squeeze(-1).squeeze(-1)
                # tmp_max,_=torch.max(tmp_mean_reuse,axis=1)
                # tmp_max=tmp_max.numpy()
                #
                #
                # # tmp_max = tmp_max.reshape(1, -1).numpy()
                # aa = np.linalg.norm(tmp_max)
                # if aa != 0:
                #     tmp_max = tmp_max / aa
                # else:
                #     tmp_max = np.zeros_like(tmp_max)
                output_mean_direction2.append(tmp_mean)

        return  output_mean,output_mean_direction2
















class GradCam_yuhan_kernel_version5_PLUS():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, index=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别
        output = self.net(input.cuda())
        output_max=torch.max(output)
        output_min=torch.min(output)
        output_norm=torch.div((output-output_min).float(),(output_max-output_min))
        output_norm=-1 * (output_max-output_min) * torch.exp(output_norm) * (2 - (output/(output_max - output_min)))
        ################################################################3
        # output_norm=(output_norm * -1).cpu().tolist()
        # output_norm=torch.tensor(output_norm).float().cuda()
        ###################################################################
        one_hot = torch.sum(output_norm)
        self.net.zero_grad()
        one_hot.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #
        output_mean=[]
        output_mean_direction2=[]

        # output_max=[]
        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['features.26.weight','features.28.weight']:
                tmp=(param * param.grad).cpu().data
                tmp_mean_reuse=torch.mean(tmp,axis=( 2, 3))
                tmp_mean=torch.mean(tmp_mean_reuse,axis=1).numpy()
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                # tmp_max = torch.max(tmp, axis=( 2, 3))
                # tmp_max = torch.nn.functional.adaptive_max_pool2d(tmp, output_size=1).squeeze(-1).squeeze(-1)
                # tmp_max,_=torch.max(tmp_mean_reuse,axis=1)
                # tmp_max=tmp_max.numpy()
                #
                #
                # # tmp_max = tmp_max.reshape(1, -1).numpy()
                # aa = np.linalg.norm(tmp_max)
                # if aa != 0:
                #     tmp_max = tmp_max / aa
                # else:
                #     tmp_max = np.zeros_like(tmp_max)
                output_mean.append(tmp_mean)
                # output_max.append(tmp_max)
                tmp_mean = torch.mean(tmp_mean_reuse, axis=0).numpy()
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                # tmp_max = torch.max(tmp, axis=( 2, 3))
                # tmp_max = torch.nn.functional.adaptive_max_pool2d(tmp, output_size=1).squeeze(-1).squeeze(-1)
                # tmp_max,_=torch.max(tmp_mean_reuse,axis=1)
                # tmp_max=tmp_max.numpy()
                #
                #
                # # tmp_max = tmp_max.reshape(1, -1).numpy()
                # aa = np.linalg.norm(tmp_max)
                # if aa != 0:
                #     tmp_max = tmp_max / aa
                # else:
                #     tmp_max = np.zeros_like(tmp_max)
                output_mean_direction2.append(tmp_mean)

        return  output_mean,output_mean_direction2





class GradCam_yuhan_kernel_version6():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, index=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别
        output = self.net(input.cuda())
        output_max=torch.max(output)
        output_min=torch.min(output)
        output_norm=torch.div((output-output_min).float(),(output_max-output_min))
        ################################################################3
        output_norm=(output_norm * -1).cpu().tolist()
        output_norm=torch.tensor(output_norm).float().cuda()
        ###################################################################
        one_hot = torch.sum(output * output_norm)
        self.net.zero_grad()
        one_hot.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #
        output_mean=[]
        output_mean_direction2=[]

        # output_max=[]
        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['features.26.weight','features.28.weight']:
                tmp=(param * param.grad).cpu().data
                tmp_mean_reuse=torch.mean(tmp,axis=( 2, 3))
                tmp_mean_direction2 = torch.mean(tmp_mean_reuse, axis=0)
                tmp_mean_direction1=torch.mean(tmp_mean_reuse,axis=1)

                tmp_mean_reuse_direction1=tmp_mean_reuse * tmp_mean_direction1.unsqueeze(0).T
                tmp_mean_direction1_with_weight = torch.mean(tmp_mean_reuse_direction1, dim=0)
                tmp_mean_direction1_with_weight = tmp_mean_direction1_with_weight.numpy()
                aa = np.linalg.norm(tmp_mean_direction1_with_weight)
                if aa != 0:
                    tmp_mean_direction1_with_weight = tmp_mean_direction1_with_weight / aa
                else:
                    tmp_mean_direction1_with_weight = np.zeros_like(tmp_mean_direction1_with_weight)

                # tmp_max = torch.max(tmp, axis=( 2, 3))
                # tmp_max = torch.nn.functional.adaptive_max_pool2d(tmp, output_size=1).squeeze(-1).squeeze(-1)
                # tmp_max,_=torch.max(tmp_mean_reuse,axis=1)
                # tmp_max=tmp_max.numpy()
                #
                #
                # # tmp_max = tmp_max.reshape(1, -1).numpy()
                # aa = np.linalg.norm(tmp_max)
                # if aa != 0:
                #     tmp_max = tmp_max / aa
                # else:
                #     tmp_max = np.zeros_like(tmp_max)
                output_mean.append(tmp_mean_direction1_with_weight)
                # output_max.append(tmp_max)
                # tmp_mean_direction2 = torch.mean(tmp_mean_reuse, axis=0).numpy()

                tmp_mean_reuse_direction2 = tmp_mean_reuse * tmp_mean_direction2.unsqueeze(0)
                tmp_mean_direction2_with_weight = torch.mean(tmp_mean_reuse_direction2, dim=1)
                tmp_mean_direction2_with_weight = tmp_mean_direction2_with_weight.numpy()
                aa = np.linalg.norm(tmp_mean_direction2_with_weight)
                if aa != 0:
                    tmp_mean_direction2_with_weight = tmp_mean_direction2_with_weight / aa
                else:
                    tmp_mean_direction2_with_weight = np.zeros_like(tmp_mean_direction2_with_weight)

                # tmp_max = torch.max(tmp, axis=( 2, 3))
                # tmp_max = torch.nn.functional.adaptive_max_pool2d(tmp, output_size=1).squeeze(-1).squeeze(-1)
                # tmp_max,_=torch.max(tmp_mean_reuse,axis=1)
                # tmp_max=tmp_max.numpy()
                #
                #
                # # tmp_max = tmp_max.reshape(1, -1).numpy()
                # aa = np.linalg.norm(tmp_max)
                # if aa != 0:
                #     tmp_max = tmp_max / aa
                # else:
                #     tmp_max = np.zeros_like(tmp_max)
                output_mean_direction2.append(tmp_mean_direction2_with_weight)

        return  output_mean,output_mean_direction2




class GradCam_yuhan_kernel_version7():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, index=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别
        output = self.net(input.cuda())
        output_max=torch.max(output)
        output_min=torch.min(output)
        output_norm=torch.div((output-output_min).float(),(output_max-output_min))
        ################################################################3
        output_norm=(output_norm * -1).cpu().tolist()
        output_norm=torch.tensor(output_norm).float().cuda()
        ###################################################################
        one_hot = torch.sum(output * output_norm)
        self.net.zero_grad()
        one_hot.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #
        output_mean=[]
        output_mean_direction2=[]

        # output_max=[]
        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['features.26.weight','features.28.weight']:
                tmp=(param * param.grad).cpu().data
                tmp_mean_reuse=torch.mean(tmp,axis=( 2, 3))
                tmp_mean_direction2 = torch.mean(tmp_mean_reuse, axis=0)
                tmp_mean_direction2=(tmp_mean_direction2>torch.mean(tmp_mean_direction2)).float()
                tmp_mean_direction1=torch.mean(tmp_mean_reuse,axis=1)
                tmp_mean_direction1=(tmp_mean_direction1>torch.mean(tmp_mean_direction1)).float()

                tmp_mean_reuse_direction1=tmp_mean_reuse * tmp_mean_direction1.unsqueeze(0).T
                tmp_mean_direction1_with_weight = torch.mean(tmp_mean_reuse_direction1, dim=0)
                tmp_mean_direction1_with_weight = tmp_mean_direction1_with_weight.numpy()
                aa = np.linalg.norm(tmp_mean_direction1_with_weight)
                if aa != 0:
                    tmp_mean_direction1_with_weight = tmp_mean_direction1_with_weight / aa
                else:
                    tmp_mean_direction1_with_weight = np.zeros_like(tmp_mean_direction1_with_weight)

                # tmp_max = torch.max(tmp, axis=( 2, 3))
                # tmp_max = torch.nn.functional.adaptive_max_pool2d(tmp, output_size=1).squeeze(-1).squeeze(-1)
                # tmp_max,_=torch.max(tmp_mean_reuse,axis=1)
                # tmp_max=tmp_max.numpy()
                #
                #
                # # tmp_max = tmp_max.reshape(1, -1).numpy()
                # aa = np.linalg.norm(tmp_max)
                # if aa != 0:
                #     tmp_max = tmp_max / aa
                # else:
                #     tmp_max = np.zeros_like(tmp_max)
                output_mean.append(tmp_mean_direction1_with_weight)
                # output_max.append(tmp_max)
                # tmp_mean_direction2 = torch.mean(tmp_mean_reuse, axis=0).numpy()

                tmp_mean_reuse_direction2 = tmp_mean_reuse * tmp_mean_direction2.unsqueeze(0)
                tmp_mean_direction2_with_weight = torch.mean(tmp_mean_reuse_direction2, dim=1)
                tmp_mean_direction2_with_weight = tmp_mean_direction2_with_weight.numpy()
                aa = np.linalg.norm(tmp_mean_direction2_with_weight)
                if aa != 0:
                    tmp_mean_direction2_with_weight = tmp_mean_direction2_with_weight / aa
                else:
                    tmp_mean_direction2_with_weight = np.zeros_like(tmp_mean_direction2_with_weight)

                # tmp_max = torch.max(tmp, axis=( 2, 3))
                # tmp_max = torch.nn.functional.adaptive_max_pool2d(tmp, output_size=1).squeeze(-1).squeeze(-1)
                # tmp_max,_=torch.max(tmp_mean_reuse,axis=1)
                # tmp_max=tmp_max.numpy()
                #
                #
                # # tmp_max = tmp_max.reshape(1, -1).numpy()
                # aa = np.linalg.norm(tmp_max)
                # if aa != 0:
                #     tmp_max = tmp_max / aa
                # else:
                #     tmp_max = np.zeros_like(tmp_max)
                output_mean_direction2.append(tmp_mean_direction2_with_weight)

        return  output_mean,output_mean_direction2





class GradCam_yuhan_kernel_version7_plus():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, index=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别
        output = self.net(input.cuda())
        output_max=torch.max(output)
        output_min=torch.min(output)
        output_norm=torch.div((output-output_min).float(),(output_max-output_min))
        ################################################################3
        output_norm=(output_norm * -1).cpu().tolist()
        output_norm=torch.tensor(output_norm).float().cuda()
        ###################################################################
        one_hot = torch.sum(output * output_norm)
        self.net.zero_grad()
        one_hot.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #
        output_mean=[]
        output_mean_direction2=[]

        # output_max=[]
        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['features.26.weight','features.28.weight']:
                tmp=(param * param.grad).cpu().data
                grad_abs=torch.abs(param.grad.cpu().data)#(512,512,3,3)
                grad_abs_reuse=torch.mean(grad_abs,axis=( 2, 3))#(512,512)

                tmp_mean_reuse=torch.mean(tmp,axis=( 2, 3))
                tmp_mean_direction2 = torch.mean(grad_abs_reuse, axis=0)
                tmp_mean_direction2=(tmp_mean_direction2>torch.mean(tmp_mean_direction2)).float()
                tmp_mean_direction1=torch.mean(grad_abs_reuse,axis=1)
                tmp_mean_direction1=(tmp_mean_direction1>torch.mean(tmp_mean_direction1)).float()

                tmp_mean_reuse_direction1=tmp_mean_reuse * tmp_mean_direction1.unsqueeze(0).T
                tmp_mean_direction1_with_weight = torch.mean(tmp_mean_reuse_direction1, dim=0)
                tmp_mean_direction1_with_weight = tmp_mean_direction1_with_weight.numpy()
                aa = np.linalg.norm(tmp_mean_direction1_with_weight)
                if aa != 0:
                    tmp_mean_direction1_with_weight = tmp_mean_direction1_with_weight / aa
                else:
                    tmp_mean_direction1_with_weight = np.zeros_like(tmp_mean_direction1_with_weight)

                # tmp_max = torch.max(tmp, axis=( 2, 3))
                # tmp_max = torch.nn.functional.adaptive_max_pool2d(tmp, output_size=1).squeeze(-1).squeeze(-1)
                # tmp_max,_=torch.max(tmp_mean_reuse,axis=1)
                # tmp_max=tmp_max.numpy()
                #
                #
                # # tmp_max = tmp_max.reshape(1, -1).numpy()
                # aa = np.linalg.norm(tmp_max)
                # if aa != 0:
                #     tmp_max = tmp_max / aa
                # else:
                #     tmp_max = np.zeros_like(tmp_max)
                output_mean.append(tmp_mean_direction1_with_weight)
                # output_max.append(tmp_max)
                # tmp_mean_direction2 = torch.mean(tmp_mean_reuse, axis=0).numpy()

                tmp_mean_reuse_direction2 = tmp_mean_reuse * tmp_mean_direction2.unsqueeze(0)
                tmp_mean_direction2_with_weight = torch.mean(tmp_mean_reuse_direction2, dim=1)
                tmp_mean_direction2_with_weight = tmp_mean_direction2_with_weight.numpy()
                aa = np.linalg.norm(tmp_mean_direction2_with_weight)
                if aa != 0:
                    tmp_mean_direction2_with_weight = tmp_mean_direction2_with_weight / aa
                else:
                    tmp_mean_direction2_with_weight = np.zeros_like(tmp_mean_direction2_with_weight)

                # tmp_max = torch.max(tmp, axis=( 2, 3))
                # tmp_max = torch.nn.functional.adaptive_max_pool2d(tmp, output_size=1).squeeze(-1).squeeze(-1)
                # tmp_max,_=torch.max(tmp_mean_reuse,axis=1)
                # tmp_max=tmp_max.numpy()
                #
                #
                # # tmp_max = tmp_max.reshape(1, -1).numpy()
                # aa = np.linalg.norm(tmp_max)
                # if aa != 0:
                #     tmp_max = tmp_max / aa
                # else:
                #     tmp_max = np.zeros_like(tmp_max)
                output_mean_direction2.append(tmp_mean_direction2_with_weight)

        return  output_mean,output_mean_direction2




class GradCam_yuhan_kernel_version7_plus_max():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, index=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别
        output = self.net(input.cuda())
        output_max=torch.max(output)
        output_min=torch.min(output)
        output_norm=torch.div((output-output_min).float(),(output_max-output_min))
        ################################################################3
        output_norm=(output_norm * -1).cpu().tolist()
        output_norm=torch.tensor(output_norm).float().cuda()
        ###################################################################
        one_hot = torch.sum(output * output_norm)
        self.net.zero_grad()
        one_hot.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #
        output_mean=[]
        output_mean_direction2=[]

        # output_max=[]
        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['features.26.weight','features.28.weight']:
                tmp=(param * param.grad).cpu().data
                grad_abs=torch.abs(param.grad.cpu().data)#(512,512,3,3)
                grad_abs_reuse=torch.mean(grad_abs,axis=( 2, 3))#(512,512)

                tmp_mean_reuse=torch.mean(tmp,axis=( 2, 3))
                tmp_mean_direction2 = torch.mean(grad_abs_reuse, axis=0)
                tmp_mean_direction2=(tmp_mean_direction2>torch.mean(tmp_mean_direction2)).float()
                tmp_mean_direction1=torch.mean(grad_abs_reuse,axis=1)
                tmp_mean_direction1=(tmp_mean_direction1>torch.mean(tmp_mean_direction1)).float()

                tmp_mean_reuse_direction1=tmp_mean_reuse * tmp_mean_direction1.unsqueeze(0).T
                tmp_mean_direction1_with_weight = torch.mean(tmp_mean_reuse_direction1, dim=0)
                tmp_mean_direction1_with_weight = tmp_mean_direction1_with_weight.numpy()
                aa = np.linalg.norm(tmp_mean_direction1_with_weight)
                if aa != 0:
                    tmp_mean_direction1_with_weight = tmp_mean_direction1_with_weight / aa
                else:
                    tmp_mean_direction1_with_weight = np.zeros_like(tmp_mean_direction1_with_weight)

                # tmp_max = torch.max(tmp, axis=( 2, 3))
                # tmp_max = torch.nn.functional.adaptive_max_pool2d(tmp, output_size=1).squeeze(-1).squeeze(-1)
                # tmp_max,_=torch.max(tmp_mean_reuse,axis=1)
                # tmp_max=tmp_max.numpy()
                #
                #
                # # tmp_max = tmp_max.reshape(1, -1).numpy()
                # aa = np.linalg.norm(tmp_max)
                # if aa != 0:
                #     tmp_max = tmp_max / aa
                # else:
                #     tmp_max = np.zeros_like(tmp_max)
                output_mean.append(tmp_mean_direction1_with_weight)
                # output_max.append(tmp_max)
                # tmp_mean_direction2 = torch.mean(tmp_mean_reuse, axis=0).numpy()

                tmp_mean_reuse_direction2 = tmp_mean_reuse * tmp_mean_direction2.unsqueeze(0)
                tmp_mean_direction2_with_weight = torch.mean(tmp_mean_reuse_direction2, dim=1)
                tmp_mean_direction2_with_weight = tmp_mean_direction2_with_weight.numpy()
                aa = np.linalg.norm(tmp_mean_direction2_with_weight)
                if aa != 0:
                    tmp_mean_direction2_with_weight = tmp_mean_direction2_with_weight / aa
                else:
                    tmp_mean_direction2_with_weight = np.zeros_like(tmp_mean_direction2_with_weight)

                # tmp_max = torch.max(tmp, axis=( 2, 3))
                # tmp_max = torch.nn.functional.adaptive_max_pool2d(tmp, output_size=1).squeeze(-1).squeeze(-1)
                # tmp_max,_=torch.max(tmp_mean_reuse,axis=1)
                # tmp_max=tmp_max.numpy()
                #
                #
                # # tmp_max = tmp_max.reshape(1, -1).numpy()
                # aa = np.linalg.norm(tmp_max)
                # if aa != 0:
                #     tmp_max = tmp_max / aa
                # else:
                #     tmp_max = np.zeros_like(tmp_max)
                output_mean_direction2.append(tmp_mean_direction2_with_weight)

        return  output_mean,output_mean_direction2








class GradCam_yuhan_kernel_version8():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, index=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别
        output = self.net(input.cuda())
        output_max=torch.max(output)
        output_min=torch.min(output)
        output_norm=torch.div((output-output_min).float(),(output_max-output_min))
        ################################################################3
        output_norm=(output_norm * -1).cpu().tolist()
        output_norm=torch.tensor(output_norm).float().cuda()
        ###################################################################
        one_hot = torch.sum(output * output_norm)
        self.net.zero_grad()
        one_hot.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #
        output_mean=[]
        output_mean_direction2=[]

        # output_max=[]
        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['features.26.weight','features.28.weight']:
                tmp=(param * param.grad).cpu().data
                tmp_mean_reuse=torch.mean(tmp,axis=( 2, 3))
                tmp_mean_direction2 = torch.mean(tmp_mean_reuse, axis=0)
                tmp_mean_direction2 = (tmp_mean_direction2 - torch.min(tmp_mean_direction2)) / (torch.max(tmp_mean_direction2) - torch.min(tmp_mean_direction2))
                # tmp_mean_direction2=(tmp_mean_direction2>torch.mean(tmp_mean_direction2)).int()
                tmp_mean_direction1=torch.mean(tmp_mean_reuse,axis=1)
                tmp_mean_direction1 = (tmp_mean_direction1 - torch.min(tmp_mean_direction1)) / (torch.max(tmp_mean_direction1) - torch.min(tmp_mean_direction1))
                # tmp_mean_direction1=(tmp_mean_direction1>torch.mean(tmp_mean_direction1)).int()

                tmp_mean_reuse_direction1=tmp_mean_reuse * tmp_mean_direction1.unsqueeze(0).T
                tmp_mean_direction1_with_weight = torch.mean(tmp_mean_reuse_direction1, dim=0)
                tmp_mean_direction1_with_weight = tmp_mean_direction1_with_weight.numpy()
                aa = np.linalg.norm(tmp_mean_direction1_with_weight)
                if aa != 0:
                    tmp_mean_direction1_with_weight = tmp_mean_direction1_with_weight / aa
                else:
                    tmp_mean_direction1_with_weight = np.zeros_like(tmp_mean_direction1_with_weight)

                # tmp_max = torch.max(tmp, axis=( 2, 3))
                # tmp_max = torch.nn.functional.adaptive_max_pool2d(tmp, output_size=1).squeeze(-1).squeeze(-1)
                # tmp_max,_=torch.max(tmp_mean_reuse,axis=1)
                # tmp_max=tmp_max.numpy()
                #
                #
                # # tmp_max = tmp_max.reshape(1, -1).numpy()
                # aa = np.linalg.norm(tmp_max)
                # if aa != 0:
                #     tmp_max = tmp_max / aa
                # else:
                #     tmp_max = np.zeros_like(tmp_max)
                output_mean.append(tmp_mean_direction1_with_weight)
                # output_max.append(tmp_max)
                # tmp_mean_direction2 = torch.mean(tmp_mean_reuse, axis=0).numpy()

                tmp_mean_reuse_direction2 = tmp_mean_reuse * tmp_mean_direction2.unsqueeze(0)
                tmp_mean_direction2_with_weight = torch.mean(tmp_mean_reuse_direction2, dim=1)
                tmp_mean_direction2_with_weight = tmp_mean_direction2_with_weight.numpy()
                aa = np.linalg.norm(tmp_mean_direction2_with_weight)
                if aa != 0:
                    tmp_mean_direction2_with_weight = tmp_mean_direction2_with_weight / aa
                else:
                    tmp_mean_direction2_with_weight = np.zeros_like(tmp_mean_direction2_with_weight)

                # tmp_max = torch.max(tmp, axis=( 2, 3))
                # tmp_max = torch.nn.functional.adaptive_max_pool2d(tmp, output_size=1).squeeze(-1).squeeze(-1)
                # tmp_max,_=torch.max(tmp_mean_reuse,axis=1)
                # tmp_max=tmp_max.numpy()
                #
                #
                # # tmp_max = tmp_max.reshape(1, -1).numpy()
                # aa = np.linalg.norm(tmp_max)
                # if aa != 0:
                #     tmp_max = tmp_max / aa
                # else:
                #     tmp_max = np.zeros_like(tmp_max)
                output_mean_direction2.append(tmp_mean_direction2_with_weight)

        return  output_mean,output_mean_direction2


class GradCam_yuhan_kernel_version9():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, index=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别
        output = self.net(input.cuda())
        output_max=torch.max(output)
        output_min=torch.min(output)
        output_norm=torch.div((output-output_min).float(),(output_max-output_min))
        ################################################################3
        output_norm=(output_norm * -1).cpu().tolist()
        output_norm=torch.tensor(output_norm).float().cuda()
        ###################################################################
        one_hot = torch.sum(output * output_norm)
        self.net.zero_grad()
        one_hot.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #
        output_mean=[]
        output_mean_direction2=[]

        # output_max=[]
        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['features.26.weight','features.28.weight']:
                tmp=(param * param.grad).cpu().data
                tmp_mean_reuse=torch.mean(tmp,axis=( 2, 3))
                tmp_mean_direction2 = torch.mean(tmp_mean_reuse, axis=0)
                tmp_mean_direction1=torch.mean(tmp_mean_reuse,axis=1)

                tmp_mean_reuse_direction1=tmp_mean_reuse * tmp_mean_direction2.unsqueeze(0).T
                tmp_mean_direction1_with_weight = torch.mean(tmp_mean_reuse_direction1, dim=0)
                tmp_mean_direction1_with_weight = tmp_mean_direction1_with_weight.numpy()
                aa = np.linalg.norm(tmp_mean_direction1_with_weight)
                if aa != 0:
                    tmp_mean_direction1_with_weight = tmp_mean_direction1_with_weight / aa
                else:
                    tmp_mean_direction1_with_weight = np.zeros_like(tmp_mean_direction1_with_weight)

                # tmp_max = torch.max(tmp, axis=( 2, 3))
                # tmp_max = torch.nn.functional.adaptive_max_pool2d(tmp, output_size=1).squeeze(-1).squeeze(-1)
                # tmp_max,_=torch.max(tmp_mean_reuse,axis=1)
                # tmp_max=tmp_max.numpy()
                #
                #
                # # tmp_max = tmp_max.reshape(1, -1).numpy()
                # aa = np.linalg.norm(tmp_max)
                # if aa != 0:
                #     tmp_max = tmp_max / aa
                # else:
                #     tmp_max = np.zeros_like(tmp_max)
                output_mean.append(tmp_mean_direction1_with_weight)
                # output_max.append(tmp_max)
                # tmp_mean_direction2 = torch.mean(tmp_mean_reuse, axis=0).numpy()

                tmp_mean_reuse_direction2 = tmp_mean_reuse * tmp_mean_direction1.unsqueeze(0)
                tmp_mean_direction2_with_weight = torch.mean(tmp_mean_reuse_direction2, dim=1)
                tmp_mean_direction2_with_weight = tmp_mean_direction2_with_weight.numpy()
                aa = np.linalg.norm(tmp_mean_direction2_with_weight)
                if aa != 0:
                    tmp_mean_direction2_with_weight = tmp_mean_direction2_with_weight / aa
                else:
                    tmp_mean_direction2_with_weight = np.zeros_like(tmp_mean_direction2_with_weight)

                # tmp_max = torch.max(tmp, axis=( 2, 3))
                # tmp_max = torch.nn.functional.adaptive_max_pool2d(tmp, output_size=1).squeeze(-1).squeeze(-1)
                # tmp_max,_=torch.max(tmp_mean_reuse,axis=1)
                # tmp_max=tmp_max.numpy()
                #
                #
                # # tmp_max = tmp_max.reshape(1, -1).numpy()
                # aa = np.linalg.norm(tmp_max)
                # if aa != 0:
                #     tmp_max = tmp_max / aa
                # else:
                #     tmp_max = np.zeros_like(tmp_max)
                output_mean_direction2.append(tmp_mean_direction2_with_weight)

        return  output_mean,output_mean_direction2





class GradCam_yuhan_kernel_version10():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, index=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别
        output = self.net(input.cuda())
        output_max=torch.max(output)
        output_min=torch.min(output)
        output_norm=torch.div((output-output_min).float(),(output_max-output_min))
        ################################################################3
        output_norm=(output_norm * -1).cpu().tolist()
        output_norm=torch.tensor(output_norm).float().cuda()
        ###################################################################
        one_hot = torch.sum(output * output_norm)
        self.net.zero_grad()
        one_hot.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #
        output_mean=[]
        output_mean_direction2=[]

        # output_max=[]
        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['features.26.weight','features.28.weight']:
                tmp=(param * param.grad).cpu().data
                tmp_mean_reuse=torch.mean(tmp,axis=( 2, 3))
                tmp_mean_direction2 = torch.mean(tmp_mean_reuse, axis=0)
                tmp_mean_direction2=(tmp_mean_direction2>torch.mean(tmp_mean_direction2)).float()
                tmp_mean_direction1=torch.mean(tmp_mean_reuse,axis=1)
                tmp_mean_direction1=(tmp_mean_direction1>torch.mean(tmp_mean_direction1)).float()

                tmp_mean_reuse_direction1=tmp_mean_reuse * tmp_mean_direction2.unsqueeze(0).T
                tmp_mean_direction1_with_weight = torch.mean(tmp_mean_reuse_direction1, dim=0)
                tmp_mean_direction1_with_weight = tmp_mean_direction1_with_weight.numpy()
                aa = np.linalg.norm(tmp_mean_direction1_with_weight)
                if aa != 0:
                    tmp_mean_direction1_with_weight = tmp_mean_direction1_with_weight / aa
                else:
                    tmp_mean_direction1_with_weight = np.zeros_like(tmp_mean_direction1_with_weight)

                # tmp_max = torch.max(tmp, axis=( 2, 3))
                # tmp_max = torch.nn.functional.adaptive_max_pool2d(tmp, output_size=1).squeeze(-1).squeeze(-1)
                # tmp_max,_=torch.max(tmp_mean_reuse,axis=1)
                # tmp_max=tmp_max.numpy()
                #
                #
                # # tmp_max = tmp_max.reshape(1, -1).numpy()
                # aa = np.linalg.norm(tmp_max)
                # if aa != 0:
                #     tmp_max = tmp_max / aa
                # else:
                #     tmp_max = np.zeros_like(tmp_max)
                output_mean.append(tmp_mean_direction1_with_weight)
                # output_max.append(tmp_max)
                # tmp_mean_direction2 = torch.mean(tmp_mean_reuse, axis=0).numpy()

                tmp_mean_reuse_direction2 = tmp_mean_reuse * tmp_mean_direction1.unsqueeze(0)
                tmp_mean_direction2_with_weight = torch.mean(tmp_mean_reuse_direction2, dim=1)
                tmp_mean_direction2_with_weight = tmp_mean_direction2_with_weight.numpy()
                aa = np.linalg.norm(tmp_mean_direction2_with_weight)
                if aa != 0:
                    tmp_mean_direction2_with_weight = tmp_mean_direction2_with_weight / aa
                else:
                    tmp_mean_direction2_with_weight = np.zeros_like(tmp_mean_direction2_with_weight)

                # tmp_max = torch.max(tmp, axis=( 2, 3))
                # tmp_max = torch.nn.functional.adaptive_max_pool2d(tmp, output_size=1).squeeze(-1).squeeze(-1)
                # tmp_max,_=torch.max(tmp_mean_reuse,axis=1)
                # tmp_max=tmp_max.numpy()
                #
                #
                # # tmp_max = tmp_max.reshape(1, -1).numpy()
                # aa = np.linalg.norm(tmp_max)
                # if aa != 0:
                #     tmp_max = tmp_max / aa
                # else:
                #     tmp_max = np.zeros_like(tmp_max)
                output_mean_direction2.append(tmp_mean_direction2_with_weight)

        return  output_mean,output_mean_direction2








class GradCam_yuhan_kernel_version11():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, index=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别
        output = self.net(input.cuda())
        output_max=torch.max(output)
        output_min=torch.min(output)
        output_norm=torch.div((output-output_min).float(),(output_max-output_min))
        ################################################################3
        output_norm=(output_norm * -1).cpu().tolist()
        output_norm=torch.tensor(output_norm).float().cuda()
        ###################################################################
        one_hot = torch.sum(output * output_norm)
        self.net.zero_grad()
        one_hot.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #
        output_mean=[]
        output_mean_direction2=[]

        output_max=[]
        output_max_direction2=[]
        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['features.26.weight','features.28.weight']:
                tmp=(param * param.grad).cpu().data
                tmp_mean_reuse=torch.mean(tmp,axis=( 2, 3))
                tmp_mean=torch.mean(tmp_mean_reuse,axis=1).numpy()
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)


                tmp_max = torch.nn.functional.adaptive_max_pool2d(tmp, output_size=1).squeeze(-1).squeeze(-1)
                tmp_min = (-1) * torch.nn.functional.adaptive_max_pool2d(-1 * tmp, output_size=1).squeeze(-1).squeeze(-1)
                tmp_max_min_reuse = tmp_max - tmp_min
                tmp_max, _ = torch.max(tmp_max_min_reuse, axis=1)
                tmp_max = tmp_max.numpy()
                aa = np.linalg.norm(tmp_max)
                if aa != 0:
                    tmp_max = tmp_max / aa
                else:
                    tmp_max = np.zeros_like(tmp_max)

                output_mean.append(tmp_mean)
                output_max.append(tmp_max)
#############################################################3
                tmp_mean = torch.mean(tmp_mean_reuse, axis=0).numpy()
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)


                tmp_max, _ = torch.max(tmp_max_min_reuse, axis=0)
                tmp_max = tmp_max.numpy()

                aa = np.linalg.norm(tmp_max)
                if aa != 0:
                    tmp_max = tmp_max / aa
                else:
                    tmp_max = np.zeros_like(tmp_max)


                output_mean_direction2.append(tmp_mean)
                output_max_direction2.append(tmp_max)

        return  output_mean,output_mean_direction2,output_max,output_max_direction2




class GradCam_yuhan_kernel_version12():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, index=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别
        output = self.net(input.cuda())
        output_max=torch.max(output)
        output_min=torch.min(output)
        output_norm=torch.div((output-output_min).float(),(output_max-output_min))
        ################################################################3
        output_norm=(output_norm * -1).cpu().tolist()
        output_norm=torch.tensor(output_norm).float().cuda()
        ###################################################################
        one_hot = torch.sum(output * output_norm)
        self.net.zero_grad()
        one_hot.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #
        output_mean=[]
        output_mean_direction2=[]

        output_mean_grad_only=[]
        output_mean_direction2_grad_only=[]
        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['features.26.weight','features.28.weight']:
                tmp=(param * param.grad).cpu().data
                tmp_mean_reuse=torch.mean(tmp,axis=( 2, 3))
                tmp_mean=torch.mean(tmp_mean_reuse,axis=1).numpy()
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean.append(tmp_mean)
#############################################################3
                tmp_mean = torch.mean(tmp_mean_reuse, axis=0).numpy()
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean_direction2.append(tmp_mean)
########***********************************///////////////////****************+++++++++++
                tmp = (param.grad).cpu().data
                tmp_mean_reuse = torch.mean(tmp, axis=(2, 3))
                tmp_mean = torch.mean(tmp_mean_reuse, axis=1).numpy()
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean_grad_only.append(tmp_mean)
                #############################################################3
                tmp_mean = torch.mean(tmp_mean_reuse, axis=0).numpy()
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean_direction2_grad_only.append(tmp_mean)

        return  output_mean,output_mean_direction2,output_mean_grad_only,output_mean_direction2_grad_only








class GradCam_yuhan_kernel_version13():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, index=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别
        features,output = self.net(input.cuda())
        ###************
        feature_maps_L28 = features[
            0][0]  # 如果输入图像的shape为[batch_size,3,224,224]的tensor的话，关于输出的feature map,shape为  #[batch_size,512,14,14]左右
        feature_maps_L28_channelwise_mean=torch.mean(feature_maps_L28,dim=(1,2))
        highlight_chanelwise_L28 = (feature_maps_L28 > feature_maps_L28_channelwise_mean.view(-1, 1, 1)).float()
        feature_maps_L28_sum=torch.sum(feature_maps_L28, dim=0)
        highlight_L28 = (feature_maps_L28_sum > torch.mean(feature_maps_L28_sum)).float()
        h28 = highlight_chanelwise_L28 * highlight_L28.unsqueeze(0)
        chanelweight_l28=torch.sum(h28,dim=(1,2))
        chanelweight_l28=chanelweight_l28/torch.sum(highlight_L28)
        chanelweight_l28=(chanelweight_l28>torch.mean(chanelweight_l28)).float()


        feature_maps_L31 = features[
            1][0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        feature_maps_L31_channelwise_mean=torch.mean(feature_maps_L31,dim=(1,2))
        highlight_chanelwise_L31 = (feature_maps_L31 > feature_maps_L31_channelwise_mean.view(-1, 1, 1)).float()#得到的结果依然是【512,7,7】的一个二值矩阵
        feature_maps_L31_sum=torch.sum(feature_maps_L31, dim=0)
        highlight_L31 = (feature_maps_L31_sum > torch.mean(feature_maps_L31_sum)).float()
        h31 = highlight_chanelwise_L31 * highlight_L31.unsqueeze(0)
        chanelweight_l31=torch.sum(h31,dim=(1,2))
        chanelweight_l31=chanelweight_l31/torch.sum(highlight_L31)
        chanelweight_l31=(chanelweight_l31>torch.mean(chanelweight_l31)).float()

        ###************

        output_max=torch.max(output)
        output_min=torch.min(output)
        output_norm=torch.div((output-output_min).float(),(output_max-output_min))
        ################################################################3
        output_norm=(output_norm * -1).cpu().tolist()
        output_norm=torch.tensor(output_norm).float().cuda()
        ###################################################################
        one_hot = torch.sum(output * output_norm)
        self.net.zero_grad()
        one_hot.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #
        output_mean=[]
        output_mean_direction2=[]

        output_mean_grad_only=[]
        output_mean_direction2_grad_only=[]
        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['features.26.weight','features.28.weight']:
                tmp=(param * param.grad).cpu().data
                # tmp=(param).cpu().data
                tmp_mean_reuse=torch.mean(tmp,axis=( 2, 3))
                tmp_mean=torch.mean(tmp_mean_reuse,axis=1).numpy()
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean.append(tmp_mean)
#############################################################3
                #zhelizheli
                if name=='features.26.weight':
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l28.cpu().unsqueeze(0).T
                else:
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l31.cpu().unsqueeze(0).T
                tmp_mean = torch.mean(tmp_mean_reuse_direction2, axis=0).numpy()#54.**
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean_direction2.append(tmp_mean)
########***********************************///////////////////****************+++++++++++
                tmp = (param.grad).cpu().data
                tmp_mean_reuse = torch.mean(tmp, axis=(2, 3))
                tmp_mean = torch.mean(tmp_mean_reuse, axis=1).numpy()
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean_grad_only.append(tmp_mean)
                #############################################################3
                if name == 'features.26.weight':
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l28.cpu().unsqueeze(0).T
                else:
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l31.cpu().unsqueeze(0).T
                tmp_mean = torch.mean(tmp_mean_reuse_direction2, axis=0).numpy()
                # tmp_mean = torch.mean(tmp_mean_reuse, axis=0).numpy()#58.96

                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean_direction2_grad_only.append(tmp_mean)

        return  output_mean,output_mean_direction2,output_mean_grad_only,output_mean_direction2_grad_only









class GradCam_yuhan_kernel_version14():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, index=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别
        features,output = self.net(input.cuda())
        ###************
        feature_maps_L28 = features[
            0][0]  # 如果输入图像的shape为[batch_size,3,224,224]的tensor的话，关于输出的feature map,shape为  #[batch_size,512,14,14]左右
        feature_maps_L28_channelwise_mean=torch.mean(feature_maps_L28,dim=(1,2))
        highlight_chanelwise_L28 = (feature_maps_L28 > feature_maps_L28_channelwise_mean.view(-1, 1, 1)).float()
        feature_maps_L28_sum=torch.sum(feature_maps_L28, dim=0)
        highlight_L28 = (feature_maps_L28_sum > torch.mean(feature_maps_L28_sum)).float()
        h28 = highlight_chanelwise_L28 * highlight_L28.unsqueeze(0)
        chanelweight_l28=torch.sum(h28,dim=(1,2))
        chanelweight_l28=chanelweight_l28/torch.sum(highlight_L28)
        chanelweight_l28=(chanelweight_l28-torch.min(chanelweight_l28))/(torch.max(chanelweight_l28)-torch.min(chanelweight_l28))
        # chanelweight_l28=(chanelweight_l28>torch.mean(chanelweight_l28)).float()
        chanelweight_l28_bool = (chanelweight_l28 > torch.mean(chanelweight_l28)).float()
        chanelweight_l28_bool_2 = torch.ones_like(chanelweight_l28_bool) - chanelweight_l28_bool
        chanelweight_l28 = chanelweight_l28_bool_2 * chanelweight_l28
        chanelweight_l28 = chanelweight_l28 + chanelweight_l28_bool



        feature_maps_L31 = features[
            1][0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        feature_maps_L31_channelwise_mean=torch.mean(feature_maps_L31,dim=(1,2))
        highlight_chanelwise_L31 = (feature_maps_L31 > feature_maps_L31_channelwise_mean.view(-1, 1, 1)).float()#得到的结果依然是【512,7,7】的一个二值矩阵
        feature_maps_L31_sum=torch.sum(feature_maps_L31, dim=0)
        highlight_L31 = (feature_maps_L31_sum > torch.mean(feature_maps_L31_sum)).float()
        h31 = highlight_chanelwise_L31 * highlight_L31.unsqueeze(0)
        chanelweight_l31=torch.sum(h31,dim=(1,2))
        chanelweight_l31=chanelweight_l31/torch.sum(highlight_L31)
        chanelweight_l31=(chanelweight_l31-torch.min(chanelweight_l31))/(torch.max(chanelweight_l31)-torch.min(chanelweight_l31))

        chanelweight_l31_bool=(chanelweight_l31>torch.mean(chanelweight_l31)).float()
        chanelweight_l31_bool_2=torch.ones_like(chanelweight_l31_bool)-chanelweight_l31_bool
        chanelweight_l31=chanelweight_l31_bool_2 * chanelweight_l31
        chanelweight_l31=chanelweight_l31 + chanelweight_l31_bool

        ###************

        output_max=torch.max(output)
        output_min=torch.min(output)
        output_norm=torch.div((output-output_min).float(),(output_max-output_min))
        ################################################################3
        output_norm=(output_norm * -1).cpu().tolist()
        output_norm=torch.tensor(output_norm).float().cuda()
        ###################################################################
        one_hot = torch.sum(output * output_norm)
        self.net.zero_grad()
        one_hot.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #
        output_mean=[]
        output_mean_direction2=[]

        output_mean_grad_only=[]
        output_mean_direction2_grad_only=[]
        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['features.26.weight','features.28.weight']:
                # tmp=(param * param.grad).cpu().data
                tmp=(param).cpu().data
                tmp_mean_reuse=torch.mean(tmp,axis=( 2, 3))
                tmp_mean=torch.mean(tmp_mean_reuse,axis=1).numpy()
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean.append(tmp_mean)
#############################################################3
                #zhelizheli
                if name=='features.26.weight':
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l28.cpu().unsqueeze(0).T
                else:
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l31.cpu().unsqueeze(0).T
                tmp_mean = torch.mean(tmp_mean_reuse_direction2, axis=0).numpy()#54.**
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean_direction2.append(tmp_mean)
########***********************************///////////////////****************+++++++++++
                tmp = (param.grad).cpu().data
                tmp_mean_reuse = torch.mean(tmp, axis=(2, 3))
                tmp_mean = torch.mean(tmp_mean_reuse, axis=1).numpy()
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean_grad_only.append(tmp_mean)
                #############################################################3
                if name == 'features.26.weight':
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l28.cpu().unsqueeze(0).T
                else:
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l31.cpu().unsqueeze(0).T
                tmp_mean = torch.mean(tmp_mean_reuse_direction2, axis=0).numpy()
                # tmp_mean = torch.mean(tmp_mean_reuse, axis=0).numpy()#58.96

                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean_direction2_grad_only.append(tmp_mean)

        return  output_mean,output_mean_direction2,output_mean_grad_only,output_mean_direction2_grad_only









class GradCam_yuhan_kernel_version15():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, index=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别
        features,output = self.net(input.cuda())
        ###************
        feature_maps_L28 = features[
            0][0]  # 如果输入图像的shape为[batch_size,3,224,224]的tensor的话，关于输出的feature map,shape为  #[batch_size,512,14,14]左右
        feature_maps_L28_channelwise_mean=torch.mean(feature_maps_L28,dim=(1,2))
        highlight_chanelwise_L28 = (feature_maps_L28 > feature_maps_L28_channelwise_mean.view(-1, 1, 1)).float()
        feature_maps_L28_sum=torch.sum(feature_maps_L28, dim=0)
        highlight_L28 = (feature_maps_L28_sum > torch.mean(feature_maps_L28_sum)).float()
        h28 = highlight_chanelwise_L28 * highlight_L28.unsqueeze(0)
        feature_maps_L28=feature_maps_L28 * h28
        chanelweight_l28 = torch.nn.functional.adaptive_max_pool2d(feature_maps_L28, output_size=1).squeeze(-1).squeeze(-1)
        # chanelweight_l28=torch.sum(h28,dim=(1,2))
        # chanelweight_l28=chanelweight_l28/torch.sum(highlight_L28)
        chanelweight_l28=(chanelweight_l28-torch.min(chanelweight_l28))/(torch.max(chanelweight_l28)-torch.min(chanelweight_l28))
        # # chanelweight_l28=(chanelweight_l28>torch.mean(chanelweight_l28)).float()
        # chanelweight_l28_bool = (chanelweight_l28 > torch.mean(chanelweight_l28)).float()
        # chanelweight_l28_bool_2 = torch.ones_like(chanelweight_l28_bool) - chanelweight_l28_bool
        # chanelweight_l28 = chanelweight_l28_bool_2 * chanelweight_l28
        # chanelweight_l28 = chanelweight_l28 + chanelweight_l28_bool



        feature_maps_L31 = features[
            1][0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        feature_maps_L31_channelwise_mean=torch.mean(feature_maps_L31,dim=(1,2))
        highlight_chanelwise_L31 = (feature_maps_L31 > feature_maps_L31_channelwise_mean.view(-1, 1, 1)).float()#得到的结果依然是【512,7,7】的一个二值矩阵
        feature_maps_L31_sum=torch.sum(feature_maps_L31, dim=0)
        highlight_L31 = (feature_maps_L31_sum > torch.mean(feature_maps_L31_sum)).float()
        h31 = highlight_chanelwise_L31 * highlight_L31.unsqueeze(0)
        feature_maps_L31 = feature_maps_L31 * h31
        chanelweight_l31 = torch.nn.functional.adaptive_max_pool2d(feature_maps_L31, output_size=1).squeeze(-1).squeeze(
            -1)
        # chanelweight_l31=torch.sum(h31,dim=(1,2))
        # chanelweight_l31=chanelweight_l31/torch.sum(highlight_L31)
        chanelweight_l31=(chanelweight_l31-torch.min(chanelweight_l31))/(torch.max(chanelweight_l31)-torch.min(chanelweight_l31))
        #
        # chanelweight_l31_bool=(chanelweight_l31>torch.mean(chanelweight_l31)).float()
        # chanelweight_l31_bool_2=torch.ones_like(chanelweight_l31_bool)-chanelweight_l31_bool
        # chanelweight_l31=chanelweight_l31_bool_2 * chanelweight_l31
        # chanelweight_l31=chanelweight_l31 + chanelweight_l31_bool

        ###************

        output_max=torch.max(output)
        output_min=torch.min(output)
        output_norm=torch.div((output-output_min).float(),(output_max-output_min))
        ################################################################3
        output_norm=(output_norm * -1).cpu().tolist()
        output_norm=torch.tensor(output_norm).float().cuda()
        ###################################################################
        one_hot = torch.sum(output * output_norm)
        self.net.zero_grad()
        one_hot.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #
        output_mean=[]
        output_mean_direction2=[]

        output_mean_grad_only=[]
        output_mean_direction2_grad_only=[]
        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['features.26.weight','features.28.weight']:
                # tmp=(param * param.grad).cpu().data
                tmp=(param).cpu().data
                tmp_mean_reuse=torch.mean(tmp,axis=( 2, 3))
                tmp_mean=torch.mean(tmp_mean_reuse,axis=1).numpy()
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean.append(tmp_mean)
#############################################################3
                #zhelizheli
                if name=='features.26.weight':
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l28.cpu().unsqueeze(0).T
                else:
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l31.cpu().unsqueeze(0).T
                tmp_mean = torch.mean(tmp_mean_reuse_direction2, axis=0).numpy()#54.**
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean_direction2.append(tmp_mean)
########***********************************///////////////////****************+++++++++++
                tmp = (param.grad).cpu().data
                tmp_mean_reuse = torch.mean(tmp, axis=(2, 3))
                tmp_mean = torch.mean(tmp_mean_reuse, axis=1).numpy()
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean_grad_only.append(tmp_mean)
                #############################################################3
                if name == 'features.26.weight':
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l28.cpu().unsqueeze(0).T
                else:
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l31.cpu().unsqueeze(0).T
                tmp_mean = torch.mean(tmp_mean_reuse_direction2, axis=0).numpy()
                # tmp_mean = torch.mean(tmp_mean_reuse, axis=0).numpy()#58.96

                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean_direction2_grad_only.append(tmp_mean)

        return  output_mean,output_mean_direction2,output_mean_grad_only,output_mean_direction2_grad_only




class GradCam_yuhan_kernel_version16():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, index=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别
        features,output = self.net(input.cuda())
        ###************
        feature_maps_L28 = features[
            0][0]  # 如果输入图像的shape为[batch_size,3,224,224]的tensor的话，关于输出的feature map,shape为  #[batch_size,512,14,14]左右
        # print(feature_maps_L28.shape)
        feature_maps_L28_channelwise_mean=torch.mean(feature_maps_L28,dim=(1,2))
        # print(feature_maps_L28_channelwise_mean.shape)
        highlight_chanelwise_L28 = (feature_maps_L28 > feature_maps_L28_channelwise_mean.view(-1, 1, 1)).float()
        # print(highlight_chanelwise_L28.shape)#为这512通道的每一个通道生成了一个（由该通道均值决定的）专属于该通道的二值掩码矩阵，将该通道的显著性区域标注了出来
        feature_maps_L28_sum=torch.sum(feature_maps_L28, dim=0)
        # print(feature_maps_L28_sum.shape)
        highlight_L28 = (feature_maps_L28_sum > torch.mean(feature_maps_L28_sum)).float()
        # print(highlight_L28.shape)
        h28 = highlight_chanelwise_L28 * highlight_L28.unsqueeze(0) #h28是一个【512，h,w】的二值矩阵，这是与总体二值矩阵交集得到的，可能在某一个通道上是一个全零矩阵
        # （一个完全对背景噪声响应的通道）
        # print(h28.shape)
        feature_maps_L28=feature_maps_L28 * h28  #同为shape为【512,7,7】的feature_map 与通道级二值矩阵（总体显著性区域与通道级显著性区域的交集）进行点乘，
        # chanelweight_l28 = torch.nn.functional.adaptive_max_pool2d(feature_maps_L28, output_size=1).squeeze(-1).squeeze(-1)
        chanelweight_l28=torch.sum(feature_maps_L28,dim=(1,2))/torch.sum(h28,dim=(1,2))
        # print(chanelweight_l28)
        # print(chanelweight_l28.shape)
        #我们用0代替了nan值
        chanelweight_l28 = torch.where(torch.isnan(chanelweight_l28), torch.full_like(chanelweight_l28, 0), chanelweight_l28)
        # chanelweight_l28=chanelweight_l28/torch.sum(highlight_L28)
        # print(chanelweight_l28)
        chanelweight_l28=(chanelweight_l28-torch.min(chanelweight_l28))/(torch.max(chanelweight_l28)-torch.min(chanelweight_l28))
        # # chanelweight_l28=(chanelweight_l28>torch.mean(chanelweight_l28)).float()
        # chanelweight_l28_bool = (chanelweight_l28 > torch.mean(chanelweight_l28)).float()
        # chanelweight_l28_bool_2 = torch.ones_like(chanelweight_l28_bool) - chanelweight_l28_bool
        # chanelweight_l28 = chanelweight_l28_bool_2 * chanelweight_l28
        # chanelweight_l28 = chanelweight_l28 + chanelweight_l28_bool



        feature_maps_L31 = features[
            1][0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        feature_maps_L31_channelwise_mean=torch.mean(feature_maps_L31,dim=(1,2))
        highlight_chanelwise_L31 = (feature_maps_L31 > feature_maps_L31_channelwise_mean.view(-1, 1, 1)).float()#得到的结果依然是【512,7,7】的一个二值矩阵
        feature_maps_L31_sum=torch.sum(feature_maps_L31, dim=0)
        highlight_L31 = (feature_maps_L31_sum > torch.mean(feature_maps_L31_sum)).float()
        h31 = highlight_chanelwise_L31 * highlight_L31.unsqueeze(0)
        feature_maps_L31 = feature_maps_L31 * h31
        # chanelweight_l31 = torch.nn.functional.adaptive_max_pool2d(feature_maps_L31, output_size=1).squeeze(-1).squeeze(
        #     -1)
        chanelweight_l31=torch.sum(feature_maps_L31,dim=(1,2))/torch.sum( h31 ,dim=(1,2))
        chanelweight_l31 = torch.where(torch.isnan(chanelweight_l31), torch.full_like(chanelweight_l31, 0), chanelweight_l31)

        # chanelweight_l31=torch.sum(h31,dim=(1,2))
        # chanelweight_l31=chanelweight_l31/torch.sum(highlight_L31)
        chanelweight_l31=(chanelweight_l31-torch.min(chanelweight_l31))/(torch.max(chanelweight_l31)-torch.min(chanelweight_l31))

        #
        # chanelweight_l31_bool=(chanelweight_l31>torch.mean(chanelweight_l31)).float()
        # chanelweight_l31_bool_2=torch.ones_like(chanelweight_l31_bool)-chanelweight_l31_bool
        # chanelweight_l31=chanelweight_l31_bool_2 * chanelweight_l31
        # chanelweight_l31=chanelweight_l31 + chanelweight_l31_bool

        ###************

        output_max=torch.max(output)
        output_min=torch.min(output)
        output_norm=torch.div((output-output_min).float(),(output_max-output_min))
        ################################################################3
        output_norm=(output_norm * -1).cpu().tolist()
        output_norm=torch.tensor(output_norm).float().cuda()
        ###################################################################
        one_hot = torch.sum(output * output_norm)
        self.net.zero_grad()
        one_hot.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #
        output_mean=[]
        output_mean_direction2=[]

        output_mean_grad_only=[]
        output_mean_direction2_grad_only=[]
        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['features.26.weight','features.28.weight']:
                # tmp=(param * param.grad).cpu().data
                tmp=(param).cpu().data
                print(tmp.shape)#[512,512,3,3]
                tmp_mean_reuse=torch.mean(tmp,axis=( 2, 3))
                print(tmp_mean_reuse.shape)#[512,512],
                # 这个矩阵的每一行代表某一层的一组卷积核，这一行是由这一层的【512,3,3】得到的，一个512维的向量
                #这个矩阵一共512行，代表512组的卷积核
                tmp_mean=torch.mean(tmp_mean_reuse,axis=1).numpy()
                #这是将模型从右往左的压成一个向量，而不是从上往下对所有的512组卷积核取平均；
                #再加上直接对测试阶段恒定不变的网络参数进行平均（未借助辅助信息进行加权），自然没用了
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean.append(tmp_mean)
#############################################################3
                #zhelizheli
                if name=='features.26.weight':
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l28.cpu().unsqueeze(0).T
                    #【512,512】的矩阵的每一行乘以一个固定的权重，也就是一组卷积核乘以一组固定的权重；这个权重是如何得到的呢?是这样的，该层的一组卷积核一组决定了该层输出特征图的某一个通道
                    #如果能够通过某种方式得到该通道在特征图整体的512个通道中的重要性的话，也就是说，我们得到了该通道的权重，也就是该组卷积核的权重，也就是tmp_mean_reuse 这个矩阵的某一行的权重。
                    #那么这个权重是如何得到的呢？这样的，scda不是得到了一个【h，w】的聚合特征图掩码矩阵嘛；其实我们完全可以得到一个通道级的而非整体的聚合特征图掩码矩阵，也就是一个【512，h，w】
                    #的掩码矩阵，其实就是生成一个通道级均值，特征图的每一个通道，高于该均值的位置掩码为1，否则为0；这就得到了【512，h,w】的掩码矩阵
                    #接下来呢，就是通道级掩码矩阵的每一个通道与整体的[h,w]的掩码进行按位与；最终得到了一个【512，h，w】的处理之后的掩码矩阵；
                    #这就决定了通道级掩码矩阵的每一个通道的每一个为1的位置，都是与图像的主要目标而非背景有关的；我们可以据此进一步的获知输出特征图的每一个通道激活的都是目标还是背景，也就是该通道
                    #进而就是该通道对应的该组滤波器的重要程度
                    #接下来就是最后的一步了，我们将处理之后的【512，h,w】的通道级掩码矩阵与输出特征图进行点乘
                    #进而对于处理后的特征图的每一个通道的不为零的元素们，假设该通道共N个非零值，对这N个值求和在除以N；进而得到了该通道的一个权重
                    #如果某通道所有值都为0的话，就单纯的将该通道的权重设置为0便好啦
                    #最终得到了512个权重（当然啦，还得进行一步0~1的归一化啦），它既是每一个通道的权重，也是每一组卷积核的权重啦
                    #当然这最后一步还有很多可以尝试的空间啦
                else:
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l31.cpu().unsqueeze(0).T
                tmp_mean = torch.mean(tmp_mean_reuse_direction2, axis=0).numpy()#54.**
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean_direction2.append(tmp_mean)
########***********************************///////////////////****************+++++++++++
                tmp = (param.grad).cpu().data  #【512,512,h,w】
                tmp_mean_reuse = torch.mean(tmp, axis=(2, 3))#[512,512]
                tmp_mean = torch.mean(tmp_mean_reuse, axis=1).numpy()#[512,],左右横向的
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean_grad_only.append(tmp_mean)
                #############################################################3
                if name == 'features.26.weight':
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l28.cpu().unsqueeze(0).T
                else:
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l31.cpu().unsqueeze(0).T
                tmp_mean = torch.mean(tmp_mean_reuse_direction2, axis=0).numpy()
                # tmp_mean = torch.mean(tmp_mean_reuse, axis=0).numpy()#58.96

                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean_direction2_grad_only.append(tmp_mean)

        return  output_mean,output_mean_direction2,output_mean_grad_only,output_mean_direction2_grad_only





class GradCam_yuhan_kernel_version16_2():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, index=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别
        features,output = self.net(input.cuda())
        ###************
        feature_maps_L28 = features[
            0][0]  # 如果输入图像的shape为[batch_size,3,224,224]的tensor的话，关于输出的feature map,shape为  #[batch_size,512,14,14]左右
        # print(feature_maps_L28.shape)
        feature_maps_L28_channelwise_mean=torch.mean(feature_maps_L28,dim=(1,2))
        # print(feature_maps_L28_channelwise_mean.shape)
        highlight_chanelwise_L28 = (feature_maps_L28 > feature_maps_L28_channelwise_mean.view(-1, 1, 1)).float()
        # print(highlight_chanelwise_L28.shape)#为这512通道的每一个通道生成了一个（由该通道均值决定的）专属于该通道的二值掩码矩阵，将该通道的显著性区域标注了出来
        feature_maps_L28_sum=torch.sum(feature_maps_L28, dim=0)
        # print(feature_maps_L28_sum.shape)
        highlight_L28 = (feature_maps_L28_sum > torch.mean(feature_maps_L28_sum)).float()
        # print(highlight_L28.shape)
        h28 = highlight_chanelwise_L28 * highlight_L28.unsqueeze(0) #h28是一个【512，h,w】的二值矩阵，这是与总体二值矩阵交集得到的，可能在某一个通道上是一个全零矩阵
        # （一个完全对背景噪声响应的通道）
        # print(h28.shape)
        feature_maps_L28=feature_maps_L28 * h28  #同为shape为【512,7,7】的feature_map 与通道级二值矩阵（总体显著性区域与通道级显著性区域的交集）进行点乘，
        # chanelweight_l28 = torch.nn.functional.adaptive_max_pool2d(feature_maps_L28, output_size=1).squeeze(-1).squeeze(-1)
        chanelweight_l28=torch.sum(feature_maps_L28,dim=(1,2))/torch.sum(h28,dim=(1,2))
        # print(chanelweight_l28)
        # print(chanelweight_l28.shape)
        #我们用0代替了nan值
        chanelweight_l28 = torch.where(torch.isnan(chanelweight_l28), torch.full_like(chanelweight_l28, 0), chanelweight_l28)
        # chanelweight_l28=chanelweight_l28/torch.sum(highlight_L28)
        # print(chanelweight_l28)
        chanelweight_l28=(chanelweight_l28-torch.min(chanelweight_l28))/(torch.max(chanelweight_l28)-torch.min(chanelweight_l28))
        # # chanelweight_l28=(chanelweight_l28>torch.mean(chanelweight_l28)).float()
        # chanelweight_l28_bool = (chanelweight_l28 > torch.mean(chanelweight_l28)).float()
        # chanelweight_l28_bool_2 = torch.ones_like(chanelweight_l28_bool) - chanelweight_l28_bool
        # chanelweight_l28 = chanelweight_l28_bool_2 * chanelweight_l28
        # chanelweight_l28 = chanelweight_l28 + chanelweight_l28_bool



        feature_maps_L31 = features[
            1][0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        feature_maps_L31_channelwise_mean=torch.mean(feature_maps_L31,dim=(1,2))
        highlight_chanelwise_L31 = (feature_maps_L31 > feature_maps_L31_channelwise_mean.view(-1, 1, 1)).float()#得到的结果依然是【512,7,7】的一个二值矩阵
        feature_maps_L31_sum=torch.sum(feature_maps_L31, dim=0)
        highlight_L31 = (feature_maps_L31_sum > torch.mean(feature_maps_L31_sum)).float()
        h31 = highlight_chanelwise_L31 * highlight_L31.unsqueeze(0)
        feature_maps_L31 = feature_maps_L31 * h31
        # chanelweight_l31 = torch.nn.functional.adaptive_max_pool2d(feature_maps_L31, output_size=1).squeeze(-1).squeeze(
        #     -1)
        chanelweight_l31=torch.sum(feature_maps_L31,dim=(1,2))/torch.sum( h31 ,dim=(1,2))
        chanelweight_l31 = torch.where(torch.isnan(chanelweight_l31), torch.full_like(chanelweight_l31, 0), chanelweight_l31)

        # chanelweight_l31=torch.sum(h31,dim=(1,2))
        # chanelweight_l31=chanelweight_l31/torch.sum(highlight_L31)
        chanelweight_l31=(chanelweight_l31-torch.min(chanelweight_l31))/(torch.max(chanelweight_l31)-torch.min(chanelweight_l31))

        #
        # chanelweight_l31_bool=(chanelweight_l31>torch.mean(chanelweight_l31)).float()
        # chanelweight_l31_bool_2=torch.ones_like(chanelweight_l31_bool)-chanelweight_l31_bool
        # chanelweight_l31=chanelweight_l31_bool_2 * chanelweight_l31
        # chanelweight_l31=chanelweight_l31 + chanelweight_l31_bool

        ###************

        output_max=torch.max(output)
        output_min=torch.min(output)
        output_norm=torch.div((output-output_min).float(),(output_max-output_min))
        ################################################################3
        output_norm=(output_norm * -1).cpu().tolist()
        output_norm=torch.tensor(output_norm).float().cuda()
        ###################################################################
        one_hot = torch.sum(output * output_norm)
        self.net.zero_grad()
        one_hot.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #
        output_mean=[]
        output_mean_direction2=[]

        output_mean_grad_only=[]
        output_mean_direction2_grad_only=[]
        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['features.26.weight','features.28.weight']:
                # tmp=(param * param.grad).cpu().data
                tmp=(param).cpu().data
                # print(tmp.shape)#[512,512,3,3]
                tmp_mean_reuse=torch.mean(tmp,axis=( 2, 3))
                # print(tmp_mean_reuse.shape)#[512,512],
                # 这个矩阵的每一行代表某一层的一组卷积核，这一行是由这一层的【512,3,3】得到的，一个512维的向量
                #这个矩阵一共512行，代表512组的卷积核

                # tmp_mean=torch.mean(tmp_mean_reuse,axis=1).numpy()
                # #这是将模型从右往左的压成一个向量，而不是从上往下对所有的512组卷积核取平均；
                # #再加上直接对测试阶段恒定不变的网络参数进行平均（未借助辅助信息进行加权），自然没用了
                # aa = np.linalg.norm(tmp_mean)
                # if aa != 0:
                #     tmp_mean = tmp_mean / aa
                # else:
                #     tmp_mean = np.zeros_like(tmp_mean)
                #
                # output_mean.append(tmp_mean)
#############################################################3
                #zhelizheli
                if name=='features.26.weight':
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l28.cpu().unsqueeze(0).T
                    #【512,512】的矩阵的每一行乘以一个固定的权重，也就是一组卷积核乘以一组固定的权重；这个权重是如何得到的呢?是这样的，该层的一组卷积核一组决定了该层输出特征图的某一个通道
                    #如果能够通过某种方式得到该通道在特征图整体的512个通道中的重要性的话，也就是说，我们得到了该通道的权重，也就是该组卷积核的权重，也就是tmp_mean_reuse 这个矩阵的某一行的权重。
                    #那么这个权重是如何得到的呢？这样的，scda不是得到了一个【h，w】的聚合特征图掩码矩阵嘛；其实我们完全可以得到一个通道级的而非整体的聚合特征图掩码矩阵，也就是一个【512，h，w】
                    #的掩码矩阵，其实就是生成一个通道级均值，特征图的每一个通道，高于该均值的位置掩码为1，否则为0；这就得到了【512，h,w】的掩码矩阵
                    #接下来呢，就是通道级掩码矩阵的每一个通道与整体的[h,w]的掩码进行按位与；最终得到了一个【512，h，w】的处理之后的掩码矩阵；
                    #这就决定了通道级掩码矩阵的每一个通道的每一个为1的位置，都是与图像的主要目标而非背景有关的；我们可以据此进一步的获知输出特征图的每一个通道激活的都是目标还是背景，也就是该通道
                    #进而就是该通道对应的该组滤波器的重要程度
                    #接下来就是最后的一步了，我们将处理之后的【512，h,w】的通道级掩码矩阵与输出特征图进行点乘
                    #进而对于处理后的特征图的每一个通道的不为零的元素们，假设该通道共N个非零值，对这N个值求和在除以N；进而得到了该通道的一个权重
                    #如果某通道所有值都为0的话，就单纯的将该通道的权重设置为0便好啦
                    #最终得到了512个权重（当然啦，还得进行一步0~1的归一化啦），它既是每一个通道的权重，也是每一组卷积核的权重啦
                    #当然这最后一步还有很多可以尝试的空间啦
                else:
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l31.cpu().unsqueeze(0).T
                tmp_mean = torch.mean(tmp_mean_reuse_direction2, axis=0).numpy()#54.**
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean_direction2.append(tmp_mean)

                tmp_mean = torch.mean(tmp_mean_reuse_direction2, axis=1).numpy()
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean.append(tmp_mean)
########***********************************///////////////////****************+++++++++++
                tmp = (param.grad).cpu().data  #【512,512,h,w】
                tmp_mean_reuse = torch.mean(tmp, axis=(2, 3))#[512,512]
                tmp_mean = torch.mean(tmp_mean_reuse, axis=1).numpy()#[512,],左右横向的
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean_grad_only.append(tmp_mean)
                #############################################################3
                if name == 'features.26.weight':
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l28.cpu().unsqueeze(0).T
                else:
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l31.cpu().unsqueeze(0).T
                tmp_mean = torch.mean(tmp_mean_reuse_direction2, axis=0).numpy()
                # tmp_mean = torch.mean(tmp_mean_reuse, axis=0).numpy()#58.96

                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean_direction2_grad_only.append(tmp_mean)

        return  output_mean,output_mean_direction2,output_mean_grad_only,output_mean_direction2_grad_only





class GradCam_yuhan_kernel_version16_2_0():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, index=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别
        features,output = self.net(input.cuda())
        ###************
        feature_maps_L28 = features[
            0][0]  # 如果输入图像的shape为[batch_size,3,224,224]的tensor的话，关于输出的feature map,shape为  #[batch_size,512,14,14]左右
        # print(feature_maps_L28.shape)
        # print(feature_maps_L28)

        feature_maps_L28_channelwise_mean=torch.mean(feature_maps_L28,dim=(1,2))
        # print(feature_maps_L28_channelwise_mean.shape)
        highlight_chanelwise_L28 = (feature_maps_L28 > feature_maps_L28_channelwise_mean.view(-1, 1, 1)).float()
        # print(highlight_chanelwise_L28.shape)#为这512通道的每一个通道生成了一个（由该通道均值决定的）专属于该通道的二值掩码矩阵，将该通道的显著性区域标注了出来
        feature_maps_L28_sum=torch.sum(feature_maps_L28, dim=0)
        # print(feature_maps_L28_sum.shape)
        highlight_L28 = (feature_maps_L28_sum > torch.mean(feature_maps_L28_sum)).float()
        # print(highlight_L28.shape)
        h28 = highlight_chanelwise_L28 * highlight_L28.unsqueeze(0) #h28是一个【512，h,w】的二值矩阵，这是与总体二值矩阵交集得到的，可能在某一个通道上是一个全零矩阵
        # （一个完全对背景噪声响应的通道）
        # print(h28.shape)
        feature_maps_L28=feature_maps_L28 * h28  #同为shape为【512,7,7】的feature_map 与通道级二值矩阵（总体显著性区域与通道级显著性区域的交集）进行点乘，
        # chanelweight_l28 = torch.nn.functional.adaptive_max_pool2d(feature_maps_L28, output_size=1).squeeze(-1).squeeze(-1)
        chanelweight_l28=torch.sum(feature_maps_L28,dim=(1,2))/torch.sum(h28,dim=(1,2))
        # print(chanelweight_l28)
        # print(chanelweight_l28.shape)
        #我们用0代替了nan值
        chanelweight_l28 = torch.where(torch.isnan(chanelweight_l28), torch.full_like(chanelweight_l28, 0), chanelweight_l28)
        # chanelweight_l28=chanelweight_l28/torch.sum(highlight_L28)
        # print(chanelweight_l28)
        chanelweight_l28=(chanelweight_l28-torch.min(chanelweight_l28))/(torch.max(chanelweight_l28)-torch.min(chanelweight_l28))
        # # chanelweight_l28=(chanelweight_l28>torch.mean(chanelweight_l28)).float()
        # chanelweight_l28_bool = (chanelweight_l28 > torch.mean(chanelweight_l28)).float()
        # chanelweight_l28_bool_2 = torch.ones_like(chanelweight_l28_bool) - chanelweight_l28_bool
        # chanelweight_l28 = chanelweight_l28_bool_2 * chanelweight_l28
        # chanelweight_l28 = chanelweight_l28 + chanelweight_l28_bool



        feature_maps_L31 = features[
            1][0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        feature_maps_L31_channelwise_mean=torch.mean(feature_maps_L31,dim=(1,2))
        highlight_chanelwise_L31 = (feature_maps_L31 > feature_maps_L31_channelwise_mean.view(-1, 1, 1)).float()#得到的结果依然是【512,7,7】的一个二值矩阵
        feature_maps_L31_sum=torch.sum(feature_maps_L31, dim=0)
        highlight_L31 = (feature_maps_L31_sum > torch.mean(feature_maps_L31_sum)).float()
        h31 = highlight_chanelwise_L31 * highlight_L31.unsqueeze(0)
        feature_maps_L31 = feature_maps_L31 * h31
        # chanelweight_l31 = torch.nn.functional.adaptive_max_pool2d(feature_maps_L31, output_size=1).squeeze(-1).squeeze(
        #     -1)
        chanelweight_l31=torch.sum(feature_maps_L31,dim=(1,2))/torch.sum( h31 ,dim=(1,2))
        chanelweight_l31 = torch.where(torch.isnan(chanelweight_l31), torch.full_like(chanelweight_l31, 0), chanelweight_l31)

        # chanelweight_l31=torch.sum(h31,dim=(1,2))
        # chanelweight_l31=chanelweight_l31/torch.sum(highlight_L31)
        chanelweight_l31=(chanelweight_l31-torch.min(chanelweight_l31))/(torch.max(chanelweight_l31)-torch.min(chanelweight_l31))

        #
        # chanelweight_l31_bool=(chanelweight_l31>torch.mean(chanelweight_l31)).float()
        # chanelweight_l31_bool_2=torch.ones_like(chanelweight_l31_bool)-chanelweight_l31_bool
        # chanelweight_l31=chanelweight_l31_bool_2 * chanelweight_l31
        # chanelweight_l31=chanelweight_l31 + chanelweight_l31_bool

        ###************
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #
        output_mean=[]
        output_mean_bias=[]


        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['features.26.weight', 'features.28.weight']:
                # tmp=(param * param.grad).cpu().data
                tmp = (param).cpu().data
                # print(tmp.shape)#[512,512,3,3]
                tmp_mean_reuse = torch.mean(tmp, axis=(2, 3))
                # print(tmp_mean_reuse.shape)#[512,512],
                # 这个矩阵的每一行代表某一层的一组卷积核，这一行是由这一层的【512,3,3】得到的，一个512维的向量
                # 这个矩阵一共512行，代表512组的卷积核

                # tmp_mean=torch.mean(tmp_mean_reuse,axis=1).numpy()
                # #这是将模型从右往左的压成一个向量，而不是从上往下对所有的512组卷积核取平均；
                # #再加上直接对测试阶段恒定不变的网络参数进行平均（未借助辅助信息进行加权），自然没用了
                # aa = np.linalg.norm(tmp_mean)
                # if aa != 0:
                #     tmp_mean = tmp_mean / aa
                # else:
                #     tmp_mean = np.zeros_like(tmp_mean)
                #
                # output_mean.append(tmp_mean)
                #############################################################3
                # zhelizheli
                if name == 'features.26.weight':
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l28.cpu().unsqueeze(0).T
                    # 【512,512】的矩阵的每一行乘以一个固定的权重，也就是一组卷积核乘以一组固定的权重；这个权重是如何得到的呢?是这样的，该层的一组卷积核一组决定了该层输出特征图的某一个通道
                    # 如果能够通过某种方式得到该通道在特征图整体的512个通道中的重要性的话，也就是说，我们得到了该通道的权重，也就是该组卷积核的权重，也就是tmp_mean_reuse 这个矩阵的某一行的权重。
                    # 那么这个权重是如何得到的呢？这样的，scda不是得到了一个【h，w】的聚合特征图掩码矩阵嘛；其实我们完全可以得到一个通道级的而非整体的聚合特征图掩码矩阵，也就是一个【512，h，w】
                    # 的掩码矩阵，其实就是生成一个通道级均值，特征图的每一个通道，高于该均值的位置掩码为1，否则为0；这就得到了【512，h,w】的掩码矩阵
                    # 接下来呢，就是通道级掩码矩阵的每一个通道与整体的[h,w]的掩码进行按位与；最终得到了一个【512，h，w】的处理之后的掩码矩阵；
                    # 这就决定了通道级掩码矩阵的每一个通道的每一个为1的位置，都是与图像的主要目标而非背景有关的；我们可以据此进一步的获知输出特征图的每一个通道激活的都是目标还是背景，也就是该通道
                    # 进而就是该通道对应的该组滤波器的重要程度
                    # 接下来就是最后的一步了，我们将处理之后的【512，h,w】的通道级掩码矩阵与输出特征图进行点乘
                    # 进而对于处理后的特征图的每一个通道的不为零的元素们，假设该通道共N个非零值，对这N个值求和在除以N；进而得到了该通道的一个权重
                    # 如果某通道所有值都为0的话，就单纯的将该通道的权重设置为0便好啦
                    # 最终得到了512个权重（当然啦，还得进行一步0~1的归一化啦），它既是每一个通道的权重，也是每一组卷积核的权重啦
                    # 当然这最后一步还有很多可以尝试的空间啦
                else:
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l31.cpu().unsqueeze(0).T

                tmp_mean = torch.mean(tmp_mean_reuse_direction2, axis=1).numpy()
                # aa = np.linalg.norm(tmp_mean)
                # if aa != 0:
                #     tmp_mean = tmp_mean / aa
                # else:
                #     tmp_mean = np.zeros_like(tmp_mean)
                output_mean.append(tmp_mean)

            if name in ['features.26.bias','features.28.bias']:
                # tmp=(param * param.grad).cpu().data
                tmp_bias=(param).cpu().data
                # print(tmp.shape)#[512,512,3,3]

                if name=='features.26.bias':
                    tmp_mean_reuse_direction2 = tmp_bias * chanelweight_l28.cpu()
                else:
                    tmp_mean_reuse_direction2 = tmp_bias * chanelweight_l31.cpu()


                tmp_mean = tmp_mean_reuse_direction2.numpy()
                # aa = np.linalg.norm(tmp_mean)
                # if aa != 0:
                #     tmp_mean = tmp_mean / aa
                # else:
                #     tmp_mean = np.zeros_like(tmp_mean)



                output_mean_bias.append(tmp_mean)
########***********************************///////////////////****************+++++++++++
        output_final=[]
        for i in range(len(output_mean)):
            tmp=output_mean[i]+output_mean_bias[i]
            aa = np.linalg.norm(tmp)
            if aa != 0:
                tmp = tmp / aa
            else:
                tmp = np.zeros_like(tmp)
            output_final.append(tmp)

        return  output_mean
#
# 层: features.0.weight torch.Size([64, 3, 3, 3])
# 层: features.0.bias torch.Size([64])
# 层: features.2.weight torch.Size([64, 64, 3, 3])
# 层: features.2.bias torch.Size([64])
# 层: features.5.weight torch.Size([128, 64, 3, 3])
# 层: features.5.bias torch.Size([128])
# 层: features.7.weight torch.Size([128, 128, 3, 3])
# 层: features.7.bias torch.Size([128])
# 层: features.10.weight torch.Size([256, 128, 3, 3])
# 层: features.10.bias torch.Size([256])
# 层: features.12.weight torch.Size([256, 256, 3, 3])
# 层: features.12.bias torch.Size([256])
# 层: features.14.weight torch.Size([256, 256, 3, 3])
# 层: features.14.bias torch.Size([256])
# 层: features.17.weight torch.Size([512, 256, 3, 3])
# 层: features.17.bias torch.Size([512])
# 层: features.19.weight torch.Size([512, 512, 3, 3])
# 层: features.19.bias torch.Size([512])
# 层: features.21.weight torch.Size([512, 512, 3, 3])
# 层: features.21.bias torch.Size([512])
# 层: features.24.weight torch.Size([512, 512, 3, 3])
# 层: features.24.bias torch.Size([512])
# 层: features.26.weight torch.Size([512, 512, 3, 3])
# 层: features.26.bias torch.Size([512])
# 层: features.28.weight torch.Size([512, 512, 3, 3])
# 层: features.28.bias torch.Size([512])
# 层: img_classifier.weight torch.Size([100, 512])
# 层: img_classifier.bias torch.Size([100])









class GradCam_yuhan_kernel_version16_2_1():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, index=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别
        features,output = self.net(input.cuda())
        ###************
        feature_maps_L28 = features[
            0][0]  # 如果输入图像的shape为[batch_size,3,224,224]的tensor的话，关于输出的feature map,shape为  #[batch_size,512,14,14]左右
        # print(feature_maps_L28.shape)
        # print(feature_maps_L28)

        feature_maps_L28_channelwise_mean=torch.mean(feature_maps_L28,dim=(1,2))
        # print(feature_maps_L28_channelwise_mean.shape)
        highlight_chanelwise_L28 = (feature_maps_L28 > feature_maps_L28_channelwise_mean.view(-1, 1, 1)).float()
        # print(highlight_chanelwise_L28.shape)#为这512通道的每一个通道生成了一个（由该通道均值决定的）专属于该通道的二值掩码矩阵，将该通道的显著性区域标注了出来
        feature_maps_L28_sum=torch.sum(feature_maps_L28, dim=0)
        # print(feature_maps_L28_sum.shape)
        highlight_L28 = (feature_maps_L28_sum > torch.mean(feature_maps_L28_sum)).float()
        # print(highlight_L28.shape)
        h28 = highlight_chanelwise_L28 * highlight_L28.unsqueeze(0) #h28是一个【512，h,w】的二值矩阵，这是与总体二值矩阵交集得到的，可能在某一个通道上是一个全零矩阵
        # （一个完全对背景噪声响应的通道）
        # print(h28.shape)
        feature_maps_L28=feature_maps_L28 * h28  #同为shape为【512,7,7】的feature_map 与通道级二值矩阵（总体显著性区域与通道级显著性区域的交集）进行点乘，
        # chanelweight_l28 = torch.nn.functional.adaptive_max_pool2d(feature_maps_L28, output_size=1).squeeze(-1).squeeze(-1)
        chanelweight_l28=torch.sum(feature_maps_L28,dim=(1,2))/torch.sum(h28,dim=(1,2))
        # print(chanelweight_l28)
        # print(chanelweight_l28.shape)
        #我们用0代替了nan值
        chanelweight_l28 = torch.where(torch.isnan(chanelweight_l28), torch.full_like(chanelweight_l28, 0), chanelweight_l28)
        # chanelweight_l28=chanelweight_l28/torch.sum(highlight_L28)
        # print(chanelweight_l28)
        chanelweight_l28=(chanelweight_l28-torch.min(chanelweight_l28))/(torch.max(chanelweight_l28)-torch.min(chanelweight_l28))
        # # chanelweight_l28=(chanelweight_l28>torch.mean(chanelweight_l28)).float()
        # chanelweight_l28_bool = (chanelweight_l28 > torch.mean(chanelweight_l28)).float()
        # chanelweight_l28_bool_2 = torch.ones_like(chanelweight_l28_bool) - chanelweight_l28_bool
        # chanelweight_l28 = chanelweight_l28_bool_2 * chanelweight_l28
        # chanelweight_l28 = chanelweight_l28 + chanelweight_l28_bool
        # print(chanelweight_l28)



        feature_maps_L31 = features[
            1][0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        feature_maps_L31_channelwise_mean=torch.mean(feature_maps_L31,dim=(1,2))
        highlight_chanelwise_L31 = (feature_maps_L31 > feature_maps_L31_channelwise_mean.view(-1, 1, 1)).float()#得到的结果依然是【512,7,7】的一个二值矩阵
        feature_maps_L31_sum=torch.sum(feature_maps_L31, dim=0)
        highlight_L31 = (feature_maps_L31_sum > torch.mean(feature_maps_L31_sum)).float()
        h31 = highlight_chanelwise_L31 * highlight_L31.unsqueeze(0)
        feature_maps_L31 = feature_maps_L31 * h31
        # chanelweight_l31 = torch.nn.functional.adaptive_max_pool2d(feature_maps_L31, output_size=1).squeeze(-1).squeeze(
        #     -1)
        chanelweight_l31=torch.sum(feature_maps_L31,dim=(1,2))/torch.sum( h31 ,dim=(1,2))
        chanelweight_l31 = torch.where(torch.isnan(chanelweight_l31), torch.full_like(chanelweight_l31, 0), chanelweight_l31)

        # chanelweight_l31=torch.sum(h31,dim=(1,2))
        # chanelweight_l31=chanelweight_l31/torch.sum(highlight_L31)
        chanelweight_l31=(chanelweight_l31-torch.min(chanelweight_l31))/(torch.max(chanelweight_l31)-torch.min(chanelweight_l31))

        #
        # chanelweight_l31_bool=(chanelweight_l31>torch.mean(chanelweight_l31)).float()
        # chanelweight_l31_bool_2=torch.ones_like(chanelweight_l31_bool)-chanelweight_l31_bool
        # chanelweight_l31=chanelweight_l31_bool_2 * chanelweight_l31
        # chanelweight_l31=chanelweight_l31 + chanelweight_l31_bool
        # print(chanelweight_l31)
        # print(torch.sum((chanelweight_l31/np.linalg.norm(chanelweight_l31.cpu().data.numpy()))*(chanelweight_l28/np.linalg.norm(chanelweight_l28.cpu().data.numpy()))))

        ###************
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #
        output_mean=[]


        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['features.26.weight','features.28.weight']:
                # tmp=(param * param.grad).cpu().data
                tmp=(param).cpu().data
                # print(tmp.shape)#[512,512,3,3]
                tmp_mean_reuse=torch.mean(tmp,axis=( 2, 3))
                # print(tmp_mean_reuse.shape)#[512,512],
                # 这个矩阵的每一行代表某一层的一组卷积核，这一行是由这一层的【512,3,3】得到的，一个512维的向量
                #这个矩阵一共512行，代表512组的卷积核

                # tmp_mean=torch.mean(tmp_mean_reuse,axis=1).numpy()
                # #这是将模型从右往左的压成一个向量，而不是从上往下对所有的512组卷积核取平均；
                # #再加上直接对测试阶段恒定不变的网络参数进行平均（未借助辅助信息进行加权），自然没用了
                # aa = np.linalg.norm(tmp_mean)
                # if aa != 0:
                #     tmp_mean = tmp_mean / aa
                # else:
                #     tmp_mean = np.zeros_like(tmp_mean)
                #
                # output_mean.append(tmp_mean)
#############################################################3
                #zhelizheli
                if name=='features.26.weight':
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l28.cpu().unsqueeze(0).T
                    #【512,512】的矩阵的每一行乘以一个固定的权重，也就是一组卷积核乘以一组固定的权重；这个权重是如何得到的呢?是这样的，该层的一组卷积核一组决定了该层输出特征图的某一个通道
                    #如果能够通过某种方式得到该通道在特征图整体的512个通道中的重要性的话，也就是说，我们得到了该通道的权重，也就是该组卷积核的权重，也就是tmp_mean_reuse 这个矩阵的某一行的权重。
                    #那么这个权重是如何得到的呢？这样的，scda不是得到了一个【h，w】的聚合特征图掩码矩阵嘛；其实我们完全可以得到一个通道级的而非整体的聚合特征图掩码矩阵，也就是一个【512，h，w】
                    #的掩码矩阵，其实就是生成一个通道级均值，特征图的每一个通道，高于该均值的位置掩码为1，否则为0；这就得到了【512，h,w】的掩码矩阵
                    #接下来呢，就是通道级掩码矩阵的每一个通道与整体的[h,w]的掩码进行按位与；最终得到了一个【512，h，w】的处理之后的掩码矩阵；
                    #这就决定了通道级掩码矩阵的每一个通道的每一个为1的位置，都是与图像的主要目标而非背景有关的；我们可以据此进一步的获知输出特征图的每一个通道激活的都是目标还是背景，也就是该通道
                    #进而就是该通道对应的该组滤波器的重要程度
                    #接下来就是最后的一步了，我们将处理之后的【512，h,w】的通道级掩码矩阵与输出特征图进行点乘
                    #进而对于处理后的特征图的每一个通道的不为零的元素们，假设该通道共N个非零值，对这N个值求和在除以N；进而得到了该通道的一个权重
                    #如果某通道所有值都为0的话，就单纯的将该通道的权重设置为0便好啦
                    #最终得到了512个权重（当然啦，还得进行一步0~1的归一化啦），它既是每一个通道的权重，也是每一组卷积核的权重啦
                    #当然这最后一步还有很多可以尝试的空间啦
                else:
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l31.cpu().unsqueeze(0).T


                tmp_mean = torch.mean(tmp_mean_reuse_direction2, axis=1).numpy()
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean.append(tmp_mean)
########***********************************///////////////////****************+++++++++++


        return  output_mean








class GradCam_yuhan_kernel_version16_2_17():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, random=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别
        features,output = self.net(input.cuda())
        ###************
        feature_maps_L28 = features[
            0][0]  # 如果输入图像的shape为[batch_size,3,224,224]的tensor的话，关于输出的feature map,shape为  #[batch_size,512,14,14]左右
        # print(feature_maps_L28.shape)
        # print(feature_maps_L28)

        feature_maps_L28_channelwise_mean=torch.mean(feature_maps_L28,dim=(1,2))
        # print(feature_maps_L28_channelwise_mean.shape)
        highlight_chanelwise_L28 = (feature_maps_L28 > feature_maps_L28_channelwise_mean.view(-1, 1, 1)).float()
        # print(highlight_chanelwise_L28.shape)#为这512通道的每一个通道生成了一个（由该通道均值决定的）专属于该通道的二值掩码矩阵，将该通道的显著性区域标注了出来
        feature_maps_L28_sum=torch.sum(feature_maps_L28, dim=0)
        # print(feature_maps_L28_sum.shape)
        highlight_L28 = (feature_maps_L28_sum > torch.mean(feature_maps_L28_sum)).float()
        # print(highlight_L28.shape)
        h28 = highlight_chanelwise_L28 * highlight_L28.unsqueeze(0) #h28是一个【512，h,w】的二值矩阵，这是与总体二值矩阵交集得到的，可能在某一个通道上是一个全零矩阵
        # （一个完全对背景噪声响应的通道）
        # print(h28.shape)
        feature_maps_L28=feature_maps_L28 * h28  #同为shape为【512,7,7】的feature_map 与通道级二值矩阵（总体显著性区域与通道级显著性区域的交集）进行点乘，
        # chanelweight_l28 = torch.nn.functional.adaptive_max_pool2d(feature_maps_L28, output_size=1).squeeze(-1).squeeze(-1)
        chanelweight_l28=torch.sum(feature_maps_L28,dim=(1,2))/torch.sum(h28,dim=(1,2))
        # print(chanelweight_l28)
        # print(chanelweight_l28.shape)
        #我们用0代替了nan值
        chanelweight_l28 = torch.where(torch.isnan(chanelweight_l28), torch.full_like(chanelweight_l28, 0), chanelweight_l28)
        # chanelweight_l28=chanelweight_l28/torch.sum(highlight_L28)
        # print(chanelweight_l28)
        chanelweight_l28=(chanelweight_l28-torch.min(chanelweight_l28))/(torch.max(chanelweight_l28)-torch.min(chanelweight_l28))
        # # chanelweight_l28=(chanelweight_l28>torch.mean(chanelweight_l28)).float()
        # chanelweight_l28_bool = (chanelweight_l28 > torch.mean(chanelweight_l28)).float()
        # chanelweight_l28_bool_2 = torch.ones_like(chanelweight_l28_bool) - chanelweight_l28_bool
        # chanelweight_l28 = chanelweight_l28_bool_2 * chanelweight_l28
        # chanelweight_l28 = chanelweight_l28 + chanelweight_l28_bool
        # print(chanelweight_l28)



        feature_maps_L31 = features[
            1][0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        feature_maps_L31_channelwise_mean=torch.mean(feature_maps_L31,dim=(1,2))
        highlight_chanelwise_L31 = (feature_maps_L31 > feature_maps_L31_channelwise_mean.view(-1, 1, 1)).float()#得到的结果依然是【512,7,7】的一个二值矩阵
        feature_maps_L31_sum=torch.sum(feature_maps_L31, dim=0)
        highlight_L31 = (feature_maps_L31_sum > torch.mean(feature_maps_L31_sum)).float()
        h31 = highlight_chanelwise_L31 * highlight_L31.unsqueeze(0)
        feature_maps_L31 = feature_maps_L31 * h31
        # chanelweight_l31 = torch.nn.functional.adaptive_max_pool2d(feature_maps_L31, output_size=1).squeeze(-1).squeeze(
        #     -1)
        chanelweight_l31=torch.sum(feature_maps_L31,dim=(1,2))/torch.sum( h31 ,dim=(1,2))
        chanelweight_l31 = torch.where(torch.isnan(chanelweight_l31), torch.full_like(chanelweight_l31, 0), chanelweight_l31)

        # chanelweight_l31=torch.sum(h31,dim=(1,2))
        # chanelweight_l31=chanelweight_l31/torch.sum(highlight_L31)
        chanelweight_l31=(chanelweight_l31-torch.min(chanelweight_l31))/(torch.max(chanelweight_l31)-torch.min(chanelweight_l31))

        #
        # chanelweight_l31_bool=(chanelweight_l31>torch.mean(chanelweight_l31)).float()
        # chanelweight_l31_bool_2=torch.ones_like(chanelweight_l31_bool)-chanelweight_l31_bool
        # chanelweight_l31=chanelweight_l31_bool_2 * chanelweight_l31
        # chanelweight_l31=chanelweight_l31 + chanelweight_l31_bool
        # print(chanelweight_l31)
        # print(torch.sum((chanelweight_l31/np.linalg.norm(chanelweight_l31.cpu().data.numpy()))*(chanelweight_l28/np.linalg.norm(chanelweight_l28.cpu().data.numpy()))))

        ###************
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #
        output_mean=[]


        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['features.26.weight','features.28.weight']:
                # tmp=(param * param.grad).cpu().data
                # tmp=(param).cpu().data
                # # print(tmp.shape)#[512,512,3,3]
                # tmp_mean_reuse=torch.mean(tmp,axis=( 2, 3))
                tmp_mean_reuse=random
                # print(tmp_mean_reuse.shape)#[512,512],
                # 这个矩阵的每一行代表某一层的一组卷积核，这一行是由这一层的【512,3,3】得到的，一个512维的向量
                #这个矩阵一共512行，代表512组的卷积核

                # tmp_mean=torch.mean(tmp_mean_reuse,axis=1).numpy()
                # #这是将模型从右往左的压成一个向量，而不是从上往下对所有的512组卷积核取平均；
                # #再加上直接对测试阶段恒定不变的网络参数进行平均（未借助辅助信息进行加权），自然没用了
                # aa = np.linalg.norm(tmp_mean)
                # if aa != 0:
                #     tmp_mean = tmp_mean / aa
                # else:
                #     tmp_mean = np.zeros_like(tmp_mean)
                #
                # output_mean.append(tmp_mean)
#############################################################3
                #zhelizheli
                if name=='features.26.weight':
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l28.cpu().unsqueeze(0)
                    #【512,512】的矩阵的每一行乘以一个固定的权重，也就是一组卷积核乘以一组固定的权重；这个权重是如何得到的呢?是这样的，该层的一组卷积核一组决定了该层输出特征图的某一个通道
                    #如果能够通过某种方式得到该通道在特征图整体的512个通道中的重要性的话，也就是说，我们得到了该通道的权重，也就是该组卷积核的权重，也就是tmp_mean_reuse 这个矩阵的某一行的权重。
                    #那么这个权重是如何得到的呢？这样的，scda不是得到了一个【h，w】的聚合特征图掩码矩阵嘛；其实我们完全可以得到一个通道级的而非整体的聚合特征图掩码矩阵，也就是一个【512，h，w】
                    #的掩码矩阵，其实就是生成一个通道级均值，特征图的每一个通道，高于该均值的位置掩码为1，否则为0；这就得到了【512，h,w】的掩码矩阵
                    #接下来呢，就是通道级掩码矩阵的每一个通道与整体的[h,w]的掩码进行按位与；最终得到了一个【512，h，w】的处理之后的掩码矩阵；
                    #这就决定了通道级掩码矩阵的每一个通道的每一个为1的位置，都是与图像的主要目标而非背景有关的；我们可以据此进一步的获知输出特征图的每一个通道激活的都是目标还是背景，也就是该通道
                    #进而就是该通道对应的该组滤波器的重要程度
                    #接下来就是最后的一步了，我们将处理之后的【512，h,w】的通道级掩码矩阵与输出特征图进行点乘
                    #进而对于处理后的特征图的每一个通道的不为零的元素们，假设该通道共N个非零值，对这N个值求和在除以N；进而得到了该通道的一个权重
                    #如果某通道所有值都为0的话，就单纯的将该通道的权重设置为0便好啦
                    #最终得到了512个权重（当然啦，还得进行一步0~1的归一化啦），它既是每一个通道的权重，也是每一组卷积核的权重啦
                    #当然这最后一步还有很多可以尝试的空间啦
                else:
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l31.cpu().unsqueeze(0)


                tmp_mean = torch.mean(tmp_mean_reuse_direction2, dim=1).numpy()
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean.append(tmp_mean)
########***********************************///////////////////****************+++++++++++


        return  output_mean









class GradCam_yuhan_kernel_version16_2_15():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, index=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别
        features,output = self.net(input.cuda())
        ###************
        feature_maps_L28 = features[
            0][0]  # 如果输入图像的shape为[batch_size,3,224,224]的tensor的话，关于输出的feature map,shape为  #[batch_size,512,14,14]左右
        # print(feature_maps_L28.shape)
        # print(feature_maps_L28)

        feature_maps_L28_channelwise_mean=torch.mean(feature_maps_L28,dim=(1,2))
        # print(feature_maps_L28_channelwise_mean.shape)
        highlight_chanelwise_L28 = (feature_maps_L28 > feature_maps_L28_channelwise_mean.view(-1, 1, 1)).float()
        # print(highlight_chanelwise_L28.shape)#为这512通道的每一个通道生成了一个（由该通道均值决定的）专属于该通道的二值掩码矩阵，将该通道的显著性区域标注了出来
        feature_maps_L28_sum=torch.sum(feature_maps_L28, dim=0)
        # print(feature_maps_L28_sum.shape)
        highlight_L28 = (feature_maps_L28_sum > torch.mean(feature_maps_L28_sum)).float()
        # print(highlight_L28.shape)
        h28 = highlight_chanelwise_L28 * highlight_L28.unsqueeze(0) #h28是一个【512，h,w】的二值矩阵，这是与总体二值矩阵交集得到的，可能在某一个通道上是一个全零矩阵
        # （一个完全对背景噪声响应的通道）
        # print(h28.shape)
        feature_maps_L28=feature_maps_L28 * h28  #同为shape为【512,7,7】的feature_map 与通道级二值矩阵（总体显著性区域与通道级显著性区域的交集）进行点乘，
        # chanelweight_l28 = torch.nn.functional.adaptive_max_pool2d(feature_maps_L28, output_size=1).squeeze(-1).squeeze(-1)
        chanelweight_l28=torch.sum(feature_maps_L28,dim=(1,2))/torch.sum(h28,dim=(1,2))
        # print(chanelweight_l28)
        # print(chanelweight_l28.shape)
        #我们用0代替了nan值
        chanelweight_l28 = torch.where(torch.isnan(chanelweight_l28), torch.full_like(chanelweight_l28, 0), chanelweight_l28)
        # chanelweight_l28=chanelweight_l28/torch.sum(highlight_L28)
        # print(chanelweight_l28)
        chanelweight_l28=(chanelweight_l28-torch.min(chanelweight_l28))/(torch.max(chanelweight_l28)-torch.min(chanelweight_l28))
        # # chanelweight_l28=(chanelweight_l28>torch.mean(chanelweight_l28)).float()
        # chanelweight_l28_bool = (chanelweight_l28 > torch.mean(chanelweight_l28)).float()
        # chanelweight_l28_bool_2 = torch.ones_like(chanelweight_l28_bool) - chanelweight_l28_bool
        # chanelweight_l28 = chanelweight_l28_bool_2 * chanelweight_l28
        # chanelweight_l28 = chanelweight_l28 + chanelweight_l28_bool
        # print(chanelweight_l28)



        feature_maps_L31 = features[
            1][0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        feature_maps_L31_channelwise_mean=torch.mean(feature_maps_L31,dim=(1,2))
        highlight_chanelwise_L31 = (feature_maps_L31 > feature_maps_L31_channelwise_mean.view(-1, 1, 1)).float()#得到的结果依然是【512,7,7】的一个二值矩阵
        feature_maps_L31_sum=torch.sum(feature_maps_L31, dim=0)
        highlight_L31 = (feature_maps_L31_sum > torch.mean(feature_maps_L31_sum)).float()
        h31 = highlight_chanelwise_L31 * highlight_L31.unsqueeze(0)
        feature_maps_L31 = feature_maps_L31 * h31
        # chanelweight_l31 = torch.nn.functional.adaptive_max_pool2d(feature_maps_L31, output_size=1).squeeze(-1).squeeze(
        #     -1)
        chanelweight_l31=torch.sum(feature_maps_L31,dim=(1,2))/torch.sum( h31 ,dim=(1,2))
        chanelweight_l31 = torch.where(torch.isnan(chanelweight_l31), torch.full_like(chanelweight_l31, 0), chanelweight_l31)

        # chanelweight_l31=torch.sum(h31,dim=(1,2))
        # chanelweight_l31=chanelweight_l31/torch.sum(highlight_L31)
        chanelweight_l31=(chanelweight_l31-torch.min(chanelweight_l31))/(torch.max(chanelweight_l31)-torch.min(chanelweight_l31))

        #
        # chanelweight_l31_bool=(chanelweight_l31>torch.mean(chanelweight_l31)).float()
        # chanelweight_l31_bool_2=torch.ones_like(chanelweight_l31_bool)-chanelweight_l31_bool
        # chanelweight_l31=chanelweight_l31_bool_2 * chanelweight_l31
        # chanelweight_l31=chanelweight_l31 + chanelweight_l31_bool
        # print(chanelweight_l31)
        # print(torch.sum((chanelweight_l31/np.linalg.norm(chanelweight_l31.cpu().data.numpy()))*(chanelweight_l28/np.linalg.norm(chanelweight_l28.cpu().data.numpy()))))

        ###************
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #
        output_mean=[]


        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['features.26.weight','features.28.weight']:
                # tmp=(param * param.grad).cpu().data
                tmp=(param).cpu().data
                # print(tmp.shape)#[512,512,3,3]
                tmp_mean_reuse=tmp.view(tmp.size()[0],-1)
                # tmp_mean_reuse=torch.mean(tmp,dim=( 2, 3))
                # tmp_mean_reuse = torch.std(tmp, axis=(2, 3))
                # tmp_mean_reuse=F.adaptive_max_pool2d(tmp,output_size=1).squeeze(-1).squeeze(-1)
                # tmp_mean_reuse=F.adaptive_max_pool2d(tmp,output_size=1).squeeze(-1).squeeze(-1) + F.adaptive_max_pool2d(-1 * tmp,output_size=1).squeeze(-1).squeeze(-1)

                # print(tmp_mean_reuse.shape)#[512,512],
                # 这个矩阵的每一行代表某一层的一组卷积核，这一行是由这一层的【512,3,3】得到的，一个512维的向量
                #这个矩阵一共512行，代表512组的卷积核

                # tmp_mean=torch.mean(tmp_mean_reuse,axis=1).numpy()
                # #这是将模型从右往左的压成一个向量，而不是从上往下对所有的512组卷积核取平均；
                # #再加上直接对测试阶段恒定不变的网络参数进行平均（未借助辅助信息进行加权），自然没用了
                # aa = np.linalg.norm(tmp_mean)
                # if aa != 0:
                #     tmp_mean = tmp_mean / aa
                # else:
                #     tmp_mean = np.zeros_like(tmp_mean)
                #
                # output_mean.append(tmp_mean)
#############################################################3
                #zhelizheli
                if name=='features.26.weight':
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l28.cpu().unsqueeze(0).T
                    #【512,512】的矩阵的每一行乘以一个固定的权重，也就是一组卷积核乘以一组固定的权重；这个权重是如何得到的呢?是这样的，该层的一组卷积核一组决定了该层输出特征图的某一个通道
                    #如果能够通过某种方式得到该通道在特征图整体的512个通道中的重要性的话，也就是说，我们得到了该通道的权重，也就是该组卷积核的权重，也就是tmp_mean_reuse 这个矩阵的某一行的权重。
                    #那么这个权重是如何得到的呢？这样的，scda不是得到了一个【h，w】的聚合特征图掩码矩阵嘛；其实我们完全可以得到一个通道级的而非整体的聚合特征图掩码矩阵，也就是一个【512，h，w】
                    #的掩码矩阵，其实就是生成一个通道级均值，特征图的每一个通道，高于该均值的位置掩码为1，否则为0；这就得到了【512，h,w】的掩码矩阵
                    #接下来呢，就是通道级掩码矩阵的每一个通道与整体的[h,w]的掩码进行按位与；最终得到了一个【512，h，w】的处理之后的掩码矩阵；
                    #这就决定了通道级掩码矩阵的每一个通道的每一个为1的位置，都是与图像的主要目标而非背景有关的；我们可以据此进一步的获知输出特征图的每一个通道激活的都是目标还是背景，也就是该通道
                    #进而就是该通道对应的该组滤波器的重要程度
                    #接下来就是最后的一步了，我们将处理之后的【512，h,w】的通道级掩码矩阵与输出特征图进行点乘
                    #进而对于处理后的特征图的每一个通道的不为零的元素们，假设该通道共N个非零值，对这N个值求和在除以N；进而得到了该通道的一个权重
                    #如果某通道所有值都为0的话，就单纯的将该通道的权重设置为0便好啦
                    #最终得到了512个权重（当然啦，还得进行一步0~1的归一化啦），它既是每一个通道的权重，也是每一组卷积核的权重啦
                    #当然这最后一步还有很多可以尝试的空间啦
                else:
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l31.cpu().unsqueeze(0).T


                tmp_mean = torch.mean(tmp_mean_reuse_direction2, dim=0).numpy()
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean.append(tmp_mean)
########***********************************///////////////////****************+++++++++++


        return  output_mean





#
# 层: features.0.weight torch.Size([64, 3, 3, 3])
# 层: features.0.bias torch.Size([64])
# 层: features.2.weight torch.Size([64, 64, 3, 3])
# 层: features.2.bias torch.Size([64])
# 层: features.5.weight torch.Size([128, 64, 3, 3])
# 层: features.5.bias torch.Size([128])
# 层: features.7.weight torch.Size([128, 128, 3, 3])
# 层: features.7.bias torch.Size([128])
# 层: features.10.weight torch.Size([256, 128, 3, 3])
# 层: features.10.bias torch.Size([256])
# 层: features.12.weight torch.Size([256, 256, 3, 3])
# 层: features.12.bias torch.Size([256])
# 层: features.14.weight torch.Size([256, 256, 3, 3])
# 层: features.14.bias torch.Size([256])
# 层: features.17.weight torch.Size([512, 256, 3, 3])
# 层: features.17.bias torch.Size([512])
# 层: features.19.weight torch.Size([512, 512, 3, 3])
# 层: features.19.bias torch.Size([512])
# 层: features.21.weight torch.Size([512, 512, 3, 3])
# 层: features.21.bias torch.Size([512])
# 层: features.24.weight torch.Size([512, 512, 3, 3])
# 层: features.24.bias torch.Size([512])
# 层: features.26.weight torch.Size([512, 512, 3, 3])
# 层: features.26.bias torch.Size([512])
# 层: features.28.weight torch.Size([512, 512, 3, 3])
# 层: features.28.bias torch.Size([512])
# 层: img_classifier.weight torch.Size([100, 512])
# 层: img_classifier.bias torch.Size([100])




class GradCam_yuhan_kernel_version16_2_16():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, index=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别
        features,output = self.net(input.cuda())
        ###************
        feature_maps_L26 = features[
            0][
            0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        feature_maps_L26_channelwise_mean = torch.mean(feature_maps_L26, dim=(1, 2))
        highlight_chanelwise_L26 = (feature_maps_L26 > feature_maps_L26_channelwise_mean.view(-1, 1,
                                                                                              1)).float()  # 得到的结果依然是【512,7,7】的一个二值矩阵
        feature_maps_L26_sum = torch.sum(feature_maps_L26, dim=0)
        highlight_L26 = (feature_maps_L26_sum > torch.mean(feature_maps_L26_sum)).float()
        h26 = highlight_chanelwise_L26 * highlight_L26.unsqueeze(0)
        feature_maps_L26 = feature_maps_L26 * h26
        # chanelweight_l31 = torch.nn.functional.adaptive_max_pool2d(feature_maps_L31, output_size=1).squeeze(-1).squeeze(
        #     -1)
        chanelweight_l26 = torch.sum(feature_maps_L26, dim=(1, 2)) / torch.sum(h26, dim=(1, 2))
        chanelweight_l26 = torch.where(torch.isnan(chanelweight_l26), torch.full_like(chanelweight_l26, 0),
                                       chanelweight_l26)

        # chanelweight_l31=torch.sum(h31,dim=(1,2))
        # chanelweight_l31=chanelweight_l31/torch.sum(highlight_L31)
        chanelweight_l26 = (chanelweight_l26 - torch.min(chanelweight_l26)) / (
                torch.max(chanelweight_l26) - torch.min(chanelweight_l26))


        feature_maps_L28 = features[
            1][0]  # 如果输入图像的shape为[batch_size,3,224,224]的tensor的话，关于输出的feature map,shape为  #[batch_size,512,14,14]左右
        # print(feature_maps_L28.shape)
        # print(feature_maps_L28)

        feature_maps_L28_channelwise_mean=torch.mean(feature_maps_L28,dim=(1,2))
        # print(feature_maps_L28_channelwise_mean.shape)
        highlight_chanelwise_L28 = (feature_maps_L28 > feature_maps_L28_channelwise_mean.view(-1, 1, 1)).float()
        # print(highlight_chanelwise_L28.shape)#为这512通道的每一个通道生成了一个（由该通道均值决定的）专属于该通道的二值掩码矩阵，将该通道的显著性区域标注了出来
        feature_maps_L28_sum=torch.sum(feature_maps_L28, dim=0)
        # print(feature_maps_L28_sum.shape)
        highlight_L28 = (feature_maps_L28_sum > torch.mean(feature_maps_L28_sum)).float()
        # print(highlight_L28.shape)
        h28 = highlight_chanelwise_L28 * highlight_L28.unsqueeze(0) #h28是一个【512，h,w】的二值矩阵，这是与总体二值矩阵交集得到的，可能在某一个通道上是一个全零矩阵
        # （一个完全对背景噪声响应的通道）
        # print(h28.shape)
        feature_maps_L28=feature_maps_L28 * h28  #同为shape为【512,7,7】的feature_map 与通道级二值矩阵（总体显著性区域与通道级显著性区域的交集）进行点乘，
        # chanelweight_l28 = torch.nn.functional.adaptive_max_pool2d(feature_maps_L28, output_size=1).squeeze(-1).squeeze(-1)
        chanelweight_l28=torch.sum(feature_maps_L28,dim=(1,2))/torch.sum(h28,dim=(1,2))
        # chanelweight_l28=F.adaptive_max_pool2d(feature_maps_L28
        #                                        ,output_size=1).squeeze(-1).squeeze(-1)

        # print(chanelweight_l28)
        # print(chanelweight_l28.shape)
        #我们用0代替了nan值
        chanelweight_l28 = torch.where(torch.isnan(chanelweight_l28), torch.full_like(chanelweight_l28, 0), chanelweight_l28)
        # chanelweight_l28=chanelweight_l28/torch.sum(highlight_L28)
        # print(chanelweight_l28)
        chanelweight_l28=(chanelweight_l28-torch.min(chanelweight_l28))/(torch.max(chanelweight_l28)-torch.min(chanelweight_l28))
        # chanelweight_l28=(chanelweight_l28>torch.mean(chanelweight_l28)).float()
        # chanelweight_l28_bool = (chanelweight_l28 > torch.mean(chanelweight_l28)).float()
        # chanelweight_l28_bool_2 = torch.ones_like(chanelweight_l28_bool) - chanelweight_l28_bool
        # chanelweight_l28 = chanelweight_l28_bool * chanelweight_l28
        # chanelweight_l28 = chanelweight_l28 + chanelweight_l28_bool
        # print(chanelweight_l28)



        feature_maps_L31 = features[
            2][0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        feature_maps_L31_channelwise_mean=torch.mean(feature_maps_L31,dim=(1,2))
        highlight_chanelwise_L31 = (feature_maps_L31 > feature_maps_L31_channelwise_mean.view(-1, 1, 1)).float()#得到的结果依然是【512,7,7】的一个二值矩阵
        feature_maps_L31_sum=torch.sum(feature_maps_L31, dim=0)
        highlight_L31 = (feature_maps_L31_sum > torch.mean(feature_maps_L31_sum)).float()
        h31 = highlight_chanelwise_L31 * highlight_L31.unsqueeze(0)
        feature_maps_L31 = feature_maps_L31 * h31
        # chanelweight_l31 = torch.nn.functional.adaptive_max_pool2d(feature_maps_L31, output_size=1).squeeze(-1).squeeze(
        #     -1)
        chanelweight_l31=torch.sum(feature_maps_L31,dim=(1,2))/torch.sum( h31 ,dim=(1,2))
        # chanelweight_l31 = F.adaptive_max_pool2d(feature_maps_L31
        #                                          , output_size=1).squeeze(-1).squeeze(-1)
        chanelweight_l31 = torch.where(torch.isnan(chanelweight_l31), torch.full_like(chanelweight_l31, 0), chanelweight_l31)

        # chanelweight_l31=torch.sum(h31,dim=(1,2))
        # chanelweight_l31=chanelweight_l31/torch.sum(highlight_L31)
        chanelweight_l31=(chanelweight_l31-torch.min(chanelweight_l31))/(torch.max(chanelweight_l31)-torch.min(chanelweight_l31))
        # chanelweight_l31=(chanelweight_l31>torch.mean(chanelweight_l31)).float()

        #
        # chanelweight_l31_bool=(chanelweight_l31>torch.mean(chanelweight_l31)).float()
        # chanelweight_l31_bool_2=torch.ones_like(chanelweight_l31_bool)-chanelweight_l31_bool
        # chanelweight_l31=chanelweight_l31_bool * chanelweight_l31
        # chanelweight_l31=chanelweight_l31 + chanelweight_l31_bool
        # print(chanelweight_l31)
        # print(torch.sum((chanelweight_l31/np.linalg.norm(chanelweight_l31.cpu().data.numpy()))*(chanelweight_l28/np.linalg.norm(chanelweight_l28.cpu().data.numpy()))))

        ###************
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #
        output_mean=[]
        output_mean_2=[]


        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['features.26.weight','features.28.weight']:
                # tmp=(param * param.grad).cpu().data
                tmp=(param).cpu().data
                # print(tmp.shape)#[512,512,3,3]
                # tmp_mean_reuse=tmp.view(tmp.size()[0],-1)
                tmp_mean_reuse=torch.mean(tmp,dim=( 2, 3))
                # print(tmp_mean_reuse.shape)#[512,512,3,3]

                # tmp_mean_reuse = torch.std(tmp, axis=(2, 3))
                # tmp_mean_reuse=F.adaptive_max_pool2d(tmp,output_size=1).squeeze(-1).squeeze(-1)
                # tmp_mean_reuse=F.adaptive_max_pool2d(tmp,output_size=1).squeeze(-1).squeeze(-1) + F.adaptive_max_pool2d(-1 * tmp,output_size=1).squeeze(-1).squeeze(-1)

                # print(tmp_mean_reuse.shape)#[512,512],
                # 这个矩阵的每一行代表某一层的一组卷积核，这一行是由这一层的【512,3,3】得到的，一个512维的向量
                #这个矩阵一共512行，代表512组的卷积核

                # tmp_mean=torch.mean(tmp_mean_reuse,axis=1).numpy()
                # #这是将模型从右往左的压成一个向量，而不是从上往下对所有的512组卷积核取平均；
                # #再加上直接对测试阶段恒定不变的网络参数进行平均（未借助辅助信息进行加权），自然没用了
                # aa = np.linalg.norm(tmp_mean)
                # if aa != 0:
                #     tmp_mean = tmp_mean / aa
                # else:
                #     tmp_mean = np.zeros_like(tmp_mean)
                #
                # output_mean.append(tmp_mean)
#############################################################3
                #zhelizheli
                if name=='features.26.weight':
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l28.cpu().unsqueeze(0).T
                    tmp_mean_reuse_direction2_2 = tmp_mean_reuse * chanelweight_l26.cpu().unsqueeze(0)
                    # tmp_mean_reuse_direction2 = tmp_mean_reuse

                    #【512,512】的矩阵的每一行乘以一个固定的权重，也就是一组卷积核乘以一组固定的权重；这个权重是如何得到的呢?是这样的，该层的一组卷积核一组决定了该层输出特征图的某一个通道
                    #如果能够通过某种方式得到该通道在特征图整体的512个通道中的重要性的话，也就是说，我们得到了该通道的权重，也就是该组卷积核的权重，也就是tmp_mean_reuse 这个矩阵的某一行的权重。
                    #那么这个权重是如何得到的呢？这样的，scda不是得到了一个【h，w】的聚合特征图掩码矩阵嘛；其实我们完全可以得到一个通道级的而非整体的聚合特征图掩码矩阵，也就是一个【512，h，w】
                    #的掩码矩阵，其实就是生成一个通道级均值，特征图的每一个通道，高于该均值的位置掩码为1，否则为0；这就得到了【512，h,w】的掩码矩阵
                    #接下来呢，就是通道级掩码矩阵的每一个通道与整体的[h,w]的掩码进行按位与；最终得到了一个【512，h，w】的处理之后的掩码矩阵；
                    #这就决定了通道级掩码矩阵的每一个通道的每一个为1的位置，都是与图像的主要目标而非背景有关的；我们可以据此进一步的获知输出特征图的每一个通道激活的都是目标还是背景，也就是该通道
                    #进而就是该通道对应的该组滤波器的重要程度
                    #接下来就是最后的一步了，我们将处理之后的【512，h,w】的通道级掩码矩阵与输出特征图进行点乘
                    #进而对于处理后的特征图的每一个通道的不为零的元素们，假设该通道共N个非零值，对这N个值求和在除以N；进而得到了该通道的一个权重
                    #如果某通道所有值都为0的话，就单纯的将该通道的权重设置为0便好啦
                    #最终得到了512个权重（当然啦，还得进行一步0~1的归一化啦），它既是每一个通道的权重，也是每一组卷积核的权重啦
                    #当然这最后一步还有很多可以尝试的空间啦
                else:
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l31.cpu().unsqueeze(0).T
                    tmp_mean_reuse_direction2_2 = tmp_mean_reuse * chanelweight_l28.cpu().unsqueeze(0)
                    # tmp_mean_reuse_direction2 = tmp_mean_reuse


                tmp_mean = torch.mean(tmp_mean_reuse_direction2, dim=0).numpy()
                tmp_mean_2 = torch.mean(tmp_mean_reuse_direction2_2, dim=1).numpy()

                # print(tmp_mean.shape)
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)
                aa = np.linalg.norm(tmp_mean_2)
                if aa != 0:
                    tmp_mean_2 = tmp_mean_2 / aa
                else:
                    tmp_mean_2 = np.zeros_like(tmp_mean_2)
                output_mean.append(tmp_mean)
                output_mean_2.append(tmp_mean_2)

########***********************************///////////////////****************+++++++++++


        return  output_mean,output_mean_2








class GradCam_yuhan_kernel_version16_2_14():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, index=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别
        features,output = self.net(input.cuda())
        ###************
        feature_maps_L28 = features[
            0][0]  # 如果输入图像的shape为[batch_size,3,224,224]的tensor的话，关于输出的feature map,shape为  #[batch_size,512,14,14]左右
        # print(feature_maps_L28.shape)
        # print(feature_maps_L28)

        feature_maps_L28_channelwise_mean=torch.mean(feature_maps_L28,dim=(1,2))
        # print(feature_maps_L28_channelwise_mean.shape)
        highlight_chanelwise_L28 = (feature_maps_L28 > feature_maps_L28_channelwise_mean.view(-1, 1, 1)).float()
        # print(highlight_chanelwise_L28.shape)#为这512通道的每一个通道生成了一个（由该通道均值决定的）专属于该通道的二值掩码矩阵，将该通道的显著性区域标注了出来
        feature_maps_L28_sum=torch.sum(feature_maps_L28, dim=0)
        # print(feature_maps_L28_sum.shape)
        highlight_L28 = (feature_maps_L28_sum > torch.mean(feature_maps_L28_sum)).float()
        # print(highlight_L28.shape)
        h28 = highlight_chanelwise_L28 * highlight_L28.unsqueeze(0) #h28是一个【512，h,w】的二值矩阵，这是与总体二值矩阵交集得到的，可能在某一个通道上是一个全零矩阵
        # （一个完全对背景噪声响应的通道）
        # print(h28.shape)
        feature_maps_L28=feature_maps_L28 * h28  #同为shape为【512,7,7】的feature_map 与通道级二值矩阵（总体显著性区域与通道级显著性区域的交集）进行点乘，
        # chanelweight_l28 = torch.nn.functional.adaptive_max_pool2d(feature_maps_L28, output_size=1).squeeze(-1).squeeze(-1)
        chanelweight_l28=torch.sum(feature_maps_L28,dim=(1,2))/torch.sum(h28,dim=(1,2))
        # print(chanelweight_l28)
        # print(chanelweight_l28.shape)
        #我们用0代替了nan值
        chanelweight_l28 = torch.where(torch.isnan(chanelweight_l28), torch.full_like(chanelweight_l28, 0), chanelweight_l28)
        # chanelweight_l28=chanelweight_l28/torch.sum(highlight_L28)
        # print(chanelweight_l28)
        chanelweight_l28=(chanelweight_l28-torch.min(chanelweight_l28))/(torch.max(chanelweight_l28)-torch.min(chanelweight_l28))
        # # chanelweight_l28=(chanelweight_l28>torch.mean(chanelweight_l28)).float()
        # chanelweight_l28_bool = (chanelweight_l28 > torch.mean(chanelweight_l28)).float()
        # chanelweight_l28_bool_2 = torch.ones_like(chanelweight_l28_bool) - chanelweight_l28_bool
        # chanelweight_l28 = chanelweight_l28_bool_2 * chanelweight_l28
        # chanelweight_l28 = chanelweight_l28 + chanelweight_l28_bool
        # print(chanelweight_l28)



        feature_maps_L31 = features[
            1][0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        feature_maps_L31_channelwise_mean=torch.mean(feature_maps_L31,dim=(1,2))
        highlight_chanelwise_L31 = (feature_maps_L31 > feature_maps_L31_channelwise_mean.view(-1, 1, 1)).float()#得到的结果依然是【512,7,7】的一个二值矩阵
        feature_maps_L31_sum=torch.sum(feature_maps_L31, dim=0)
        highlight_L31 = (feature_maps_L31_sum > torch.mean(feature_maps_L31_sum)).float()
        h31 = highlight_chanelwise_L31 * highlight_L31.unsqueeze(0)
        feature_maps_L31 = feature_maps_L31 * h31
        # chanelweight_l31 = torch.nn.functional.adaptive_max_pool2d(feature_maps_L31, output_size=1).squeeze(-1).squeeze(
        #     -1)
        chanelweight_l31=torch.sum(feature_maps_L31,dim=(1,2))/torch.sum( h31 ,dim=(1,2))
        chanelweight_l31 = torch.where(torch.isnan(chanelweight_l31), torch.full_like(chanelweight_l31, 0), chanelweight_l31)

        # chanelweight_l31=torch.sum(h31,dim=(1,2))
        # chanelweight_l31=chanelweight_l31/torch.sum(highlight_L31)
        chanelweight_l31=(chanelweight_l31-torch.min(chanelweight_l31))/(torch.max(chanelweight_l31)-torch.min(chanelweight_l31))

        #
        # chanelweight_l31_bool=(chanelweight_l31>torch.mean(chanelweight_l31)).float()
        # chanelweight_l31_bool_2=torch.ones_like(chanelweight_l31_bool)-chanelweight_l31_bool
        # chanelweight_l31=chanelweight_l31_bool_2 * chanelweight_l31
        # chanelweight_l31=chanelweight_l31 + chanelweight_l31_bool
        # print(chanelweight_l31)
        # print(torch.sum((chanelweight_l31/np.linalg.norm(chanelweight_l31.cpu().data.numpy()))*(chanelweight_l28/np.linalg.norm(chanelweight_l28.cpu().data.numpy()))))

        ###************
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #
        output_mean=[]
        output_mean_max=[]


        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['features.26.weight','features.28.weight']:
                # tmp=(param * param.grad).cpu().data
                tmp=(param).cpu().data
                # print(tmp.shape)#[512,512,3,3]
                tmp_mean_reuse=torch.mean(tmp,axis=( 2, 3))
                # tmp_mean_reuse_max=F.adaptive_max_pool2d(tmp,output_size=1).squeeze(-1).squeeze(-1)
                # tmp_mean_reuse=F.adaptive_max_pool2d(tmp,output_size=1).squeeze(-1).squeeze(-1) + F.adaptive_max_pool2d(-1 * tmp,output_size=1).squeeze(-1).squeeze(-1)
                tmp_mean_reuse_max = torch.std(tmp, axis=(2, 3))


                if name=='features.26.weight':
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l28.cpu().unsqueeze(0).T
                    tmp_mean_reuse_direction2_max = tmp_mean_reuse_max * chanelweight_l28.cpu().unsqueeze(0).T

                    #【512,512】的矩阵的每一行乘以一个固定的权重，也就是一组卷积核乘以一组固定的权重；这个权重是如何得到的呢?是这样的，该层的一组卷积核一组决定了该层输出特征图的某一个通道
                    #如果能够通过某种方式得到该通道在特征图整体的512个通道中的重要性的话，也就是说，我们得到了该通道的权重，也就是该组卷积核的权重，也就是tmp_mean_reuse 这个矩阵的某一行的权重。
                    #那么这个权重是如何得到的呢？这样的，scda不是得到了一个【h，w】的聚合特征图掩码矩阵嘛；其实我们完全可以得到一个通道级的而非整体的聚合特征图掩码矩阵，也就是一个【512，h，w】
                    #的掩码矩阵，其实就是生成一个通道级均值，特征图的每一个通道，高于该均值的位置掩码为1，否则为0；这就得到了【512，h,w】的掩码矩阵
                    #接下来呢，就是通道级掩码矩阵的每一个通道与整体的[h,w]的掩码进行按位与；最终得到了一个【512，h，w】的处理之后的掩码矩阵；
                    #这就决定了通道级掩码矩阵的每一个通道的每一个为1的位置，都是与图像的主要目标而非背景有关的；我们可以据此进一步的获知输出特征图的每一个通道激活的都是目标还是背景，也就是该通道
                    #进而就是该通道对应的该组滤波器的重要程度
                    #接下来就是最后的一步了，我们将处理之后的【512，h,w】的通道级掩码矩阵与输出特征图进行点乘
                    #进而对于处理后的特征图的每一个通道的不为零的元素们，假设该通道共N个非零值，对这N个值求和在除以N；进而得到了该通道的一个权重
                    #如果某通道所有值都为0的话，就单纯的将该通道的权重设置为0便好啦
                    #最终得到了512个权重（当然啦，还得进行一步0~1的归一化啦），它既是每一个通道的权重，也是每一组卷积核的权重啦
                    #当然这最后一步还有很多可以尝试的空间啦
                else:
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l31.cpu().unsqueeze(0).T
                    tmp_mean_reuse_direction2_max = tmp_mean_reuse_max * chanelweight_l31.cpu().unsqueeze(0).T


                tmp_mean = torch.mean(tmp_mean_reuse_direction2, axis=1).numpy()
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean.append(tmp_mean)

                tmp_mean_max = torch.mean(tmp_mean_reuse_direction2_max, axis=1).numpy()
                aa = np.linalg.norm(tmp_mean_max)
                if aa != 0:
                    tmp_mean_max = tmp_mean_max / aa
                else:
                    tmp_mean_max = np.zeros_like(tmp_mean_max)

                output_mean_max.append(tmp_mean_max)

########***********************************///////////////////****************+++++++++++


        return  output_mean,output_mean_max














class GradCam_yuhan_kernel_version16_2_13():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, index=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别
        features,output = self.net(input.cuda())
        ###************
        feature_maps_L28 = features[
            0][0]  # 如果输入图像的shape为[batch_size,3,224,224]的tensor的话，关于输出的feature map,shape为  #[batch_size,512,14,14]左右
        # print(feature_maps_L28.shape)
        # print(feature_maps_L28)

        feature_maps_L28_channelwise_mean=torch.mean(feature_maps_L28,dim=(1,2))
        # print(feature_maps_L28_channelwise_mean.shape)
        highlight_chanelwise_L28 = (feature_maps_L28 > feature_maps_L28_channelwise_mean.view(-1, 1, 1)).float()
        # print(highlight_chanelwise_L28.shape)#为这512通道的每一个通道生成了一个（由该通道均值决定的）专属于该通道的二值掩码矩阵，将该通道的显著性区域标注了出来
        feature_maps_L28_sum=torch.sum(feature_maps_L28, dim=0)
        # print(feature_maps_L28_sum.shape)
        highlight_L28 = (feature_maps_L28_sum > torch.mean(feature_maps_L28_sum)).float()
        # print(highlight_L28.shape)
        h28 = highlight_chanelwise_L28 * highlight_L28.unsqueeze(0) #h28是一个【512，h,w】的二值矩阵，这是与总体二值矩阵交集得到的，可能在某一个通道上是一个全零矩阵
        # （一个完全对背景噪声响应的通道）
        # print(h28.shape)
        feature_maps_L28=feature_maps_L28 * h28  #同为shape为【512,7,7】的feature_map 与通道级二值矩阵（总体显著性区域与通道级显著性区域的交集）进行点乘，
        # chanelweight_l28 = torch.nn.functional.adaptive_max_pool2d(feature_maps_L28, output_size=1).squeeze(-1).squeeze(-1)
        chanelweight_l28=torch.sum(feature_maps_L28,dim=(1,2))/torch.sum(h28,dim=(1,2))
        # print(chanelweight_l28)
        # print(chanelweight_l28.shape)
        #我们用0代替了nan值
        chanelweight_l28 = torch.where(torch.isnan(chanelweight_l28), torch.full_like(chanelweight_l28, 0), chanelweight_l28)
        # chanelweight_l28=chanelweight_l28/torch.sum(highlight_L28)
        # print(chanelweight_l28)
        chanelweight_l28=(chanelweight_l28-torch.min(chanelweight_l28))/(torch.max(chanelweight_l28)-torch.min(chanelweight_l28))
        # # chanelweight_l28=(chanelweight_l28>torch.mean(chanelweight_l28)).float()
        # chanelweight_l28_bool = (chanelweight_l28 > torch.mean(chanelweight_l28)).float()
        # chanelweight_l28_bool_2 = torch.ones_like(chanelweight_l28_bool) - chanelweight_l28_bool
        # chanelweight_l28 = chanelweight_l28_bool_2 * chanelweight_l28
        # chanelweight_l28 = chanelweight_l28 + chanelweight_l28_bool
        # print(chanelweight_l28)



        feature_maps_L31 = features[
            1][0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        feature_maps_L31_channelwise_mean=torch.mean(feature_maps_L31,dim=(1,2))
        highlight_chanelwise_L31 = (feature_maps_L31 > feature_maps_L31_channelwise_mean.view(-1, 1, 1)).float()#得到的结果依然是【512,7,7】的一个二值矩阵
        feature_maps_L31_sum=torch.sum(feature_maps_L31, dim=0)
        highlight_L31 = (feature_maps_L31_sum > torch.mean(feature_maps_L31_sum)).float()
        h31 = highlight_chanelwise_L31 * highlight_L31.unsqueeze(0)
        feature_maps_L31 = feature_maps_L31 * h31
        # chanelweight_l31 = torch.nn.functional.adaptive_max_pool2d(feature_maps_L31, output_size=1).squeeze(-1).squeeze(
        #     -1)
        chanelweight_l31=torch.sum(feature_maps_L31,dim=(1,2))/torch.sum( h31 ,dim=(1,2))
        chanelweight_l31 = torch.where(torch.isnan(chanelweight_l31), torch.full_like(chanelweight_l31, 0), chanelweight_l31)

        # chanelweight_l31=torch.sum(h31,dim=(1,2))
        # chanelweight_l31=chanelweight_l31/torch.sum(highlight_L31)
        chanelweight_l31=(chanelweight_l31-torch.min(chanelweight_l31))/(torch.max(chanelweight_l31)-torch.min(chanelweight_l31))

        #
        # chanelweight_l31_bool=(chanelweight_l31>torch.mean(chanelweight_l31)).float()
        # chanelweight_l31_bool_2=torch.ones_like(chanelweight_l31_bool)-chanelweight_l31_bool
        # chanelweight_l31=chanelweight_l31_bool_2 * chanelweight_l31
        # chanelweight_l31=chanelweight_l31 + chanelweight_l31_bool
        # print(chanelweight_l31)
        # print(torch.sum((chanelweight_l31/np.linalg.norm(chanelweight_l31.cpu().data.numpy()))*(chanelweight_l28/np.linalg.norm(chanelweight_l28.cpu().data.numpy()))))

        ###************
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #
        output_mean=[]


        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['features.26.weight','features.28.weight']:
                # tmp=(param * param.grad).cpu().data
                tmp=(param).cpu().data
                # print(tmp)
                # tmp = F.relu(tmp)
                # print(tmp)
                # print(tmp.shape)#[512,512,3,3]

                # tmp_mean_reuse=torch.mean(tmp,axis=(2,3))
                # tmp_mean_reuse=F.adaptive_max_pool2d(tmp,output_size=1).squeeze(-1).squeeze(-1)
                # tmp_mean_reuse=F.adaptive_max_pool2d(tmp,output_size=1).squeeze(-1).squeeze(-1) + F.adaptive_max_pool2d(-1 * tmp,output_size=1).squeeze(-1).squeeze(-1)
                tmp_mean_reuse=torch.std(tmp,axis=(2,3))
                # print(tmp_mean_reuse.size())

                # print(tmp_mean_reuse.shape)
                # print(tmp_mean_reuse.shape)#[512,512],
                # 这个矩阵的每一行代表某一层的一组卷积核，这一行是由这一层的【512,3,3】得到的，一个512维的向量
                #这个矩阵一共512行，代表512组的卷积核

                # tmp_mean=torch.mean(tmp_mean_reuse,axis=1).numpy()
                # #这是将模型从右往左的压成一个向量，而不是从上往下对所有的512组卷积核取平均；
                # #再加上直接对测试阶段恒定不变的网络参数进行平均（未借助辅助信息进行加权），自然没用了
                # aa = np.linalg.norm(tmp_mean)
                # if aa != 0:
                #     tmp_mean = tmp_mean / aa
                # else:
                #     tmp_mean = np.zeros_like(tmp_mean)
                #
                # output_mean.append(tmp_mean)
#############################################################3
                #zhelizheli
                if name=='features.26.weight':
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l28.cpu().unsqueeze(0).T
                    #【512,512】的矩阵的每一行乘以一个固定的权重，也就是一组卷积核乘以一组固定的权重；这个权重是如何得到的呢?是这样的，该层的一组卷积核一组决定了该层输出特征图的某一个通道
                    #如果能够通过某种方式得到该通道在特征图整体的512个通道中的重要性的话，也就是说，我们得到了该通道的权重，也就是该组卷积核的权重，也就是tmp_mean_reuse 这个矩阵的某一行的权重。
                    #那么这个权重是如何得到的呢？这样的，scda不是得到了一个【h，w】的聚合特征图掩码矩阵嘛；其实我们完全可以得到一个通道级的而非整体的聚合特征图掩码矩阵，也就是一个【512，h，w】
                    #的掩码矩阵，其实就是生成一个通道级均值，特征图的每一个通道，高于该均值的位置掩码为1，否则为0；这就得到了【512，h,w】的掩码矩阵
                    #接下来呢，就是通道级掩码矩阵的每一个通道与整体的[h,w]的掩码进行按位与；最终得到了一个【512，h，w】的处理之后的掩码矩阵；
                    #这就决定了通道级掩码矩阵的每一个通道的每一个为1的位置，都是与图像的主要目标而非背景有关的；我们可以据此进一步的获知输出特征图的每一个通道激活的都是目标还是背景，也就是该通道
                    #进而就是该通道对应的该组滤波器的重要程度
                    #接下来就是最后的一步了，我们将处理之后的【512，h,w】的通道级掩码矩阵与输出特征图进行点乘
                    #进而对于处理后的特征图的每一个通道的不为零的元素们，假设该通道共N个非零值，对这N个值求和在除以N；进而得到了该通道的一个权重
                    #如果某通道所有值都为0的话，就单纯的将该通道的权重设置为0便好啦
                    #最终得到了512个权重（当然啦，还得进行一步0~1的归一化啦），它既是每一个通道的权重，也是每一组卷积核的权重啦
                    #当然这最后一步还有很多可以尝试的空间啦
                else:
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l31.cpu().unsqueeze(0).T


                tmp_mean = torch.mean(tmp_mean_reuse_direction2, dim=1)
                # tmp_mean = (tmp_mean - torch.min(tmp_mean)) / (
                #             torch.max(tmp_mean) - torch.min(tmp_mean))

                tmp_mean = tmp_mean.numpy()
                # print(tmp_mean)
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean.append(tmp_mean)
########***********************************///////////////////****************+++++++++++


        return  output_mean











class GradCam_yuhan_kernel_version16_2_12():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, index=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别
        features,output = self.net(input.cuda())
        ###************
        feature_maps_L28 = features[
            0][0]  # 如果输入图像的shape为[batch_size,3,224,224]的tensor的话，关于输出的feature map,shape为  #[batch_size,512,14,14]左右
        # print(feature_maps_L28.shape)
        # print(feature_maps_L28)

        feature_maps_L28_channelwise_mean=torch.mean(feature_maps_L28,dim=(1,2))
        # print(feature_maps_L28_channelwise_mean.shape)
        highlight_chanelwise_L28 = (feature_maps_L28 > feature_maps_L28_channelwise_mean.view(-1, 1, 1)).float()
        # print(highlight_chanelwise_L28.shape)#为这512通道的每一个通道生成了一个（由该通道均值决定的）专属于该通道的二值掩码矩阵，将该通道的显著性区域标注了出来
        feature_maps_L28_sum=torch.sum(feature_maps_L28, dim=0)
        # print(feature_maps_L28_sum.shape)
        highlight_L28 = (feature_maps_L28_sum > torch.mean(feature_maps_L28_sum)).float()
        # print(highlight_L28.shape)
        h28 = highlight_chanelwise_L28 * highlight_L28.unsqueeze(0) #h28是一个【512，h,w】的二值矩阵，这是与总体二值矩阵交集得到的，可能在某一个通道上是一个全零矩阵
        # （一个完全对背景噪声响应的通道）
        # print(h28.shape)
        feature_maps_L28=feature_maps_L28 * h28  #同为shape为【512,7,7】的feature_map 与通道级二值矩阵（总体显著性区域与通道级显著性区域的交集）进行点乘，
        # chanelweight_l28 = torch.nn.functional.adaptive_max_pool2d(feature_maps_L28, output_size=1).squeeze(-1).squeeze(-1)
        chanelweight_l28=torch.sum(feature_maps_L28,dim=(1,2))/torch.sum(h28,dim=(1,2))
        # print(chanelweight_l28)
        # print(chanelweight_l28.shape)
        #我们用0代替了nan值
        chanelweight_l28 = torch.where(torch.isnan(chanelweight_l28), torch.full_like(chanelweight_l28, 0), chanelweight_l28)
        # chanelweight_l28=chanelweight_l28/torch.sum(highlight_L28)
        # print(chanelweight_l28)
        chanelweight_l28=(chanelweight_l28-torch.min(chanelweight_l28))/(torch.max(chanelweight_l28)-torch.min(chanelweight_l28))
        # # chanelweight_l28=(chanelweight_l28>torch.mean(chanelweight_l28)).float()
        # chanelweight_l28_bool = (chanelweight_l28 > torch.mean(chanelweight_l28)).float()
        # chanelweight_l28_bool_2 = torch.ones_like(chanelweight_l28_bool) - chanelweight_l28_bool
        # chanelweight_l28 = chanelweight_l28_bool_2 * chanelweight_l28
        # chanelweight_l28 = chanelweight_l28 + chanelweight_l28_bool
        # print(chanelweight_l28)



        feature_maps_L31 = features[
            1][0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        feature_maps_L31_channelwise_mean=torch.mean(feature_maps_L31,dim=(1,2))
        highlight_chanelwise_L31 = (feature_maps_L31 > feature_maps_L31_channelwise_mean.view(-1, 1, 1)).float()#得到的结果依然是【512,7,7】的一个二值矩阵
        feature_maps_L31_sum=torch.sum(feature_maps_L31, dim=0)
        highlight_L31 = (feature_maps_L31_sum > torch.mean(feature_maps_L31_sum)).float()
        h31 = highlight_chanelwise_L31 * highlight_L31.unsqueeze(0)
        feature_maps_L31 = feature_maps_L31 * h31
        # chanelweight_l31 = torch.nn.functional.adaptive_max_pool2d(feature_maps_L31, output_size=1).squeeze(-1).squeeze(
        #     -1)
        chanelweight_l31=torch.sum(feature_maps_L31,dim=(1,2))/torch.sum( h31 ,dim=(1,2))
        chanelweight_l31 = torch.where(torch.isnan(chanelweight_l31), torch.full_like(chanelweight_l31, 0), chanelweight_l31)

        # chanelweight_l31=torch.sum(h31,dim=(1,2))
        # chanelweight_l31=chanelweight_l31/torch.sum(highlight_L31)
        chanelweight_l31=(chanelweight_l31-torch.min(chanelweight_l31))/(torch.max(chanelweight_l31)-torch.min(chanelweight_l31))

        #
        # chanelweight_l31_bool=(chanelweight_l31>torch.mean(chanelweight_l31)).float()
        # chanelweight_l31_bool_2=torch.ones_like(chanelweight_l31_bool)-chanelweight_l31_bool
        # chanelweight_l31=chanelweight_l31_bool_2 * chanelweight_l31
        # chanelweight_l31=chanelweight_l31 + chanelweight_l31_bool
        # print(chanelweight_l31)
        # print(torch.sum((chanelweight_l31/np.linalg.norm(chanelweight_l31.cpu().data.numpy()))*(chanelweight_l28/np.linalg.norm(chanelweight_l28.cpu().data.numpy()))))

        ###************
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #
        output_mean=[]


        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['features.26.weight','features.28.weight']:
                # tmp=(param * param.grad).cpu().data
                tmp=(param).cpu().data
                # print(tmp.shape)#[512,512,3,3]
                tmp_mean_reuse=torch.mean(tmp,dim=[ 2, 3])
                # print(tmp_mean_reuse.shape)#[512,512],
                # 这个矩阵的每一行代表某一层的一组卷积核，这一行是由这一层的【512,3,3】得到的，一个512维的向量
                #这个矩阵一共512行，代表512组的卷积核

                # tmp_mean=torch.mean(tmp_mean_reuse,axis=1).numpy()
                # #这是将模型从右往左的压成一个向量，而不是从上往下对所有的512组卷积核取平均；
                # #再加上直接对测试阶段恒定不变的网络参数进行平均（未借助辅助信息进行加权），自然没用了
                # aa = np.linalg.norm(tmp_mean)
                # if aa != 0:
                #     tmp_mean = tmp_mean / aa
                # else:
                #     tmp_mean = np.zeros_like(tmp_mean)
                #
                # output_mean.append(tmp_mean)
#############################################################3
                #zhelizheli
                if name=='features.26.weight':
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l28.cpu().unsqueeze(0).T
                    #【512,512】的矩阵的每一行乘以一个固定的权重，也就是一组卷积核乘以一组固定的权重；这个权重是如何得到的呢?是这样的，该层的一组卷积核一组决定了该层输出特征图的某一个通道
                    #如果能够通过某种方式得到该通道在特征图整体的512个通道中的重要性的话，也就是说，我们得到了该通道的权重，也就是该组卷积核的权重，也就是tmp_mean_reuse 这个矩阵的某一行的权重。
                    #那么这个权重是如何得到的呢？这样的，scda不是得到了一个【h，w】的聚合特征图掩码矩阵嘛；其实我们完全可以得到一个通道级的而非整体的聚合特征图掩码矩阵，也就是一个【512，h，w】
                    #的掩码矩阵，其实就是生成一个通道级均值，特征图的每一个通道，高于该均值的位置掩码为1，否则为0；这就得到了【512，h,w】的掩码矩阵
                    #接下来呢，就是通道级掩码矩阵的每一个通道与整体的[h,w]的掩码进行按位与；最终得到了一个【512，h，w】的处理之后的掩码矩阵；
                    #这就决定了通道级掩码矩阵的每一个通道的每一个为1的位置，都是与图像的主要目标而非背景有关的；我们可以据此进一步的获知输出特征图的每一个通道激活的都是目标还是背景，也就是该通道
                    #进而就是该通道对应的该组滤波器的重要程度
                    #接下来就是最后的一步了，我们将处理之后的【512，h,w】的通道级掩码矩阵与输出特征图进行点乘
                    #进而对于处理后的特征图的每一个通道的不为零的元素们，假设该通道共N个非零值，对这N个值求和在除以N；进而得到了该通道的一个权重
                    #如果某通道所有值都为0的话，就单纯的将该通道的权重设置为0便好啦
                    #最终得到了512个权重（当然啦，还得进行一步0~1的归一化啦），它既是每一个通道的权重，也是每一组卷积核的权重啦
                    #当然这最后一步还有很多可以尝试的空间啦
                else:
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l31.cpu().unsqueeze(0).T


                tmp_mean = torch.mean(tmp_mean_reuse_direction2[:,0:int((tmp_mean_reuse_direction2.size()[1])/4)], dim=1)
                tmp_mean_2 = torch.mean(tmp_mean_reuse_direction2[:,int((tmp_mean_reuse_direction2.size()[1])/4) : int(tmp_mean_reuse_direction2.size()[1]/2)], dim=1)
                tmp_mean_3 = torch.mean(tmp_mean_reuse_direction2[:,int((tmp_mean_reuse_direction2.size()[1])/2) : int(3*tmp_mean_reuse_direction2.size()[1]/4)], dim=1)
                tmp_mean_4 = torch.mean(tmp_mean_reuse_direction2[:,int((3*tmp_mean_reuse_direction2.size()[1])/4) : int(tmp_mean_reuse_direction2.size()[1])], dim=1)

                # print(tmp_mean.shape)
                # print(tmp_mean_2.shape)
                # print(tmp_mean_3.shape)
                # print(tmp_mean_4.shape)

                tmp_mean=torch.cat((tmp_mean,tmp_mean_2,tmp_mean_3,tmp_mean_4))
                # print(tmp_mean.size())
                tmp_mean=tmp_mean.numpy()
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean.append(tmp_mean)
########***********************************///////////////////****************+++++++++++


        return  output_mean













class GradCam_yuhan_kernel_version16_2_10():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, index=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别
        features,output = self.net(input.cuda())
        ###************
        feature_maps_L26 = features[
            0][
            0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        feature_maps_L26_channelwise_mean = torch.mean(feature_maps_L26, dim=(1, 2))
        highlight_chanelwise_L26 = (feature_maps_L26 > feature_maps_L26_channelwise_mean.view(-1, 1,
                                                                                              1)).float()  # 得到的结果依然是【512,7,7】的一个二值矩阵
        feature_maps_L26_sum = torch.sum(feature_maps_L26, dim=0)
        highlight_L26 = (feature_maps_L26_sum > torch.mean(feature_maps_L26_sum)).float()
        h26 = highlight_chanelwise_L26 * highlight_L26.unsqueeze(0)
        feature_maps_L26 = feature_maps_L26 * h26
        # chanelweight_l31 = torch.nn.functional.adaptive_max_pool2d(feature_maps_L31, output_size=1).squeeze(-1).squeeze(
        #     -1)
        chanelweight_l26 = torch.sum(feature_maps_L26, dim=(1, 2)) / torch.sum(h26, dim=(1, 2))
        chanelweight_l26 = torch.where(torch.isnan(chanelweight_l26), torch.full_like(chanelweight_l26, 0),
                                       chanelweight_l26)

        # chanelweight_l31=torch.sum(h31,dim=(1,2))
        # chanelweight_l31=chanelweight_l31/torch.sum(highlight_L31)
        chanelweight_l26 = (chanelweight_l26 - torch.min(chanelweight_l26)) / (
                    torch.max(chanelweight_l26) - torch.min(chanelweight_l26))

        feature_maps_L28 = features[
            1][0]  # 如果输入图像的shape为[batch_size,3,224,224]的tensor的话，关于输出的feature map,shape为  #[batch_size,512,14,14]左右
        # print(feature_maps_L28.shape)
        # print(feature_maps_L28)

        feature_maps_L28_channelwise_mean=torch.mean(feature_maps_L28,dim=(1,2))
        # print(feature_maps_L28_channelwise_mean.shape)
        highlight_chanelwise_L28 = (feature_maps_L28 > feature_maps_L28_channelwise_mean.view(-1, 1, 1)).float()
        # print(highlight_chanelwise_L28.shape)#为这512通道的每一个通道生成了一个（由该通道均值决定的）专属于该通道的二值掩码矩阵，将该通道的显著性区域标注了出来
        feature_maps_L28_sum=torch.sum(feature_maps_L28, dim=0)
        # print(feature_maps_L28_sum.shape)
        highlight_L28 = (feature_maps_L28_sum > torch.mean(feature_maps_L28_sum)).float()
        # print(highlight_L28.shape)
        h28 = highlight_chanelwise_L28 * highlight_L28.unsqueeze(0) #h28是一个【512，h,w】的二值矩阵，这是与总体二值矩阵交集得到的，可能在某一个通道上是一个全零矩阵
        # （一个完全对背景噪声响应的通道）
        # print(h28.shape)
        feature_maps_L28=feature_maps_L28 * h28  #同为shape为【512,7,7】的feature_map 与通道级二值矩阵（总体显著性区域与通道级显著性区域的交集）进行点乘，
        # chanelweight_l28 = torch.nn.functional.adaptive_max_pool2d(feature_maps_L28, output_size=1).squeeze(-1).squeeze(-1)
        chanelweight_l28=torch.sum(feature_maps_L28,dim=(1,2))/torch.sum(h28,dim=(1,2))
        # print(chanelweight_l28)
        # print(chanelweight_l28.shape)
        #我们用0代替了nan值
        chanelweight_l28 = torch.where(torch.isnan(chanelweight_l28), torch.full_like(chanelweight_l28, 0), chanelweight_l28)
        # chanelweight_l28=chanelweight_l28/torch.sum(highlight_L28)
        # print(chanelweight_l28)
        chanelweight_l28=(chanelweight_l28-torch.min(chanelweight_l28))/(torch.max(chanelweight_l28)-torch.min(chanelweight_l28))
        # # chanelweight_l28=(chanelweight_l28>torch.mean(chanelweight_l28)).float()
        # chanelweight_l28_bool = (chanelweight_l28 > torch.mean(chanelweight_l28)).float()
        # chanelweight_l28_bool_2 = torch.ones_like(chanelweight_l28_bool) - chanelweight_l28_bool
        # chanelweight_l28 = chanelweight_l28_bool_2 * chanelweight_l28
        # chanelweight_l28 = chanelweight_l28 + chanelweight_l28_bool
        # print(chanelweight_l28)



        feature_maps_L31 = features[
            2][0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        feature_maps_L31_channelwise_mean=torch.mean(feature_maps_L31,dim=(1,2))
        highlight_chanelwise_L31 = (feature_maps_L31 > feature_maps_L31_channelwise_mean.view(-1, 1, 1)).float()#得到的结果依然是【512,7,7】的一个二值矩阵
        feature_maps_L31_sum=torch.sum(feature_maps_L31, dim=0)
        highlight_L31 = (feature_maps_L31_sum > torch.mean(feature_maps_L31_sum)).float()
        h31 = highlight_chanelwise_L31 * highlight_L31.unsqueeze(0)
        feature_maps_L31 = feature_maps_L31 * h31
        # chanelweight_l31 = torch.nn.functional.adaptive_max_pool2d(feature_maps_L31, output_size=1).squeeze(-1).squeeze(
        #     -1)
        chanelweight_l31=torch.sum(feature_maps_L31,dim=(1,2))/torch.sum( h31 ,dim=(1,2))
        chanelweight_l31 = torch.where(torch.isnan(chanelweight_l31), torch.full_like(chanelweight_l31, 0), chanelweight_l31)

        # chanelweight_l31=torch.sum(h31,dim=(1,2))
        # chanelweight_l31=chanelweight_l31/torch.sum(highlight_L31)
        chanelweight_l31=(chanelweight_l31-torch.min(chanelweight_l31))/(torch.max(chanelweight_l31)-torch.min(chanelweight_l31))
        # print(chanelweight_l31)

        #
        # chanelweight_l31_bool=(chanelweight_l31>torch.mean(chanelweight_l31)).float()
        # chanelweight_l31_bool_2=torch.ones_like(chanelweight_l31_bool)-chanelweight_l31_bool
        # chanelweight_l31=chanelweight_l31_bool_2 * chanelweight_l31
        # chanelweight_l31=chanelweight_l31 + chanelweight_l31_bool

        ###************
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #
        output_mean=[]
        output_mean_2=[]


        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['features.26.weight','features.28.weight']:
                # tmp=(param * param.grad).cpu().data
                tmp=(param).cpu().data
                # print(tmp.shape)#[512,512,3,3]
                tmp_mean_reuse=torch.mean(tmp,axis=( 2, 3))
                # print(tmp_mean_reuse.shape)#[512,512],
                # 这个矩阵的每一行代表某一层的一组卷积核，这一行是由这一层的【512,3,3】得到的，一个512维的向量
                #这个矩阵一共512行，代表512组的卷积核

                # tmp_mean=torch.mean(tmp_mean_reuse,axis=1).numpy()
                # #这是将模型从右往左的压成一个向量，而不是从上往下对所有的512组卷积核取平均；
                # #再加上直接对测试阶段恒定不变的网络参数进行平均（未借助辅助信息进行加权），自然没用了
                # aa = np.linalg.norm(tmp_mean)
                # if aa != 0:
                #     tmp_mean = tmp_mean / aa
                # else:
                #     tmp_mean = np.zeros_like(tmp_mean)
                #
                # output_mean.append(tmp_mean)
#############################################################3
                #zhelizheli
                if name=='features.26.weight':
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l28.cpu().unsqueeze(0).T
                    # tmp_mean_reuse_direction2 = tmp_mean_reuse_direction2 * chanelweight_l26.cpu().unsqueeze(0)
                    # tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l28.cpu().unsqueeze(0)

                    #【512,512】的矩阵的每一行乘以一个固定的权重，也就是一组卷积核乘以一组固定的权重；这个权重是如何得到的呢?是这样的，该层的一组卷积核一组决定了该层输出特征图的某一个通道
                    #如果能够通过某种方式得到该通道在特征图整体的512个通道中的重要性的话，也就是说，我们得到了该通道的权重，也就是该组卷积核的权重，也就是tmp_mean_reuse 这个矩阵的某一行的权重。
                    #那么这个权重是如何得到的呢？这样的，scda不是得到了一个【h，w】的聚合特征图掩码矩阵嘛；其实我们完全可以得到一个通道级的而非整体的聚合特征图掩码矩阵，也就是一个【512，h，w】
                    #的掩码矩阵，其实就是生成一个通道级均值，特征图的每一个通道，高于该均值的位置掩码为1，否则为0；这就得到了【512，h,w】的掩码矩阵
                    #接下来呢，就是通道级掩码矩阵的每一个通道与整体的[h,w]的掩码进行按位与；最终得到了一个【512，h，w】的处理之后的掩码矩阵；
                    #这就决定了通道级掩码矩阵的每一个通道的每一个为1的位置，都是与图像的主要目标而非背景有关的；我们可以据此进一步的获知输出特征图的每一个通道激活的都是目标还是背景，也就是该通道
                    #进而就是该通道对应的该组滤波器的重要程度
                    #接下来就是最后的一步了，我们将处理之后的【512，h,w】的通道级掩码矩阵与输出特征图进行点乘
                    #进而对于处理后的特征图的每一个通道的不为零的元素们，假设该通道共N个非零值，对这N个值求和在除以N；进而得到了该通道的一个权重
                    #如果某通道所有值都为0的话，就单纯的将该通道的权重设置为0便好啦
                    #最终得到了512个权重（当然啦，还得进行一步0~1的归一化啦），它既是每一个通道的权重，也是每一组卷积核的权重啦
                    #当然这最后一步还有很多可以尝试的空间啦
                else:
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l31.cpu().unsqueeze(0).T
                    # tmp_mean_reuse_direction2 = tmp_mean_reuse_direction2 * chanelweight_l28.cpu().unsqueeze(0)
                    # tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l31.cpu().unsqueeze(0)


                tmp_mean = torch.mean(tmp_mean_reuse_direction2, axis=1).numpy()
                tmp_mean_2 = torch.mean(tmp_mean_reuse_direction2, axis=0).numpy()

                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                bb = np.linalg.norm(tmp_mean_2)
                if bb != 0:
                    tmp_mean_2 = tmp_mean_2 / bb
                else:
                    tmp_mean_2 = np.zeros_like(tmp_mean_2)

                output_mean.append(tmp_mean)
                output_mean_2.append(tmp_mean_2)
########***********************************///////////////////****************+++++++++++


        return  output_mean,output_mean_2








class GradCam_yuhan_kernel_version16_2_11():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, index=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别
        features,output = self.net(input.cuda())
        ###************


        feature_maps_L28 = features[
            0][0]  # 如果输入图像的shape为[batch_size,3,224,224]的tensor的话，关于输出的feature map,shape为  #[batch_size,512,14,14]左右
        # print(feature_maps_L28.shape)
        # print(feature_maps_L28)

        feature_maps_L28_channelwise_mean=torch.mean(feature_maps_L28,dim=(1,2))
        # print(feature_maps_L28_channelwise_mean.shape)
        highlight_chanelwise_L28 = (feature_maps_L28 > feature_maps_L28_channelwise_mean.view(-1, 1, 1)).float()
        # print(highlight_chanelwise_L28.shape)#为这512通道的每一个通道生成了一个（由该通道均值决定的）专属于该通道的二值掩码矩阵，将该通道的显著性区域标注了出来
        feature_maps_L28_sum=torch.sum(feature_maps_L28, dim=0)
        # print(feature_maps_L28_sum.shape)
        highlight_L28 = (feature_maps_L28_sum > torch.mean(feature_maps_L28_sum)).float()
        # print(highlight_L28.shape)
        h28 = highlight_chanelwise_L28 * highlight_L28.unsqueeze(0) #h28是一个【512，h,w】的二值矩阵，这是与总体二值矩阵交集得到的，可能在某一个通道上是一个全零矩阵
        # （一个完全对背景噪声响应的通道）
        # print(h28.shape)
        feature_maps_L28=feature_maps_L28 * h28  #同为shape为【512,7,7】的feature_map 与通道级二值矩阵（总体显著性区域与通道级显著性区域的交集）进行点乘，
        # chanelweight_l28 = torch.nn.functional.adaptive_max_pool2d(feature_maps_L28, output_size=1).squeeze(-1).squeeze(-1)
        chanelweight_l28=torch.sum(feature_maps_L28,dim=(1,2))/torch.sum(h28,dim=(1,2))
        # print(chanelweight_l28)
        # print(chanelweight_l28.shape)
        #我们用0代替了nan值
        chanelweight_l28 = torch.where(torch.isnan(chanelweight_l28), torch.full_like(chanelweight_l28, 0), chanelweight_l28)
        # chanelweight_l28=chanelweight_l28/torch.sum(highlight_L28)
        # print(chanelweight_l28)
        chanelweight_l28=(chanelweight_l28-torch.min(chanelweight_l28))/(torch.max(chanelweight_l28)-torch.min(chanelweight_l28))
        # # chanelweight_l28=(chanelweight_l28>torch.mean(chanelweight_l28)).float()
        # chanelweight_l28_bool = (chanelweight_l28 > torch.mean(chanelweight_l28)).float()
        # chanelweight_l28_bool_2 = torch.ones_like(chanelweight_l28_bool) - chanelweight_l28_bool
        # chanelweight_l28 = chanelweight_l28_bool_2 * chanelweight_l28
        # chanelweight_l28 = chanelweight_l28 + chanelweight_l28_bool
        # print(chanelweight_l28)



        feature_maps_L31 = features[
            1][0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        feature_maps_L31_channelwise_mean=torch.mean(feature_maps_L31,dim=(1,2))
        highlight_chanelwise_L31 = (feature_maps_L31 > feature_maps_L31_channelwise_mean.view(-1, 1, 1)).float()#得到的结果依然是【512,7,7】的一个二值矩阵
        feature_maps_L31_sum=torch.sum(feature_maps_L31, dim=0)
        highlight_L31 = (feature_maps_L31_sum > torch.mean(feature_maps_L31_sum)).float()
        h31 = highlight_chanelwise_L31 * highlight_L31.unsqueeze(0)
        feature_maps_L31 = feature_maps_L31 * h31
        # chanelweight_l31 = torch.nn.functional.adaptive_max_pool2d(feature_maps_L31, output_size=1).squeeze(-1).squeeze(
        #     -1)
        chanelweight_l31=torch.sum(feature_maps_L31,dim=(1,2))/torch.sum( h31 ,dim=(1,2))
        chanelweight_l31 = torch.where(torch.isnan(chanelweight_l31), torch.full_like(chanelweight_l31, 0), chanelweight_l31)

        # chanelweight_l31=torch.sum(h31,dim=(1,2))
        # chanelweight_l31=chanelweight_l31/torch.sum(highlight_L31)
        chanelweight_l31=(chanelweight_l31-torch.min(chanelweight_l31))/(torch.max(chanelweight_l31)-torch.min(chanelweight_l31))
        # print(chanelweight_l31)

        #
        # chanelweight_l31_bool=(chanelweight_l31>torch.mean(chanelweight_l31)).float()
        # chanelweight_l31_bool_2=torch.ones_like(chanelweight_l31_bool)-chanelweight_l31_bool
        # chanelweight_l31=chanelweight_l31_bool_2 * chanelweight_l31
        # chanelweight_l31=chanelweight_l31 + chanelweight_l31_bool

        ###************
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #
        output_mean=[]
        output_mean_2=[]


        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['features.26.weight','features.28.weight']:
                # tmp=(param * param.grad).cpu().data
                tmp=(param).cpu().data
                # print(tmp.shape)#[512,512,3,3]
                tmp_mean_reuse=torch.mean(tmp,axis=( 2, 3))
                # print(tmp_mean_reuse.shape)#[512,512],
                # 这个矩阵的每一行代表某一层的一组卷积核，这一行是由这一层的【512,3,3】得到的，一个512维的向量
                #这个矩阵一共512行，代表512组的卷积核

                # tmp_mean=torch.mean(tmp_mean_reuse,axis=1).numpy()
                # #这是将模型从右往左的压成一个向量，而不是从上往下对所有的512组卷积核取平均；
                # #再加上直接对测试阶段恒定不变的网络参数进行平均（未借助辅助信息进行加权），自然没用了
                # aa = np.linalg.norm(tmp_mean)
                # if aa != 0:
                #     tmp_mean = tmp_mean / aa
                # else:
                #     tmp_mean = np.zeros_like(tmp_mean)
                #
                # output_mean.append(tmp_mean)
#############################################################3
                #zhelizheli
                if name=='features.26.weight':
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l28.cpu().unsqueeze(0).T
                    # tmp_mean_reuse_direction2 = tmp_mean_reuse_direction2 * chanelweight_l26.cpu().unsqueeze(0)
                    # tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l28.cpu().unsqueeze(0)

                    #【512,512】的矩阵的每一行乘以一个固定的权重，也就是一组卷积核乘以一组固定的权重；这个权重是如何得到的呢?是这样的，该层的一组卷积核一组决定了该层输出特征图的某一个通道
                    #如果能够通过某种方式得到该通道在特征图整体的512个通道中的重要性的话，也就是说，我们得到了该通道的权重，也就是该组卷积核的权重，也就是tmp_mean_reuse 这个矩阵的某一行的权重。
                    #那么这个权重是如何得到的呢？这样的，scda不是得到了一个【h，w】的聚合特征图掩码矩阵嘛；其实我们完全可以得到一个通道级的而非整体的聚合特征图掩码矩阵，也就是一个【512，h，w】
                    #的掩码矩阵，其实就是生成一个通道级均值，特征图的每一个通道，高于该均值的位置掩码为1，否则为0；这就得到了【512，h,w】的掩码矩阵
                    #接下来呢，就是通道级掩码矩阵的每一个通道与整体的[h,w]的掩码进行按位与；最终得到了一个【512，h，w】的处理之后的掩码矩阵；
                    #这就决定了通道级掩码矩阵的每一个通道的每一个为1的位置，都是与图像的主要目标而非背景有关的；我们可以据此进一步的获知输出特征图的每一个通道激活的都是目标还是背景，也就是该通道
                    #进而就是该通道对应的该组滤波器的重要程度
                    #接下来就是最后的一步了，我们将处理之后的【512，h,w】的通道级掩码矩阵与输出特征图进行点乘
                    #进而对于处理后的特征图的每一个通道的不为零的元素们，假设该通道共N个非零值，对这N个值求和在除以N；进而得到了该通道的一个权重
                    #如果某通道所有值都为0的话，就单纯的将该通道的权重设置为0便好啦
                    #最终得到了512个权重（当然啦，还得进行一步0~1的归一化啦），它既是每一个通道的权重，也是每一组卷积核的权重啦
                    #当然这最后一步还有很多可以尝试的空间啦
                else:
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l31.cpu().unsqueeze(0).T
                    # tmp_mean_reuse_direction2 = tmp_mean_reuse_direction2 * chanelweight_l28.cpu().unsqueeze(0)
                    # tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l31.cpu().unsqueeze(0)


                tmp_mean = torch.mean(tmp_mean_reuse_direction2, axis=1).numpy()
                tmp_mean_2 = torch.mean(tmp_mean_reuse_direction2, axis=0).numpy()

                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                bb = np.linalg.norm(tmp_mean_2)
                if bb != 0:
                    tmp_mean_2 = tmp_mean_2 / bb
                else:
                    tmp_mean_2 = np.zeros_like(tmp_mean_2)

                output_mean.append(tmp_mean)
                output_mean_2.append(tmp_mean_2)
########***********************************///////////////////****************+++++++++++


        return  output_mean,output_mean_2









class GradCam_yuhan_kernel_version16_2_9():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, index=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别
        features,output = self.net(input.cuda())
        ###************
        feature_maps_L28 = features[
            1][0]  # 如果输入图像的shape为[batch_size,3,224,224]的tensor的话，关于输出的feature map,shape为  #[batch_size,512,14,14]左右
        # print(feature_maps_L28.shape)
        # print(feature_maps_L28)

        feature_maps_L28_channelwise_mean=torch.mean(feature_maps_L28,dim=(1,2))
        # print(feature_maps_L28_channelwise_mean.shape)
        highlight_chanelwise_L28 = (feature_maps_L28 > feature_maps_L28_channelwise_mean.view(-1, 1, 1)).float()
        # print(highlight_chanelwise_L28.shape)#为这512通道的每一个通道生成了一个（由该通道均值决定的）专属于该通道的二值掩码矩阵，将该通道的显著性区域标注了出来
        feature_maps_L28_sum=torch.sum(feature_maps_L28, dim=0)
        # print(feature_maps_L28_sum.shape)
        highlight_L28 = (feature_maps_L28_sum > torch.mean(feature_maps_L28_sum)).float()
        # print(highlight_L28.shape)
        h28 = highlight_chanelwise_L28 * highlight_L28.unsqueeze(0) #h28是一个【512，h,w】的二值矩阵，这是与总体二值矩阵交集得到的，可能在某一个通道上是一个全零矩阵
        # （一个完全对背景噪声响应的通道）
        # print(h28.shape)
        feature_maps_L28=feature_maps_L28 * h28  #同为shape为【512,7,7】的feature_map 与通道级二值矩阵（总体显著性区域与通道级显著性区域的交集）进行点乘，
        # chanelweight_l28 = torch.nn.functional.adaptive_max_pool2d(feature_maps_L28, output_size=1).squeeze(-1).squeeze(-1)
        chanelweight_l28=torch.sum(feature_maps_L28,dim=(1,2))/torch.sum(h28,dim=(1,2))
        # print(chanelweight_l28)
        # print(chanelweight_l28.shape)
        #我们用0代替了nan值
        chanelweight_l28 = torch.where(torch.isnan(chanelweight_l28), torch.full_like(chanelweight_l28, 0), chanelweight_l28)
        # chanelweight_l28=chanelweight_l28/torch.sum(highlight_L28)
        # print(chanelweight_l28)
        chanelweight_l28=(chanelweight_l28-torch.min(chanelweight_l28))/(torch.max(chanelweight_l28)-torch.min(chanelweight_l28))
        # # chanelweight_l28=(chanelweight_l28>torch.mean(chanelweight_l28)).float()
        # chanelweight_l28_bool = (chanelweight_l28 > torch.mean(chanelweight_l28)).float()
        # chanelweight_l28_bool_2 = torch.ones_like(chanelweight_l28_bool) - chanelweight_l28_bool
        # chanelweight_l28 = chanelweight_l28_bool_2 * chanelweight_l28
        # chanelweight_l28 = chanelweight_l28 + chanelweight_l28_bool



        feature_maps_L31 = features[
            2][0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        feature_maps_L31_channelwise_mean=torch.mean(feature_maps_L31,dim=(1,2))
        highlight_chanelwise_L31 = (feature_maps_L31 > feature_maps_L31_channelwise_mean.view(-1, 1, 1)).float()#得到的结果依然是【512,7,7】的一个二值矩阵
        feature_maps_L31_sum=torch.sum(feature_maps_L31, dim=0)
        highlight_L31 = (feature_maps_L31_sum > torch.mean(feature_maps_L31_sum)).float()
        h31 = highlight_chanelwise_L31 * highlight_L31.unsqueeze(0)
        feature_maps_L31 = feature_maps_L31 * h31
        # chanelweight_l31 = torch.nn.functional.adaptive_max_pool2d(feature_maps_L31, output_size=1).squeeze(-1).squeeze(
        #     -1)
        chanelweight_l31=torch.sum(feature_maps_L31,dim=(1,2))/torch.sum( h31 ,dim=(1,2))
        chanelweight_l31 = torch.where(torch.isnan(chanelweight_l31), torch.full_like(chanelweight_l31, 0), chanelweight_l31)

        # chanelweight_l31=torch.sum(h31,dim=(1,2))
        # chanelweight_l31=chanelweight_l31/torch.sum(highlight_L31)
        chanelweight_l31=(chanelweight_l31-torch.min(chanelweight_l31))/(torch.max(chanelweight_l31)-torch.min(chanelweight_l31))

        #
        # chanelweight_l31_bool=(chanelweight_l31>torch.mean(chanelweight_l31)).float()
        # chanelweight_l31_bool_2=torch.ones_like(chanelweight_l31_bool)-chanelweight_l31_bool
        # chanelweight_l31=chanelweight_l31_bool_2 * chanelweight_l31
        # chanelweight_l31=chanelweight_l31 + chanelweight_l31_bool

        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['features.26.weight']:
                # tmp=(param * param.grad).cpu().data
                tmp = (param).data
                # print(tmp.shape)#[512,512,3,3]
                feature_maps_L26 = features[
                    0]
                filters = tmp.transpose(1,0)#[512,512,3,3]
                inputs = feature_maps_L26
                output_tmp = F.conv2d(inputs, filters, padding=1)
                output_tmp = F.relu(output_tmp)
                feature_maps_L28 = output_tmp[0]
                feature_maps_L28_channelwise_mean = torch.mean(feature_maps_L28, dim=(1, 2))
                # print(feature_maps_L28_channelwise_mean.shape)
                highlight_chanelwise_L28 = (feature_maps_L28 > feature_maps_L28_channelwise_mean.view(-1, 1, 1)).float()
                # print(highlight_chanelwise_L28.shape)#为这512通道的每一个通道生成了一个（由该通道均值决定的）专属于该通道的二值掩码矩阵，将该通道的显著性区域标注了出来
                feature_maps_L28_sum = torch.sum(features[1][0], dim=0)
                # print(feature_maps_L28_sum.shape)
                highlight_L28 = (feature_maps_L28_sum > torch.mean(feature_maps_L28_sum)).float()
                # print(highlight_L28.shape)
                h28 = highlight_chanelwise_L28 * highlight_L28.unsqueeze(
                    0)  # h28是一个【512，h,w】的二值矩阵，这是与总体二值矩阵交集得到的，可能在某一个通道上是一个全零矩阵
                # （一个完全对背景噪声响应的通道）
                # print(h28.shape)
                feature_maps_L28 = feature_maps_L28 * h28  # 同为shape为【512,7,7】的feature_map 与通道级二值矩阵（总体显著性区域与通道级显著性区域的交集）进行点乘，
                # chanelweight_l28 = torch.nn.functional.adaptive_max_pool2d(feature_maps_L28, output_size=1).squeeze(-1).squeeze(-1)
                chanelweight_l28_hori = torch.sum(feature_maps_L28, dim=(1, 2)) / torch.sum(h28, dim=(1, 2))
                # print(chanelweight_l28)
                # print(chanelweight_l28.shape)
                # 我们用0代替了nan值
                chanelweight_l28_hori = torch.where(torch.isnan(chanelweight_l28_hori), torch.full_like(chanelweight_l28_hori, 0),
                                               chanelweight_l28_hori)
                # chanelweight_l28=chanelweight_l28/torch.sum(highlight_L28)
                # print(chanelweight_l28)
                chanelweight_l28_hori = (chanelweight_l28_hori - torch.min(chanelweight_l28_hori)) / (
                            torch.max(chanelweight_l28_hori) - torch.min(chanelweight_l28_hori))


            if name in ['features.28.weight']:
                # tmp=(param * param.grad).cpu().data
                tmp = (param).data
                # print(tmp.shape)#[512,512,3,3]
                feature_maps_L28 = features[
                    1]
                filters = tmp.transpose(1,0)#[512,512,3,3]
                inputs = feature_maps_L28
                output_tmp = F.conv2d(inputs, filters, padding=1)
                output_tmp = F.relu(output_tmp)
                feature_maps_L31 = output_tmp[0]
                feature_maps_L31_channelwise_mean = torch.mean(feature_maps_L31, dim=(1, 2))
                # print(feature_maps_L28_channelwise_mean.shape)
                highlight_chanelwise_L31 = (feature_maps_L31 > feature_maps_L31_channelwise_mean.view(-1, 1, 1)).float()
                # print(highlight_chanelwise_L28.shape)#为这512通道的每一个通道生成了一个（由该通道均值决定的）专属于该通道的二值掩码矩阵，将该通道的显著性区域标注了出来
                feature_maps_L31_sum = torch.sum(features[2][0], dim=0)
                # print(feature_maps_L28_sum.shape)
                highlight_L31 = (feature_maps_L31_sum > torch.mean(feature_maps_L31_sum)).float()
                # print(highlight_L28.shape)
                h31 = highlight_chanelwise_L31 * highlight_L31.unsqueeze(
                    0)  # h28是一个【512，h,w】的二值矩阵，这是与总体二值矩阵交集得到的，可能在某一个通道上是一个全零矩阵
                # （一个完全对背景噪声响应的通道）
                # print(h28.shape)
                feature_maps_L31 = feature_maps_L31 * h31  # 同为shape为【512,7,7】的feature_map 与通道级二值矩阵（总体显著性区域与通道级显著性区域的交集）进行点乘，
                # chanelweight_l28 = torch.nn.functional.adaptive_max_pool2d(feature_maps_L28, output_size=1).squeeze(-1).squeeze(-1)
                chanelweight_l31_hori = torch.sum(feature_maps_L31, dim=(1, 2)) / torch.sum(h31, dim=(1, 2))
                # print(chanelweight_l28)
                # print(chanelweight_l28.shape)
                # 我们用0代替了nan值
                chanelweight_l31_hori = torch.where(torch.isnan(chanelweight_l31_hori), torch.full_like(chanelweight_l31_hori, 0),
                                               chanelweight_l31_hori)
                # chanelweight_l28=chanelweight_l28/torch.sum(highlight_L28)
                # print(chanelweight_l28)
                chanelweight_l31_hori = (chanelweight_l31_hori - torch.min(chanelweight_l31_hori)) / (
                            torch.max(chanelweight_l31_hori) - torch.min(chanelweight_l31_hori))

        ###************
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #
        output_mean=[]


        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['features.26.weight','features.28.weight']:
                # tmp=(param * param.grad).cpu().data
                tmp=(param).cpu().data
                # print(tmp.shape)#[512,512,3,3]
                tmp_mean_reuse=torch.mean(tmp,axis=( 2, 3))
                # print(tmp_mean_reuse.shape)#[512,512],

                if name=='features.26.weight':
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l28_hori.cpu().unsqueeze(0)
                    #【512,512】的矩阵的每一行乘以一个固定的权重，也就是一组卷积核乘以一组固定的权重；这个权重是如何得到的呢?是这样的，该层的一组卷积核一组决定了该层输出特征图的某一个通道
                    #如果能够通过某种方式得到该通道在特征图整体的512个通道中的重要性的话，也就是说，我们得到了该通道的权重，也就是该组卷积核的权重，也就是tmp_mean_reuse 这个矩阵的某一行的权重。
                    #那么这个权重是如何得到的呢？这样的，scda不是得到了一个【h，w】的聚合特征图掩码矩阵嘛；其实我们完全可以得到一个通道级的而非整体的聚合特征图掩码矩阵，也就是一个【512，h，w】
                    #的掩码矩阵，其实就是生成一个通道级均值，特征图的每一个通道，高于该均值的位置掩码为1，否则为0；这就得到了【512，h,w】的掩码矩阵
                    #接下来呢，就是通道级掩码矩阵的每一个通道与整体的[h,w]的掩码进行按位与；最终得到了一个【512，h，w】的处理之后的掩码矩阵；
                    #这就决定了通道级掩码矩阵的每一个通道的每一个为1的位置，都是与图像的主要目标而非背景有关的；我们可以据此进一步的获知输出特征图的每一个通道激活的都是目标还是背景，也就是该通道
                    #进而就是该通道对应的该组滤波器的重要程度
                    #接下来就是最后的一步了，我们将处理之后的【512，h,w】的通道级掩码矩阵与输出特征图进行点乘
                    #进而对于处理后的特征图的每一个通道的不为零的元素们，假设该通道共N个非零值，对这N个值求和在除以N；进而得到了该通道的一个权重
                    #如果某通道所有值都为0的话，就单纯的将该通道的权重设置为0便好啦
                    #最终得到了512个权重（当然啦，还得进行一步0~1的归一化啦），它既是每一个通道的权重，也是每一组卷积核的权重啦
                    #当然这最后一步还有很多可以尝试的空间啦
                else:
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l31_hori.cpu().unsqueeze(0)


                tmp_mean = torch.mean(tmp_mean_reuse_direction2, axis=0).numpy()
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean.append(tmp_mean)
########***********************************///////////////////****************+++++++++++


        return  output_mean









class GradCam_yuhan_kernel_version16_2_8():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, index=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别
        features,output = self.net(input.cuda())
        ###************
        feature_maps_L28 = features[
            0][0]  # 如果输入图像的shape为[batch_size,3,224,224]的tensor的话，关于输出的feature map,shape为  #[batch_size,512,14,14]左右
        # print(feature_maps_L28.shape)
        # print(feature_maps_L28)

        feature_maps_L28_channelwise_mean=torch.mean(feature_maps_L28,dim=(1,2))
        # print(feature_maps_L28_channelwise_mean.shape)
        highlight_chanelwise_L28 = (feature_maps_L28 > feature_maps_L28_channelwise_mean.view(-1, 1, 1)).float()
        # print(highlight_chanelwise_L28.shape)#为这512通道的每一个通道生成了一个（由该通道均值决定的）专属于该通道的二值掩码矩阵，将该通道的显著性区域标注了出来
        feature_maps_L28_sum=torch.sum(feature_maps_L28, dim=0)
        # print(feature_maps_L28_sum.shape)
        highlight_L28 = (feature_maps_L28_sum > torch.mean(feature_maps_L28_sum)).float()
        # print(highlight_L28.shape)
        h28 = highlight_chanelwise_L28 * highlight_L28.unsqueeze(0) #h28是一个【512，h,w】的二值矩阵，这是与总体二值矩阵交集得到的，可能在某一个通道上是一个全零矩阵
        # （一个完全对背景噪声响应的通道）
        # print(h28.shape)
        feature_maps_L28=feature_maps_L28 * h28  #同为shape为【512,7,7】的feature_map 与通道级二值矩阵（总体显著性区域与通道级显著性区域的交集）进行点乘，
        # chanelweight_l28 = torch.nn.functional.adaptive_max_pool2d(feature_maps_L28, output_size=1).squeeze(-1).squeeze(-1)
        chanelweight_l28=torch.sum(feature_maps_L28,dim=(1,2))/torch.sum(h28,dim=(1,2))
        # print(chanelweight_l28)
        # print(chanelweight_l28.shape)
        #我们用0代替了nan值
        chanelweight_l28 = torch.where(torch.isnan(chanelweight_l28), torch.full_like(chanelweight_l28, 0), chanelweight_l28)
        # chanelweight_l28=chanelweight_l28/torch.sum(highlight_L28)
        # print(chanelweight_l28)
        chanelweight_l28=(chanelweight_l28-torch.min(chanelweight_l28))/(torch.max(chanelweight_l28)-torch.min(chanelweight_l28))
        # # chanelweight_l28=(chanelweight_l28>torch.mean(chanelweight_l28)).float()
        # chanelweight_l28_bool = (chanelweight_l28 > torch.mean(chanelweight_l28)).float()
        # chanelweight_l28_bool_2 = torch.ones_like(chanelweight_l28_bool) - chanelweight_l28_bool
        # chanelweight_l28 = chanelweight_l28_bool_2 * chanelweight_l28
        # chanelweight_l28 = chanelweight_l28 + chanelweight_l28_bool



        feature_maps_L31 = features[
            1][0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        feature_maps_L31_channelwise_mean=torch.mean(feature_maps_L31,dim=(1,2))
        highlight_chanelwise_L31 = (feature_maps_L31 > feature_maps_L31_channelwise_mean.view(-1, 1, 1)).float()#得到的结果依然是【512,7,7】的一个二值矩阵
        feature_maps_L31_sum=torch.sum(feature_maps_L31, dim=0)
        highlight_L31 = (feature_maps_L31_sum > torch.mean(feature_maps_L31_sum)).float()
        h31 = highlight_chanelwise_L31 * highlight_L31.unsqueeze(0)
        feature_maps_L31 = feature_maps_L31 * h31
        # chanelweight_l31 = torch.nn.functional.adaptive_max_pool2d(feature_maps_L31, output_size=1).squeeze(-1).squeeze(
        #     -1)
        chanelweight_l31=torch.sum(feature_maps_L31,dim=(1,2))/torch.sum( h31 ,dim=(1,2))
        chanelweight_l31 = torch.where(torch.isnan(chanelweight_l31), torch.full_like(chanelweight_l31, 0), chanelweight_l31)

        # chanelweight_l31=torch.sum(h31,dim=(1,2))
        # chanelweight_l31=chanelweight_l31/torch.sum(highlight_L31)
        chanelweight_l31=(chanelweight_l31-torch.min(chanelweight_l31))/(torch.max(chanelweight_l31)-torch.min(chanelweight_l31))

        #
        # chanelweight_l31_bool=(chanelweight_l31>torch.mean(chanelweight_l31)).float()
        # chanelweight_l31_bool_2=torch.ones_like(chanelweight_l31_bool)-chanelweight_l31_bool
        # chanelweight_l31=chanelweight_l31_bool_2 * chanelweight_l31
        # chanelweight_l31=chanelweight_l31 + chanelweight_l31_bool

        ###************
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #
        output_mean=[]


        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['features.26.weight','features.28.weight']:
                # tmp=(param * param.grad).cpu().data
                tmp=(param).cpu().data
                # print(tmp.shape)#[512,512,3,3]
                tmp_mean_reuse=torch.mean(tmp,axis=( 2, 3))
                # print(tmp_mean_reuse.shape)#[512,512],
                # 这个矩阵的每一行代表某一层的一组卷积核，这一行是由这一层的【512,3,3】得到的，一个512维的向量
                #这个矩阵一共512行，代表512组的卷积核

                # tmp_mean=torch.mean(tmp_mean_reuse,axis=1).numpy()
                # #这是将模型从右往左的压成一个向量，而不是从上往下对所有的512组卷积核取平均；
                # #再加上直接对测试阶段恒定不变的网络参数进行平均（未借助辅助信息进行加权），自然没用了
                # aa = np.linalg.norm(tmp_mean)
                # if aa != 0:
                #     tmp_mean = tmp_mean / aa
                # else:
                #     tmp_mean = np.zeros_like(tmp_mean)
                #
                # output_mean.append(tmp_mean)
#############################################################3
                #zhelizheli
                if name=='features.26.weight':
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l28.cpu().unsqueeze(0).T
                    #【512,512】的矩阵的每一行乘以一个固定的权重，也就是一组卷积核乘以一组固定的权重；这个权重是如何得到的呢?是这样的，该层的一组卷积核一组决定了该层输出特征图的某一个通道
                    #如果能够通过某种方式得到该通道在特征图整体的512个通道中的重要性的话，也就是说，我们得到了该通道的权重，也就是该组卷积核的权重，也就是tmp_mean_reuse 这个矩阵的某一行的权重。
                    #那么这个权重是如何得到的呢？这样的，scda不是得到了一个【h，w】的聚合特征图掩码矩阵嘛；其实我们完全可以得到一个通道级的而非整体的聚合特征图掩码矩阵，也就是一个【512，h，w】
                    #的掩码矩阵，其实就是生成一个通道级均值，特征图的每一个通道，高于该均值的位置掩码为1，否则为0；这就得到了【512，h,w】的掩码矩阵
                    #接下来呢，就是通道级掩码矩阵的每一个通道与整体的[h,w]的掩码进行按位与；最终得到了一个【512，h，w】的处理之后的掩码矩阵；
                    #这就决定了通道级掩码矩阵的每一个通道的每一个为1的位置，都是与图像的主要目标而非背景有关的；我们可以据此进一步的获知输出特征图的每一个通道激活的都是目标还是背景，也就是该通道
                    #进而就是该通道对应的该组滤波器的重要程度
                    #接下来就是最后的一步了，我们将处理之后的【512，h,w】的通道级掩码矩阵与输出特征图进行点乘
                    #进而对于处理后的特征图的每一个通道的不为零的元素们，假设该通道共N个非零值，对这N个值求和在除以N；进而得到了该通道的一个权重
                    #如果某通道所有值都为0的话，就单纯的将该通道的权重设置为0便好啦
                    #最终得到了512个权重（当然啦，还得进行一步0~1的归一化啦），它既是每一个通道的权重，也是每一组卷积核的权重啦
                    #当然这最后一步还有很多可以尝试的空间啦
                else:
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l31.cpu().unsqueeze(0).T


                tmp_mean,_ = torch.max(tmp_mean_reuse_direction2, axis=1)
                tmp_mean=tmp_mean.numpy()
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean.append(tmp_mean)
########***********************************///////////////////****************+++++++++++


        return  output_mean










class GradCam_yuhan_kernel_version16_2_7():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, index=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别
        features,output = self.net(input.cuda())
        ###************
        feature_maps_L28 = features[
            0][0]  # 如果输入图像的shape为[batch_size,3,224,224]的tensor的话，关于输出的feature map,shape为  #[batch_size,512,14,14]左右
        # print(feature_maps_L28.shape)
        # print(feature_maps_L28)

        feature_maps_L28_channelwise_mean=torch.mean(feature_maps_L28,dim=(1,2))
        # print(feature_maps_L28_channelwise_mean.shape)
        highlight_chanelwise_L28 = (feature_maps_L28 > feature_maps_L28_channelwise_mean.view(-1, 1, 1)).float()
        # print(highlight_chanelwise_L28.shape)#为这512通道的每一个通道生成了一个（由该通道均值决定的）专属于该通道的二值掩码矩阵，将该通道的显著性区域标注了出来
        feature_maps_L28_sum=torch.sum(feature_maps_L28, dim=0)
        # print(feature_maps_L28_sum.shape)
        highlight_L28 = (feature_maps_L28_sum > torch.mean(feature_maps_L28_sum)).float()
        # print(highlight_L28.shape)
        h28 = highlight_chanelwise_L28 * highlight_L28.unsqueeze(0) #h28是一个【512，h,w】的二值矩阵，这是与总体二值矩阵交集得到的，可能在某一个通道上是一个全零矩阵
        # （一个完全对背景噪声响应的通道）
        # print(h28.shape)
        feature_maps_L28=feature_maps_L28 * h28  #同为shape为【512,7,7】的feature_map 与通道级二值矩阵（总体显著性区域与通道级显著性区域的交集）进行点乘，
        # chanelweight_l28 = torch.nn.functional.adaptive_max_pool2d(feature_maps_L28, output_size=1).squeeze(-1).squeeze(-1)
        chanelweight_l28=torch.sum(feature_maps_L28,dim=(1,2))/torch.sum(h28,dim=(1,2))
        # print(chanelweight_l28)
        # print(chanelweight_l28.shape)
        #我们用0代替了nan值
        chanelweight_l28 = torch.where(torch.isnan(chanelweight_l28), torch.full_like(chanelweight_l28, 0), chanelweight_l28)
        # chanelweight_l28=chanelweight_l28/torch.sum(highlight_L28)
        # print(chanelweight_l28)
        chanelweight_l28=(chanelweight_l28-torch.min(chanelweight_l28))/(torch.max(chanelweight_l28)-torch.min(chanelweight_l28))
        # # chanelweight_l28=(chanelweight_l28>torch.mean(chanelweight_l28)).float()
        # chanelweight_l28_bool = (chanelweight_l28 > torch.mean(chanelweight_l28)).float()
        # chanelweight_l28_bool_2 = torch.ones_like(chanelweight_l28_bool) - chanelweight_l28_bool
        # chanelweight_l28 = chanelweight_l28_bool_2 * chanelweight_l28
        # chanelweight_l28 = chanelweight_l28 + chanelweight_l28_bool



        feature_maps_L31 = features[
            1][0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        feature_maps_L31_channelwise_mean=torch.mean(feature_maps_L31,dim=(1,2))
        highlight_chanelwise_L31 = (feature_maps_L31 > feature_maps_L31_channelwise_mean.view(-1, 1, 1)).float()#得到的结果依然是【512,7,7】的一个二值矩阵
        feature_maps_L31_sum=torch.sum(feature_maps_L31, dim=0)
        highlight_L31 = (feature_maps_L31_sum > torch.mean(feature_maps_L31_sum)).float()
        h31 = highlight_chanelwise_L31 * highlight_L31.unsqueeze(0)
        feature_maps_L31 = feature_maps_L31 * h31
        # chanelweight_l31 = torch.nn.functional.adaptive_max_pool2d(feature_maps_L31, output_size=1).squeeze(-1).squeeze(
        #     -1)
        chanelweight_l31=torch.sum(feature_maps_L31,dim=(1,2))/torch.sum( h31 ,dim=(1,2))
        chanelweight_l31 = torch.where(torch.isnan(chanelweight_l31), torch.full_like(chanelweight_l31, 0), chanelweight_l31)

        # chanelweight_l31=torch.sum(h31,dim=(1,2))
        # chanelweight_l31=chanelweight_l31/torch.sum(highlight_L31)
        chanelweight_l31=(chanelweight_l31-torch.min(chanelweight_l31))/(torch.max(chanelweight_l31)-torch.min(chanelweight_l31))

        #
        # chanelweight_l31_bool=(chanelweight_l31>torch.mean(chanelweight_l31)).float()
        # chanelweight_l31_bool_2=torch.ones_like(chanelweight_l31_bool)-chanelweight_l31_bool
        # chanelweight_l31=chanelweight_l31_bool_2 * chanelweight_l31
        # chanelweight_l31=chanelweight_l31 + chanelweight_l31_bool

        ###************
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #
        output_mean=[]


        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['features.26.weight','features.28.weight']:
                # tmp=(param * param.grad).cpu().data
                tmp=(param).cpu().data
                # print(tmp.shape)#[512,512,3,3]
                # tmp_mean_reuse=torch.mean(tmp,axis=( 2, 3))
                tmp_mean_reuse=tmp.view(tmp.size()[0],-1)

                # print(tmp_mean_reuse.shape)#[512,512],
                # 这个矩阵的每一行代表某一层的一组卷积核，这一行是由这一层的【512,3,3】得到的，一个512维的向量
                #这个矩阵一共512行，代表512组的卷积核

                # tmp_mean=torch.mean(tmp_mean_reuse,axis=1).numpy()
                # #这是将模型从右往左的压成一个向量，而不是从上往下对所有的512组卷积核取平均；
                # #再加上直接对测试阶段恒定不变的网络参数进行平均（未借助辅助信息进行加权），自然没用了
                # aa = np.linalg.norm(tmp_mean)
                # if aa != 0:
                #     tmp_mean = tmp_mean / aa
                # else:
                #     tmp_mean = np.zeros_like(tmp_mean)
                #
                # output_mean.append(tmp_mean)
#############################################################3
                #zhelizheli
                if name=='features.26.weight':
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l28.cpu().unsqueeze(0).T
                    #【512,512】的矩阵的每一行乘以一个固定的权重，也就是一组卷积核乘以一组固定的权重；这个权重是如何得到的呢?是这样的，该层的一组卷积核一组决定了该层输出特征图的某一个通道
                    #如果能够通过某种方式得到该通道在特征图整体的512个通道中的重要性的话，也就是说，我们得到了该通道的权重，也就是该组卷积核的权重，也就是tmp_mean_reuse 这个矩阵的某一行的权重。
                    #那么这个权重是如何得到的呢？这样的，scda不是得到了一个【h，w】的聚合特征图掩码矩阵嘛；其实我们完全可以得到一个通道级的而非整体的聚合特征图掩码矩阵，也就是一个【512，h，w】
                    #的掩码矩阵，其实就是生成一个通道级均值，特征图的每一个通道，高于该均值的位置掩码为1，否则为0；这就得到了【512，h,w】的掩码矩阵
                    #接下来呢，就是通道级掩码矩阵的每一个通道与整体的[h,w]的掩码进行按位与；最终得到了一个【512，h，w】的处理之后的掩码矩阵；
                    #这就决定了通道级掩码矩阵的每一个通道的每一个为1的位置，都是与图像的主要目标而非背景有关的；我们可以据此进一步的获知输出特征图的每一个通道激活的都是目标还是背景，也就是该通道
                    #进而就是该通道对应的该组滤波器的重要程度
                    #接下来就是最后的一步了，我们将处理之后的【512，h,w】的通道级掩码矩阵与输出特征图进行点乘
                    #进而对于处理后的特征图的每一个通道的不为零的元素们，假设该通道共N个非零值，对这N个值求和在除以N；进而得到了该通道的一个权重
                    #如果某通道所有值都为0的话，就单纯的将该通道的权重设置为0便好啦
                    #最终得到了512个权重（当然啦，还得进行一步0~1的归一化啦），它既是每一个通道的权重，也是每一组卷积核的权重啦
                    #当然这最后一步还有很多可以尝试的空间啦
                else:
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l31.cpu().unsqueeze(0).T


                tmp_mean = torch.mean(tmp_mean_reuse_direction2, axis=0).numpy()
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean.append(tmp_mean)
########***********************************///////////////////****************+++++++++++


        return  output_mean








class GradCam_yuhan_kernel_version16_2_5():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, mode='mean'):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别
        features,output = self.net(input.cuda())
        ###************
        feature_maps_L28 = features[
            0][0]  # 如果输入图像的shape为[batch_size,3,224,224]的tensor的话，关于输出的feature map,shape为  #[batch_size,512,14,14]左右
        # print(feature_maps_L28.shape)
        # print(feature_maps_L28)

        feature_maps_L28_channelwise_mean=torch.mean(feature_maps_L28,dim=(1,2))
        # print(feature_maps_L28_channelwise_mean.shape)
        highlight_chanelwise_L28 = (feature_maps_L28 > feature_maps_L28_channelwise_mean.view(-1, 1, 1)).float()
        # print(highlight_chanelwise_L28.shape)#为这512通道的每一个通道生成了一个（由该通道均值决定的）专属于该通道的二值掩码矩阵，将该通道的显著性区域标注了出来
        feature_maps_L28_sum=torch.sum(feature_maps_L28, dim=0)
        # print(feature_maps_L28_sum.shape)
        highlight_L28 = (feature_maps_L28_sum > torch.mean(feature_maps_L28_sum)).float()
        # print(highlight_L28.shape)
        h28 = highlight_chanelwise_L28 * highlight_L28.unsqueeze(0) #h28是一个【512，h,w】的二值矩阵，这是与总体二值矩阵交集得到的，可能在某一个通道上是一个全零矩阵
        # （一个完全对背景噪声响应的通道）
        # print(h28.shape)
        feature_maps_L28=feature_maps_L28 * h28  #同为shape为【512,7,7】的feature_map 与通道级二值矩阵（总体显著性区域与通道级显著性区域的交集）进行点乘，
        # chanelweight_l28 = torch.nn.functional.adaptive_max_pool2d(feature_maps_L28, output_size=1).squeeze(-1).squeeze(-1)
        chanelweight_l28=torch.sum(feature_maps_L28,dim=(1,2))/torch.sum(h28,dim=(1,2))
        # print(chanelweight_l28)
        # print(chanelweight_l28.shape)
        #我们用0代替了nan值
        chanelweight_l28 = torch.where(torch.isnan(chanelweight_l28), torch.full_like(chanelweight_l28, 0), chanelweight_l28)
        chanelweight_l28_tmp=chanelweight_l28.cpu().data.numpy()
        aa = np.linalg.norm(chanelweight_l28_tmp)
        if aa != 0:
            chanelweight_l28_norm = chanelweight_l28_tmp / aa
        else:
            chanelweight_l28_norm = np.zeros_like(chanelweight_l28_tmp)

        # chanelweight_l28=chanelweight_l28/torch.sum(highlight_L28)
        # print(chanelweight_l28)
        chanelweight_l28=(chanelweight_l28-torch.min(chanelweight_l28))/(torch.max(chanelweight_l28)-torch.min(chanelweight_l28))
        # # chanelweight_l28=(chanelweight_l28>torch.mean(chanelweight_l28)).float()
        # chanelweight_l28_bool = (chanelweight_l28 > torch.mean(chanelweight_l28)).float()
        # chanelweight_l28_bool_2 = torch.ones_like(chanelweight_l28_bool) - chanelweight_l28_bool
        # chanelweight_l28 = chanelweight_l28_bool_2 * chanelweight_l28
        # chanelweight_l28 = chanelweight_l28 + chanelweight_l28_bool



        feature_maps_L31 = features[
            1][0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        feature_maps_L31_channelwise_mean=torch.mean(feature_maps_L31,dim=(1,2))
        highlight_chanelwise_L31 = (feature_maps_L31 > feature_maps_L31_channelwise_mean.view(-1, 1, 1)).float()#得到的结果依然是【512,7,7】的一个二值矩阵
        feature_maps_L31_sum=torch.sum(feature_maps_L31, dim=0)
        highlight_L31 = (feature_maps_L31_sum > torch.mean(feature_maps_L31_sum)).float()
        h31 = highlight_chanelwise_L31 * highlight_L31.unsqueeze(0)
        feature_maps_L31 = feature_maps_L31 * h31
        # chanelweight_l31 = torch.nn.functional.adaptive_max_pool2d(feature_maps_L31, output_size=1).squeeze(-1).squeeze(
        #     -1)
        chanelweight_l31=torch.sum(feature_maps_L31,dim=(1,2))/torch.sum( h31 ,dim=(1,2))
        chanelweight_l31 = torch.where(torch.isnan(chanelweight_l31), torch.full_like(chanelweight_l31, 0), chanelweight_l31)
        chanelweight_l31_tmp=chanelweight_l31.cpu().numpy()

        aa = np.linalg.norm(chanelweight_l31_tmp)
        if aa != 0:
            chanelweight_l31_norm = chanelweight_l31_tmp / aa
        else:
            chanelweight_l31_norm = np.zeros_like(chanelweight_l31_tmp)

        # chanelweight_l31=torch.sum(h31,dim=(1,2))
        # chanelweight_l31=chanelweight_l31/torch.sum(highlight_L31)
        chanelweight_l31=(chanelweight_l31-torch.min(chanelweight_l31))/(torch.max(chanelweight_l31)-torch.min(chanelweight_l31))

        #
        # chanelweight_l31_bool=(chanelweight_l31>torch.mean(chanelweight_l31)).float()
        # chanelweight_l31_bool_2=torch.ones_like(chanelweight_l31_bool)-chanelweight_l31_bool
        # chanelweight_l31=chanelweight_l31_bool_2 * chanelweight_l31
        # chanelweight_l31=chanelweight_l31 + chanelweight_l31_bool

        ###************
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #
        output_mean=[]


        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['features.26.weight','features.28.weight']:
                # tmp=(param * param.grad).cpu().data
                tmp=(param).cpu().data
                # print(tmp.shape)#[512,512,3,3]
                tmp_mean_reuse=torch.mean(tmp,axis=( 2, 3))
                # print(tmp_mean_reuse.shape)#[512,512],
                # 这个矩阵的每一行代表某一层的一组卷积核，这一行是由这一层的【512,3,3】得到的，一个512维的向量
                #这个矩阵一共512行，代表512组的卷积核

                # tmp_mean=torch.mean(tmp_mean_reuse,axis=1).numpy()
                # #这是将模型从右往左的压成一个向量，而不是从上往下对所有的512组卷积核取平均；
                # #再加上直接对测试阶段恒定不变的网络参数进行平均（未借助辅助信息进行加权），自然没用了
                # aa = np.linalg.norm(tmp_mean)
                # if aa != 0:
                #     tmp_mean = tmp_mean / aa
                # else:
                #     tmp_mean = np.zeros_like(tmp_mean)
                #
                # output_mean.append(tmp_mean)
#############################################################3
                #zhelizheli
                if name=='features.26.weight':
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l28.cpu().unsqueeze(0).T
                    #【512,512】的矩阵的每一行乘以一个固定的权重，也就是一组卷积核乘以一组固定的权重；这个权重是如何得到的呢?是这样的，该层的一组卷积核一组决定了该层输出特征图的某一个通道
                    #如果能够通过某种方式得到该通道在特征图整体的512个通道中的重要性的话，也就是说，我们得到了该通道的权重，也就是该组卷积核的权重，也就是tmp_mean_reuse 这个矩阵的某一行的权重。
                    #那么这个权重是如何得到的呢？这样的，scda不是得到了一个【h，w】的聚合特征图掩码矩阵嘛；其实我们完全可以得到一个通道级的而非整体的聚合特征图掩码矩阵，也就是一个【512，h，w】
                    #的掩码矩阵，其实就是生成一个通道级均值，特征图的每一个通道，高于该均值的位置掩码为1，否则为0；这就得到了【512，h,w】的掩码矩阵
                    #接下来呢，就是通道级掩码矩阵的每一个通道与整体的[h,w]的掩码进行按位与；最终得到了一个【512，h，w】的处理之后的掩码矩阵；
                    #这就决定了通道级掩码矩阵的每一个通道的每一个为1的位置，都是与图像的主要目标而非背景有关的；我们可以据此进一步的获知输出特征图的每一个通道激活的都是目标还是背景，也就是该通道
                    #进而就是该通道对应的该组滤波器的重要程度
                    #接下来就是最后的一步了，我们将处理之后的【512，h,w】的通道级掩码矩阵与输出特征图进行点乘
                    #进而对于处理后的特征图的每一个通道的不为零的元素们，假设该通道共N个非零值，对这N个值求和在除以N；进而得到了该通道的一个权重
                    #如果某通道所有值都为0的话，就单纯的将该通道的权重设置为0便好啦
                    #最终得到了512个权重（当然啦，还得进行一步0~1的归一化啦），它既是每一个通道的权重，也是每一组卷积核的权重啦
                    #当然这最后一步还有很多可以尝试的空间啦
                else:
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l31.cpu().unsqueeze(0).T


                tmp_mean = torch.mean(tmp_mean_reuse_direction2, axis=1).numpy()
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean.append(tmp_mean)
########***********************************///////////////////****************+++++++++++


        return  output_mean,chanelweight_l28_norm,chanelweight_l31_norm











class GradCam_yuhan_kernel_version16_2_6():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, mode='mean'):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别
        features,output = self.net(input.cuda())
        ###************
        feature_maps_L28 = features[
            0][0]  # 如果输入图像的shape为[batch_size,3,224,224]的tensor的话，关于输出的feature map,shape为  #[batch_size,512,14,14]左右
        # print(feature_maps_L28.shape)
        # print(feature_maps_L28)

        feature_maps_L28_channelwise_mean=torch.mean(feature_maps_L28,dim=(1,2))
        # print(feature_maps_L28_channelwise_mean.shape)
        highlight_chanelwise_L28 = (feature_maps_L28 > feature_maps_L28_channelwise_mean.view(-1, 1, 1)).float()
        # print(highlight_chanelwise_L28.shape)#为这512通道的每一个通道生成了一个（由该通道均值决定的）专属于该通道的二值掩码矩阵，将该通道的显著性区域标注了出来
        feature_maps_L28_sum=torch.sum(feature_maps_L28, dim=0)
        # print(feature_maps_L28_sum.shape)
        highlight_L28 = (feature_maps_L28_sum > torch.mean(feature_maps_L28_sum)).float()
        # print(highlight_L28.shape)
        h28 = highlight_chanelwise_L28 * highlight_L28.unsqueeze(0) #h28是一个【512，h,w】的二值矩阵，这是与总体二值矩阵交集得到的，可能在某一个通道上是一个全零矩阵
        # （一个完全对背景噪声响应的通道）
        # print(h28.shape)
        feature_maps_L28=feature_maps_L28 * h28  #同为shape为【512,7,7】的feature_map 与通道级二值矩阵（总体显著性区域与通道级显著性区域的交集）进行点乘，
        # chanelweight_l28 = torch.nn.functional.adaptive_max_pool2d(feature_maps_L28, output_size=1).squeeze(-1).squeeze(-1)
        # chanelweight_l28=torch.sum(feature_maps_L28,dim=(1,2))/torch.sum(h28,dim=(1,2))
        chanelweight_l28=torch.nn.functional.adaptive_max_pool2d(feature_maps_L28,output_size=1).view(-1,)
        # print(chanelweight_l28)
        # print(chanelweight_l28.shape)
        #我们用0代替了nan值
        chanelweight_l28 = torch.where(torch.isnan(chanelweight_l28), torch.full_like(chanelweight_l28, 0), chanelweight_l28)
        chanelweight_l28_tmp=chanelweight_l28.cpu().data.numpy()
        aa = np.linalg.norm(chanelweight_l28_tmp)
        if aa != 0:
            chanelweight_l28_norm = chanelweight_l28_tmp / aa
        else:
            chanelweight_l28_norm = np.zeros_like(chanelweight_l28_tmp)

        # chanelweight_l28=chanelweight_l28/torch.sum(highlight_L28)
        # print(chanelweight_l28)
        chanelweight_l28=(chanelweight_l28-torch.min(chanelweight_l28))/(torch.max(chanelweight_l28)-torch.min(chanelweight_l28))
        # # chanelweight_l28=(chanelweight_l28>torch.mean(chanelweight_l28)).float()
        # chanelweight_l28_bool = (chanelweight_l28 > torch.mean(chanelweight_l28)).float()
        # chanelweight_l28_bool_2 = torch.ones_like(chanelweight_l28_bool) - chanelweight_l28_bool
        # chanelweight_l28 = chanelweight_l28_bool_2 * chanelweight_l28
        # chanelweight_l28 = chanelweight_l28 + chanelweight_l28_bool



        feature_maps_L31 = features[
            1][0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        feature_maps_L31_channelwise_mean=torch.mean(feature_maps_L31,dim=(1,2))
        highlight_chanelwise_L31 = (feature_maps_L31 > feature_maps_L31_channelwise_mean.view(-1, 1, 1)).float()#得到的结果依然是【512,7,7】的一个二值矩阵
        feature_maps_L31_sum=torch.sum(feature_maps_L31, dim=0)
        highlight_L31 = (feature_maps_L31_sum > torch.mean(feature_maps_L31_sum)).float()
        h31 = highlight_chanelwise_L31 * highlight_L31.unsqueeze(0)
        feature_maps_L31 = feature_maps_L31 * h31
        # chanelweight_l31 = torch.nn.functional.adaptive_max_pool2d(feature_maps_L31, output_size=1).squeeze(-1).squeeze(
        #     -1)
        # chanelweight_l31=torch.sum(feature_maps_L31,dim=(1,2))/torch.sum( h31 ,dim=(1,2))
        chanelweight_l31=torch.nn.functional.adaptive_max_pool2d(feature_maps_L31,output_size=1).view(-1,)

        chanelweight_l31 = torch.where(torch.isnan(chanelweight_l31), torch.full_like(chanelweight_l31, 0), chanelweight_l31)
        chanelweight_l31_tmp=chanelweight_l31.cpu().numpy()

        aa = np.linalg.norm(chanelweight_l31_tmp)
        if aa != 0:
            chanelweight_l31_norm = chanelweight_l31_tmp / aa
        else:
            chanelweight_l31_norm = np.zeros_like(chanelweight_l31_tmp)

        # chanelweight_l31=torch.sum(h31,dim=(1,2))
        # chanelweight_l31=chanelweight_l31/torch.sum(highlight_L31)
        chanelweight_l31=(chanelweight_l31-torch.min(chanelweight_l31))/(torch.max(chanelweight_l31)-torch.min(chanelweight_l31))

        #
        # chanelweight_l31_bool=(chanelweight_l31>torch.mean(chanelweight_l31)).float()
        # chanelweight_l31_bool_2=torch.ones_like(chanelweight_l31_bool)-chanelweight_l31_bool
        # chanelweight_l31=chanelweight_l31_bool_2 * chanelweight_l31
        # chanelweight_l31=chanelweight_l31 + chanelweight_l31_bool

        ###************
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #
        output_mean=[]


        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['features.26.weight','features.28.weight']:
                # tmp=(param * param.grad).cpu().data
                tmp=(param).cpu().data
                # print(tmp.shape)#[512,512,3,3]
                tmp_mean_reuse=torch.mean(tmp,axis=( 2, 3))
                # print(tmp_mean_reuse.shape)#[512,512],
                # 这个矩阵的每一行代表某一层的一组卷积核，这一行是由这一层的【512,3,3】得到的，一个512维的向量
                #这个矩阵一共512行，代表512组的卷积核

                # tmp_mean=torch.mean(tmp_mean_reuse,axis=1).numpy()
                # #这是将模型从右往左的压成一个向量，而不是从上往下对所有的512组卷积核取平均；
                # #再加上直接对测试阶段恒定不变的网络参数进行平均（未借助辅助信息进行加权），自然没用了
                # aa = np.linalg.norm(tmp_mean)
                # if aa != 0:
                #     tmp_mean = tmp_mean / aa
                # else:
                #     tmp_mean = np.zeros_like(tmp_mean)
                #
                # output_mean.append(tmp_mean)
#############################################################3
                #zhelizheli
                if name=='features.26.weight':
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l28.cpu().unsqueeze(0).T
                    #【512,512】的矩阵的每一行乘以一个固定的权重，也就是一组卷积核乘以一组固定的权重；这个权重是如何得到的呢?是这样的，该层的一组卷积核一组决定了该层输出特征图的某一个通道
                    #如果能够通过某种方式得到该通道在特征图整体的512个通道中的重要性的话，也就是说，我们得到了该通道的权重，也就是该组卷积核的权重，也就是tmp_mean_reuse 这个矩阵的某一行的权重。
                    #那么这个权重是如何得到的呢？这样的，scda不是得到了一个【h，w】的聚合特征图掩码矩阵嘛；其实我们完全可以得到一个通道级的而非整体的聚合特征图掩码矩阵，也就是一个【512，h，w】
                    #的掩码矩阵，其实就是生成一个通道级均值，特征图的每一个通道，高于该均值的位置掩码为1，否则为0；这就得到了【512，h,w】的掩码矩阵
                    #接下来呢，就是通道级掩码矩阵的每一个通道与整体的[h,w]的掩码进行按位与；最终得到了一个【512，h，w】的处理之后的掩码矩阵；
                    #这就决定了通道级掩码矩阵的每一个通道的每一个为1的位置，都是与图像的主要目标而非背景有关的；我们可以据此进一步的获知输出特征图的每一个通道激活的都是目标还是背景，也就是该通道
                    #进而就是该通道对应的该组滤波器的重要程度
                    #接下来就是最后的一步了，我们将处理之后的【512，h,w】的通道级掩码矩阵与输出特征图进行点乘
                    #进而对于处理后的特征图的每一个通道的不为零的元素们，假设该通道共N个非零值，对这N个值求和在除以N；进而得到了该通道的一个权重
                    #如果某通道所有值都为0的话，就单纯的将该通道的权重设置为0便好啦
                    #最终得到了512个权重（当然啦，还得进行一步0~1的归一化啦），它既是每一个通道的权重，也是每一组卷积核的权重啦
                    #当然这最后一步还有很多可以尝试的空间啦
                else:
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l31.cpu().unsqueeze(0).T


                tmp_mean = torch.mean(tmp_mean_reuse_direction2, axis=1).numpy()
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean.append(tmp_mean)
########***********************************///////////////////****************+++++++++++


        return  output_mean,chanelweight_l28_norm,chanelweight_l31_norm















class GradCam_yuhan_kernel_version16_2_4():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, index=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别
        features,output = self.net(input.cuda())

        feature_maps_L28 = features[
            0]  # 如果输入图像的shape为[batch_size,3,224,224]的tensor的话，关于输出的feature map,shape为  #[batch_size,512,14,14]左右
        batch_size, c_L28, h_L28, w_L28 = feature_maps_L28.size()
        feature_maps_L31 = features[
            1]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        batch_size, c_L31, h_L31, w_L31 = feature_maps_L31.size()

        # Pool5   #feature_maps_L31  [512,7,7]==[512,]
        feature_maps_L31_sum = torch.sum(feature_maps_L31[0],
                                         0)  # 如果要将CAM或者gard_cam引入SCDA，需要更改的程序也就仅仅是这一步
        L31_sum_mean = torch.mean(feature_maps_L31_sum)  # 返回的是scalar tensor ,这是与matlab的mean不同的地方
        highlight = torch.zeros(feature_maps_L31_sum.size())  # 生成了一个h*w的全零矩阵
        # print(highlight)
        highlight_index = torch.nonzero(feature_maps_L31_sum >= L31_sum_mean)  # 一个K*2的tensor,每一行一个符合要求的点
        a, b = highlight_index.size()
        # print(highlight_index.size())
        jj = (feature_maps_L31_sum > L31_sum_mean).cpu().data.numpy()
        highlight = torch.from_numpy(jj + 0).cuda().float()


        highlight_conn_L31 = highlight
        # highlight_conn_L31 = torch.tensor(largestConnectComponent(highlight.cpu().numpy()) + 0)#[7*7],我们需要将其变成[512,7,7]

        feature_maps_L31 = feature_maps_L31[0].cpu().data.numpy()  # [1,512,7,7]==>[512,7,7]
        highlight_conn_L31_beifen = highlight_conn_L31
        highlight_conn_L31 = highlight_conn_L31.cpu().data.numpy()
        highlight_conn_L31 = highlight_conn_L31.reshape(1, h_L31, w_L31) * np.ones_like(
            feature_maps_L31)  # [7,7]=>[1,7,7]=>[512,7,7]

        feature_maps_L31 = feature_maps_L31 * highlight_conn_L31

        feature_maps_L31_mean = np.sum(feature_maps_L31, axis=(1, 2)) / a
        feature_maps_L31_max = np.max(feature_maps_L31, axis=(1, 2))
        chanelweight_l31 = torch.where(torch.isnan(torch.from_numpy(feature_maps_L31_mean)), torch.full_like(torch.from_numpy(feature_maps_L31_mean), 0),
                                       torch.from_numpy(feature_maps_L31_mean))
        # chanelweight_l28=chanelweight_l28/torch.sum(highlight_L28)
        # print(chanelweight_l28)
        chanelweight_l31 = (chanelweight_l31 - torch.min(chanelweight_l31)) / (
                    torch.max(chanelweight_l31) - torch.min(chanelweight_l31))
        chanelweight_l31=chanelweight_l31.cuda()

        # % Relu5_2
        feature_maps_L28_sum = torch.sum(feature_maps_L28[0],
                                         0)  # 如果要将CAM或者gard_cam引入SCDA，需要更改的程序也就仅仅是这一步
        L28_sum_mean = torch.mean(feature_maps_L28_sum)  # 返回的是scalar tensor ,这是与matlab的mean不同的地方
        highlight = torch.zeros(feature_maps_L28_sum.size())  # 生成了一个h*w的全零矩阵
        highlight_index = torch.nonzero(feature_maps_L28_sum >= L28_sum_mean)  # 一个K*2的tensor,每一行一个符合要求的点
        a, b = highlight_index.size()
        # for i in range(a):  # a就是被选择出来的点的个数，这些点我们需要将其标记为1
        #     highlight[highlight_index[i][0], highlight_index[i][1]] = 1
        jj = (feature_maps_L28_sum > L28_sum_mean).cpu().data.numpy()
        highlight = torch.from_numpy(jj + 0).cuda().float()
        highlight_conn_L31 = highlight_conn_L31_beifen
        highlight_conn_L31_to_L28 = interpolate(highlight_conn_L31.view(1, 1, h_L31, w_L31).float(),
                                                size=[h_L28, w_L28], mode="nearest").view(h_L28, w_L28)
        highlight_conn_L28 = highlight.mul(highlight_conn_L31_to_L28.cuda())  # 逐点按元素想乘，两个都是二值矩阵，可不就是按位与嘛
        feature_maps_L28 = feature_maps_L28[0].cpu().data.numpy()  # [1,512,7,7]==>[512,7,7]
        highlight_conn_L28 = highlight_conn_L28.cpu().data.numpy()
        highlight_conn_L28 = highlight_conn_L28.reshape(1, h_L28, w_L28) * np.ones_like(
            feature_maps_L28)  # [7,7]=>[1,7,7]=>[512,7,7]

        feature_maps_L28 = feature_maps_L28 * highlight_conn_L28

        feature_maps_L28_mean = np.sum(feature_maps_L28, axis=(1, 2)) / a

        feature_maps_L28_max = np.max(feature_maps_L28, axis=(1, 2))
        chanelweight_l28 = torch.where(torch.isnan(torch.from_numpy(feature_maps_L28_mean)), torch.full_like(torch.from_numpy(feature_maps_L28_mean), 0),
                                       torch.from_numpy(feature_maps_L28_mean))
        # chanelweight_l28=chanelweight_l28/torch.sum(highlight_L28)
        # print(chanelweight_l28)
        chanelweight_l28 = (chanelweight_l28 - torch.min(chanelweight_l28)) / (
                    torch.max(chanelweight_l28) - torch.min(chanelweight_l28))
        chanelweight_l28=chanelweight_l28.cuda()

        ###************
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #
        output_mean=[]


        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['features.26.weight','features.28.weight']:
                # tmp=(param * param.grad).cpu().data
                tmp=(param).cpu().data
                # print(tmp.shape)#[512,512,3,3]
                tmp_mean_reuse=torch.mean(tmp,axis=( 2, 3))
                # print(tmp_mean_reuse.shape)#[512,512],
                # 这个矩阵的每一行代表某一层的一组卷积核，这一行是由这一层的【512,3,3】得到的，一个512维的向量
                #这个矩阵一共512行，代表512组的卷积核

                # tmp_mean=torch.mean(tmp_mean_reuse,axis=1).numpy()
                # #这是将模型从右往左的压成一个向量，而不是从上往下对所有的512组卷积核取平均；
                # #再加上直接对测试阶段恒定不变的网络参数进行平均（未借助辅助信息进行加权），自然没用了
                # aa = np.linalg.norm(tmp_mean)
                # if aa != 0:
                #     tmp_mean = tmp_mean / aa
                # else:
                #     tmp_mean = np.zeros_like(tmp_mean)
                #
                # output_mean.append(tmp_mean)
#############################################################3
                #zhelizheli
                if name=='features.26.weight':
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l28.cpu().unsqueeze(0).T
                    #【512,512】的矩阵的每一行乘以一个固定的权重，也就是一组卷积核乘以一组固定的权重；这个权重是如何得到的呢?是这样的，该层的一组卷积核一组决定了该层输出特征图的某一个通道
                    #如果能够通过某种方式得到该通道在特征图整体的512个通道中的重要性的话，也就是说，我们得到了该通道的权重，也就是该组卷积核的权重，也就是tmp_mean_reuse 这个矩阵的某一行的权重。
                    #那么这个权重是如何得到的呢？这样的，scda不是得到了一个【h，w】的聚合特征图掩码矩阵嘛；其实我们完全可以得到一个通道级的而非整体的聚合特征图掩码矩阵，也就是一个【512，h，w】
                    #的掩码矩阵，其实就是生成一个通道级均值，特征图的每一个通道，高于该均值的位置掩码为1，否则为0；这就得到了【512，h,w】的掩码矩阵
                    #接下来呢，就是通道级掩码矩阵的每一个通道与整体的[h,w]的掩码进行按位与；最终得到了一个【512，h，w】的处理之后的掩码矩阵；
                    #这就决定了通道级掩码矩阵的每一个通道的每一个为1的位置，都是与图像的主要目标而非背景有关的；我们可以据此进一步的获知输出特征图的每一个通道激活的都是目标还是背景，也就是该通道
                    #进而就是该通道对应的该组滤波器的重要程度
                    #接下来就是最后的一步了，我们将处理之后的【512，h,w】的通道级掩码矩阵与输出特征图进行点乘
                    #进而对于处理后的特征图的每一个通道的不为零的元素们，假设该通道共N个非零值，对这N个值求和在除以N；进而得到了该通道的一个权重
                    #如果某通道所有值都为0的话，就单纯的将该通道的权重设置为0便好啦
                    #最终得到了512个权重（当然啦，还得进行一步0~1的归一化啦），它既是每一个通道的权重，也是每一组卷积核的权重啦
                    #当然这最后一步还有很多可以尝试的空间啦
                else:
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l31.cpu().unsqueeze(0).T


                tmp_mean = torch.mean(tmp_mean_reuse_direction2, axis=1).numpy()
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean.append(tmp_mean)
########***********************************///////////////////****************+++++++++++


        return  output_mean






class GradCam_yuhan_kernel_version16_2_1_1():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, index=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别
        features,output = self.net(input.cuda())
        ###************
        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['features.26.bias']:
                tmp_bias_26 = (param).data
                print(tmp_bias_26)
            if name in ['features.28.bias']:
                tmp_bias_28 = (param).data

        feature_maps_L28 = features[
            0][0]  # 如果输入图像的shape为[batch_size,3,224,224]的tensor的话，关于输出的feature map,shape为  #[batch_size,512,14,14]左右
        feature_maps_L28 = feature_maps_L28 + tmp_bias_26.view(-1,1,1)
        # print(feature_maps_L28.shape)
        # print(feature_maps_L28)

        feature_maps_L28_channelwise_mean=torch.mean(feature_maps_L28,dim=(1,2))
        # print(feature_maps_L28_channelwise_mean.shape)
        highlight_chanelwise_L28 = (feature_maps_L28 > feature_maps_L28_channelwise_mean.view(-1, 1, 1)).float()
        # print(highlight_chanelwise_L28.shape)#为这512通道的每一个通道生成了一个（由该通道均值决定的）专属于该通道的二值掩码矩阵，将该通道的显著性区域标注了出来
        feature_maps_L28_sum=torch.sum(feature_maps_L28, dim=0)
        # print(feature_maps_L28_sum.shape)
        highlight_L28 = (feature_maps_L28_sum > torch.mean(feature_maps_L28_sum)).float()
        # print(highlight_L28.shape)
        h28 = highlight_chanelwise_L28 * highlight_L28.unsqueeze(0) #h28是一个【512，h,w】的二值矩阵，这是与总体二值矩阵交集得到的，可能在某一个通道上是一个全零矩阵
        # （一个完全对背景噪声响应的通道）
        # print(h28.shape)
        feature_maps_L28=feature_maps_L28 * h28  #同为shape为【512,7,7】的feature_map 与通道级二值矩阵（总体显著性区域与通道级显著性区域的交集）进行点乘，
        # chanelweight_l28 = torch.nn.functional.adaptive_max_pool2d(feature_maps_L28, output_size=1).squeeze(-1).squeeze(-1)
        chanelweight_l28=torch.sum(feature_maps_L28,dim=(1,2))/torch.sum(h28,dim=(1,2))
        # print(chanelweight_l28)
        # print(chanelweight_l28.shape)
        #我们用0代替了nan值
        chanelweight_l28 = torch.where(torch.isnan(chanelweight_l28), torch.full_like(chanelweight_l28, 0), chanelweight_l28)
        # chanelweight_l28=chanelweight_l28/torch.sum(highlight_L28)
        # print(chanelweight_l28)
        chanelweight_l28=(chanelweight_l28-torch.min(chanelweight_l28))/(torch.max(chanelweight_l28)-torch.min(chanelweight_l28))
        # # chanelweight_l28=(chanelweight_l28>torch.mean(chanelweight_l28)).float()
        # chanelweight_l28_bool = (chanelweight_l28 > torch.mean(chanelweight_l28)).float()
        # chanelweight_l28_bool_2 = torch.ones_like(chanelweight_l28_bool) - chanelweight_l28_bool
        # chanelweight_l28 = chanelweight_l28_bool_2 * chanelweight_l28
        # chanelweight_l28 = chanelweight_l28 + chanelweight_l28_bool



        feature_maps_L31 = features[
            1][0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        feature_maps_L31 = feature_maps_L31 + tmp_bias_28.view(-1, 1, 1)
        feature_maps_L31_channelwise_mean=torch.mean(feature_maps_L31,dim=(1,2))
        highlight_chanelwise_L31 = (feature_maps_L31 > feature_maps_L31_channelwise_mean.view(-1, 1, 1)).float()#得到的结果依然是【512,7,7】的一个二值矩阵
        feature_maps_L31_sum=torch.sum(feature_maps_L31, dim=0)
        highlight_L31 = (feature_maps_L31_sum > torch.mean(feature_maps_L31_sum)).float()
        h31 = highlight_chanelwise_L31 * highlight_L31.unsqueeze(0)
        feature_maps_L31 = feature_maps_L31 * h31
        # chanelweight_l31 = torch.nn.functional.adaptive_max_pool2d(feature_maps_L31, output_size=1).squeeze(-1).squeeze(
        #     -1)
        chanelweight_l31=torch.sum(feature_maps_L31,dim=(1,2))/torch.sum( h31 ,dim=(1,2))
        chanelweight_l31 = torch.where(torch.isnan(chanelweight_l31), torch.full_like(chanelweight_l31, 0), chanelweight_l31)

        # chanelweight_l31=torch.sum(h31,dim=(1,2))
        # chanelweight_l31=chanelweight_l31/torch.sum(highlight_L31)
        chanelweight_l31=(chanelweight_l31-torch.min(chanelweight_l31))/(torch.max(chanelweight_l31)-torch.min(chanelweight_l31))

        #
        # chanelweight_l31_bool=(chanelweight_l31>torch.mean(chanelweight_l31)).float()
        # chanelweight_l31_bool_2=torch.ones_like(chanelweight_l31_bool)-chanelweight_l31_bool
        # chanelweight_l31=chanelweight_l31_bool_2 * chanelweight_l31
        # chanelweight_l31=chanelweight_l31 + chanelweight_l31_bool

        ###************
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #
        output_mean=[]


        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['features.26.weight','features.28.weight']:
                # tmp=(param * param.grad).cpu().data
                tmp=(param).cpu().data
                # print(tmp.shape)#[512,512,3,3]
                tmp_mean_reuse=torch.mean(tmp,axis=( 2, 3))
                # print(tmp_mean_reuse.shape)#[512,512],
                # 这个矩阵的每一行代表某一层的一组卷积核，这一行是由这一层的【512,3,3】得到的，一个512维的向量
                #这个矩阵一共512行，代表512组的卷积核

                # tmp_mean=torch.mean(tmp_mean_reuse,axis=1).numpy()
                # #这是将模型从右往左的压成一个向量，而不是从上往下对所有的512组卷积核取平均；
                # #再加上直接对测试阶段恒定不变的网络参数进行平均（未借助辅助信息进行加权），自然没用了
                # aa = np.linalg.norm(tmp_mean)
                # if aa != 0:
                #     tmp_mean = tmp_mean / aa
                # else:
                #     tmp_mean = np.zeros_like(tmp_mean)
                #
                # output_mean.append(tmp_mean)
#############################################################3
                #zhelizheli
                if name=='features.26.weight':
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l28.cpu().unsqueeze(0).T
                    #【512,512】的矩阵的每一行乘以一个固定的权重，也就是一组卷积核乘以一组固定的权重；这个权重是如何得到的呢?是这样的，该层的一组卷积核一组决定了该层输出特征图的某一个通道
                    #如果能够通过某种方式得到该通道在特征图整体的512个通道中的重要性的话，也就是说，我们得到了该通道的权重，也就是该组卷积核的权重，也就是tmp_mean_reuse 这个矩阵的某一行的权重。
                    #那么这个权重是如何得到的呢？这样的，scda不是得到了一个【h，w】的聚合特征图掩码矩阵嘛；其实我们完全可以得到一个通道级的而非整体的聚合特征图掩码矩阵，也就是一个【512，h，w】
                    #的掩码矩阵，其实就是生成一个通道级均值，特征图的每一个通道，高于该均值的位置掩码为1，否则为0；这就得到了【512，h,w】的掩码矩阵
                    #接下来呢，就是通道级掩码矩阵的每一个通道与整体的[h,w]的掩码进行按位与；最终得到了一个【512，h，w】的处理之后的掩码矩阵；
                    #这就决定了通道级掩码矩阵的每一个通道的每一个为1的位置，都是与图像的主要目标而非背景有关的；我们可以据此进一步的获知输出特征图的每一个通道激活的都是目标还是背景，也就是该通道
                    #进而就是该通道对应的该组滤波器的重要程度
                    #接下来就是最后的一步了，我们将处理之后的【512，h,w】的通道级掩码矩阵与输出特征图进行点乘
                    #进而对于处理后的特征图的每一个通道的不为零的元素们，假设该通道共N个非零值，对这N个值求和在除以N；进而得到了该通道的一个权重
                    #如果某通道所有值都为0的话，就单纯的将该通道的权重设置为0便好啦
                    #最终得到了512个权重（当然啦，还得进行一步0~1的归一化啦），它既是每一个通道的权重，也是每一组卷积核的权重啦
                    #当然这最后一步还有很多可以尝试的空间啦
                else:
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l31.cpu().unsqueeze(0).T


                tmp_mean = torch.mean(tmp_mean_reuse_direction2, axis=1).numpy()
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean.append(tmp_mean)
########***********************************///////////////////****************+++++++++++


        return  output_mean









class GradCam_yuhan_kernel_version16_2_2():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, index=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别
        features,output = self.net(input.cuda())
        ###************

        feature_maps_L26 = features[
            0][0]  # 如果输入图像的shape为[batch_size,3,224,224]的tensor的话，关于输出的feature map,shape为  #[batch_size,512,14,14]左右
        # print(feature_maps_L26.shape)
        # print(feature_maps_L26)

        feature_maps_L26_channelwise_mean = torch.mean(feature_maps_L26, dim=(1, 2))
        # print(feature_maps_L26_channelwise_mean.shape)
        highlight_chanelwise_L26 = (feature_maps_L26 > feature_maps_L26_channelwise_mean.view(-1, 1, 1)).float()
        # print(highlight_chanelwise_L26.shape)#为这512通道的每一个通道生成了一个（由该通道均值决定的）专属于该通道的二值掩码矩阵，将该通道的显著性区域标注了出来
        feature_maps_L26_sum = torch.sum(feature_maps_L26, dim=0)
        # print(feature_maps_L26_sum.shape)
        highlight_L26 = (feature_maps_L26_sum > torch.mean(feature_maps_L26_sum)).float()
        # print(highlight_L26.shape)
        h26 = highlight_chanelwise_L26 * highlight_L26.unsqueeze(
            0)  # h26是一个【512，h,w】的二值矩阵，这是与总体二值矩阵交集得到的，可能在某一个通道上是一个全零矩阵
        # （一个完全对背景噪声响应的通道）
        # print(h26.shape)
        feature_maps_L26 = feature_maps_L26 * h26  # 同为shape为【512,7,7】的feature_map 与通道级二值矩阵（总体显著性区域与通道级显著性区域的交集）进行点乘，
        # chanelweight_l26 = torch.nn.functional.adaptive_max_pool2d(feature_maps_L26, output_size=1).squeeze(-1).squeeze(-1)
        chanelweight_l26 = torch.sum(feature_maps_L26, dim=(1, 2)) / torch.sum(h26, dim=(1, 2))
        # print(chanelweight_l26)
        # print(chanelweight_l26.shape)
        # 我们用0代替了nan值
        chanelweight_l26 = torch.where(torch.isnan(chanelweight_l26), torch.full_like(chanelweight_l26, 0),
                                       chanelweight_l26)
        # chanelweight_l26=chanelweight_l26/torch.sum(highlight_L26)
        # print(chanelweight_l26)
        chanelweight_l26 = (chanelweight_l26 - torch.min(chanelweight_l26)) / (
                    torch.max(chanelweight_l26) - torch.min(chanelweight_l26))
        # # chanelweight_l26=(chanelweight_l26>torch.mean(chanelweight_l26)).float()
        # chanelweight_l26_bool = (chanelweight_l26 > torch.mean(chanelweight_l26)).float()
        # chanelweight_l26_bool_2 = torch.ones_like(chanelweight_l26_bool) - chanelweight_l26_bool
        # chanelweight_l26 = chanelweight_l26_bool_2 * chanelweight_l26
        # chanelweight_l26 = chanelweight_l26 + chanelweight_l26_bool



        feature_maps_L28 = features[
            1][0]  # 如果输入图像的shape为[batch_size,3,224,224]的tensor的话，关于输出的feature map,shape为  #[batch_size,512,14,14]左右
        # print(feature_maps_L28.shape)
        # print(feature_maps_L28)

        feature_maps_L28_channelwise_mean=torch.mean(feature_maps_L28,dim=(1,2))
        # print(feature_maps_L28_channelwise_mean.shape)
        highlight_chanelwise_L28 = (feature_maps_L28 > feature_maps_L28_channelwise_mean.view(-1, 1, 1)).float()
        # print(highlight_chanelwise_L28.shape)#为这512通道的每一个通道生成了一个（由该通道均值决定的）专属于该通道的二值掩码矩阵，将该通道的显著性区域标注了出来
        feature_maps_L28_sum=torch.sum(feature_maps_L28, dim=0)
        # print(feature_maps_L28_sum.shape)
        highlight_L28 = (feature_maps_L28_sum > torch.mean(feature_maps_L28_sum)).float()
        # print(highlight_L28.shape)
        h28 = highlight_chanelwise_L28 * highlight_L28.unsqueeze(0) #h28是一个【512，h,w】的二值矩阵，这是与总体二值矩阵交集得到的，可能在某一个通道上是一个全零矩阵
        # （一个完全对背景噪声响应的通道）
        # print(h28.shape)
        feature_maps_L28=feature_maps_L28 * h28  #同为shape为【512,7,7】的feature_map 与通道级二值矩阵（总体显著性区域与通道级显著性区域的交集）进行点乘，
        # chanelweight_l28 = torch.nn.functional.adaptive_max_pool2d(feature_maps_L28, output_size=1).squeeze(-1).squeeze(-1)
        chanelweight_l28=torch.sum(feature_maps_L28,dim=(1,2))/torch.sum(h28,dim=(1,2))
        # print(chanelweight_l28)
        # print(chanelweight_l28.shape)
        #我们用0代替了nan值
        chanelweight_l28 = torch.where(torch.isnan(chanelweight_l28), torch.full_like(chanelweight_l28, 0), chanelweight_l28)
        # chanelweight_l28=chanelweight_l28/torch.sum(highlight_L28)
        # print(chanelweight_l28)
        chanelweight_l28=(chanelweight_l28-torch.min(chanelweight_l28))/(torch.max(chanelweight_l28)-torch.min(chanelweight_l28))
        # # chanelweight_l28=(chanelweight_l28>torch.mean(chanelweight_l28)).float()
        # chanelweight_l28_bool = (chanelweight_l28 > torch.mean(chanelweight_l28)).float()
        # chanelweight_l28_bool_2 = torch.ones_like(chanelweight_l28_bool) - chanelweight_l28_bool
        # chanelweight_l28 = chanelweight_l28_bool_2 * chanelweight_l28
        # chanelweight_l28 = chanelweight_l28 + chanelweight_l28_bool



        feature_maps_L31 = features[
            2][0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        feature_maps_L31_channelwise_mean=torch.mean(feature_maps_L31,dim=(1,2))
        highlight_chanelwise_L31 = (feature_maps_L31 > feature_maps_L31_channelwise_mean.view(-1, 1, 1)).float()#得到的结果依然是【512,7,7】的一个二值矩阵
        feature_maps_L31_sum=torch.sum(feature_maps_L31, dim=0)
        highlight_L31 = (feature_maps_L31_sum > torch.mean(feature_maps_L31_sum)).float()
        h31 = highlight_chanelwise_L31 * highlight_L31.unsqueeze(0)
        feature_maps_L31 = feature_maps_L31 * h31
        # chanelweight_l31 = torch.nn.functional.adaptive_max_pool2d(feature_maps_L31, output_size=1).squeeze(-1).squeeze(
        #     -1)
        chanelweight_l31=torch.sum(feature_maps_L31,dim=(1,2))/torch.sum( h31 ,dim=(1,2))
        chanelweight_l31 = torch.where(torch.isnan(chanelweight_l31), torch.full_like(chanelweight_l31, 0), chanelweight_l31)

        # chanelweight_l31=torch.sum(h31,dim=(1,2))
        # chanelweight_l31=chanelweight_l31/torch.sum(highlight_L31)
        chanelweight_l31=(chanelweight_l31-torch.min(chanelweight_l31))/(torch.max(chanelweight_l31)-torch.min(chanelweight_l31))

        #
        # chanelweight_l31_bool=(chanelweight_l31>torch.mean(chanelweight_l31)).float()
        # chanelweight_l31_bool_2=torch.ones_like(chanelweight_l31_bool)-chanelweight_l31_bool
        # chanelweight_l31=chanelweight_l31_bool_2 * chanelweight_l31
        # chanelweight_l31=chanelweight_l31 + chanelweight_l31_bool

        ###************
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #
        output_mean=[]


        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['features.24.weight','features.26.weight','features.28.weight']:
                # tmp=(param * param.grad).cpu().data
                tmp=(param).cpu().data
                # print(tmp.shape)#[512,512,3,3]
                tmp_mean_reuse=torch.mean(tmp,axis=( 2, 3))
                # print(tmp_mean_reuse.shape)#[512,512],
                # 这个矩阵的每一行代表某一层的一组卷积核，这一行是由这一层的【512,3,3】得到的，一个512维的向量
                #这个矩阵一共512行，代表512组的卷积核

                # tmp_mean=torch.mean(tmp_mean_reuse,axis=1).numpy()
                # #这是将模型从右往左的压成一个向量，而不是从上往下对所有的512组卷积核取平均；
                # #再加上直接对测试阶段恒定不变的网络参数进行平均（未借助辅助信息进行加权），自然没用了
                # aa = np.linalg.norm(tmp_mean)
                # if aa != 0:
                #     tmp_mean = tmp_mean / aa
                # else:
                #     tmp_mean = np.zeros_like(tmp_mean)
                #
                # output_mean.append(tmp_mean)
#############################################################3
                #zhelizheli
                if name=='features.26.weight':
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l28.cpu().unsqueeze(0).T
                    #【512,512】的矩阵的每一行乘以一个固定的权重，也就是一组卷积核乘以一组固定的权重；这个权重是如何得到的呢?是这样的，该层的一组卷积核一组决定了该层输出特征图的某一个通道
                    #如果能够通过某种方式得到该通道在特征图整体的512个通道中的重要性的话，也就是说，我们得到了该通道的权重，也就是该组卷积核的权重，也就是tmp_mean_reuse 这个矩阵的某一行的权重。
                    #那么这个权重是如何得到的呢？这样的，scda不是得到了一个【h，w】的聚合特征图掩码矩阵嘛；其实我们完全可以得到一个通道级的而非整体的聚合特征图掩码矩阵，也就是一个【512，h，w】
                    #的掩码矩阵，其实就是生成一个通道级均值，特征图的每一个通道，高于该均值的位置掩码为1，否则为0；这就得到了【512，h,w】的掩码矩阵
                    #接下来呢，就是通道级掩码矩阵的每一个通道与整体的[h,w]的掩码进行按位与；最终得到了一个【512，h，w】的处理之后的掩码矩阵；
                    #这就决定了通道级掩码矩阵的每一个通道的每一个为1的位置，都是与图像的主要目标而非背景有关的；我们可以据此进一步的获知输出特征图的每一个通道激活的都是目标还是背景，也就是该通道
                    #进而就是该通道对应的该组滤波器的重要程度
                    #接下来就是最后的一步了，我们将处理之后的【512，h,w】的通道级掩码矩阵与输出特征图进行点乘
                    #进而对于处理后的特征图的每一个通道的不为零的元素们，假设该通道共N个非零值，对这N个值求和在除以N；进而得到了该通道的一个权重
                    #如果某通道所有值都为0的话，就单纯的将该通道的权重设置为0便好啦
                    #最终得到了512个权重（当然啦，还得进行一步0~1的归一化啦），它既是每一个通道的权重，也是每一组卷积核的权重啦
                    #当然这最后一步还有很多可以尝试的空间啦
                elif name=='features.28.weight':
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l31.cpu().unsqueeze(0).T
                elif name=='features.24.weight':
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l26.cpu().unsqueeze(0).T



                tmp_mean = torch.mean(tmp_mean_reuse_direction2, axis=1).numpy()
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean.append(tmp_mean)
########***********************************///////////////////****************+++++++++++


        return  output_mean






class GradCam_yuhan_kernel_version16_4():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, index=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别
        features,output = self.net(input.cuda())
        ###************
        feature_maps_L28 = features[
            0][0]  # 如果输入图像的shape为[batch_size,3,224,224]的tensor的话，关于输出的feature map,shape为  #[batch_size,512,14,14]左右
        # print(feature_maps_L28.shape)
        feature_maps_L28_channelwise_mean=torch.mean(feature_maps_L28,dim=(1,2))
        # print(feature_maps_L28_channelwise_mean.shape)
        highlight_chanelwise_L28 = (feature_maps_L28 > feature_maps_L28_channelwise_mean.view(-1, 1, 1)).float()
        # print(highlight_chanelwise_L28.shape)#为这512通道的每一个通道生成了一个（由该通道均值决定的）专属于该通道的二值掩码矩阵，将该通道的显著性区域标注了出来
        feature_maps_L28_sum=torch.sum(feature_maps_L28, dim=0)
        # print(feature_maps_L28_sum.shape)
        highlight_L28 = (feature_maps_L28_sum > torch.mean(feature_maps_L28_sum)).float()
        # print(highlight_L28.shape)
        h28 = highlight_chanelwise_L28 * highlight_L28.unsqueeze(0) #h28是一个【512，h,w】的二值矩阵，这是与总体二值矩阵交集得到的，可能在某一个通道上是一个全零矩阵
        # （一个完全对背景噪声响应的通道）
        # print(h28.shape)
        feature_maps_L28=feature_maps_L28 * h28  #同为shape为【512,7,7】的feature_map 与通道级二值矩阵（总体显著性区域与通道级显著性区域的交集）进行点乘，
        # chanelweight_l28 = torch.nn.functional.adaptive_max_pool2d(feature_maps_L28, output_size=1).squeeze(-1).squeeze(-1)
        chanelweight_l28=torch.sum(feature_maps_L28,dim=(1,2))/torch.sum(h28,dim=(1,2))
        # print(chanelweight_l28)
        # print(chanelweight_l28.shape)
        #我们用0代替了nan值
        chanelweight_l28 = torch.where(torch.isnan(chanelweight_l28), torch.full_like(chanelweight_l28, 0), chanelweight_l28)
        # chanelweight_l28=chanelweight_l28/torch.sum(highlight_L28)
        # print(chanelweight_l28)
        # chanelweight_l28=(chanelweight_l28-torch.min(chanelweight_l28))/(torch.max(chanelweight_l28)-torch.min(chanelweight_l28))
        chanelweight_l28 = (chanelweight_l28 > torch.mean(chanelweight_l28)).float()

        # # chanelweight_l28=(chanelweight_l28>torch.mean(chanelweight_l28)).float()
        # chanelweight_l28 = (chanelweight_l28 > torch.mean(chanelweight_l28)).float()
        # chanelweight_l28_bool_2 = torch.ones_like(chanelweight_l28_bool) - chanelweight_l28_bool
        # chanelweight_l28 = chanelweight_l28_bool_2 * chanelweight_l28
        # chanelweight_l28 = chanelweight_l28 + chanelweight_l28_bool



        feature_maps_L31 = features[
            1][0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        feature_maps_L31_channelwise_mean=torch.mean(feature_maps_L31,dim=(1,2))
        highlight_chanelwise_L31 = (feature_maps_L31 > feature_maps_L31_channelwise_mean.view(-1, 1, 1)).float()#得到的结果依然是【512,7,7】的一个二值矩阵
        feature_maps_L31_sum=torch.sum(feature_maps_L31, dim=0)
        highlight_L31 = (feature_maps_L31_sum > torch.mean(feature_maps_L31_sum)).float()
        h31 = highlight_chanelwise_L31 * highlight_L31.unsqueeze(0)
        feature_maps_L31 = feature_maps_L31 * h31
        # chanelweight_l31 = torch.nn.functional.adaptive_max_pool2d(feature_maps_L31, output_size=1).squeeze(-1).squeeze(
        #     -1)
        chanelweight_l31=torch.sum(feature_maps_L31,dim=(1,2))/torch.sum( h31 ,dim=(1,2))
        chanelweight_l31 = torch.where(torch.isnan(chanelweight_l31), torch.full_like(chanelweight_l31, 0), chanelweight_l31)

        # chanelweight_l31=torch.sum(h31,dim=(1,2))
        # chanelweight_l31=chanelweight_l31/torch.sum(highlight_L31)
        # chanelweight_l31=(chanelweight_l31-torch.min(chanelweight_l31))/(torch.max(chanelweight_l31)-torch.min(chanelweight_l31))
        chanelweight_l31 = (chanelweight_l31 > torch.mean(chanelweight_l31)).float()


        #
        # chanelweight_l31=(chanelweight_l31>torch.mean(chanelweight_l31)).float()
        # chanelweight_l31_bool_2=torch.ones_like(chanelweight_l31_bool)-chanelweight_l31_bool
        # chanelweight_l31=chanelweight_l31_bool_2 * chanelweight_l31
        # chanelweight_l31=chanelweight_l31 + chanelweight_l31_bool

        ###************
        return chanelweight_l28,chanelweight_l31,features



class GradCam_yuhan_kernel_version16_3():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, index=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别
        features,output = self.net(input.cuda())
        ###************
        feature_maps_L28 = features[
            0][0]  # 如果输入图像的shape为[batch_size,3,224,224]的tensor的话，关于输出的feature map,shape为  #[batch_size,512,14,14]左右
        print(feature_maps_L28.shape)
        feature_maps_L28_channelwise_mean=torch.mean(feature_maps_L28,dim=(1,2))
        print(feature_maps_L28_channelwise_mean.shape)
        highlight_chanelwise_L28 = (feature_maps_L28 > feature_maps_L28_channelwise_mean.view(-1, 1, 1)).float()
        print(highlight_chanelwise_L28.shape)#为这512通道的每一个通道生成了一个（由该通道均值决定的）专属于该通道的二值掩码矩阵，将该通道的显著性区域标注了出来
        feature_maps_L28_sum=torch.sum(feature_maps_L28, dim=0)
        print(feature_maps_L28_sum.shape)
        highlight_L28 = (feature_maps_L28_sum > torch.mean(feature_maps_L28_sum)).float()
        print(highlight_L28.shape)
        h28 = highlight_chanelwise_L28 * highlight_L28.unsqueeze(0) #h28是一个【512，h,w】的二值矩阵，这是与总体二值矩阵交集得到的，可能在某一个通道上是一个全零矩阵
        # （一个完全对背景噪声响应的通道）
        print(h28.shape)
        feature_maps_L28=feature_maps_L28 * h28  #同为shape为【512,7,7】的feature_map 与通道级二值矩阵（总体显著性区域与通道级显著性区域的交集）进行点乘，
        # chanelweight_l28 = torch.nn.functional.adaptive_max_pool2d(feature_maps_L28, output_size=1).squeeze(-1).squeeze(-1)
        chanelweight_l28=torch.sum(feature_maps_L28,dim=(1,2))/torch.sum(h28,dim=(1,2))
        print(chanelweight_l28)
        print(chanelweight_l28.shape)
        #我们用0代替了nan值
        chanelweight_l28 = torch.where(torch.isnan(chanelweight_l28), torch.full_like(chanelweight_l28, 0), chanelweight_l28)
        # chanelweight_l28=chanelweight_l28/torch.sum(highlight_L28)
        # print(chanelweight_l28)
        # chanelweight_l28=(chanelweight_l28-torch.min(chanelweight_l28))/(torch.max(chanelweight_l28)-torch.min(chanelweight_l28))
        # # chanelweight_l28=(chanelweight_l28>torch.mean(chanelweight_l28)).float()
        chanelweight_l28 = (chanelweight_l28 > torch.mean(chanelweight_l28)).float()
        # chanelweight_l28_bool_2 = torch.ones_like(chanelweight_l28_bool) - chanelweight_l28_bool
        # chanelweight_l28 = chanelweight_l28_bool_2 * chanelweight_l28
        # chanelweight_l28 = chanelweight_l28 + chanelweight_l28_bool



        feature_maps_L31 = features[
            1][0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        feature_maps_L31_channelwise_mean=torch.mean(feature_maps_L31,dim=(1,2))
        highlight_chanelwise_L31 = (feature_maps_L31 > feature_maps_L31_channelwise_mean.view(-1, 1, 1)).float()#得到的结果依然是【512,7,7】的一个二值矩阵
        feature_maps_L31_sum=torch.sum(feature_maps_L31, dim=0)
        highlight_L31 = (feature_maps_L31_sum > torch.mean(feature_maps_L31_sum)).float()
        h31 = highlight_chanelwise_L31 * highlight_L31.unsqueeze(0)
        feature_maps_L31 = feature_maps_L31 * h31
        # chanelweight_l31 = torch.nn.functional.adaptive_max_pool2d(feature_maps_L31, output_size=1).squeeze(-1).squeeze(
        #     -1)
        chanelweight_l31=torch.sum(feature_maps_L31,dim=(1,2))/torch.sum( h31 ,dim=(1,2))
        chanelweight_l31 = torch.where(torch.isnan(chanelweight_l31), torch.full_like(chanelweight_l31, 0), chanelweight_l31)

        # chanelweight_l31=torch.sum(h31,dim=(1,2))
        # chanelweight_l31=chanelweight_l31/torch.sum(highlight_L31)
        # chanelweight_l31=(chanelweight_l31-torch.min(chanelweight_l31))/(torch.max(chanelweight_l31)-torch.min(chanelweight_l31))

        #
        chanelweight_l31=(chanelweight_l31>torch.mean(chanelweight_l31)).float()
        # chanelweight_l31_bool_2=torch.ones_like(chanelweight_l31_bool)-chanelweight_l31_bool
        # chanelweight_l31=chanelweight_l31_bool_2 * chanelweight_l31
        # chanelweight_l31=chanelweight_l31 + chanelweight_l31_bool

        ###************

        output_max=torch.max(output)
        output_min=torch.min(output)
        output_norm=torch.div((output-output_min).float(),(output_max-output_min))
        ################################################################3
        output_norm=(output_norm * -1).cpu().tolist()
        output_norm=torch.tensor(output_norm).float().cuda()
        ###################################################################
        one_hot = torch.sum(output * output_norm)
        self.net.zero_grad()
        one_hot.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #
        output_mean=[]
        output_mean_direction2=[]

        output_mean_grad_only=[]
        output_mean_direction2_grad_only=[]
        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['features.26.weight','features.28.weight']:
                # tmp=(param * param.grad).cpu().data
                tmp=(param).cpu().data
                # print(tmp.shape)#[512,512,3,3]
                tmp_mean_reuse=torch.mean(tmp,axis=( 2, 3))
                # print(tmp_mean_reuse.shape)#[512,512],
                # 这个矩阵的每一行代表某一层的一组卷积核，这一行是由这一层的【512,3,3】得到的，一个512维的向量
                #这个矩阵一共512行，代表512组的卷积核

                # tmp_mean=torch.mean(tmp_mean_reuse,axis=1).numpy()
                # #这是将模型从右往左的压成一个向量，而不是从上往下对所有的512组卷积核取平均；
                # #再加上直接对测试阶段恒定不变的网络参数进行平均（未借助辅助信息进行加权），自然没用了
                # aa = np.linalg.norm(tmp_mean)
                # if aa != 0:
                #     tmp_mean = tmp_mean / aa
                # else:
                #     tmp_mean = np.zeros_like(tmp_mean)
                #
                # output_mean.append(tmp_mean)
#############################################################3
                #zhelizheli
                if name=='features.26.weight':
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l28.cpu().unsqueeze(0).T
                    #【512,512】的矩阵的每一行乘以一个固定的权重，也就是一组卷积核乘以一组固定的权重；这个权重是如何得到的呢?是这样的，该层的一组卷积核一组决定了该层输出特征图的某一个通道
                    #如果能够通过某种方式得到该通道在特征图整体的512个通道中的重要性的话，也就是说，我们得到了该通道的权重，也就是该组卷积核的权重，也就是tmp_mean_reuse 这个矩阵的某一行的权重。
                    #那么这个权重是如何得到的呢？这样的，scda不是得到了一个【h，w】的聚合特征图掩码矩阵嘛；其实我们完全可以得到一个通道级的而非整体的聚合特征图掩码矩阵，也就是一个【512，h，w】
                    #的掩码矩阵，其实就是生成一个通道级均值，特征图的每一个通道，高于该均值的位置掩码为1，否则为0；这就得到了【512，h,w】的掩码矩阵
                    #接下来呢，就是通道级掩码矩阵的每一个通道与整体的[h,w]的掩码进行按位与；最终得到了一个【512，h，w】的处理之后的掩码矩阵；
                    #这就决定了通道级掩码矩阵的每一个通道的每一个为1的位置，都是与图像的主要目标而非背景有关的；我们可以据此进一步的获知输出特征图的每一个通道激活的都是目标还是背景，也就是该通道
                    #进而就是该通道对应的该组滤波器的重要程度
                    #接下来就是最后的一步了，我们将处理之后的【512，h,w】的通道级掩码矩阵与输出特征图进行点乘
                    #进而对于处理后的特征图的每一个通道的不为零的元素们，假设该通道共N个非零值，对这N个值求和在除以N；进而得到了该通道的一个权重
                    #如果某通道所有值都为0的话，就单纯的将该通道的权重设置为0便好啦
                    #最终得到了512个权重（当然啦，还得进行一步0~1的归一化啦），它既是每一个通道的权重，也是每一组卷积核的权重啦
                    #当然这最后一步还有很多可以尝试的空间啦
                else:
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l31.cpu().unsqueeze(0).T
                tmp_mean = torch.mean(tmp_mean_reuse_direction2, axis=0).numpy()#54.**
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean_direction2.append(tmp_mean)

                tmp_mean = torch.mean(tmp_mean_reuse_direction2, axis=1).numpy()
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean.append(tmp_mean)
########***********************************///////////////////****************+++++++++++
                tmp = (param.grad).cpu().data  #【512,512,h,w】
                tmp_mean_reuse = torch.mean(tmp, axis=(2, 3))#[512,512]
                tmp_mean = torch.mean(tmp_mean_reuse, axis=1).numpy()#[512,],左右横向的
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean_grad_only.append(tmp_mean)
                #############################################################3
                if name == 'features.26.weight':
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l28.cpu().unsqueeze(0).T
                else:
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l31.cpu().unsqueeze(0).T
                tmp_mean = torch.mean(tmp_mean_reuse_direction2, axis=0).numpy()
                # tmp_mean = torch.mean(tmp_mean_reuse, axis=0).numpy()#58.96

                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean_direction2_grad_only.append(tmp_mean)

        return  output_mean,output_mean_direction2,output_mean_grad_only,output_mean_direction2_grad_only



# aaaaa=np.array([0])
# aaaaa.reshape()




class GradCam_yuhan_kernel_version17():#通过三层特征图指导两层参数的聚合
    def __init__(self, net):
        self.net = net

    def __call__(self, input, index=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别
        features,output = self.net(input.cuda())
        ###************
        feature_maps_L26 = features[
            0][0]  # 如果输入图像的shape为[batch_size,3,224,224]的tensor的话，关于输出的feature map,shape为  #[batch_size,512,14,14]左右
        # print(feature_maps_L26.shape)
        feature_maps_L26_channelwise_mean = torch.mean(feature_maps_L26, dim=(1, 2))
        # print(feature_maps_L26_channelwise_mean.shape)
        highlight_chanelwise_L26 = (feature_maps_L26 > feature_maps_L26_channelwise_mean.view(-1, 1, 1)).float()
        # print(highlight_chanelwise_L26.shape)#为这512通道的每一个通道生成了一个（由该通道均值决定的）专属于该通道的二值掩码矩阵，将该通道的显著性区域标注了出来
        feature_maps_L26_sum = torch.sum(feature_maps_L26, dim=0)
        # print(feature_maps_L26_sum.shape)
        highlight_L26 = (feature_maps_L26_sum > torch.mean(feature_maps_L26_sum)).float()
        # print(highlight_L26.shape)
        h26 = highlight_chanelwise_L26 * highlight_L26.unsqueeze(
            0)  # h26是一个【512，h,w】的二值矩阵，这是与总体二值矩阵交集得到的，可能在某一个通道上是一个全零矩阵
        # （一个完全对背景噪声响应的通道）
        # print(h26.shape)
        feature_maps_L26 = feature_maps_L26 * h26  # 同为shape为【512,7,7】的feature_map 与通道级二值矩阵（总体显著性区域与通道级显著性区域的交集）进行点乘，
        # chanelweight_l26 = torch.nn.functional.adaptive_max_pool2d(feature_maps_L26, output_size=1).squeeze(-1).squeeze(-1)
        chanelweight_l26 = torch.sum(feature_maps_L26, dim=(1, 2)) / torch.sum(h26, dim=(1, 2))
        # print(chanelweight_l26)
        # print(chanelweight_l26.shape)
        # 我们用0代替了nan值
        chanelweight_l26 = torch.where(torch.isnan(chanelweight_l26), torch.full_like(chanelweight_l26, 0),
                                       chanelweight_l26)
        # chanelweight_l26=chanelweight_l26/torch.sum(highlight_L26)
        # print(chanelweight_l26)
        chanelweight_l26 = (chanelweight_l26 - torch.min(chanelweight_l26)) / (
                    torch.max(chanelweight_l26) - torch.min(chanelweight_l26))
        # # chanelweight_l26=(chanelweight_l26>torch.mean(chanelweight_l26)).float()
        # chanelweight_l26_bool = (chanelweight_l26 > torch.mean(chanelweight_l26)).float()
        # chanelweight_l26_bool_2 = torch.ones_like(chanelweight_l26_bool) - chanelweight_l26_bool
        # chanelweight_l26 = chanelweight_l26_bool_2 * chanelweight_l26
        # chanelweight_l26 = chanelweight_l26 + chanelweight_l26_bool



        feature_maps_L28 = features[
            1][0]  # 如果输入图像的shape为[batch_size,3,224,224]的tensor的话，关于输出的feature map,shape为  #[batch_size,512,14,14]左右
        # print(feature_maps_L28.shape)
        feature_maps_L28_channelwise_mean=torch.mean(feature_maps_L28,dim=(1,2))
        # print(feature_maps_L28_channelwise_mean.shape)
        highlight_chanelwise_L28 = (feature_maps_L28 > feature_maps_L28_channelwise_mean.view(-1, 1, 1)).float()
        # print(highlight_chanelwise_L28.shape)#为这512通道的每一个通道生成了一个（由该通道均值决定的）专属于该通道的二值掩码矩阵，将该通道的显著性区域标注了出来
        feature_maps_L28_sum=torch.sum(feature_maps_L28, dim=0)
        # print(feature_maps_L28_sum.shape)
        highlight_L28 = (feature_maps_L28_sum > torch.mean(feature_maps_L28_sum)).float()
        # print(highlight_L28.shape)
        h28 = highlight_chanelwise_L28 * highlight_L28.unsqueeze(0) #h28是一个【512，h,w】的二值矩阵，这是与总体二值矩阵交集得到的，可能在某一个通道上是一个全零矩阵
        # （一个完全对背景噪声响应的通道）
        # print(h28.shape)
        feature_maps_L28=feature_maps_L28 * h28  #同为shape为【512,7,7】的feature_map 与通道级二值矩阵（总体显著性区域与通道级显著性区域的交集）进行点乘，
        # chanelweight_l28 = torch.nn.functional.adaptive_max_pool2d(feature_maps_L28, output_size=1).squeeze(-1).squeeze(-1)
        chanelweight_l28=torch.sum(feature_maps_L28,dim=(1,2))/torch.sum(h28,dim=(1,2))
        # print(chanelweight_l28)
        # print(chanelweight_l28.shape)
        #我们用0代替了nan值
        chanelweight_l28 = torch.where(torch.isnan(chanelweight_l28), torch.full_like(chanelweight_l28, 0), chanelweight_l28)
        # chanelweight_l28=chanelweight_l28/torch.sum(highlight_L28)
        # print(chanelweight_l28)
        chanelweight_l28=(chanelweight_l28-torch.min(chanelweight_l28))/(torch.max(chanelweight_l28)-torch.min(chanelweight_l28))
        # # chanelweight_l28=(chanelweight_l28>torch.mean(chanelweight_l28)).float()
        # chanelweight_l28_bool = (chanelweight_l28 > torch.mean(chanelweight_l28)).float()
        # chanelweight_l28_bool_2 = torch.ones_like(chanelweight_l28_bool) - chanelweight_l28_bool
        # chanelweight_l28 = chanelweight_l28_bool_2 * chanelweight_l28
        # chanelweight_l28 = chanelweight_l28 + chanelweight_l28_bool



        feature_maps_L31 = features[
            2][0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        feature_maps_L31_channelwise_mean=torch.mean(feature_maps_L31,dim=(1,2))
        highlight_chanelwise_L31 = (feature_maps_L31 > feature_maps_L31_channelwise_mean.view(-1, 1, 1)).float()#得到的结果依然是【512,7,7】的一个二值矩阵
        feature_maps_L31_sum=torch.sum(feature_maps_L31, dim=0)
        highlight_L31 = (feature_maps_L31_sum > torch.mean(feature_maps_L31_sum)).float()
        h31 = highlight_chanelwise_L31 * highlight_L31.unsqueeze(0)
        feature_maps_L31 = feature_maps_L31 * h31
        # chanelweight_l31 = torch.nn.functional.adaptive_max_pool2d(feature_maps_L31, output_size=1).squeeze(-1).squeeze(
        #     -1)
        chanelweight_l31=torch.sum(feature_maps_L31,dim=(1,2))/torch.sum( h31 ,dim=(1,2))
        chanelweight_l31 = torch.where(torch.isnan(chanelweight_l31), torch.full_like(chanelweight_l31, 0), chanelweight_l31)

        # chanelweight_l31=torch.sum(h31,dim=(1,2))
        # chanelweight_l31=chanelweight_l31/torch.sum(highlight_L31)
        chanelweight_l31=(chanelweight_l31-torch.min(chanelweight_l31))/(torch.max(chanelweight_l31)-torch.min(chanelweight_l31))

        #
        # chanelweight_l31_bool=(chanelweight_l31>torch.mean(chanelweight_l31)).float()
        # chanelweight_l31_bool_2=torch.ones_like(chanelweight_l31_bool)-chanelweight_l31_bool
        # chanelweight_l31=chanelweight_l31_bool_2 * chanelweight_l31
        # chanelweight_l31=chanelweight_l31 + chanelweight_l31_bool

        ###************
        # print(chanelweight_l26)
        # print(chanelweight_l28)
        output_max=torch.max(output)
        output_min=torch.min(output)
        output_norm=torch.div((output-output_min).float(),(output_max-output_min))
        ################################################################3
        output_norm=(output_norm * -1).cpu().tolist()
        output_norm=torch.tensor(output_norm).float().cuda()
        ###################################################################
        one_hot = torch.sum(output * output_norm)
        self.net.zero_grad()
        one_hot.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #
        output_mean=[]
        output_mean_direction2=[]

        output_mean_grad_only=[]
        output_mean_direction2_grad_only=[]
        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['features.26.weight', 'features.28.weight']:
                # tmp=(param * param.grad).cpu().data
                tmp = (param).cpu().data
                # print(tmp.shape)#[512,512,3,3]
                tmp_mean_reuse = torch.mean(tmp, axis=(2, 3))
                # print(tmp_mean_reuse.shape)#[512,512],
                # 这个矩阵的每一行代表某一层的一组卷积核，这一行是由这一层的【512,3,3】得到的，一个512维的向量
                # 这个矩阵一共512行，代表512组的卷积核

                # tmp_mean=torch.mean(tmp_mean_reuse,axis=1).numpy()
                # #这是将模型从右往左的压成一个向量，而不是从上往下对所有的512组卷积核取平均；
                # #再加上直接对测试阶段恒定不变的网络参数进行平均（未借助辅助信息进行加权），自然没用了
                # aa = np.linalg.norm(tmp_mean)
                # if aa != 0:
                #     tmp_mean = tmp_mean / aa
                # else:
                #     tmp_mean = np.zeros_like(tmp_mean)
                #
                # output_mean.append(tmp_mean)
                #############################################################3
                # zhelizheli
                if name == 'features.26.weight':
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l26.cpu().unsqueeze(0).T
                    # 【512,512】的矩阵的每一行乘以一个固定的权重，也就是一组卷积核乘以一组固定的权重；这个权重是如何得到的呢?是这样的，该层的一组卷积核一组决定了该层输出特征图的某一个通道
                    # 如果能够通过某种方式得到该通道在特征图整体的512个通道中的重要性的话，也就是说，我们得到了该通道的权重，也就是该组卷积核的权重，也就是tmp_mean_reuse 这个矩阵的某一行的权重。
                    # 那么这个权重是如何得到的呢？这样的，scda不是得到了一个【h，w】的聚合特征图掩码矩阵嘛；其实我们完全可以得到一个通道级的而非整体的聚合特征图掩码矩阵，也就是一个【512，h，w】
                    # 的掩码矩阵，其实就是生成一个通道级均值，特征图的每一个通道，高于该均值的位置掩码为1，否则为0；这就得到了【512，h,w】的掩码矩阵
                    # 接下来呢，就是通道级掩码矩阵的每一个通道与整体的[h,w]的掩码进行按位与；最终得到了一个【512，h，w】的处理之后的掩码矩阵；
                    # 这就决定了通道级掩码矩阵的每一个通道的每一个为1的位置，都是与图像的主要目标而非背景有关的；我们可以据此进一步的获知输出特征图的每一个通道激活的都是目标还是背景，也就是该通道
                    # 进而就是该通道对应的该组滤波器的重要程度
                    # 接下来就是最后的一步了，我们将处理之后的【512，h,w】的通道级掩码矩阵与输出特征图进行点乘
                    # 进而对于处理后的特征图的每一个通道的不为零的元素们，假设该通道共N个非零值，对这N个值求和在除以N；进而得到了该通道的一个权重
                    # 如果某通道所有值都为0的话，就单纯的将该通道的权重设置为0便好啦
                    # 最终得到了512个权重（当然啦，还得进行一步0~1的归一化啦），它既是每一个通道的权重，也是每一组卷积核的权重啦
                    # 当然这最后一步还有很多可以尝试的空间啦
                else:
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l28.cpu().unsqueeze(0).T
                tmp_mean = torch.mean(tmp_mean_reuse_direction2, axis=0).numpy()  # #纵向的挤压，将【512,512】===》【512，】，512组卷积核被挤压到一起
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean_direction2.append(tmp_mean)

                tmp_mean = torch.mean(tmp_mean_reuse_direction2, axis=1).numpy()#横向的挤压，将【512,512】===》【512，】，一组卷积核的512个数最终被挤压为1个数
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean.append(tmp_mean)
                ########***********************************///////////////////****************+++++++++++
                tmp = (param.grad).cpu().data  # 【512,512,h,w】
                tmp_mean_reuse = torch.mean(tmp, axis=(2, 3))  # [512,512]
                tmp_mean = torch.mean(tmp_mean_reuse, axis=1).numpy()  # [512,],左右横向的
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean_grad_only.append(tmp_mean)
                #############################################################3
                if name == 'features.26.weight':
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l26.cpu().unsqueeze(0).T
                else:
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l28.cpu().unsqueeze(0).T
                tmp_mean = torch.mean(tmp_mean_reuse_direction2, axis=0).numpy()
                # tmp_mean = torch.mean(tmp_mean_reuse, axis=0).numpy()#58.96

                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean_direction2_grad_only.append(tmp_mean)

        return output_mean, output_mean_direction2, output_mean_grad_only, output_mean_direction2_grad_only
















class GradCam_yuhan_kernel_version17_2():#通过三层特征图指导两层参数的聚合
    def __init__(self, net):
        self.net = net

    def __call__(self, input, index=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别
        features,output = self.net(input.cuda())
        ###************
        feature_maps_L26 = features[
            0][0]  # 如果输入图像的shape为[batch_size,3,224,224]的tensor的话，关于输出的feature map,shape为  #[batch_size,512,14,14]左右
        # print(feature_maps_L26.shape)
        feature_maps_L26_channelwise_mean = torch.mean(feature_maps_L26, dim=(1, 2))
        # print(feature_maps_L26_channelwise_mean.shape)
        highlight_chanelwise_L26 = (feature_maps_L26 > feature_maps_L26_channelwise_mean.view(-1, 1, 1)).float()
        # print(highlight_chanelwise_L26.shape)#为这512通道的每一个通道生成了一个（由该通道均值决定的）专属于该通道的二值掩码矩阵，将该通道的显著性区域标注了出来
        feature_maps_L26_sum = torch.sum(feature_maps_L26, dim=0)
        # print(feature_maps_L26_sum.shape)
        highlight_L26 = (feature_maps_L26_sum > torch.mean(feature_maps_L26_sum)).float()
        # print(highlight_L26.shape)
        h26 = highlight_chanelwise_L26 * highlight_L26.unsqueeze(
            0)  # h26是一个【512，h,w】的二值矩阵，这是与总体二值矩阵交集得到的，可能在某一个通道上是一个全零矩阵
        # （一个完全对背景噪声响应的通道）
        # print(h26.shape)
        feature_maps_L26 = feature_maps_L26 * h26  # 同为shape为【512,7,7】的feature_map 与通道级二值矩阵（总体显著性区域与通道级显著性区域的交集）进行点乘，
        # chanelweight_l26 = torch.nn.functional.adaptive_max_pool2d(feature_maps_L26, output_size=1).squeeze(-1).squeeze(-1)
        chanelweight_l26 = torch.sum(feature_maps_L26, dim=(1, 2)) / torch.sum(h26, dim=(1, 2))
        # print(chanelweight_l26)
        # print(chanelweight_l26.shape)
        # 我们用0代替了nan值
        chanelweight_l26 = torch.where(torch.isnan(chanelweight_l26), torch.full_like(chanelweight_l26, 0),
                                       chanelweight_l26)
        # chanelweight_l26=chanelweight_l26/torch.sum(highlight_L26)
        # print(chanelweight_l26)
        chanelweight_l26 = (chanelweight_l26 - torch.min(chanelweight_l26)) / (
                    torch.max(chanelweight_l26) - torch.min(chanelweight_l26))
        # # chanelweight_l26=(chanelweight_l26>torch.mean(chanelweight_l26)).float()
        # chanelweight_l26_bool = (chanelweight_l26 > torch.mean(chanelweight_l26)).float()
        # chanelweight_l26_bool_2 = torch.ones_like(chanelweight_l26_bool) - chanelweight_l26_bool
        # chanelweight_l26 = chanelweight_l26_bool_2 * chanelweight_l26
        # chanelweight_l26 = chanelweight_l26 + chanelweight_l26_bool



        feature_maps_L28 = features[
            1][0]  # 如果输入图像的shape为[batch_size,3,224,224]的tensor的话，关于输出的feature map,shape为  #[batch_size,512,14,14]左右
        # print(feature_maps_L28.shape)
        feature_maps_L28_channelwise_mean=torch.mean(feature_maps_L28,dim=(1,2))
        # print(feature_maps_L28_channelwise_mean.shape)
        highlight_chanelwise_L28 = (feature_maps_L28 > feature_maps_L28_channelwise_mean.view(-1, 1, 1)).float()
        # print(highlight_chanelwise_L28.shape)#为这512通道的每一个通道生成了一个（由该通道均值决定的）专属于该通道的二值掩码矩阵，将该通道的显著性区域标注了出来
        feature_maps_L28_sum=torch.sum(feature_maps_L28, dim=0)
        # print(feature_maps_L28_sum.shape)
        highlight_L28 = (feature_maps_L28_sum > torch.mean(feature_maps_L28_sum)).float()
        # print(highlight_L28.shape)
        h28 = highlight_chanelwise_L28 * highlight_L28.unsqueeze(0) #h28是一个【512，h,w】的二值矩阵，这是与总体二值矩阵交集得到的，可能在某一个通道上是一个全零矩阵
        # （一个完全对背景噪声响应的通道）
        # print(h28.shape)
        feature_maps_L28=feature_maps_L28 * h28  #同为shape为【512,7,7】的feature_map 与通道级二值矩阵（总体显著性区域与通道级显著性区域的交集）进行点乘，
        # chanelweight_l28 = torch.nn.functional.adaptive_max_pool2d(feature_maps_L28, output_size=1).squeeze(-1).squeeze(-1)
        chanelweight_l28=torch.sum(feature_maps_L28,dim=(1,2))/torch.sum(h28,dim=(1,2))
        # print(chanelweight_l28)
        # print(chanelweight_l28.shape)
        #我们用0代替了nan值
        chanelweight_l28 = torch.where(torch.isnan(chanelweight_l28), torch.full_like(chanelweight_l28, 0), chanelweight_l28)
        # chanelweight_l28=chanelweight_l28/torch.sum(highlight_L28)
        # print(chanelweight_l28)
        chanelweight_l28=(chanelweight_l28-torch.min(chanelweight_l28))/(torch.max(chanelweight_l28)-torch.min(chanelweight_l28))
        # # chanelweight_l28=(chanelweight_l28>torch.mean(chanelweight_l28)).float()
        # chanelweight_l28_bool = (chanelweight_l28 > torch.mean(chanelweight_l28)).float()
        # chanelweight_l28_bool_2 = torch.ones_like(chanelweight_l28_bool) - chanelweight_l28_bool
        # chanelweight_l28 = chanelweight_l28_bool_2 * chanelweight_l28
        # chanelweight_l28 = chanelweight_l28 + chanelweight_l28_bool



        feature_maps_L31 = features[
            2][0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        feature_maps_L31_channelwise_mean=torch.mean(feature_maps_L31,dim=(1,2))
        highlight_chanelwise_L31 = (feature_maps_L31 > feature_maps_L31_channelwise_mean.view(-1, 1, 1)).float()#得到的结果依然是【512,7,7】的一个二值矩阵
        feature_maps_L31_sum=torch.sum(feature_maps_L31, dim=0)
        highlight_L31 = (feature_maps_L31_sum > torch.mean(feature_maps_L31_sum)).float()
        h31 = highlight_chanelwise_L31 * highlight_L31.unsqueeze(0)
        feature_maps_L31 = feature_maps_L31 * h31
        # chanelweight_l31 = torch.nn.functional.adaptive_max_pool2d(feature_maps_L31, output_size=1).squeeze(-1).squeeze(
        #     -1)
        chanelweight_l31=torch.sum(feature_maps_L31,dim=(1,2))/torch.sum( h31 ,dim=(1,2))
        chanelweight_l31 = torch.where(torch.isnan(chanelweight_l31), torch.full_like(chanelweight_l31, 0), chanelweight_l31)

        # chanelweight_l31=torch.sum(h31,dim=(1,2))
        # chanelweight_l31=chanelweight_l31/torch.sum(highlight_L31)
        chanelweight_l31=(chanelweight_l31-torch.min(chanelweight_l31))/(torch.max(chanelweight_l31)-torch.min(chanelweight_l31))

        #
        # chanelweight_l31_bool=(chanelweight_l31>torch.mean(chanelweight_l31)).float()
        # chanelweight_l31_bool_2=torch.ones_like(chanelweight_l31_bool)-chanelweight_l31_bool
        # chanelweight_l31=chanelweight_l31_bool_2 * chanelweight_l31
        # chanelweight_l31=chanelweight_l31 + chanelweight_l31_bool

        ###************

        output_max=torch.max(output)
        output_min=torch.min(output)
        output_norm=torch.div((output-output_min).float(),(output_max-output_min))
        ################################################################3
        output_norm=(output_norm * -1).cpu().tolist()
        output_norm=torch.tensor(output_norm).float().cuda()
        ###################################################################
        one_hot = torch.sum(output * output_norm)
        self.net.zero_grad()
        one_hot.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #
        output_mean=[]
        output_mean_direction2=[]

        output_mean_grad_only=[]
        output_mean_direction2_grad_only=[]
        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['features.26.weight','features.28.weight']:
                # tmp=(param * param.grad).cpu().data
                tmp=(param).cpu().data
                print(tmp.shape)#[512,512,3,3]
                tmp_mean_reuse=torch.mean(tmp,axis=( 2, 3))
                print(tmp_mean_reuse.shape)#[512,512],
                # 这个矩阵的每一行代表某一层的一组卷积核，这一行是由这一层的【512,3,3】得到的，一个512维的向量
                #这个矩阵一共512行，代表512组的卷积核
                tmp_mean=torch.mean(tmp_mean_reuse,axis=1).numpy()
                #这是将模型从右往左的压成一个向量，而不是从上往下对所有的512组卷积核取平均；
                #再加上直接对测试阶段恒定不变的网络参数进行平均（未借助辅助信息进行加权），自然没用了
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean.append(tmp_mean)
#############################################################3
                #zhelizheli
                if name=='features.26.weight':
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l26.cpu().unsqueeze(0)
                    #【512,512】的矩阵的每一行乘以一个固定的权重，也就是一组卷积核乘以一组固定的权重；这个权重是如何得到的呢?是这样的，该层的一组卷积核一组决定了该层输出特征图的某一个通道
                    #如果能够通过某种方式得到该通道在特征图整体的512个通道中的重要性的话，也就是说，我们得到了该通道的权重，也就是该组卷积核的权重，也就是tmp_mean_reuse 这个矩阵的某一行的权重。
                    #那么这个权重是如何得到的呢？这样的，scda不是得到了一个【h，w】的聚合特征图掩码矩阵嘛；其实我们完全可以得到一个通道级的而非整体的聚合特征图掩码矩阵，也就是一个【512，h，w】
                    #的掩码矩阵，其实就是生成一个通道级均值，特征图的每一个通道，高于该均值的位置掩码为1，否则为0；这就得到了【512，h,w】的掩码矩阵
                    #接下来呢，就是通道级掩码矩阵的每一个通道与整体的[h,w]的掩码进行按位与；最终得到了一个【512，h，w】的处理之后的掩码矩阵；
                    #这就决定了通道级掩码矩阵的每一个通道的每一个为1的位置，都是与图像的主要目标而非背景有关的；我们可以据此进一步的获知输出特征图的每一个通道激活的都是目标还是背景，也就是该通道
                    #进而就是该通道对应的该组滤波器的重要程度
                    #接下来就是最后的一步了，我们将处理之后的【512，h,w】的通道级掩码矩阵与输出特征图进行点乘
                    #进而对于处理后的特征图的每一个通道的不为零的元素们，假设该通道共N个非零值，对这N个值求和在除以N；进而得到了该通道的一个权重
                    #如果某通道所有值都为0的话，就单纯的将该通道的权重设置为0便好啦
                    #最终得到了512个权重（当然啦，还得进行一步0~1的归一化啦），它既是每一个通道的权重，也是每一组卷积核的权重啦
                    #当然这最后一步还有很多可以尝试的空间啦
                else:
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l28.cpu().unsqueeze(0)
                tmp_mean = torch.mean(tmp_mean_reuse_direction2, axis=0).numpy()#54.**
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean_direction2.append(tmp_mean)
########***********************************///////////////////****************+++++++++++
                tmp = (param.grad).cpu().data  #【512,512,h,w】
                tmp_mean_reuse = torch.mean(tmp, axis=(2, 3))#[512,512]
                tmp_mean = torch.mean(tmp_mean_reuse, axis=1).numpy()#[512,],左右横向的
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean_grad_only.append(tmp_mean)
                #############################################################3
                if name == 'features.26.weight':
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l26.cpu().unsqueeze(0)
                else:
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l28.cpu().unsqueeze(0)
                tmp_mean = torch.mean(tmp_mean_reuse_direction2, axis=0).numpy()
                # tmp_mean = torch.mean(tmp_mean_reuse, axis=0).numpy()#58.96

                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean_direction2_grad_only.append(tmp_mean)

        return  output_mean,output_mean_direction2,output_mean_grad_only,output_mean_direction2_grad_only








class GradCam_yuhan_kernel_version18():#通过三层特征图指导两层参数的聚合
    def __init__(self, net):
        self.net = net

    def __call__(self, input, index=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别
        features,output = self.net(input.cuda())
        ###************
        feature_maps_L26 = features[
            0][0]  # 如果输入图像的shape为[batch_size,3,224,224]的tensor的话，关于输出的feature map,shape为  #[batch_size,512,14,14]左右
        # print(feature_maps_L26.shape)
        feature_maps_L26_channelwise_mean = torch.mean(feature_maps_L26, dim=(1, 2))
        # print(feature_maps_L26_channelwise_mean.shape)
        highlight_chanelwise_L26 = (feature_maps_L26 > feature_maps_L26_channelwise_mean.view(-1, 1, 1)).float()
        # print(highlight_chanelwise_L26.shape)#为这512通道的每一个通道生成了一个（由该通道均值决定的）专属于该通道的二值掩码矩阵，将该通道的显著性区域标注了出来
        feature_maps_L26_sum = torch.sum(feature_maps_L26, dim=0)
        # print(feature_maps_L26_sum.shape)
        highlight_L26 = (feature_maps_L26_sum > torch.mean(feature_maps_L26_sum)).float()
        # print(highlight_L26.shape)
        h26 = highlight_chanelwise_L26 * highlight_L26.unsqueeze(
            0)  # h26是一个【512，h,w】的二值矩阵，这是与总体二值矩阵交集得到的，可能在某一个通道上是一个全零矩阵
        # （一个完全对背景噪声响应的通道）
        # print(h26.shape)
        feature_maps_L26 = feature_maps_L26 * h26  # 同为shape为【512,7,7】的feature_map 与通道级二值矩阵（总体显著性区域与通道级显著性区域的交集）进行点乘，
        # chanelweight_l26 = torch.nn.functional.adaptive_max_pool2d(feature_maps_L26, output_size=1).squeeze(-1).squeeze(-1)
        chanelweight_l26 = torch.sum(feature_maps_L26, dim=(1, 2)) / torch.sum(h26, dim=(1, 2))
        # print(chanelweight_l26)
        # print(chanelweight_l26.shape)
        # 我们用0代替了nan值
        chanelweight_l26 = torch.where(torch.isnan(chanelweight_l26), torch.full_like(chanelweight_l26, 0),
                                       chanelweight_l26)
        # chanelweight_l26=chanelweight_l26/torch.sum(highlight_L26)
        # print(chanelweight_l26)
        chanelweight_l26 = (chanelweight_l26 - torch.min(chanelweight_l26)) / (
                    torch.max(chanelweight_l26) - torch.min(chanelweight_l26))
        # # chanelweight_l26=(chanelweight_l26>torch.mean(chanelweight_l26)).float()
        # chanelweight_l26_bool = (chanelweight_l26 > torch.mean(chanelweight_l26)).float()
        # chanelweight_l26_bool_2 = torch.ones_like(chanelweight_l26_bool) - chanelweight_l26_bool
        # chanelweight_l26 = chanelweight_l26_bool_2 * chanelweight_l26
        # chanelweight_l26 = chanelweight_l26 + chanelweight_l26_bool



        feature_maps_L28 = features[
            1][0]  # 如果输入图像的shape为[batch_size,3,224,224]的tensor的话，关于输出的feature map,shape为  #[batch_size,512,14,14]左右
        # print(feature_maps_L28.shape)
        feature_maps_L28_channelwise_mean=torch.mean(feature_maps_L28,dim=(1,2))
        # print(feature_maps_L28_channelwise_mean.shape)
        highlight_chanelwise_L28 = (feature_maps_L28 > feature_maps_L28_channelwise_mean.view(-1, 1, 1)).float()
        # print(highlight_chanelwise_L28.shape)#为这512通道的每一个通道生成了一个（由该通道均值决定的）专属于该通道的二值掩码矩阵，将该通道的显著性区域标注了出来
        feature_maps_L28_sum=torch.sum(feature_maps_L28, dim=0)
        # print(feature_maps_L28_sum.shape)
        highlight_L28 = (feature_maps_L28_sum > torch.mean(feature_maps_L28_sum)).float()
        # print(highlight_L28.shape)
        h28 = highlight_chanelwise_L28 * highlight_L28.unsqueeze(0) #h28是一个【512，h,w】的二值矩阵，这是与总体二值矩阵交集得到的，可能在某一个通道上是一个全零矩阵
        # （一个完全对背景噪声响应的通道）
        # print(h28.shape)
        feature_maps_L28=feature_maps_L28 * h28  #同为shape为【512,7,7】的feature_map 与通道级二值矩阵（总体显著性区域与通道级显著性区域的交集）进行点乘，
        # chanelweight_l28 = torch.nn.functional.adaptive_max_pool2d(feature_maps_L28, output_size=1).squeeze(-1).squeeze(-1)
        chanelweight_l28=torch.sum(feature_maps_L28,dim=(1,2))/torch.sum(h28,dim=(1,2))
        # print(chanelweight_l28)
        # print(chanelweight_l28.shape)
        #我们用0代替了nan值
        chanelweight_l28 = torch.where(torch.isnan(chanelweight_l28), torch.full_like(chanelweight_l28, 0), chanelweight_l28)
        # chanelweight_l28=chanelweight_l28/torch.sum(highlight_L28)
        # print(chanelweight_l28)
        chanelweight_l28=(chanelweight_l28-torch.min(chanelweight_l28))/(torch.max(chanelweight_l28)-torch.min(chanelweight_l28))
        # # chanelweight_l28=(chanelweight_l28>torch.mean(chanelweight_l28)).float()
        # chanelweight_l28_bool = (chanelweight_l28 > torch.mean(chanelweight_l28)).float()
        # chanelweight_l28_bool_2 = torch.ones_like(chanelweight_l28_bool) - chanelweight_l28_bool
        # chanelweight_l28 = chanelweight_l28_bool_2 * chanelweight_l28
        # chanelweight_l28 = chanelweight_l28 + chanelweight_l28_bool



        feature_maps_L31 = features[
            2][0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        feature_maps_L31_channelwise_mean=torch.mean(feature_maps_L31,dim=(1,2))
        highlight_chanelwise_L31 = (feature_maps_L31 > feature_maps_L31_channelwise_mean.view(-1, 1, 1)).float()#得到的结果依然是【512,7,7】的一个二值矩阵
        feature_maps_L31_sum=torch.sum(feature_maps_L31, dim=0)
        highlight_L31 = (feature_maps_L31_sum > torch.mean(feature_maps_L31_sum)).float()
        h31 = highlight_chanelwise_L31 * highlight_L31.unsqueeze(0)
        feature_maps_L31 = feature_maps_L31 * h31
        # chanelweight_l31 = torch.nn.functional.adaptive_max_pool2d(feature_maps_L31, output_size=1).squeeze(-1).squeeze(
        #     -1)
        chanelweight_l31=torch.sum(feature_maps_L31,dim=(1,2))/torch.sum( h31 ,dim=(1,2))
        chanelweight_l31 = torch.where(torch.isnan(chanelweight_l31), torch.full_like(chanelweight_l31, 0), chanelweight_l31)

        # chanelweight_l31=torch.sum(h31,dim=(1,2))
        # chanelweight_l31=chanelweight_l31/torch.sum(highlight_L31)
        chanelweight_l31=(chanelweight_l31-torch.min(chanelweight_l31))/(torch.max(chanelweight_l31)-torch.min(chanelweight_l31))

        #
        # chanelweight_l31_bool=(chanelweight_l31>torch.mean(chanelweight_l31)).float()
        # chanelweight_l31_bool_2=torch.ones_like(chanelweight_l31_bool)-chanelweight_l31_bool
        # chanelweight_l31=chanelweight_l31_bool_2 * chanelweight_l31
        # chanelweight_l31=chanelweight_l31 + chanelweight_l31_bool

        ###************

        output_max=torch.max(output)
        output_min=torch.min(output)
        output_norm=torch.div((output-output_min).float(),(output_max-output_min))
        ################################################################3
        output_norm=(output_norm * -1).cpu().tolist()
        output_norm=torch.tensor(output_norm).float().cuda()
        ###################################################################
        one_hot = torch.sum(output * output_norm)
        self.net.zero_grad()
        one_hot.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #
        output_mean=[]
        output_mean_direction2=[]

        output_mean_grad_only=[]
        output_mean_direction2_grad_only=[]
        for name, param in self.net.named_parameters():
            # print('层:', name, param.size())
            # print('权值梯度', param.grad)
            # print('权值', param)
            if name in ['features.26.weight','features.28.weight']:
                # tmp=(param * param.grad).cpu().data
                tmp=(param).cpu().data
                print(tmp.shape)#[512,512,3,3]
                tmp_mean_reuse=torch.mean(tmp,axis=( 2, 3))
                print(tmp_mean_reuse.shape)#[512,512],
                # 这个矩阵的每一行代表某一层的一组卷积核，这一行是由这一层的【512,3,3】得到的，一个512维的向量
                #这个矩阵一共512行，代表512组的卷积核
                tmp_mean=torch.mean(tmp_mean_reuse,axis=1).numpy()
                #这是将模型从右往左的压成一个向量，而不是从上往下对所有的512组卷积核取平均；
                #再加上直接对测试阶段恒定不变的网络参数进行平均（未借助辅助信息进行加权），自然没用了
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean.append(tmp_mean)
#############################################################3
                #zhelizheli
                if name=='features.26.weight':
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l28.cpu().unsqueeze(0).T
                    #【512,512】的矩阵的每一行乘以一个固定的权重，也就是一组卷积核乘以一组固定的权重；这个权重是如何得到的呢?是这样的，该层的一组卷积核一组决定了该层输出特征图的某一个通道
                    #如果能够通过某种方式得到该通道在特征图整体的512个通道中的重要性的话，也就是说，我们得到了该通道的权重，也就是该组卷积核的权重，也就是tmp_mean_reuse 这个矩阵的某一行的权重。
                    #那么这个权重是如何得到的呢？这样的，scda不是得到了一个【h，w】的聚合特征图掩码矩阵嘛；其实我们完全可以得到一个通道级的而非整体的聚合特征图掩码矩阵，也就是一个【512，h，w】
                    #的掩码矩阵，其实就是生成一个通道级均值，特征图的每一个通道，高于该均值的位置掩码为1，否则为0；这就得到了【512，h,w】的掩码矩阵
                    #接下来呢，就是通道级掩码矩阵的每一个通道与整体的[h,w]的掩码进行按位与；最终得到了一个【512，h，w】的处理之后的掩码矩阵；
                    #这就决定了通道级掩码矩阵的每一个通道的每一个为1的位置，都是与图像的主要目标而非背景有关的；我们可以据此进一步的获知输出特征图的每一个通道激活的都是目标还是背景，也就是该通道
                    #进而就是该通道对应的该组滤波器的重要程度
                    #接下来就是最后的一步了，我们将处理之后的【512，h,w】的通道级掩码矩阵与输出特征图进行点乘
                    #进而对于处理后的特征图的每一个通道的不为零的元素们，假设该通道共N个非零值，对这N个值求和在除以N；进而得到了该通道的一个权重
                    #如果某通道所有值都为0的话，就单纯的将该通道的权重设置为0便好啦
                    #最终得到了512个权重（当然啦，还得进行一步0~1的归一化啦），它既是每一个通道的权重，也是每一组卷积核的权重啦
                    #当然这最后一步还有很多可以尝试的空间啦
                else:
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l31.cpu().unsqueeze(0).T
                tmp_mean = torch.mean(tmp_mean_reuse_direction2, axis=0).numpy()#54.**
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean_direction2.append(tmp_mean)
########***********************************///////////////////****************+++++++++++
                tmp = (param.grad).cpu().data  #【512,512,h,w】
                tmp_mean_reuse = torch.mean(tmp, axis=(2, 3))#[512,512]
                tmp_mean = torch.mean(tmp_mean_reuse, axis=1).numpy()#[512,],左右横向的
                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean_grad_only.append(tmp_mean)
                #############################################################3
                if name == 'features.26.weight':
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l28.cpu().unsqueeze(0).T
                else:
                    tmp_mean_reuse_direction2 = tmp_mean_reuse * chanelweight_l31.cpu().unsqueeze(0).T
                tmp_mean = torch.mean(tmp_mean_reuse_direction2, axis=0).numpy()
                # tmp_mean = torch.mean(tmp_mean_reuse, axis=0).numpy()#58.96

                aa = np.linalg.norm(tmp_mean)
                if aa != 0:
                    tmp_mean = tmp_mean / aa
                else:
                    tmp_mean = np.zeros_like(tmp_mean)

                output_mean_direction2_grad_only.append(tmp_mean)

        return  output_mean,output_mean_direction2,output_mean_grad_only,output_mean_direction2_grad_only












def returnCAM1(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    # size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
        #[1 512]*[512,h*w]===>[1   h*w]====>[h w]聚合特征图，一个灰度图像，灰度的取值范围是【0,255】
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        cam = np.uint8(255 * cam)
        # output_cam.append(cv2.resize(cam_img, size_upsample))
        output_cam.append(cam)
    return output_cam #返回的是一个列表，列表里面可以有多个元素，对应着多个该张图片可能的类别，列表中的每一个元素都是一个聚合特征图，
    # 每一个聚合特征图都是一个（将不同的聚合特征图的取值范围设置的相同:我直觉上觉得这一步处理是合理的）灰度图像，
    # 按照对应类别的 概率值进行聚合特征图们的加权组合，最终的结果，求均值，得到聚合特征图掩码。再接下来就是SCDA了
    #不过这个CAM只能应用于pool5

def get_cam1(net, img_preds, cnnFeature_maps_L28,cnnFeature_maps_L31):
    # batch_size, c_L31, h_L31, w_L31   cnnFeature_maps_L31.size()
    params = list(net.parameters())
    weight_softmax = np.squeeze(params[-2].data.cpu().numpy())#[200,512]的ndarray,这是全连接层的参数，矩阵的每一行对应512个权重参数
    h_x = F.softmax(img_preds, dim=1).data.squeeze()#h_x就是一个一维向量
    probs, idx = h_x.sort(0, True)#一维向量，就一个维度，0维度，在这个维度上进行排序，得到【200，】的概率值tensor probs,【200，】的类别索引tensor idx,我们知道了最有可能是哪一个类别
    #比方说idx的第一个元素是m,那么我们所需要的就是weight_softmax的第m+1行的参数（当然直接weight_softmax[m]就行了），以这些参数进行pool5输出特征图的加权组合，得到聚合特征图
    # output: the prediction
    # for i in range(0, 2):
    #     line = '{:.3f} -> {}'.format(probs[i], classes[idx[i].item()])
    #     print(line)
    miki=[]
    for i in range(len(probs) - 1):
        if probs[i]/probs[i + 1] < 2:
            miki.append(idx[i].item())
        else:
            break   #只要出现了一个大于等于2的，直接跳出循环，后面的概率值都不考虑了
    CAMs = returnCAM(cnnFeature_maps_L31, weight_softmax, miki)

    # render the CAM and output
    # print('output CAM.jpg for the top1 prediction: %s' % classes[idx[0].item()])
    # img = cv2.imread(root_img)
    # height, width, _ = img.shape
    # CAM = cv2.resize(CAMs[0], (width, height))
    # heatmap = cv2.applyColorMap(CAM, cv2.COLORMAP_JET)
    # result = heatmap * 0.3 + img * 0.5
    # cv2.imwrite('cam.jpg', result)
    for i in range(len(CAMs)):
        if i==0:
            cam=CAMs[i]*probs[i]
        else:
            cam=cam+CAMs[i]*probs[i]

    cam=cam/len(CAMs)

    return  cam#现在我们只能对L31的输出特征图进行聚合











class GradCam_yuhan_multiple_classes_WACD_3_no_once_withweight_and_weight_tmp():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, index=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别
        features, output = self.net(input.cuda())
        # output_temp=output
        # output=torch.exp(output)#这一步真的很重要，为了损失函数的无穷阶可导
        # 不不不，我理解错了，是原论文里的公式17，那样一种简单的形式，将高阶导数化为一阶导的高次幂；这种形式是隐式的将输出的概率预测输入了一个指数函数层，用指数函数层
        # 的某一的输出进行反向传播，得到了公式17中的那样一种结果；但是我们却不需要显式的定义指数函数层，因为公式17已经推导出来了；公式17中显式的包含的是概率预测值关于输出特征图某一元素的偏导数
        # 这是直接我们对神经网络的输出的概率反向传播就可以的了；
        # 我也是在exp()出现了数据溢出这一警告时才发现这一问题的存在，是我理解的偏差了

        ##relu5_2与pool5的输出特征图，类别预测结果（对于cub而言，就是一个200维的向量）

        feature_maps_L16 = features[
            0]  # 如果输入图像的shape为[batch_size,3,224,224]的tensor的话，关于输出的feature map,shape为  #[batch_size,512,14,14]左右
        batch_size, c_L16, h_L16, w_L16 = feature_maps_L16.size()
        feature_maps_L28 = features[
            1]  # 如果输入图像的shape为[batch_size,3,224,224]的tensor的话，关于输出的feature map,shape为  #[batch_size,512,14,14]左右
        batch_size, c_L28, h_L28, w_L28 = feature_maps_L28.size()
        feature_maps_L31 = features[
            2]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        batch_size, c_L31, h_L31, w_L31 = feature_maps_L31.size()
        batch_size, num_classes = output.size()

        # if index == None:
        #     index = np.argmax(output.cpu().data.numpy())#比方说output是：【1,200】，那么np.argmax返回的是单单一个scalar,output[0][index]便能找到这个max
        #######################################################################################################################################
        output_tmp = output.cpu().data.numpy()
        # print(np.sort(output_tmp))
        sorted_output = -np.sort(-output_tmp[0])  # 0指代的是batch_size为1的情况
        # print(sorted_output)
        sorted_output_index = np.argsort(-output_tmp[0])
        # lisalisa=np.exp(sorted_output-np.max(sorted_output))#防止数据溢出的softmax实现
        # sorted_output_probs=lisalisa/np.sum(lisalisa)
        # print(sorted_output_probs[0:10])
        norm_sorted_output = []
        yuri = np.zeros_like(sorted_output)
        for i in range(len(sorted_output) - 1):
            if i == 0:
                yuri[i] = 1
                # miki=sorted_output_probs[i]
                norm_sorted_output.append(sorted_output[i])
            if i <= num_classes - 2:  # 可以调节的超参数
            # if i <= num_classes - 97:  # 可以调节的超参数

                # if sorted_output[i]/sorted_output[i+1]<1.5 and sorted_output[i]-sorted_output[i+1]<sorted_output[i]:  #这个10是可以调节的超参数
                # if sorted_output_probs[i]/sorted_output_probs[i+1]<6:  #这个10是可以调节的超参数
                # if miki + sorted_output_probs[i + 1] <= 0.9:  # 这个10是可以调节的超参数
                yuri[i + 1] = 1
                # miki=miki+sorted_output_probs[i+1]
                norm_sorted_output.append(sorted_output[i + 1])
            else:
                break
        out_max = norm_sorted_output[0]
        out_min = norm_sorted_output[-1]
        norm_sorted_output = (norm_sorted_output - out_min) / (out_max - out_min)

        # print(yuri)
        # print(sorted_output)
        # print(norm_sorted_output)


        feature_maps_L16 = feature_maps_L16.cpu().data.numpy()[0,
                           :]  # 将四维tensor降为三维【512，h，w】，去除batch_szie那一维；这是relu5_2或者pool5的输出特征图
        feature_maps_L28 = feature_maps_L28.cpu().data.numpy()[0,
                           :]  # 将四维tensor降为三维【512，h，w】，去除batch_szie那一维；这是relu5_2或者pool5的输出特征图
        feature_maps_L31 = feature_maps_L31.cpu().data.numpy()[0,
                           :]  # 将四维tensor降为三维【512，h，w】，去除batch_szie那一维；这是relu5_2或者pool5的输出特征图
        # grad_L31_list=[]
        # grad_L28_list=[]
        feature_maps_L31_output = np.zeros_like(feature_maps_L31)
        feature_maps_L28_output = np.zeros_like(feature_maps_L28)
        feature_maps_L16_output = np.zeros_like(feature_maps_L16)


        norm_sorted_output_nosort = np.zeros((1, num_classes))
        one_hot = np.zeros((1, num_classes), dtype=np.float32)
        for kk in range(len(norm_sorted_output) - 1):  # 这个10也是可以调节的超参数
            # if sorted_output_probs[kk]/sorted_output_probs[kk+1]<2 or kk==0:
            if yuri[kk] == 1:
            # if kk == 2:

                one_hot[0][
                    sorted_output_index[kk]] = 1  # 生成了一个[1,200]维的one_hot向量，1所在的位置就是最大值所属类别或者指定类别对应的位置，就是一个数字到类别的对应关系嘛
                norm_sorted_output_nosort[0][
                    sorted_output_index[kk]] = norm_sorted_output[
                    kk]  # 生成了一个[1,200]维的one_hot向量，1所在的位置就是最大值所属类别或者指定类别对应的位置，就是一个数字到类别的对应关系嘛
            # else:
            #     break
        one_hot = torch.from_numpy(one_hot).requires_grad_(
            True)
        norm_sorted_output_nosort = norm_sorted_output_nosort.tolist()
        norm_sorted_output_nosort = torch.tensor(norm_sorted_output_nosort).requires_grad_(
            True).float().cuda()
        # norm_sorted_output_nosort=norm_sorted_output_nosort)
        output = output * norm_sorted_output_nosort
        one_hot = torch.sum(one_hot.cuda() * output)
        self.net.zero_grad()
        one_hot.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚
        grads_val_L31 = self.net.gradients[-3].cpu().data.numpy()
        grads_val_L28 = self.net.gradients[-2].cpu().data.numpy()
        grads_val_L16 = self.net.gradients[-1].cpu().data.numpy()


        feature_maps_L31_output = feature_maps_L31_output + feature_maps_L31 * \
                                  grads_val_L31[0]
        feature_maps_L28_output = feature_maps_L28_output + feature_maps_L28 * \
                                  grads_val_L28[0]
        feature_maps_L16_output = feature_maps_L16_output + feature_maps_L16 * \
                                  grads_val_L16[0]



        cam_L16 = np.sum(feature_maps_L16_output, axis=0)  # 【512，h，w】==>[h,w]
        cam_L28 = np.sum(feature_maps_L28_output, axis=0)  # 【512，h，w】==>[h,w]
        cam_L31 = np.sum(feature_maps_L31_output, axis=0)  # 【512，h，w】==>[h,w]

        # # cam_L4 = np.maximum(cam_L4, 0)  # 这个相当于原论文中的relu,直接将聚合特征图中的小于0的值置零；我待会实验下加或者不加的区别
        # cam_L4 = cam_L4 - np.min(cam_L4)
        # cam_L4 = cam_L4 / np.max(cam_L4)  # 然后呢将其归一化
        #
        # # cam_L9 = np.maximum(cam_L9, 0)  # 这个相当于原论文中的relu,直接将聚合特征图中的小于0的值置零；我待会实验下加或者不加的区别
        # cam_L9 = cam_L9 - np.min(cam_L9)
        # cam_L9 = cam_L9 / np.max(cam_L9)  # 然后呢将其归一化
        #
        # # cam_L16 = np.maximum(cam_L16, 0)  # 这个相当于原论文中的relu,直接将聚合特征图中的小于0的值置零；我待会实验下加或者不加的区别
        # cam_L16 = cam_L16 - np.min(cam_L16)
        # cam_L16 = cam_L16 / np.max(cam_L16)  # 然后呢将其归一化
        #
        # # cam_L28 = np.maximum(cam_L28, 0)  # 这个相当于原论文中的relu,直接将聚合特征图中的小于0的值置零；我待会实验下加或者不加的区别
        # cam_L28 = cam_L28 - np.min(cam_L28)
        # cam_L28 = cam_L28 / np.max(cam_L28)  # 然后呢将其归一化
        #
        # # cam_L31 = np.maximum(cam_L31, 0)  # 这个相当于原论文中的relu,直接将聚合特征图中的小于0的值置零；我待会实验下加或者不加的区别
        # cam_L31 = cam_L31 - np.min(cam_L31)
        # cam_L31 = cam_L31 / (np.max(cam_L31)+1e-20)  # 然后呢将其归一化

        # cam_L31_out=cam_L31
        # cam_L28_out=cam_L28
        # cam_L16_out=cam_L16
        # cam_L9_out=cam_L9
        # cam_L4_out=cam_L4

        # h_L31, w_L31 = cam_L31.shape
        # h_L28, w_L28 = cam_L28.shape
        # h_L16, w_L16 = cam_L16.shape
        # h_L9, w_L9 = cam_L9.shape
        # h_L4, w_L4 = cam_L4.shape
        #
        # cam_L31 = cam_L31.reshape(1, h_L31, w_L31) * np.ones_like(
        #     feature_maps_L31_output)
        # cam_L28 = cam_L28.reshape(1, h_L28, w_L28) * np.ones_like(
        #     feature_maps_L28_output)
        # cam_L16 = cam_L16.reshape(1, h_L16, w_L16) * np.ones_like(
        #     feature_maps_L16_output)
        # cam_L9 = cam_L9.reshape(1, h_L9, w_L9) * np.ones_like(
        #     feature_maps_L9_output)
        # cam_L4 = cam_L4.reshape(1, h_L4, w_L4) * np.ones_like(
        #     feature_maps_L4_output)
        # feature_maps_L31_output = feature_maps_L31_output * cam_L31
        # feature_maps_L28_output = feature_maps_L28_output * cam_L28
        # feature_maps_L16_output = feature_maps_L16_output * cam_L16
        # feature_maps_L9_output = feature_maps_L9_output * cam_L9
        # feature_maps_L4_output = feature_maps_L4_output * cam_L4

        return  cam_L16,cam_L28,cam_L31, feature_maps_L16_output, feature_maps_L28_output, feature_maps_L31_output















class GradCam_yuhan_multiple_classes_WACD_3_no_once_withweight_and_softweight():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, index=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别
        features, output = self.net(input.cuda())
        # output_temp=output
        # output=torch.exp(output)#这一步真的很重要，为了损失函数的无穷阶可导
        # 不不不，我理解错了，是原论文里的公式17，那样一种简单的形式，将高阶导数化为一阶导的高次幂；这种形式是隐式的将输出的概率预测输入了一个指数函数层，用指数函数层
        # 的某一的输出进行反向传播，得到了公式17中的那样一种结果；但是我们却不需要显式的定义指数函数层，因为公式17已经推导出来了；公式17中显式的包含的是概率预测值关于输出特征图某一元素的偏导数
        # 这是直接我们对神经网络的输出的概率反向传播就可以的了；
        # 我也是在exp()出现了数据溢出这一警告时才发现这一问题的存在，是我理解的偏差了

        ##relu5_2与pool5的输出特征图，类别预测结果（对于cub而言，就是一个200维的向量）

        feature_maps_L16 = features[
            0]  # 如果输入图像的shape为[batch_size,3,224,224]的tensor的话，关于输出的feature map,shape为  #[batch_size,512,14,14]左右
        batch_size, c_L16, h_L16, w_L16 = feature_maps_L16.size()
        feature_maps_L28 = features[
            1]  # 如果输入图像的shape为[batch_size,3,224,224]的tensor的话，关于输出的feature map,shape为  #[batch_size,512,14,14]左右
        batch_size, c_L28, h_L28, w_L28 = feature_maps_L28.size()
        feature_maps_L31 = features[
            2]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        batch_size, c_L31, h_L31, w_L31 = feature_maps_L31.size()
        batch_size, num_classes = output.size()

        # if index == None:
        #     index = np.argmax(output.cpu().data.numpy())#比方说output是：【1,200】，那么np.argmax返回的是单单一个scalar,output[0][index]便能找到这个max
        #######################################################################################################################################
        output_tmp = output.cpu().data.numpy()
        # print(np.sort(output_tmp))
        sorted_output = -np.sort(-output_tmp[0])  # 0指代的是batch_size为1的情况
        # print(sorted_output)
        sorted_output_index = np.argsort(-output_tmp[0])
        # lisalisa=np.exp(sorted_output-np.max(sorted_output))#防止数据溢出的softmax实现
        # sorted_output_probs=lisalisa/np.sum(lisalisa)
        # print(sorted_output_probs[0:10])
        norm_sorted_output = []
        yuri = np.zeros_like(sorted_output)
        for i in range(len(sorted_output) - 1):
            if i == 0:
                yuri[i] = 1
                # miki=sorted_output_probs[i]
                norm_sorted_output.append(sorted_output[i])
            if i <= num_classes - 2:  # 可以调节的超参数
                # if sorted_output[i]/sorted_output[i+1]<1.5 and sorted_output[i]-sorted_output[i+1]<sorted_output[i]:  #这个10是可以调节的超参数
                # if sorted_output_probs[i]/sorted_output_probs[i+1]<6:  #这个10是可以调节的超参数
                # if miki + sorted_output_probs[i + 1] <= 0.9:  # 这个10是可以调节的超参数
                yuri[i + 1] = 1
                # miki=miki+sorted_output_probs[i+1]
                norm_sorted_output.append(sorted_output[i + 1])
            else:
                break
        out_max = norm_sorted_output[0]
        out_min = norm_sorted_output[-1]
        norm_sorted_output = (norm_sorted_output - out_min) / (out_max - out_min)

        # print(yuri)
        # print(sorted_output)
        # print(norm_sorted_output)


        feature_maps_L16 = feature_maps_L16.cpu().data.numpy()[0,
                           :]  # 将四维tensor降为三维【512，h，w】，去除batch_szie那一维；这是relu5_2或者pool5的输出特征图
        feature_maps_L28 = feature_maps_L28.cpu().data.numpy()[0,
                           :]  # 将四维tensor降为三维【512，h，w】，去除batch_szie那一维；这是relu5_2或者pool5的输出特征图
        feature_maps_L31 = feature_maps_L31.cpu().data.numpy()[0,
                           :]  # 将四维tensor降为三维【512，h，w】，去除batch_szie那一维；这是relu5_2或者pool5的输出特征图
        # grad_L31_list=[]
        # grad_L28_list=[]
        feature_maps_L31_output = np.zeros_like(feature_maps_L31)
        feature_maps_L28_output = np.zeros_like(feature_maps_L28)
        feature_maps_L16_output = np.zeros_like(feature_maps_L16)


        norm_sorted_output_nosort = np.zeros((1, num_classes))
        one_hot = np.zeros((1, num_classes), dtype=np.float32)
        for kk in range(len(norm_sorted_output) - 1):  # 这个10也是可以调节的超参数
            # if sorted_output_probs[kk]/sorted_output_probs[kk+1]<2 or kk==0:
            if yuri[kk] == 1:
                one_hot[0][
                    sorted_output_index[kk]] = 1  # 生成了一个[1,200]维的one_hot向量，1所在的位置就是最大值所属类别或者指定类别对应的位置，就是一个数字到类别的对应关系嘛
                norm_sorted_output_nosort[0][
                    sorted_output_index[kk]] = norm_sorted_output[
                    kk]  # 生成了一个[1,200]维的one_hot向量，1所在的位置就是最大值所属类别或者指定类别对应的位置，就是一个数字到类别的对应关系嘛
            else:
                break
        one_hot = torch.from_numpy(one_hot).requires_grad_(
            True)
        norm_sorted_output_nosort = norm_sorted_output_nosort.tolist()
        norm_sorted_output_nosort = torch.tensor(norm_sorted_output_nosort).requires_grad_(
            True).float().cuda()
        # norm_sorted_output_nosort=norm_sorted_output_nosort)
        output = output * norm_sorted_output_nosort
        one_hot = torch.sum(one_hot.cuda() * output)
        self.net.zero_grad()
        one_hot.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚
        grads_val_L31 = self.net.gradients[-3].cpu().data.numpy()
        grads_val_L28 = self.net.gradients[-2].cpu().data.numpy()
        grads_val_L16 = self.net.gradients[-1].cpu().data.numpy()


        feature_maps_L31_output = feature_maps_L31_output + feature_maps_L31 * \
                                  grads_val_L31[0]
        feature_maps_L28_output = feature_maps_L28_output + feature_maps_L28 * \
                                  grads_val_L28[0]
        feature_maps_L16_output = feature_maps_L16_output + feature_maps_L16 * \
                                  grads_val_L16[0]



        cam_L16 = np.sum(feature_maps_L16_output, axis=0)  # 【512，h，w】==>[h,w]
        cam_L28 = np.sum(feature_maps_L28_output, axis=0)  # 【512，h，w】==>[h,w]
        cam_L31 = np.sum(feature_maps_L31_output, axis=0)  # 【512，h，w】==>[h,w]

        # # cam_L4 = np.maximum(cam_L4, 0)  # 这个相当于原论文中的relu,直接将聚合特征图中的小于0的值置零；我待会实验下加或者不加的区别
        # cam_L4 = cam_L4 - np.min(cam_L4)
        # cam_L4 = cam_L4 / np.max(cam_L4)  # 然后呢将其归一化
        #
        # # cam_L9 = np.maximum(cam_L9, 0)  # 这个相当于原论文中的relu,直接将聚合特征图中的小于0的值置零；我待会实验下加或者不加的区别
        # cam_L9 = cam_L9 - np.min(cam_L9)
        # cam_L9 = cam_L9 / np.max(cam_L9)  # 然后呢将其归一化
        #
        # # cam_L16 = np.maximum(cam_L16, 0)  # 这个相当于原论文中的relu,直接将聚合特征图中的小于0的值置零；我待会实验下加或者不加的区别
        cam_L16=cam_L16 * -1
        cam_L16 = cam_L16 - np.min(cam_L16)
        cam_L16 = cam_L16 / np.max(cam_L16)  # 然后呢将其归一化
        #
        # # cam_L28 = np.maximum(cam_L28, 0)  # 这个相当于原论文中的relu,直接将聚合特征图中的小于0的值置零；我待会实验下加或者不加的区别
        cam_L28=cam_L28 * -1
        cam_L28 = cam_L28 - np.min(cam_L28)
        cam_L28 = cam_L28 / np.max(cam_L28)  # 然后呢将其归一化
        #
        # # cam_L31 = np.maximum(cam_L31, 0)  # 这个相当于原论文中的relu,直接将聚合特征图中的小于0的值置零；我待会实验下加或者不加的区别
        cam_L31=cam_L31 * -1
        cam_L31 = cam_L31 - np.min(cam_L31)
        cam_L31 = cam_L31 / (np.max(cam_L31))  # 然后呢将其归一化

        # cam_L31_out=cam_L31
        # cam_L28_out=cam_L28
        # cam_L16_out=cam_L16
        # cam_L9_out=cam_L9
        # cam_L4_out=cam_L4

        # h_L31, w_L31 = cam_L31.shape
        # h_L28, w_L28 = cam_L28.shape
        # h_L16, w_L16 = cam_L16.shape
        # h_L9, w_L9 = cam_L9.shape
        # h_L4, w_L4 = cam_L4.shape
        #
        # cam_L31 = cam_L31.reshape(1, h_L31, w_L31) * np.ones_like(
        #     feature_maps_L31_output)
        # cam_L28 = cam_L28.reshape(1, h_L28, w_L28) * np.ones_like(
        #     feature_maps_L28_output)
        # cam_L16 = cam_L16.reshape(1, h_L16, w_L16) * np.ones_like(
        #     feature_maps_L16_output)
        # cam_L9 = cam_L9.reshape(1, h_L9, w_L9) * np.ones_like(
        #     feature_maps_L9_output)
        # cam_L4 = cam_L4.reshape(1, h_L4, w_L4) * np.ones_like(
        #     feature_maps_L4_output)
        # feature_maps_L31_output = feature_maps_L31_output * cam_L31
        # feature_maps_L28_output = feature_maps_L28_output * cam_L28
        # feature_maps_L16_output = feature_maps_L16_output * cam_L16
        # feature_maps_L9_output = feature_maps_L9_output * cam_L9
        # feature_maps_L4_output = feature_maps_L4_output * cam_L4

        return  cam_L16,cam_L28,cam_L31, feature_maps_L16_output, feature_maps_L28_output, feature_maps_L31_output




















class GradCam_yuhan_multiple_classes_WACD_3_no_once_withnewweight_and_weight():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, index=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别
        features, output = self.net(input.cuda())
        # output_temp=output
        # output=torch.exp(output)#这一步真的很重要，为了损失函数的无穷阶可导
        # 不不不，我理解错了，是原论文里的公式17，那样一种简单的形式，将高阶导数化为一阶导的高次幂；这种形式是隐式的将输出的概率预测输入了一个指数函数层，用指数函数层
        # 的某一的输出进行反向传播，得到了公式17中的那样一种结果；但是我们却不需要显式的定义指数函数层，因为公式17已经推导出来了；公式17中显式的包含的是概率预测值关于输出特征图某一元素的偏导数
        # 这是直接我们对神经网络的输出的概率反向传播就可以的了；
        # 我也是在exp()出现了数据溢出这一警告时才发现这一问题的存在，是我理解的偏差了

        ##relu5_2与pool5的输出特征图，类别预测结果（对于cub而言，就是一个200维的向量）

        feature_maps_L16 = features[
            0]  # 如果输入图像的shape为[batch_size,3,224,224]的tensor的话，关于输出的feature map,shape为  #[batch_size,512,14,14]左右
        batch_size, c_L16, h_L16, w_L16 = feature_maps_L16.size()
        feature_maps_L28 = features[
            1]  # 如果输入图像的shape为[batch_size,3,224,224]的tensor的话，关于输出的feature map,shape为  #[batch_size,512,14,14]左右
        batch_size, c_L28, h_L28, w_L28 = feature_maps_L28.size()
        feature_maps_L31 = features[
            2]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        batch_size, c_L31, h_L31, w_L31 = feature_maps_L31.size()
        batch_size, num_classes = output.size()

        # if index == None:
        #     index = np.argmax(output.cpu().data.numpy())#比方说output是：【1,200】，那么np.argmax返回的是单单一个scalar,output[0][index]便能找到这个max
        #######################################################################################################################################
        output_tmp = output.cpu().data.numpy() * ( -1 )
        # print(np.sort(output_tmp))
        sorted_output = -np.sort(-output_tmp[0])  # 0指代的是batch_size为1的情况
        # print(sorted_output)
        sorted_output_index = np.argsort(-output_tmp[0])
        # lisalisa=np.exp(sorted_output-np.max(sorted_output))#防止数据溢出的softmax实现
        # sorted_output_probs=lisalisa/np.sum(lisalisa)
        # print(sorted_output_probs[0:10])
        norm_sorted_output = []
        yuri = np.zeros_like(sorted_output)
        for i in range(len(sorted_output) - 1):
            if i == 0:
                yuri[i] = 1
                # miki=sorted_output_probs[i]
                norm_sorted_output.append(sorted_output[i])
            if i <= num_classes - 2:  # 可以调节的超参数
                # if sorted_output[i]/sorted_output[i+1]<1.5 and sorted_output[i]-sorted_output[i+1]<sorted_output[i]:  #这个10是可以调节的超参数
                # if sorted_output_probs[i]/sorted_output_probs[i+1]<6:  #这个10是可以调节的超参数
                # if miki + sorted_output_probs[i + 1] <= 0.9:  # 这个10是可以调节的超参数
                yuri[i + 1] = 1
                # miki=miki+sorted_output_probs[i+1]
                norm_sorted_output.append(sorted_output[i + 1])
            else:
                break
        out_max = norm_sorted_output[0]
        out_min = norm_sorted_output[-1]
        norm_sorted_output = (norm_sorted_output - out_min) / (out_max - out_min)

        # print(yuri)
        # print(sorted_output)
        # print(norm_sorted_output)


        feature_maps_L16 = feature_maps_L16.cpu().data.numpy()[0,
                           :]  # 将四维tensor降为三维【512，h，w】，去除batch_szie那一维；这是relu5_2或者pool5的输出特征图
        feature_maps_L28 = feature_maps_L28.cpu().data.numpy()[0,
                           :]  # 将四维tensor降为三维【512，h，w】，去除batch_szie那一维；这是relu5_2或者pool5的输出特征图
        feature_maps_L31 = feature_maps_L31.cpu().data.numpy()[0,
                           :]  # 将四维tensor降为三维【512，h，w】，去除batch_szie那一维；这是relu5_2或者pool5的输出特征图
        # grad_L31_list=[]
        # grad_L28_list=[]
        feature_maps_L31_output = np.zeros_like(feature_maps_L31)
        feature_maps_L28_output = np.zeros_like(feature_maps_L28)
        feature_maps_L16_output = np.zeros_like(feature_maps_L16)


        norm_sorted_output_nosort = np.zeros((1, num_classes))
        one_hot = np.zeros((1, num_classes), dtype=np.float32)
        for kk in range(len(norm_sorted_output) - 1):  # 这个10也是可以调节的超参数
            # if sorted_output_probs[kk]/sorted_output_probs[kk+1]<2 or kk==0:
            if yuri[kk] == 1:
                one_hot[0][
                    sorted_output_index[kk]] = 1  # 生成了一个[1,200]维的one_hot向量，1所在的位置就是最大值所属类别或者指定类别对应的位置，就是一个数字到类别的对应关系嘛
                norm_sorted_output_nosort[0][
                    sorted_output_index[kk]] = norm_sorted_output[
                    kk]  # 生成了一个[1,200]维的one_hot向量，1所在的位置就是最大值所属类别或者指定类别对应的位置，就是一个数字到类别的对应关系嘛
            else:
                break
        one_hot = torch.from_numpy(one_hot).requires_grad_(
            True)
        norm_sorted_output_nosort = norm_sorted_output_nosort.tolist()
        norm_sorted_output_nosort = torch.tensor(norm_sorted_output_nosort).requires_grad_(
            True).float().cuda()
        # norm_sorted_output_nosort=norm_sorted_output_nosort)
        output = output * norm_sorted_output_nosort
        one_hot = torch.sum(one_hot.cuda() * output)
        self.net.zero_grad()
        one_hot.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚
        grads_val_L31 = self.net.gradients[-3].cpu().data.numpy()
        grads_val_L28 = self.net.gradients[-2].cpu().data.numpy()
        grads_val_L16 = self.net.gradients[-1].cpu().data.numpy()


        feature_maps_L31_output = feature_maps_L31_output + feature_maps_L31 * \
                                  grads_val_L31[0]
        feature_maps_L28_output = feature_maps_L28_output + feature_maps_L28 * \
                                  grads_val_L28[0]
        feature_maps_L16_output = feature_maps_L16_output + feature_maps_L16 * \
                                  grads_val_L16[0]



        cam_L16 = np.sum(feature_maps_L16_output, axis=0)  # 【512，h，w】==>[h,w]
        cam_L28 = np.sum(feature_maps_L28_output, axis=0)  # 【512，h，w】==>[h,w]
        cam_L31 = np.sum(feature_maps_L31_output, axis=0)  # 【512，h，w】==>[h,w]

        # # cam_L4 = np.maximum(cam_L4, 0)  # 这个相当于原论文中的relu,直接将聚合特征图中的小于0的值置零；我待会实验下加或者不加的区别
        # cam_L4 = cam_L4 - np.min(cam_L4)
        # cam_L4 = cam_L4 / np.max(cam_L4)  # 然后呢将其归一化
        #
        # # cam_L9 = np.maximum(cam_L9, 0)  # 这个相当于原论文中的relu,直接将聚合特征图中的小于0的值置零；我待会实验下加或者不加的区别
        # cam_L9 = cam_L9 - np.min(cam_L9)
        # cam_L9 = cam_L9 / np.max(cam_L9)  # 然后呢将其归一化
        #
        # # cam_L16 = np.maximum(cam_L16, 0)  # 这个相当于原论文中的relu,直接将聚合特征图中的小于0的值置零；我待会实验下加或者不加的区别
        # cam_L16 = cam_L16 - np.min(cam_L16)
        # cam_L16 = cam_L16 / np.max(cam_L16)  # 然后呢将其归一化
        #
        # # cam_L28 = np.maximum(cam_L28, 0)  # 这个相当于原论文中的relu,直接将聚合特征图中的小于0的值置零；我待会实验下加或者不加的区别
        # cam_L28 = cam_L28 - np.min(cam_L28)
        # cam_L28 = cam_L28 / np.max(cam_L28)  # 然后呢将其归一化
        #
        # # cam_L31 = np.maximum(cam_L31, 0)  # 这个相当于原论文中的relu,直接将聚合特征图中的小于0的值置零；我待会实验下加或者不加的区别
        # cam_L31 = cam_L31 - np.min(cam_L31)
        # cam_L31 = cam_L31 / (np.max(cam_L31)+1e-20)  # 然后呢将其归一化

        # cam_L31_out=cam_L31
        # cam_L28_out=cam_L28
        # cam_L16_out=cam_L16
        # cam_L9_out=cam_L9
        # cam_L4_out=cam_L4

        # h_L31, w_L31 = cam_L31.shape
        # h_L28, w_L28 = cam_L28.shape
        # h_L16, w_L16 = cam_L16.shape
        # h_L9, w_L9 = cam_L9.shape
        # h_L4, w_L4 = cam_L4.shape
        #
        # cam_L31 = cam_L31.reshape(1, h_L31, w_L31) * np.ones_like(
        #     feature_maps_L31_output)
        # cam_L28 = cam_L28.reshape(1, h_L28, w_L28) * np.ones_like(
        #     feature_maps_L28_output)
        # cam_L16 = cam_L16.reshape(1, h_L16, w_L16) * np.ones_like(
        #     feature_maps_L16_output)
        # cam_L9 = cam_L9.reshape(1, h_L9, w_L9) * np.ones_like(
        #     feature_maps_L9_output)
        # cam_L4 = cam_L4.reshape(1, h_L4, w_L4) * np.ones_like(
        #     feature_maps_L4_output)
        # feature_maps_L31_output = feature_maps_L31_output * cam_L31
        # feature_maps_L28_output = feature_maps_L28_output * cam_L28
        # feature_maps_L16_output = feature_maps_L16_output * cam_L16
        # feature_maps_L9_output = feature_maps_L9_output * cam_L9
        # feature_maps_L4_output = feature_maps_L4_output * cam_L4

        return  cam_L16,cam_L28,cam_L31, feature_maps_L16_output, feature_maps_L28_output, feature_maps_L31_output













class GradCam_yuhan_multiple_classes_WACD_3_no_once_noweight_and_weight():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, index=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别
        features, output = self.net(input.cuda())
        # output_temp=output
        # output=torch.exp(output)#这一步真的很重要，为了损失函数的无穷阶可导
        # 不不不，我理解错了，是原论文里的公式17，那样一种简单的形式，将高阶导数化为一阶导的高次幂；这种形式是隐式的将输出的概率预测输入了一个指数函数层，用指数函数层
        # 的某一的输出进行反向传播，得到了公式17中的那样一种结果；但是我们却不需要显式的定义指数函数层，因为公式17已经推导出来了；公式17中显式的包含的是概率预测值关于输出特征图某一元素的偏导数
        # 这是直接我们对神经网络的输出的概率反向传播就可以的了；
        # 我也是在exp()出现了数据溢出这一警告时才发现这一问题的存在，是我理解的偏差了

        ##relu5_2与pool5的输出特征图，类别预测结果（对于cub而言，就是一个200维的向量）

        feature_maps_L16 = features[
            0]  # 如果输入图像的shape为[batch_size,3,224,224]的tensor的话，关于输出的feature map,shape为  #[batch_size,512,14,14]左右
        batch_size, c_L16, h_L16, w_L16 = feature_maps_L16.size()
        feature_maps_L28 = features[
            1]  # 如果输入图像的shape为[batch_size,3,224,224]的tensor的话，关于输出的feature map,shape为  #[batch_size,512,14,14]左右
        batch_size, c_L28, h_L28, w_L28 = feature_maps_L28.size()
        feature_maps_L31 = features[
            2]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        batch_size, c_L31, h_L31, w_L31 = feature_maps_L31.size()
        batch_size, num_classes = output.size()

        # if index == None:
        #     index = np.argmax(output.cpu().data.numpy())#比方说output是：【1,200】，那么np.argmax返回的是单单一个scalar,output[0][index]便能找到这个max
        #######################################################################################################################################
        output_tmp = output.cpu().data.numpy()
        # print(np.sort(output_tmp))
        sorted_output = -np.sort(-output_tmp[0])  # 0指代的是batch_size为1的情况
        # print(sorted_output)
        sorted_output_index = np.argsort(-output_tmp[0])
        # lisalisa=np.exp(sorted_output-np.max(sorted_output))#防止数据溢出的softmax实现
        # sorted_output_probs=lisalisa/np.sum(lisalisa)
        # print(sorted_output_probs[0:10])
        norm_sorted_output = []
        yuri = np.zeros_like(sorted_output)
        for i in range(len(sorted_output) - 1):
            if i == 0:
                yuri[i] = 1
                # miki=sorted_output_probs[i]
                norm_sorted_output.append(sorted_output[i])
            if i <= num_classes - 2:  # 可以调节的超参数
                # if sorted_output[i]/sorted_output[i+1]<1.5 and sorted_output[i]-sorted_output[i+1]<sorted_output[i]:  #这个10是可以调节的超参数
                # if sorted_output_probs[i]/sorted_output_probs[i+1]<6:  #这个10是可以调节的超参数
                # if miki + sorted_output_probs[i + 1] <= 0.9:  # 这个10是可以调节的超参数
                yuri[i + 1] = 1
                # miki=miki+sorted_output_probs[i+1]
                norm_sorted_output.append(sorted_output[i + 1])
            else:
                break
        out_max = norm_sorted_output[0]
        out_min = norm_sorted_output[-1]
        norm_sorted_output = (norm_sorted_output - out_min) / (out_max - out_min)

        # print(yuri)
        # print(sorted_output)
        # print(norm_sorted_output)


        feature_maps_L16 = feature_maps_L16.cpu().data.numpy()[0,
                           :]  # 将四维tensor降为三维【512，h，w】，去除batch_szie那一维；这是relu5_2或者pool5的输出特征图
        feature_maps_L28 = feature_maps_L28.cpu().data.numpy()[0,
                           :]  # 将四维tensor降为三维【512，h，w】，去除batch_szie那一维；这是relu5_2或者pool5的输出特征图
        feature_maps_L31 = feature_maps_L31.cpu().data.numpy()[0,
                           :]  # 将四维tensor降为三维【512，h，w】，去除batch_szie那一维；这是relu5_2或者pool5的输出特征图
        # grad_L31_list=[]
        # grad_L28_list=[]
        feature_maps_L31_output = np.zeros_like(feature_maps_L31)
        feature_maps_L28_output = np.zeros_like(feature_maps_L28)
        feature_maps_L16_output = np.zeros_like(feature_maps_L16)


        norm_sorted_output_nosort = np.zeros((1, num_classes))
        one_hot = np.zeros((1, num_classes), dtype=np.float32)
        for kk in range(len(norm_sorted_output) - 1):  # 这个10也是可以调节的超参数
            # if sorted_output_probs[kk]/sorted_output_probs[kk+1]<2 or kk==0:
            if yuri[kk] == 1:
                one_hot[0][
                    sorted_output_index[kk]] = 1  # 生成了一个[1,200]维的one_hot向量，1所在的位置就是最大值所属类别或者指定类别对应的位置，就是一个数字到类别的对应关系嘛
                # norm_sorted_output_nosort[0][
                #     sorted_output_index[kk]] = norm_sorted_output[
                #     kk]  # 生成了一个[1,200]维的one_hot向量，1所在的位置就是最大值所属类别或者指定类别对应的位置，就是一个数字到类别的对应关系嘛
            else:
                break
        one_hot = torch.from_numpy(one_hot).requires_grad_(
            True)
        # norm_sorted_output_nosort = norm_sorted_output_nosort.tolist()
        # norm_sorted_output_nosort = torch.tensor(norm_sorted_output_nosort).requires_grad_(
        #     True).float().cuda()
        # norm_sorted_output_nosort=norm_sorted_output_nosort)
        # output = output * norm_sorted_output_nosort
        one_hot = torch.sum(one_hot.cuda() * output)
        self.net.zero_grad()
        one_hot.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚
        grads_val_L31 = self.net.gradients[-3].cpu().data.numpy()
        grads_val_L28 = self.net.gradients[-2].cpu().data.numpy()
        grads_val_L16 = self.net.gradients[-1].cpu().data.numpy()


        feature_maps_L31_output = feature_maps_L31_output + feature_maps_L31 * \
                                  grads_val_L31[0]
        feature_maps_L28_output = feature_maps_L28_output + feature_maps_L28 * \
                                  grads_val_L28[0]
        feature_maps_L16_output = feature_maps_L16_output + feature_maps_L16 * \
                                  grads_val_L16[0]



        cam_L16 = np.sum(feature_maps_L16_output, axis=0)  # 【512，h，w】==>[h,w]
        cam_L28 = np.sum(feature_maps_L28_output, axis=0)  # 【512，h，w】==>[h,w]
        cam_L31 = np.sum(feature_maps_L31_output, axis=0)  # 【512，h，w】==>[h,w]

        # # cam_L4 = np.maximum(cam_L4, 0)  # 这个相当于原论文中的relu,直接将聚合特征图中的小于0的值置零；我待会实验下加或者不加的区别
        # cam_L4 = cam_L4 - np.min(cam_L4)
        # cam_L4 = cam_L4 / np.max(cam_L4)  # 然后呢将其归一化
        #
        # # cam_L9 = np.maximum(cam_L9, 0)  # 这个相当于原论文中的relu,直接将聚合特征图中的小于0的值置零；我待会实验下加或者不加的区别
        # cam_L9 = cam_L9 - np.min(cam_L9)
        # cam_L9 = cam_L9 / np.max(cam_L9)  # 然后呢将其归一化
        #
        # # cam_L16 = np.maximum(cam_L16, 0)  # 这个相当于原论文中的relu,直接将聚合特征图中的小于0的值置零；我待会实验下加或者不加的区别
        # cam_L16 = cam_L16 - np.min(cam_L16)
        # cam_L16 = cam_L16 / np.max(cam_L16)  # 然后呢将其归一化
        #
        # # cam_L28 = np.maximum(cam_L28, 0)  # 这个相当于原论文中的relu,直接将聚合特征图中的小于0的值置零；我待会实验下加或者不加的区别
        # cam_L28 = cam_L28 - np.min(cam_L28)
        # cam_L28 = cam_L28 / np.max(cam_L28)  # 然后呢将其归一化
        #
        # # cam_L31 = np.maximum(cam_L31, 0)  # 这个相当于原论文中的relu,直接将聚合特征图中的小于0的值置零；我待会实验下加或者不加的区别
        # cam_L31 = cam_L31 - np.min(cam_L31)
        # cam_L31 = cam_L31 / (np.max(cam_L31)+1e-20)  # 然后呢将其归一化

        # cam_L31_out=cam_L31
        # cam_L28_out=cam_L28
        # cam_L16_out=cam_L16
        # cam_L9_out=cam_L9
        # cam_L4_out=cam_L4

        # h_L31, w_L31 = cam_L31.shape
        # h_L28, w_L28 = cam_L28.shape
        # h_L16, w_L16 = cam_L16.shape
        # h_L9, w_L9 = cam_L9.shape
        # h_L4, w_L4 = cam_L4.shape
        #
        # cam_L31 = cam_L31.reshape(1, h_L31, w_L31) * np.ones_like(
        #     feature_maps_L31_output)
        # cam_L28 = cam_L28.reshape(1, h_L28, w_L28) * np.ones_like(
        #     feature_maps_L28_output)
        # cam_L16 = cam_L16.reshape(1, h_L16, w_L16) * np.ones_like(
        #     feature_maps_L16_output)
        # cam_L9 = cam_L9.reshape(1, h_L9, w_L9) * np.ones_like(
        #     feature_maps_L9_output)
        # cam_L4 = cam_L4.reshape(1, h_L4, w_L4) * np.ones_like(
        #     feature_maps_L4_output)
        # feature_maps_L31_output = feature_maps_L31_output * cam_L31
        # feature_maps_L28_output = feature_maps_L28_output * cam_L28
        # feature_maps_L16_output = feature_maps_L16_output * cam_L16
        # feature_maps_L9_output = feature_maps_L9_output * cam_L9
        # feature_maps_L4_output = feature_maps_L4_output * cam_L4

        return  cam_L16,cam_L28,cam_L31, feature_maps_L16_output, feature_maps_L28_output, feature_maps_L31_output


















class GradCam_pp_multiple_classes_WACD_noweight():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, index=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别
        features, output = self.net(input.cuda())

        # output=torch.exp(output)#这一步真的很重要，为了损失函数的无穷阶可导
        # 不不不，我理解错了，是原论文里的公式17，那样一种简单的形式，将高阶导数化为一阶导的高次幂；这种形式是隐式的将输出的概率预测输入了一个指数函数层，用指数函数层
        # 的某一的输出进行反向传播，得到了公式17中的那样一种结果；但是我们却不需要显式的定义指数函数层，因为公式17已经推导出来了；公式17中显式的包含的是概率预测值关于输出特征图某一元素的偏导数
        # 这是直接我们对神经网络的输出的概率反向传播就可以的了；
        # 我也是在exp()出现了数据溢出这一警告时才发现这一问题的存在，是我理解的偏差了

        ##relu5_2与pool5的输出特征图，类别预测结果（对于cub而言，就是一个200维的向量）
        feature_maps_L28 = features[
            0]  # 如果输入图像的shape为[batch_size,3,224,224]的tensor的话，关于输出的feature map,shape为  #[batch_size,512,14,14]左右
        batch_size, c_L28, h_L28, w_L28 = feature_maps_L28.size()
        feature_maps_L31 = features[
            1]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        batch_size, c_L31, h_L31, w_L31 = feature_maps_L31.size()
        batch_size, num_classes = output.size()

        # if index == None:
        #     index = np.argmax(output.cpu().data.numpy())#比方说output是：【1,200】，那么np.argmax返回的是单单一个scalar,output[0][index]便能找到这个max
        #######################################################################################################################################
        output_tmp = output.cpu().data.numpy()
        # print(np.sort(output_tmp))
        sorted_output = -np.sort(-output_tmp[0])  # 0指代的是batch_size为1的情况
        # print(sorted_output)
        sorted_output_index = np.argsort(-output_tmp[0])
        # lisalisa=np.exp(sorted_output-np.max(sorted_output))#防止数据溢出的softmax实现
        # sorted_output_probs=lisalisa/np.sum(lisalisa)
        # print(sorted_output_probs[0:10])
        # norm_sorted_output = []
        yuri = np.zeros_like(sorted_output)
        for i in range(len(sorted_output) - 1):
            if i == 0:
                yuri[i] = 1
                # miki=sorted_output_probs[i]
                # norm_sorted_output.append(sorted_output[i])
            if i <= 5:  # 可以调节的超参数
                # if sorted_output[i]/sorted_output[i+1]<1.5 and sorted_output[i]-sorted_output[i+1]<sorted_output[i]:  #这个10是可以调节的超参数
                # if sorted_output_probs[i]/sorted_output_probs[i+1]<6:  #这个10是可以调节的超参数
                # if miki + sorted_output_probs[i + 1] <= 0.9:  # 这个10是可以调节的超参数
                yuri[i + 1] = 1
                # miki=miki+sorted_output_probs[i+1]
                # norm_sorted_output.append(sorted_output[i + 1])
            else:
                break
        # out_max = norm_sorted_output[0]
        # out_min = norm_sorted_output[-1]
        # norm_sorted_output = (norm_sorted_output - out_min) / (out_max - out_min)

        # print(yuri)
        # print(sorted_output)
        # print(norm_sorted_output)

        cam_L28_sum = np.zeros([h_L28, w_L28], dtype=np.float32)
        cam_L31_sum = np.zeros([h_L31, w_L31], dtype=np.float32)

        feature_maps_L28 = feature_maps_L28.cpu().data.numpy()[0,
                           :]  # 将四维tensor降为三维【512，h，w】，去除batch_szie那一维；这是relu5_2或者pool5的输出特征图
        feature_maps_L31 = feature_maps_L31.cpu().data.numpy()[0,
                           :]  # 将四维tensor降为三维【512，h，w】，去除batch_szie那一维；这是relu5_2或者pool5的输出特征图
        # grad_L31_list=[]
        # grad_L28_list=[]
        feature_maps_L31_output = np.zeros_like(feature_maps_L31)
        feature_maps_L28_output = np.zeros_like(feature_maps_L28)

        for kk in range(len(sorted_output)):  # 这个10也是可以调节的超参数
            # if sorted_output_probs[kk]/sorted_output_probs[kk+1]<2 or kk==0:
            if yuri[kk] == 1:
                one_hot = np.zeros((1, num_classes), dtype=np.float32)
                one_hot[0][
                    sorted_output_index[kk]] = 1  # 生成了一个[1,200]维的one_hot向量，1所在的位置就是最大值所属类别或者指定类别对应的位置，就是一个数字到类别的对应关系嘛
                one_hot = torch.from_numpy(one_hot).requires_grad_(
                    True)  # 我们将该onehot向量转化为一个gpu tensor,与输出的类别想乘，然后再求和，得到一个标量，损失函数就必须是
                # 标量的形式，这也是我们引入one_hot的原因，只考虑我们想要关注的那个类别的预测值；损失函数的这个标量在数值上与最大的类别预测值是相等的，反向传播时别的类别的预测值也就完全不起作用
                one_hot = torch.sum(one_hot.cuda() * output)
                self.net.zero_grad()
                one_hot.backward(retain_graph=True)  # 这里为什么要保留计算图啊？搞不清楚
                # 另外，关于这个retain_graph=True
                # 如果一次前向处理对应着多次的反向传播的话，必须进行这样的设置；但是这样的设置所带来的问题就是
                # 虽然每一次反向传播互不影响干扰，但是叶节点的梯度在每一次反向传播上是累加在一起的，
                # 所以self.net.zero_grad()，它确保了将这种累加给去除掉
                # 但是，我们现在所要获取的梯度是根节点的梯度，根节点的梯度我们是通过hook来获取的，
                # 那么，self.net.zero_grad()是否能够起到跟叶节点同样的作用呢？或者说
                # self.net.zero_grad()是否是必须的呢？因为我们知道根节点的梯度在反向传播之后是会被自动free掉的
                # 好的，我已经测试完成了，具体可见test_hook.py,  self.net.zero_grad()在我现在的情景下，对于根节点是可有可无并非必须的

                # grads_val_L28 = self.net.gradients[0].cpu().data.numpy()
                # print(grads_val_L28.shape)#(1, 512, 7, 10)
                # grads_val_L31 = self.net.gradients[1].cpu().data.numpy()
                # print(grads_val_L31.shape)#(1, 512, 14, 20)   #你看，这不就想当然啦，正好给整反了，反向传播的梯度，当然是，处在网络靠后位置的输出特征图所对应的梯度在前边啦
                # 也就是self.net.gradients存储着先是L31对应的梯度，然后才是L28对应的梯度

                # print('yu',len(self.net.gradients))
                # 现在，一次前向处理对应着多次反向传播是吧；但是，存储hook的梯度的列表却只在每一次前向处理时才会被清空
                # 如果一次前向处理对应着多次的反向传播，那么；这多次反向传播所对应的梯度是会按顺序累计存储在这样一个列表中的
                # 0  #第一次前向处理
                # yu 2     #1bp
                # yu 4      #2bp
                # yu 6       #3bp
                # 1   #第二次前向处理
                # yu 2      #1bp
                # yu 4       #2bp
                # 2   #第三次前向处理
                # yu 2
                # yu 4
                # yu 6
                # yu 8  ......................
                # 我们需要每一次反向传播所对应的两个梯度，那就应该去获取列表的最后两个元素，而不是最开始的两个元素
                # 所要取的元素所对应的列表下标：      0,1=====》变成了-2，-1
                grads_val_L31 = self.net.gradients[-2].cpu().data.numpy()
                # grad_L31_list.append(grads_val_L31)
                # if len(grad_L31_list)>1:
                #     grads_val_L31=grads_val_L31 - grad_L31_list[-2]
                # print(grads_val_L31)
                # print(grads_val_L31[0][1])

                grads_val_L28 = self.net.gradients[-1].cpu().data.numpy()
                # grad_L28_list.append(grads_val_L28)
                # if len(grad_L28_list)>1:
                #     grads_val_L28=grads_val_L28 - grad_L28_list[-2]
                # print(grads_val_L28)
                feature_maps_L31_output = feature_maps_L31_output + feature_maps_L31 * \
                                          grads_val_L31[0]
                feature_maps_L28_output = feature_maps_L28_output + feature_maps_L28 * \
                                          grads_val_L28[0]


            else:
                one_hot.backward(retain_graph=False)
                self.net.zero_grad()
                break

        return  feature_maps_L28_output, feature_maps_L31_output











class GradCam_pp_multiple_classes_WACD_new():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, index=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别
        features, output = self.net(input.cuda())

        # output=torch.exp(output)#这一步真的很重要，为了损失函数的无穷阶可导
        # 不不不，我理解错了，是原论文里的公式17，那样一种简单的形式，将高阶导数化为一阶导的高次幂；这种形式是隐式的将输出的概率预测输入了一个指数函数层，用指数函数层
        # 的某一的输出进行反向传播，得到了公式17中的那样一种结果；但是我们却不需要显式的定义指数函数层，因为公式17已经推导出来了；公式17中显式的包含的是概率预测值关于输出特征图某一元素的偏导数
        # 这是直接我们对神经网络的输出的概率反向传播就可以的了；
        # 我也是在exp()出现了数据溢出这一警告时才发现这一问题的存在，是我理解的偏差了

        ##relu5_2与pool5的输出特征图，类别预测结果（对于cub而言，就是一个200维的向量）
        feature_maps_L28 = features[
            0]  # 如果输入图像的shape为[batch_size,3,224,224]的tensor的话，关于输出的feature map,shape为  #[batch_size,512,14,14]左右
        batch_size, c_L28, h_L28, w_L28 = feature_maps_L28.size()
        feature_maps_L31 = features[
            1]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        batch_size, c_L31, h_L31, w_L31 = feature_maps_L31.size()
        batch_size, num_classes = output.size()

        # if index == None:
        #     index = np.argmax(output.cpu().data.numpy())#比方说output是：【1,200】，那么np.argmax返回的是单单一个scalar,output[0][index]便能找到这个max
        #######################################################################################################################################
        output_tmp = output.cpu().data.numpy()
        # print(np.sort(output_tmp))
        sorted_output = -np.sort(-output_tmp[0])  # 0指代的是batch_size为1的情况
        # print(sorted_output)
        sorted_output_index = np.argsort(-output_tmp[0])
        # lisalisa=np.exp(sorted_output-np.max(sorted_output))#防止数据溢出的softmax实现
        # sorted_output_probs=lisalisa/np.sum(lisalisa)
        # print(sorted_output_probs[0:10])
        norm_sorted_output = []
        yuri = np.zeros_like(sorted_output)
        for i in range(len(sorted_output) - 1):
            if i == 0:
                yuri[i] = 1
                # miki=sorted_output_probs[i]
                norm_sorted_output.append(sorted_output[i])
            if i <= 5:  # 可以调节的超参数
                # if sorted_output[i]/sorted_output[i+1]<1.5 and sorted_output[i]-sorted_output[i+1]<sorted_output[i]:  #这个10是可以调节的超参数
                # if sorted_output_probs[i]/sorted_output_probs[i+1]<6:  #这个10是可以调节的超参数
                # if miki + sorted_output_probs[i + 1] <= 0.9:  # 这个10是可以调节的超参数
                yuri[i + 1] = 1
                # miki=miki+sorted_output_probs[i+1]
                norm_sorted_output.append(sorted_output[i + 1])
            else:
                break
        out_max = norm_sorted_output[0]
        out_min = norm_sorted_output[-1]
        norm_sorted_output = (norm_sorted_output - out_min) / (out_max - out_min)

        # print(yuri)
        # print(sorted_output)
        # print(norm_sorted_output)

        cam_L28_sum = np.zeros([h_L28, w_L28], dtype=np.float32)
        cam_L31_sum = np.zeros([h_L31, w_L31], dtype=np.float32)

        feature_maps_L28 = feature_maps_L28.cpu().data.numpy()[0,
                           :]  # 将四维tensor降为三维【512，h，w】，去除batch_szie那一维；这是relu5_2或者pool5的输出特征图
        feature_maps_L31 = feature_maps_L31.cpu().data.numpy()[0,
                           :]  # 将四维tensor降为三维【512，h，w】，去除batch_szie那一维；这是relu5_2或者pool5的输出特征图
        # grad_L31_list=[]
        # grad_L28_list=[]
        feature_maps_L31_output = np.zeros_like(feature_maps_L31)
        feature_maps_L28_output = np.zeros_like(feature_maps_L28)

        for kk in range(len(norm_sorted_output)):  # 这个10也是可以调节的超参数
            # if sorted_output_probs[kk]/sorted_output_probs[kk+1]<2 or kk==0:
            if yuri[kk] == 1:
                one_hot = np.zeros((1, num_classes), dtype=np.float32)
                one_hot[0][
                    sorted_output_index[kk]] = 1  # 生成了一个[1,200]维的one_hot向量，1所在的位置就是最大值所属类别或者指定类别对应的位置，就是一个数字到类别的对应关系嘛
                one_hot = torch.from_numpy(one_hot).requires_grad_(
                    True)  # 我们将该onehot向量转化为一个gpu tensor,与输出的类别想乘，然后再求和，得到一个标量，损失函数就必须是
                # 标量的形式，这也是我们引入one_hot的原因，只考虑我们想要关注的那个类别的预测值；损失函数的这个标量在数值上与最大的类别预测值是相等的，反向传播时别的类别的预测值也就完全不起作用
                one_hot = torch.sum(one_hot.cuda() * output)
                self.net.zero_grad()
                one_hot.backward(retain_graph=True)  # 这里为什么要保留计算图啊？搞不清楚
                # 另外，关于这个retain_graph=True
                # 如果一次前向处理对应着多次的反向传播的话，必须进行这样的设置；但是这样的设置所带来的问题就是
                # 虽然每一次反向传播互不影响干扰，但是叶节点的梯度在每一次反向传播上是累加在一起的，
                # 所以self.net.zero_grad()，它确保了将这种累加给去除掉
                # 但是，我们现在所要获取的梯度是根节点的梯度，根节点的梯度我们是通过hook来获取的，
                # 那么，self.net.zero_grad()是否能够起到跟叶节点同样的作用呢？或者说
                # self.net.zero_grad()是否是必须的呢？因为我们知道根节点的梯度在反向传播之后是会被自动free掉的
                # 好的，我已经测试完成了，具体可见test_hook.py,  self.net.zero_grad()在我现在的情景下，对于根节点是可有可无并非必须的

                # grads_val_L28 = self.net.gradients[0].cpu().data.numpy()
                # print(grads_val_L28.shape)#(1, 512, 7, 10)
                # grads_val_L31 = self.net.gradients[1].cpu().data.numpy()
                # print(grads_val_L31.shape)#(1, 512, 14, 20)   #你看，这不就想当然啦，正好给整反了，反向传播的梯度，当然是，处在网络靠后位置的输出特征图所对应的梯度在前边啦
                # 也就是self.net.gradients存储着先是L31对应的梯度，然后才是L28对应的梯度

                # print('yu',len(self.net.gradients))
                # 现在，一次前向处理对应着多次反向传播是吧；但是，存储hook的梯度的列表却只在每一次前向处理时才会被清空
                # 如果一次前向处理对应着多次的反向传播，那么；这多次反向传播所对应的梯度是会按顺序累计存储在这样一个列表中的
                # 0  #第一次前向处理
                # yu 2     #1bp
                # yu 4      #2bp
                # yu 6       #3bp
                # 1   #第二次前向处理
                # yu 2      #1bp
                # yu 4       #2bp
                # 2   #第三次前向处理
                # yu 2
                # yu 4
                # yu 6
                # yu 8  ......................
                # 我们需要每一次反向传播所对应的两个梯度，那就应该去获取列表的最后两个元素，而不是最开始的两个元素
                # 所要取的元素所对应的列表下标：      0,1=====》变成了-2，-1
                grads_val_L31 = self.net.gradients[-2].cpu().data.numpy()
                # grad_L31_list.append(grads_val_L31)
                # if len(grad_L31_list)>1:
                #     grads_val_L31=grads_val_L31 - grad_L31_list[-2]
                # print(grads_val_L31)
                # print(grads_val_L31[0][1])

                grads_val_L28 = self.net.gradients[-1].cpu().data.numpy()
                # grad_L28_list.append(grads_val_L28)
                # if len(grad_L28_list)>1:
                #     grads_val_L28=grads_val_L28 - grad_L28_list[-2]
                # print(grads_val_L28)
                feature_maps_L31_output = feature_maps_L31_output + norm_sorted_output[kk] * feature_maps_L31 * \
                                          grads_val_L31[0]
                feature_maps_L28_output = feature_maps_L28_output + norm_sorted_output[kk] * feature_maps_L28 * \
                                          grads_val_L28[0]

                # if sorted_output[kk]>0:
                #     feature_maps_L31_output=feature_maps_L31_output + sorted_output[ kk ] * feature_maps_L31*grads_val_L31[0]
                #     feature_maps_L28_output=feature_maps_L28_output + sorted_output[ kk ] * feature_maps_L28*grads_val_L28[0]
                # else:
                #     feature_maps_L31_output = feature_maps_L31_output + feature_maps_L31 * \
                #                               grads_val_L31[0]
                #     feature_maps_L28_output = feature_maps_L28_output + feature_maps_L28 * \
                #                               grads_val_L28[0]

                # self.extractor.gradients返回的是一个列表，这个列表可能不止一个元素，我们取出最后一个元素来，比方说一个【1，512,7,7】的偏导数tensor,并最终将其转化为ndarray
                # 它进过进一步的处理，得到一个512维的权重向量，我们可以据此得到h*w的聚合特征图

                # batch_size, grads_c_L31, grads_h_L31, grads_w_L31 = grads_val_L31.shape
                # batch_size, grads_c_L28, grads_h_L28, grads_w_L28 = grads_val_L28.shape
                #
                # pixelwise_grads_weight_L31 = np.zeros_like(grads_val_L31, dtype=np.float32)
                # pixelwise_grads_weight_L28 = np.zeros_like(grads_val_L28, dtype=np.float32)
                # # print(pixelwise_grads_weight_L31.shape)
                # # print(pixelwise_grads_weight_L28.shape)
                #
                # # print(grads_val_L31.shape)
                # # print(grads_val_L28.shape)
                # # [1,512,h,w]
                # feature_maps_L31_sum = np.sum(feature_maps_L31, axis=(1, 2))
                # # feature_maps_L31_sum=np.expand_dims(feature_maps_L31_sum,axis=0)*np.ones_like(pixelwise_grads_weight_L31)
                # feature_maps_L31_sum = feature_maps_L31_sum.reshape(1, grads_c_L31, 1, 1) * np.ones_like(
                #     pixelwise_grads_weight_L31)
                #
                # # pixelwise_grads_weight_L31=(grads_val_L31**2)/((grads_val_L31**2)*2+(grads_val_L31**3)*feature_maps_L31_sum)
                # a = grads_val_L31 ** 2
                # b = (grads_val_L31 ** 2) * 2 + (grads_val_L31 ** 3) * feature_maps_L31_sum
                # # print(b)
                # pixelwise_grads_weight_L31 = np.divide(a, b, out=np.zeros_like(grads_val_L31), where=b != 0)
                #
                # feature_maps_L28_sum = np.sum(feature_maps_L28, axis=(1, 2))
                # # feature_maps_L28_sum=np.expand_dims(feature_maps_L28_sum,axis=0)*np.ones_like(pixelwise_grads_weight_L28)
                # feature_maps_L28_sum = feature_maps_L28_sum.reshape(1, grads_c_L28, 1, 1) * np.ones_like(
                #     pixelwise_grads_weight_L28)
                # # pixelwise_grads_weight_L28 = (grads_val_L28 ** 2) / (
                # #             (grads_val_L28 ** 2) * 2 + (grads_val_L28 ** 3) * feature_maps_L28_sum)
                # a1, b1 = grads_val_L28 ** 2, (grads_val_L28 ** 2) * 2 + (grads_val_L28 ** 3) * feature_maps_L28_sum
                # pixelwise_grads_weight_L28 = np.divide(a1, b1, out=np.zeros_like(grads_val_L28), where=b1 != 0)
                #
                # # for i in range(grads_c_L31):
                # #     for j in range(grads_h_L31):
                # #         for k in range(grads_w_L31):
                # #             if grads_val_L31[0,i,j,k]==0:
                # #                 pixelwise_grads_weight_L31[0, i, j, k]=0
                # #             else:
                # #                 pixelwise_grads_weight_L31[0,i,j,k]=(grads_val_L31[0,i,j,k]**2)\
                # #                                         /((grads_val_L31[0,i,j,k]**2)*2+(grads_val_L31[0,i,j,k]**3)*np.sum(feature_maps_L31[i,:]))
                # # print(grads_val_L28[0,:])
                # # for i in range(grads_c_L28):
                # #     for j in range(grads_h_L28):
                # #         for k in range(grads_w_L28):
                # #             if grads_val_L28[0, i, j, k] == 0:
                # #                 pixelwise_grads_weight_L28[0,i,j,k]=0
                # #             else:
                # #                 pixelwise_grads_weight_L28[0, i, j, k] = (grads_val_L28[0, i, j, k] ** 2) \
                # #                                                         / ((grads_val_L28[0, i, j, k] ** 2) * 2 + (grads_val_L28[0, i, j, k] ** 3) * np.sum(feature_maps_L28[i, :]))
                #
                # grads_val_L28 = np.maximum(grads_val_L28, 0)
                # grads_val_L31 = np.maximum(grads_val_L31, 0)  # 相当于relu操作
                # weights_L31 = pixelwise_grads_weight_L31 * grads_val_L31  # 这里是点乘操作，进行像素级的加权，最终得到的ndarray的[1,512,h,w]
                # weights_L31 = np.sum(weights_L31, axis=(2, 3))[0, :]  # 【1,512，h，w】==>[1,512]===>[512,]
                # weights_L28 = pixelwise_grads_weight_L28 * grads_val_L28
                # weights_L28 = np.sum(weights_L28, axis=(2, 3))[0, :]
                #
                # # weights_L28 = np.mean(grads_val_L28, axis=(2, 3))[0, :]#【1,512，h，w】==>[1,512]===>[512,]
                # # weights_L31 = np.mean(grads_val_L31, axis=(2, 3))[0, :]#【1,512，h，w】==>[1,512]===>[512,]
                # # print(weights_L28)
                # cam_L28 = np.zeros(feature_maps_L28.shape[1:], dtype=np.float32)  # [2h,2w]左右
                # cam_L31 = np.zeros(feature_maps_L31.shape[1:], dtype=np.float32)  # [h,w]
                #
                # for i, w in enumerate(weights_L28):
                #     cam_L28 += w * feature_maps_L28[i, :, :]  # [h,w]
                #     # if i in range(6):
                #     #     print(cam_L28)
                # for i, w in enumerate(weights_L31):
                #     cam_L31 += w * feature_maps_L31[i, :, :]  # [h,w]
                # # print(cam_L31)
                # # print(feature_maps_L28[0, :, :])
                # cam_L28 = np.maximum(cam_L28, 0)  # 这个相当于原论文中的relu,直接将聚合特征图中的小于0的值置零；我待会实验下加或者不加的区别
                # # cam_L28 = cv2.resize(cam_L28, input.shape[2:])#想要将其变为一个灰度图像，尺寸与原输入网络图像的长宽相同
                # cam_L28 = cam_L28 - np.min(cam_L28)
                # cam_L28 = cam_L28 / np.max(cam_L28)  # 然后呢将其归一化
                #
                # cam_L31 = np.maximum(cam_L31, 0)  # 这个相当于原论文中的relu,直接将聚合特征图中的小于0的值置零；我待会实验下加或者不加的区别
                # # cam_L31 = cv2.resize(cam_L31, input.shape[2:])  # 想要将其变为一个灰度图像，尺寸与原输入网络图像的长宽相同
                # cam_L31 = cam_L31 - np.min(cam_L31)
                # cam_L31 = cam_L31 / np.max(cam_L31)  # 然后呢将其归一化
                #
                # # if sorted_output[kk]>0:
                # #     cam_L28_sum=cam_L28_sum+cam_L28*sorted_output[kk]
                # #     cam_L31_sum=cam_L31_sum+cam_L31*sorted_output[kk]
                # # else:
                # #     cam_L28_sum = cam_L28_sum + cam_L28
                # #     cam_L31_sum = cam_L31_sum + cam_L31
                #
                # cam_L28_sum = cam_L28_sum + cam_L28 * norm_sorted_output[kk]
                # cam_L31_sum = cam_L31_sum + cam_L31 * norm_sorted_output[kk]
            else:
                one_hot.backward(retain_graph=False)
                self.net.zero_grad()
                break

            # print(cam_L31_sum)

            # cam_L28_sum = np.maximum(cam_L28_sum, 0)  # 这个相当于原论文中的relu,直接将聚合特征图中的小于0的值置零；我待会实验下加或者不加的区别
            #
            # # cam_L28 = cv2.resize(cam_L28, input.shape[2:])#想要将其变为一个灰度图像，尺寸与原输入网络图像的长宽相同
            # cam_L28_sum = cam_L28_sum - np.min(cam_L28_sum)
            # cam_L28_sum = cam_L28_sum / np.max(cam_L28_sum)  # 然后呢将其归一化
            #
            # cam_L31_sum = np.maximum(cam_L31_sum, 0)  # 这个相当于原论文中的relu,直接将聚合特征图中的小于0的值置零；我待会实验下加或者不加的区别
            #
            # # cam_L31 = cv2.resize(cam_L31, input.shape[2:])  # 想要将其变为一个灰度图像，尺寸与原输入网络图像的长宽相同
            # cam_L31_sum = cam_L31_sum - np.min(cam_L31_sum)
            # # print(np.max(cam_L31_sum))
            # # print(cam_L31_sum)
            # cam_L31_sum = cam_L31_sum / np.max(cam_L31_sum)  # 然后呢将其归一化
            # # print(cam_L31_sum)
            # # print(cam_L28_sum)

        return  feature_maps_L28_output, feature_maps_L31_output










class GradCam_new_many_layers_yu():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, index=None):#通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别
        lisalisa=input.requires_grad_(True).cuda()
        features, output = self.net(lisalisa)

        # output_max = torch.max(output)
        # output_min = torch.min(output)
        # output_norm = torch.div((output - output_min).float(), (output_max - output_min))
        ################################################################3
        output_norm = (output* -1 ) .cpu().tolist()
        output_norm = torch.tensor(output_norm).float().cuda()
        ###################################################################
        one_hot = torch.sum(output * output_norm)
        self.net.zero_grad()
        one_hot.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚


        features_maps=[]
        hata=lisalisa.grad * lisalisa
        hata = hata.cpu().data.numpy()[0, :]
        features_maps.append(hata)
        for i in range(len(features)):
            miki=features[i]*self.net.gradients[-1*(i+1)]
            miki=miki.cpu().data.numpy()[0,:]
            features_maps.append(miki)



        cams=[]
        for i in range(len(features_maps)):
            miki=np.sum(features_maps[i], axis=0)
            print(miki)
            miki = np.maximum(miki*-1, 0)
            miki = miki - np.min(miki)
            miki = miki / np.max(miki)
            cams.append(miki)

        return cams,features_maps













class GradCam_new_many_layers():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, index=None):#通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别
        lisalisa=input.requires_grad_(True).cuda()
        features, output = self.net(lisalisa)
        # ##relu5_2与pool5的输出特征图，类别预测结果（对于cub而言，就是一个200维的向量）
        # # feature_maps_L28 = features[0]  # 如果输入图像的shape为[batch_size,3,224,224]的tensor的话，关于输出的feature map,shape为  #[batch_size,512,14,14]左右
        # # batch_size, c_L28, h_L28, w_L28 = feature_maps_L28.size()
        # # feature_maps_L31 = features[1]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        # # batch_size, c_L31, h_L31, w_L31 = feature_maps_L31.size()
        # batch_size,num_classes=output.size()
        #
        # if index == None:
        #     index = np.argmax(output.cpu().data.numpy())#比方说output是：【1,200】，那么np.argmax返回的是单单一个scalar,output[0][index]便能找到这个max
        #
        # one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        # one_hot[0][index] = 1  #生成了一个[1,200]维的one_hot向量，1所在的位置就是最大值所属类别或者指定类别对应的位置，就是一个数字到类别的对应关系嘛
        # one_hot = torch.from_numpy(one_hot).requires_grad_(True)#我们将该onehot向量转化为一个gpu tensor,与输出的类别想乘，然后再求和，得到一个标量，损失函数就必须是
        # #标量的形式，这也是我们引入one_hot的原因，只考虑我们想要关注的那个类别的预测值；损失函数的这个标量在数值上与最大的类别预测值是相等的，反向传播时别的类别的预测值也就完全不起作用
        # one_hot = torch.sum(one_hot.cuda() * output)
        # self.net.zero_grad()
        # one_hot.backward(retain_graph=True)#这里为什么要保留计算图啊？搞不清楚

        # features,output = self.net(input.cuda())
        output_max = torch.max(output)
        output_min = torch.min(output)
        output_norm = torch.div((output - output_min).float(), (output_max - output_min))
        output_norm = torch.exp(output_norm)
        ################################################################3
        output_norm = (output_norm *-1).cpu().tolist()
        output_norm = torch.tensor(output_norm).float().cuda()
        ###################################################################
        one_hot = torch.sum(output * output_norm)
        self.net.zero_grad()
        one_hot.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #



        # grads_val_L28 = self.net.gradients[0].cpu().data.numpy()
        # print(grads_val_L28.shape)#(1, 512, 7, 10)
        # grads_val_L31 = self.net.gradients[1].cpu().data.numpy()
        # print(grads_val_L31.shape)#(1, 512, 14, 20)   #你看，这不就想当然啦，正好给整反了，反向传播的梯度，当然是，处在网络靠后位置的输出特征图所对应的梯度在前边啦
        #也就是self.net.gradients存储着先是L31对应的梯度，然后才是L28对应的梯度
        # grads_val_L31 = self.net.gradients[0]
        # grads_val_L28 = self.net.gradients[1]

        # feature_maps_L28=feature_maps_L28*grads_val_L28
        # feature_maps_L31=feature_maps_L31*grads_val_L31

        features_maps=[]
        # hata=lisalisa.grad * lisalisa
        # hata = hata.cpu().data.numpy()[0, :]
        # features_maps.append(hata)
        for i in range(len(features)):
            miki=features[i]*self.net.gradients[-1*(i+1)]
            miki=miki.cpu().data.numpy()[0,:]
            features_maps.append(miki)


        #self.extractor.gradients返回的是一个列表，这个列表可能不止一个元素，我们取出最后一个元素来，比方说一个【1，512,7,7】的偏导数tensor,并最终将其转化为ndarray
        #它进过进一步的处理，得到一个512维的权重向量，我们可以据此得到h*w的聚合特征图

        # feature_maps_L28= feature_maps_L28.cpu().data.numpy()[0, :]#将四维tensor降为三维【512，h，w】，去除batch_szie那一维；这是relu5_2或者pool5的输出特征图
        # feature_maps_L31= feature_maps_L31.cpu().data.numpy()[0, :]#将四维tensor降为三维【512，h，w】，去除batch_szie那一维；这是relu5_2或者pool5的输出特征图

        # weights_L28 = np.mean(grads_val_L28, axis=(2, 3))[0, :]#【1,512，h，w】==>[1,512]===>[512,]
        # weights_L31 = np.mean(grads_val_L31, axis=(2, 3))[0, :]#【1,512，h，w】==>[1,512]===>[512,]
        # print(weights_L28)
        # cam_L28 = np.zeros(feature_maps_L28.shape[1:], dtype=np.float32)#[2h,2w]左右
        # cam_L31 = np.zeros(feature_maps_L31.shape[1:], dtype=np.float32)#[h,w]
        cams=[]
        for i in range(len(features_maps)):
            miki=np.sum(features_maps[i], axis=0)

            # miki = np.maximum(miki, 0)
            miki = miki - np.min(miki)
            miki = miki / np.max(miki)
            print(miki)
            cams.append(miki)
        # cam_L28 = np.sum(feature_maps_L28, axis = 0 )#【512，h，w】==>[h,w]
        # cam_L31 = np.sum(feature_maps_L31, axis = 0 )#【512，h，w】==>[h,w]

        # for i, w in enumerate(weights_L28):
        #     cam_L28 += w * feature_maps_L28[i, :, :]#[h,w]

        # for i, w in enumerate(weights_L31):
        #     cam_L31 += w * feature_maps_L31[i, :, :]#[h,w]
        # print(cam_L31)
        # print(feature_maps_L28[0, :, :])
        # cam_L28 = np.maximum(cam_L28, 0)#这个相当于原论文中的relu,直接将聚合特征图中的小于0的值置零；我待会实验下加或者不加的区别
        # cam_L28 = cv2.resize(cam_L28, input.shape[2:])#想要将其变为一个灰度图像，尺寸与原输入网络图像的长宽相同
        # cam_L28 = cam_L28 - np.min(cam_L28)
        # cam_L28 = cam_L28 / np.max(cam_L28)#然后呢将其归一化

        # cam_L31 = np.maximum(cam_L31, 0)  # 这个相当于原论文中的relu,直接将聚合特征图中的小于0的值置零；我待会实验下加或者不加的区别
        # cam_L31 = cv2.resize(cam_L31, input.shape[2:])  # 想要将其变为一个灰度图像，尺寸与原输入网络图像的长宽相同
        # cam_L31 = cam_L31 - np.min(cam_L31)
        # cam_L31 = cam_L31 / np.max(cam_L31)  # 然后呢将其归一化
        return cams,features_maps



class GradCam_new_many_layers_3out():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, index=None):#通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别
        lisalisa=input.requires_grad_(True).cuda()
        features, output,_ = self.net(lisalisa)

        output_max = torch.max(output)
        output_min = torch.min(output)
        output_norm = torch.div((output - output_min).float(), (output_max - output_min))
        # output_norm = torch.exp(output_norm)
        ################################################################3
        output_norm = (output_norm *-1).cpu().tolist()
        output_norm = torch.tensor(output_norm).float().cuda()
        ###################################################################
        one_hot = torch.sum(output * output_norm)
        self.net.zero_grad()
        one_hot.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚

        features_maps=[]
        # hata=lisalisa.grad * lisalisa
        # hata = hata.cpu().data.numpy()[0, :]
        # features_maps.append(hata)
        for i in range(len(features)):
            miki=features[i]*self.net.gradients[-1*(i+1)]
            miki=miki.cpu().data.numpy()[0,:]
            # miki = np.maximum(miki, 0)

            features_maps.append(miki)

        cams=[]
        cams_relu=[]
        for i in range(len(features_maps)):
            miki=np.sum(features_maps[i], axis=0)
            # miki = np.maximum(miki, 0)
            print(miki)
            print("***********************************************")
            # print(miki)
            miki = miki - np.min(miki)
            miki = miki / np.max(miki)
            cams.append(miki)
            print(miki)
        for i in range(len(features_maps)):
            miki=np.sum(features_maps[i], axis=0)
            miki = np.maximum(miki, 0)
            print(miki)
            print("***********************************************")
            # print(miki)
            miki = miki - np.min(miki)
            miki = miki / np.max(miki)
            cams_relu.append(miki)
            print(miki)

        return cams,cams_relu







class GradCam_new_many_layers_3out0():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, index=None):#通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别
        lisalisa=input.requires_grad_(True).cuda()
        features, output,_ = self.net(lisalisa)

        output_max = torch.max(output)
        output_min = torch.min(output)
        output_norm = torch.div((output - output_min).float(), (output_max - output_min))
        # output_norm = torch.exp(output_norm)
        ################################################################3
        output_norm = (output_norm).cpu().tolist()
        output_norm = torch.tensor(output_norm).float().cuda()
        ###################################################################
        one_hot = torch.sum(output * output_norm)
        self.net.zero_grad()
        one_hot.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚

        features_maps=[]
        # hata=lisalisa.grad * lisalisa
        # hata = hata.cpu().data.numpy()[0, :]
        # features_maps.append(hata)
        for i in range(len(features)):

            grads_val = self.net.gradients[-1 * (i + 1)].cpu().data.numpy()
            feature_maps_i = features[i].cpu().data.numpy()[0,
                             :]  # 将四维tensor降为三维【512，h，w】，去除batch_szie那一维；这是relu5_2或者pool5的输出特征图
            weights_i = np.mean(grads_val, axis=(2, 3))[0, :]
            cam_i = np.zeros(feature_maps_i.shape[1:], dtype=np.float32)  # [2h,2w]左右
            for j, w in enumerate(weights_i):
                cam_i += w * feature_maps_i[j, :, :]  # [h,w]
            # cam_i = np.maximum(cam_i, 0)#这个相当于原论文中的relu,直接将聚合特征图中的小于0的值置零；我待会实验下加或者不加的区别

            # miki=features[i]*self.net.gradients[-1*(i+1)]
            # miki=miki.cpu().data.numpy()[0,:]

            # miki = np.maximum(miki, 0)

            features_maps.append(cam_i)

        cams = []
        cams_relu = []
        for i in range(len(features_maps)):
            # miki=np.sum(features_maps[i], axis=0)
            miki = features_maps[i]

            # print(miki)
            miki = np.maximum(miki, 0)
            # print(miki)
            print("***********************************************")
            miki = miki - np.min(miki)
            miki = miki / np.max(miki)
            cams_relu.append(miki)
            # print(miki)
        for i in range(len(features_maps)):
            # miki=np.sum(features_maps[i], axis=0)
            miki = features_maps[i]
            # print(miki)
            # miki = np.maximum(miki, 0)
            # print(miki)
            print("***********************************************")
            miki = miki - np.min(miki)
            miki = miki / np.max(miki)
            cams.append(miki)
            # print(miki)

        return cams, cams_relu







class PoolCam_new_many_layers_3out0():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, index=None):#通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别
        lisalisa=input.requires_grad_(True).cuda()
        features, output,_ = self.net(lisalisa)

        feature_maps_L31 = features[
            0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        # feature_maps_L31 = feature_maps_L31[0]  # [1,512,7,7]==>[512,7,7]

        feature_maps_L31_max = global_avg_pool(feature_maps_L31)
        feature_maps_L31_mean = global_max_pool(feature_maps_L31)
        # print(feature_maps_L31_max.size())
        # feature_maps_L31_max = feature_maps_L31_max.view( -1)
        # feature_maps_L31_mean = feature_maps_L31_mean.view(-1)
        feature_maps_L31_mean_max = torch.cat((feature_maps_L31_mean, feature_maps_L31_max), dim=1)
        print(feature_maps_L31.size())
        # print(feature_maps_L31_mean_max.size())
        loss2 = torch.sum(feature_maps_L31_mean_max)
        # print(loss2)
        self.net.zero_grad()
        loss2.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚

        features_maps=[]
        # hata=lisalisa.grad * lisalisa
        # hata = hata.cpu().data.numpy()[0, :]
        # features_maps.append(hata)
        for i in range(len(features)):

            grads_val = self.net.gradients[-1 * (i + 1)].cpu().data.numpy()
            feature_maps_i = features[i].cpu().data.numpy()[0,
                             :]  # 将四维tensor降为三维【512，h，w】，去除batch_szie那一维；这是relu5_2或者pool5的输出特征图
            weights_i = np.mean(grads_val, axis=(2, 3))[0, :]
            print(weights_i)
            cam_i = np.zeros(feature_maps_i.shape[1:], dtype=np.float32)  # [2h,2w]左右
            for j, w in enumerate(weights_i):
                cam_i += w * feature_maps_i[j, :, :]  # [h,w]
            # cam_i = np.maximum(cam_i, 0)#这个相当于原论文中的relu,直接将聚合特征图中的小于0的值置零；我待会实验下加或者不加的区别

            # miki=features[i]*self.net.gradients[-1*(i+1)]
            # miki=miki.cpu().data.numpy()[0,:]

            # miki = np.maximum(miki, 0)

            features_maps.append(cam_i)

        cams = []
        cams_relu = []
        for i in range(len(features_maps)):
            # miki=np.sum(features_maps[i], axis=0)
            miki = features_maps[i]

            # print(miki)
            miki = np.maximum(miki, 0)
            # print(miki)
            print("***********************************************")
            miki = miki - np.min(miki)
            miki = miki / np.max(miki)
            cams_relu.append(miki)
            # print(miki)
        for i in range(len(features_maps)):
            # miki=np.sum(features_maps[i], axis=0)
            miki = features_maps[i]
            # print(miki)
            # miki = np.maximum(miki, 0)
            # print(miki)
            print("***********************************************")
            miki = miki - np.min(miki)
            miki = miki / np.max(miki)
            cams.append(miki)
            # print(miki)

        return cams, cams_relu








class PoolCam_new_many_layers_3out0_saliency_map():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, index=None):#通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别
        lisalisa=input.requires_grad_(True).cuda()
        features, output,_ = self.net(lisalisa)

        feature_maps_L31 = features[
            0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        # Pool5   #feature_maps_L31  [512,7,7]==[512,]
        batch_size, c_L31, h_L31, w_L31 = feature_maps_L31.size()
        feature_maps_L31_sum = torch.sum(feature_maps_L31[0],
                                         0)  # 如果要将CAM或者gard_cam引入SCDA，需要更改的程序也就仅仅是这一步
        L31_sum_mean = torch.mean(feature_maps_L31_sum)  # 返回的是scalar tensor ,这是与matlab的mean不同的地方
        # largestConnectComponent将最大连通区域所对应的像素点置为true
        jj = (feature_maps_L31_sum > L31_sum_mean).cpu().data.numpy()
        feature_maps_L31 = feature_maps_L31[0].cpu().data.numpy()
        highlight_conn_L31 = jj.reshape(1, h_L31, w_L31) * np.ones_like(
            feature_maps_L31)


        feature_maps_L31 = features[
            0]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        # feature_maps_L31 = feature_maps_L31[0]  # [1,512,7,7]==>[512,7,7]
        feature_maps_L31 = feature_maps_L31 * torch.from_numpy(highlight_conn_L31).unsqueeze(0).cuda().float()
        feature_maps_L31_max = global_avg_pool(feature_maps_L31)
        feature_maps_L31_mean = global_max_pool(feature_maps_L31)
        # print(feature_maps_L31_max.size())
        # feature_maps_L31_max = feature_maps_L31_max.view( -1)
        # feature_maps_L31_mean = feature_maps_L31_mean.view(-1)
        feature_maps_L31_mean_max = torch.cat((feature_maps_L31_mean, feature_maps_L31_max), dim=1)
        # print(feature_maps_L31_mean_max.size())
        loss2 = torch.sum(feature_maps_L31_mean_max)
        # print(loss2)
        self.net.zero_grad()
        loss2.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚

        features_maps=[]
        # hata=lisalisa.grad * lisalisa
        # hata = hata.cpu().data.numpy()[0, :]
        # features_maps.append(hata)
        for i in range(len(features)):

            grads_val = self.net.gradients[-1 * (i + 1)].cpu().data.numpy()
            feature_maps_i = features[i].cpu().data.numpy()[0,
                             :]  # 将四维tensor降为三维【512，h，w】，去除batch_szie那一维；这是relu5_2或者pool5的输出特征图
            weights_i = np.mean(grads_val, axis=(2, 3))[0, :]
            cam_i = np.zeros(feature_maps_i.shape[1:], dtype=np.float32)  # [2h,2w]左右
            for j, w in enumerate(weights_i):
                cam_i += w * feature_maps_i[j, :, :]  # [h,w]
            # cam_i = np.maximum(cam_i, 0)#这个相当于原论文中的relu,直接将聚合特征图中的小于0的值置零；我待会实验下加或者不加的区别

            # miki=features[i]*self.net.gradients[-1*(i+1)]
            # miki=miki.cpu().data.numpy()[0,:]

            # miki = np.maximum(miki, 0)

            features_maps.append(cam_i)

        cams = []
        cams_relu = []
        for i in range(len(features_maps)):
            # miki=np.sum(features_maps[i], axis=0)
            miki = features_maps[i]

            # print(miki)
            miki = np.maximum(miki, 0)
            # print(miki)
            print("***********************************************")
            miki = miki - np.min(miki)
            miki = miki / np.max(miki)
            cams_relu.append(miki)
            # print(miki)
        for i in range(len(features_maps)):
            # miki=np.sum(features_maps[i], axis=0)
            miki = features_maps[i]
            # print(miki)
            # miki = np.maximum(miki, 0)
            # print(miki)
            print("***********************************************")
            miki = miki - np.min(miki)
            miki = miki / np.max(miki)
            cams.append(miki)
            # print(miki)

        return cams, cams_relu









class GradCam_new_many_layers_version_2():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, index=None):#通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别
        lisalisa=input.requires_grad_(True).cuda()
        features, output = self.net(lisalisa)
        # ##relu5_2与pool5的输出特征图，类别预测结果（对于cub而言，就是一个200维的向量）
        # # feature_maps_L28 = features[0]  # 如果输入图像的shape为[batch_size,3,224,224]的tensor的话，关于输出的feature map,shape为  #[batch_size,512,14,14]左右
        # # batch_size, c_L28, h_L28, w_L28 = feature_maps_L28.size()
        # # feature_maps_L31 = features[1]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        # # batch_size, c_L31, h_L31, w_L31 = feature_maps_L31.size()
        # batch_size,num_classes=output.size()
        #
        # if index == None:
        #     index = np.argmax(output.cpu().data.numpy())#比方说output是：【1,200】，那么np.argmax返回的是单单一个scalar,output[0][index]便能找到这个max
        #
        # one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        # one_hot[0][index] = 1  #生成了一个[1,200]维的one_hot向量，1所在的位置就是最大值所属类别或者指定类别对应的位置，就是一个数字到类别的对应关系嘛
        # one_hot = torch.from_numpy(one_hot).requires_grad_(True)#我们将该onehot向量转化为一个gpu tensor,与输出的类别想乘，然后再求和，得到一个标量，损失函数就必须是
        # #标量的形式，这也是我们引入one_hot的原因，只考虑我们想要关注的那个类别的预测值；损失函数的这个标量在数值上与最大的类别预测值是相等的，反向传播时别的类别的预测值也就完全不起作用
        # one_hot = torch.sum(one_hot.cuda() * output)
        # self.net.zero_grad()
        # one_hot.backward(retain_graph=True)#这里为什么要保留计算图啊？搞不清楚

        # features,output = self.net(input.cuda())
        output_max = torch.max(output)
        output_min = torch.min(output)
        output_norm = torch.div((output - output_min).float(), (output_max - output_min))
        output_norm = torch.exp(output_norm)
        ################################################################3
        output_norm = output_norm.cpu().tolist()
        output_norm = torch.tensor(output_norm).float().cuda()
        ###################################################################
        one_hot = torch.sum(output * output_norm)
        self.net.zero_grad()
        one_hot.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚
        # params = list(self.net.parameters())
        # weight_cnn_5_3 = np.squeeze(params[-3].data.cpu().numpy())  #



        # grads_val_L28 = self.net.gradients[0].cpu().data.numpy()
        # print(grads_val_L28.shape)#(1, 512, 7, 10)
        # grads_val_L31 = self.net.gradients[1].cpu().data.numpy()
        # print(grads_val_L31.shape)#(1, 512, 14, 20)   #你看，这不就想当然啦，正好给整反了，反向传播的梯度，当然是，处在网络靠后位置的输出特征图所对应的梯度在前边啦
        #也就是self.net.gradients存储着先是L31对应的梯度，然后才是L28对应的梯度
        # grads_val_L31 = self.net.gradients[0]
        # grads_val_L28 = self.net.gradients[1]

        # feature_maps_L28=feature_maps_L28*grads_val_L28
        # feature_maps_L31=feature_maps_L31*grads_val_L31

        features_maps=[]
        # hata=lisalisa.grad * lisalisa
        # hata = hata.cpu().data.numpy()[0, :]
        # features_maps.append(hata)
        for i in range(len(features)):
            miki=features[i]*self.net.gradients[-1*(i+1)]
            miki=miki.cpu().data.numpy()[0,:]
            features_maps.append(miki)


        #self.extractor.gradients返回的是一个列表，这个列表可能不止一个元素，我们取出最后一个元素来，比方说一个【1，512,7,7】的偏导数tensor,并最终将其转化为ndarray
        #它进过进一步的处理，得到一个512维的权重向量，我们可以据此得到h*w的聚合特征图

        # feature_maps_L28= feature_maps_L28.cpu().data.numpy()[0, :]#将四维tensor降为三维【512，h，w】，去除batch_szie那一维；这是relu5_2或者pool5的输出特征图
        # feature_maps_L31= feature_maps_L31.cpu().data.numpy()[0, :]#将四维tensor降为三维【512，h，w】，去除batch_szie那一维；这是relu5_2或者pool5的输出特征图

        # weights_L28 = np.mean(grads_val_L28, axis=(2, 3))[0, :]#【1,512，h，w】==>[1,512]===>[512,]
        # weights_L31 = np.mean(grads_val_L31, axis=(2, 3))[0, :]#【1,512，h，w】==>[1,512]===>[512,]
        # print(weights_L28)
        # cam_L28 = np.zeros(feature_maps_L28.shape[1:], dtype=np.float32)#[2h,2w]左右
        # cam_L31 = np.zeros(feature_maps_L31.shape[1:], dtype=np.float32)#[h,w]
        cams=[]
        for i in range(len(features_maps)):
            miki=np.sum(features_maps[i], axis=0)
            print(miki)
            # miki = np.maximum(miki, 0)
            miki = miki - np.min(miki)
            miki = miki / np.max(miki)
            cams.append(miki)
            print(miki)

        # cam_L28 = np.sum(feature_maps_L28, axis = 0 )#【512，h，w】==>[h,w]
        # cam_L31 = np.sum(feature_maps_L31, axis = 0 )#【512，h，w】==>[h,w]

        # for i, w in enumerate(weights_L28):
        #     cam_L28 += w * feature_maps_L28[i, :, :]#[h,w]

        # for i, w in enumerate(weights_L31):
        #     cam_L31 += w * feature_maps_L31[i, :, :]#[h,w]
        # print(cam_L31)
        # print(feature_maps_L28[0, :, :])
        # cam_L28 = np.maximum(cam_L28, 0)#这个相当于原论文中的relu,直接将聚合特征图中的小于0的值置零；我待会实验下加或者不加的区别
        # cam_L28 = cv2.resize(cam_L28, input.shape[2:])#想要将其变为一个灰度图像，尺寸与原输入网络图像的长宽相同
        # cam_L28 = cam_L28 - np.min(cam_L28)
        # cam_L28 = cam_L28 / np.max(cam_L28)#然后呢将其归一化

        # cam_L31 = np.maximum(cam_L31, 0)  # 这个相当于原论文中的relu,直接将聚合特征图中的小于0的值置零；我待会实验下加或者不加的区别
        # cam_L31 = cv2.resize(cam_L31, input.shape[2:])  # 想要将其变为一个灰度图像，尺寸与原输入网络图像的长宽相同
        # cam_L31 = cam_L31 - np.min(cam_L31)
        # cam_L31 = cam_L31 / np.max(cam_L31)  # 然后呢将其归一化
        return cams,features_maps



class GradCam_new_many_layers_version_2_3out():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, index=None):#通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别
        lisalisa=input.requires_grad_(True).cuda()
        features, output,_ = self.net(lisalisa)

        output_max = torch.max(output)
        output_min = torch.min(output)
        output_norm = torch.div((output - output_min).float(), (output_max - output_min))
        # output_norm = torch.exp(output_norm)
        ################################################################3
        output_norm = output_norm.cpu().tolist()
        output_norm = torch.tensor(output_norm).float().cuda()
        ###################################################################
        one_hot = torch.sum(output * output_norm)
        # one_hot=torch.sum(output)
        self.net.zero_grad()
        one_hot.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚

        features_maps=[]
        # hata=lisalisa.grad * lisalisa
        # hata = hata.cpu().data.numpy()[0, :]
        # features_maps.append(hata)
        for i in range(len(features)):
            miki=features[i]*self.net.gradients[-1*(i+1)]
            miki=miki.cpu().data.numpy()[0,:]

            # miki = np.maximum(miki, 0)


            features_maps.append(miki)



        cams=[]
        cams_relu=[]
        for i in range(len(features_maps)):
            miki=np.sum(features_maps[i], axis=0)
            # print(miki)
            # miki = np.maximum(miki, 0)
            print(miki)
            print("***********************************************")
            miki = miki - np.min(miki)
            miki = miki / np.max(miki)
            cams.append(miki)
            print(miki)
        for i in range(len(features_maps)):
            miki=np.sum(features_maps[i], axis=0)
            # print(miki)
            miki = np.maximum(miki, 0)
            print(miki)
            print("***********************************************")
            miki = miki - np.min(miki)
            miki = miki / np.max(miki)
            cams_relu.append(miki)
            print(miki)

        return cams,cams_relu






class GradCam_new_many_layers_version_2_3out_tmp():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, index=None):#通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别
        lisalisa=input.requires_grad_(True).cuda()
        features, output ,_= self.net(lisalisa)

        # output_max = torch.max(output)
        # output_min = torch.min(output)
        # output_norm = torch.div((output - output_min).float(), (output_max - output_min))
        # output_norm = torch.exp(output_norm)
        output_norm = torch.zeros_like(output)
        output_norm[0][index]=1
        ################################################################3
        output_norm = output_norm.cpu().tolist()
        output_norm = torch.tensor(output_norm).float().cuda()
        ###################################################################
        one_hot = torch.sum(output * output_norm)
        self.net.zero_grad()
        one_hot.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚

        features_maps=[]
        # hata=lisalisa.grad * lisalisa
        # hata = hata.cpu().data.numpy()[0, :]
        # features_maps.append(hata)
        for i in range(len(features)):
            miki=features[i]*self.net.gradients[-1*(i+1)]
            miki=miki.cpu().data.numpy()[0,:]

            # miki = np.maximum(miki, 0)

            features_maps.append(miki)



        cams=[]
        cams_relu=[]
        for i in range(len(features_maps)):
            miki=np.sum(features_maps[i], axis=0)
            # print(miki)
            miki = np.maximum(miki, 0)
            # print(miki)
            print("***********************************************")
            miki = miki - np.min(miki)
            miki = miki / np.max(miki)
            cams_relu.append(miki)
            # print(miki)
        for i in range(len(features_maps)):
            miki=np.sum(features_maps[i], axis=0)
            # print(miki)
            # miki = np.maximum(miki, 0)
            # print(miki)
            print("***********************************************")
            miki = miki - np.min(miki)
            miki = miki / np.max(miki)
            cams.append(miki)
            # print(miki)


        return cams,cams_relu










class GradCam_new_many_layers_version_2_3out_tmp1():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, index=None):#通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别
        lisalisa=input.requires_grad_(True).cuda()
        features, output ,_= self.net(lisalisa)

        # output_max = torch.max(output)
        # output_min = torch.min(output)
        # output_norm = torch.div((output - output_min).float(), (output_max - output_min))
        # output_norm = torch.exp(output_norm)
        output_norm = torch.zeros_like(output)
        output_norm[0][index]=1
        ################################################################3
        output_norm = output_norm.cpu().tolist()
        output_norm = torch.tensor(output_norm).float().cuda()
        ###################################################################
        one_hot = torch.sum(output * output_norm)
        self.net.zero_grad()
        one_hot.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚

        features_maps=[]
        # hata=lisalisa.grad * lisalisa
        # hata = hata.cpu().data.numpy()[0, :]
        # features_maps.append(hata)
        for i in range(len(features)):

            grads_val = self.net.gradients[-1*(i+1)].cpu().data.numpy()
            feature_maps_i = features[i].cpu().data.numpy()[0,
                                          :]  # 将四维tensor降为三维【512，h，w】，去除batch_szie那一维；这是relu5_2或者pool5的输出特征图
            weights_i=np.mean(grads_val,axis=(2 ,3))[0, :]
            cam_i = np.zeros(feature_maps_i.shape[1:], dtype=np.float32)#[2h,2w]左右
            for j, w in enumerate(weights_i):
                cam_i += w * feature_maps_i[j, :, :]#[h,w]
            # cam_i = np.maximum(cam_i, 0)#这个相当于原论文中的relu,直接将聚合特征图中的小于0的值置零；我待会实验下加或者不加的区别

            # miki=features[i]*self.net.gradients[-1*(i+1)]
            # miki=miki.cpu().data.numpy()[0,:]

            # miki = np.maximum(miki, 0)

            features_maps.append(cam_i)



        cams=[]
        cams_relu=[]
        for i in range(len(features_maps)):
            # miki=np.sum(features_maps[i], axis=0)
            miki=features_maps[i]

            # print(miki)
            miki = np.maximum(miki, 0)
            # print(miki)
            print("***********************************************")
            miki = miki - np.min(miki)
            miki = miki / np.max(miki)
            cams_relu.append(miki)
            # print(miki)
        for i in range(len(features_maps)):
            # miki=np.sum(features_maps[i], axis=0)
            miki=features_maps[i]
            # print(miki)
            # miki = np.maximum(miki, 0)
            # print(miki)
            print("***********************************************")
            miki = miki - np.min(miki)
            miki = miki / np.max(miki)
            cams.append(miki)
            # print(miki)


        return cams,cams_relu






 #
 # grads_val = self.net.gradients[-1*(i+1)].cpu().data.numpy()
 #            feature_maps_i = features[i].cpu().data.numpy()[0,
 #                               :]  # 将四维tensor降为三维【512，h，w】，去除batch_szie那一维；这是relu5_2或者pool5的输出特征图
 #            weights_i=np.mean(grads_val,axis=(2 ,3))[0, :]
 #            cam_i = np.zeros(feature_maps_i.shape[1:], dtype=np.float32)#[2h,2w]左右
 #            for i, w in enumerate(weights_i):
 #                cam_i += w * feature_maps_i[i, :, :]#[h,w]
 #            cam_i = np.maximum(cam_i, 0)#这个相当于原论文中的relu,直接将聚合特征图中的小于0的值置零；我待会实验下加或者不加的区别








class GradCam_new_many_layers_version_2_2out_tmp():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, index=None):#通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别
        lisalisa=input.requires_grad_(True).cuda()
        features, output = self.net(lisalisa)

        # output_max = torch.max(output)
        # output_min = torch.min(output)
        # output_norm = torch.div((output - output_min).float(), (output_max - output_min))
        # output_norm = torch.exp(output_norm)
        output_norm = torch.zeros_like(output)
        output_norm[0][47]=1
        ################################################################3
        output_norm = output_norm.cpu().tolist()
        output_norm = torch.tensor(output_norm).float().cuda()
        ###################################################################
        one_hot = torch.sum(output * output_norm)
        self.net.zero_grad()
        one_hot.backward(retain_graph=False)  # 这里为什么要保留计算图啊？搞不清楚

        features_maps=[]
        # hata=lisalisa.grad * lisalisa
        # hata = hata.cpu().data.numpy()[0, :]
        # features_maps.append(hata)
        for i in range(len(features)):
            miki=features[i]*self.net.gradients[-1*(i+1)]
            print(self.net.gradients[-1 * (i + 1)][0][0])
            print(self.net.gradients[-1 * (i + 1)].size())

            miki=miki.cpu().data.numpy()[0,:]
            features_maps.append(miki)



        cams=[]
        for i in range(len(features_maps)):
            miki=np.sum(features_maps[i], axis=0)
            # print(miki)
            # miki = np.maximum(miki, 0)
            # print(miki)
            print("***********************************************")
            miki = miki - np.min(miki)
            miki = miki / np.max(miki)
            cams.append(miki)
            # print(miki)


        return cams,features_maps











class GradCam_pp_many_layers():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, index=None):#通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别
        lisalisa=input.requires_grad_(True).cuda()
        features, output = self.net(lisalisa)

        # output=torch.exp(output)#这一步真的很重要，为了损失函数的无穷阶可导
        #不不不，我理解错了，是原论文里的公式17，那样一种简单的形式，将高阶导数化为一阶导的高次幂；这种形式是隐式的将输出的概率预测输入了一个指数函数层，用指数函数层
        #的某一的输出进行反向传播，得到了公式17中的那样一种结果；但是我们却不需要显式的定义指数函数层，因为公式17已经推导出来了；公式17中显式的包含的是概率预测值关于输出特征图某一元素的偏导数
        #这是直接我们对神经网络的输出的概率反向传播就可以的了；
        #我也是在exp()出现了数据溢出这一警告时才发现这一问题的存在，是我理解的偏差了

        ##relu5_2与pool5的输出特征图，类别预测结果（对于cub而言，就是一个200维的向量）
        # feature_maps_L28 = features[0]  # 如果输入图像的shape为[batch_size,3,224,224]的tensor的话，关于输出的feature map,shape为  #[batch_size,512,14,14]左右
        # batch_size, c_L28, h_L28, w_L28 = feature_maps_L28.size()
        # feature_maps_L31 = features[1]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        # batch_size, c_L31, h_L31, w_L31 = feature_maps_L31.size()
        batch_size,num_classes=output.size()

        if index == None:
            index = np.argmax(output.cpu().data.numpy())#比方说output是：【1,200】，那么np.argmax返回的是单单一个scalar,output[0][index]便能找到这个max

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1  #生成了一个[1,200]维的one_hot向量，1所在的位置就是最大值所属类别或者指定类别对应的位置，就是一个数字到类别的对应关系嘛
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)#我们将该onehot向量转化为一个gpu tensor,与输出的类别想乘，然后再求和，得到一个标量，损失函数就必须是
        #标量的形式，这也是我们引入one_hot的原因，只考虑我们想要关注的那个类别的预测值；损失函数的这个标量在数值上与最大的类别预测值是相等的，反向传播时别的类别的预测值也就完全不起作用
        one_hot = torch.sum(one_hot.cuda() * output)
        self.net.zero_grad()
        one_hot.backward(retain_graph=True)#这里为什么要保留计算图啊？搞不清楚
###################
        cams = []
        grads_val = lisalisa.grad.cpu().data.numpy()
        batch_size, grads_c, grads_h, grads_w = grads_val.shape
        pixelwise_grads_weight = np.zeros_like(grads_val, dtype=np.float32)
        feature_maps_i = lisalisa.cpu().data.numpy()[0,
                         :]  # 将四维tensor降为三维【512，h，w】，去除batch_szie那一维；这是relu5_2或者pool5的输出特征图
        feature_maps_i_sum = np.sum(feature_maps_i, axis=(1, 2))
        feature_maps_i_sum = feature_maps_i_sum.reshape(1, grads_c, 1, 1) * np.ones_like(pixelwise_grads_weight)
        a = grads_val ** 2
        b = (grads_val ** 2) * 2 + (grads_val ** 3) * feature_maps_i_sum
        pixelwise_grads_weight_i = np.divide(a, b, out=np.zeros_like(grads_val), where=b != 0)
        grads_val = np.maximum(grads_val, 0)
        weights_i = pixelwise_grads_weight_i * grads_val
        weights_i = np.sum(weights_i, axis=(2, 3))[0, :]
        cam_i = np.zeros(feature_maps_i.shape[1:], dtype=np.float32)  # [2h,2w]左右
        for i, w in enumerate(weights_i):
            cam_i += w * feature_maps_i[i, :, :]  # [h,w]
        cam_i = np.maximum(cam_i, 0)  # 这个相当于原论文中的relu,直接将聚合特征图中的小于0的值置零；我待会实验下加或者不加的区别
        cam_i = cam_i - np.min(cam_i)
        cam_i = cam_i / np.max(cam_i)  # 然后呢将其归一化
        cams.append(cam_i)
        # hata = lisalisa.grad * lisalisa
        # hata = hata.cpu().data.numpy()[0, :]
        # features_maps.append(hata)
        for i in range(len(features)):
            grads_val = self.net.gradients[-1*(i+1)].cpu().data.numpy()
            batch_size,grads_c,grads_h,grads_w=grads_val.shape
            pixelwise_grads_weight=np.zeros_like(grads_val,dtype=np.float32)
            feature_maps_i = features[i].cpu().data.numpy()[0,
                               :]  # 将四维tensor降为三维【512，h，w】，去除batch_szie那一维；这是relu5_2或者pool5的输出特征图
            feature_maps_i_sum=np.sum(feature_maps_i,axis=(1,2))
            feature_maps_i_sum=feature_maps_i_sum.reshape(1,grads_c,1,1) * np.ones_like(pixelwise_grads_weight)
            a=grads_val**2
            b=(grads_val**2)*2+(grads_val**3)*feature_maps_i_sum
            pixelwise_grads_weight_i=np.divide(a, b, out=np.zeros_like(grads_val), where=b != 0)
            grads_val = np.maximum(grads_val, 0)
            weights_i=pixelwise_grads_weight_i * grads_val
            weights_i=np.sum(weights_i,axis=(2 ,3))[0, :]
            cam_i = np.zeros(feature_maps_i.shape[1:], dtype=np.float32)#[2h,2w]左右
            for i, w in enumerate(weights_i):
                cam_i += w * feature_maps_i[i, :, :]#[h,w]
            cam_i = np.maximum(cam_i, 0)#这个相当于原论文中的relu,直接将聚合特征图中的小于0的值置零；我待会实验下加或者不加的区别
            cam_i = cam_i - np.min(cam_i)
            cam_i = cam_i / np.max(cam_i)#然后呢将其归一化
            cams.append(cam_i)
        features.insert(0, lisalisa)
        for i in range(len(features)):
            features[i]=features[i].cpu().data.numpy()[0,:]

        return cams,features


class GradCam_many_layers():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, index=None):#通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别
        lisalisa=input.requires_grad_(True).cuda()
        features, output = self.net(lisalisa)

        # output=torch.exp(output)#这一步真的很重要，为了损失函数的无穷阶可导
        #不不不，我理解错了，是原论文里的公式17，那样一种简单的形式，将高阶导数化为一阶导的高次幂；这种形式是隐式的将输出的概率预测输入了一个指数函数层，用指数函数层
        #的某一的输出进行反向传播，得到了公式17中的那样一种结果；但是我们却不需要显式的定义指数函数层，因为公式17已经推导出来了；公式17中显式的包含的是概率预测值关于输出特征图某一元素的偏导数
        #这是直接我们对神经网络的输出的概率反向传播就可以的了；
        #我也是在exp()出现了数据溢出这一警告时才发现这一问题的存在，是我理解的偏差了

        ##relu5_2与pool5的输出特征图，类别预测结果（对于cub而言，就是一个200维的向量）
        # feature_maps_L28 = features[0]  # 如果输入图像的shape为[batch_size,3,224,224]的tensor的话，关于输出的feature map,shape为  #[batch_size,512,14,14]左右
        # batch_size, c_L28, h_L28, w_L28 = feature_maps_L28.size()
        # feature_maps_L31 = features[1]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        # batch_size, c_L31, h_L31, w_L31 = feature_maps_L31.size()
        batch_size,num_classes=output.size()

        if index == None:
            index = np.argmax(output.cpu().data.numpy())#比方说output是：【1,200】，那么np.argmax返回的是单单一个scalar,output[0][index]便能找到这个max

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1  #生成了一个[1,200]维的one_hot向量，1所在的位置就是最大值所属类别或者指定类别对应的位置，就是一个数字到类别的对应关系嘛
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)#我们将该onehot向量转化为一个gpu tensor,与输出的类别想乘，然后再求和，得到一个标量，损失函数就必须是
        #标量的形式，这也是我们引入one_hot的原因，只考虑我们想要关注的那个类别的预测值；损失函数的这个标量在数值上与最大的类别预测值是相等的，反向传播时别的类别的预测值也就完全不起作用
        one_hot = torch.sum(one_hot.cuda() * output)
        self.net.zero_grad()
        one_hot.backward(retain_graph=True)#这里为什么要保留计算图啊？搞不清楚
###################
        cams = []
        grads_val = lisalisa.grad.cpu().data.numpy()
        batch_size, grads_c, grads_h, grads_w = grads_val.shape
        feature_maps_i = lisalisa.cpu().data.numpy()[0,
                         :]  # 将四维tensor降为三维【512，h，w】，去除batch_szie那一维；这是relu5_2或者pool5的输出特征图
        weights_i = np.mean(grads_val, axis=(2, 3))[0, :]
        cam_i = np.zeros(feature_maps_i.shape[1:], dtype=np.float32)  # [2h,2w]左右
        for i, w in enumerate(weights_i):
            cam_i += w * feature_maps_i[i, :, :]  # [h,w]
        cam_i = np.maximum(cam_i, 0)  # 这个相当于原论文中的relu,直接将聚合特征图中的小于0的值置零；我待会实验下加或者不加的区别
        cam_i = cam_i - np.min(cam_i)
        cam_i = cam_i / np.max(cam_i)  # 然后呢将其归一化
        cams.append(cam_i)
        # hata = lisalisa.grad * lisalisa
        # hata = hata.cpu().data.numpy()[0, :]
        # features_maps.append(hata)
        for i in range(len(features)):
            grads_val = self.net.gradients[-1*(i+1)].cpu().data.numpy()
            feature_maps_i = features[i].cpu().data.numpy()[0,
                               :]  # 将四维tensor降为三维【512，h，w】，去除batch_szie那一维；这是relu5_2或者pool5的输出特征图
            weights_i=np.mean(grads_val,axis=(2 ,3))[0, :]
            cam_i = np.zeros(feature_maps_i.shape[1:], dtype=np.float32)#[2h,2w]左右
            for i, w in enumerate(weights_i):
                cam_i += w * feature_maps_i[i, :, :]#[h,w]
            cam_i = np.maximum(cam_i, 0)#这个相当于原论文中的relu,直接将聚合特征图中的小于0的值置零；我待会实验下加或者不加的区别
            cam_i = cam_i - np.min(cam_i)
            cam_i = cam_i / np.max(cam_i)#然后呢将其归一化
            cams.append(cam_i)
        features.insert(0, lisalisa)
        for i in range(len(features)):
            features[i]=features[i].cpu().data.numpy()[0, :]

        return cams,features

class score_cam():
    def __init__(self, net):
        self.net = net

    def __call__(self, input, index=None):  # 通过这个index，我们可以指定任意一个类别，获得其热图，而不仅仅是最高得分所对应的类别
        lisalisa = input.requires_grad_(True).cuda()
        batch_size,c,h,w=lisalisa.size()
        features, output = self.net(lisalisa)
        ##relu5_2与pool5的输出特征图，类别预测结果（对于cub而言，就是一个200维的向量）
        feature_maps_L28 = features[0]  # 如果输入图像的shape为[batch_size,3,224,224]的tensor的话，关于输出的feature map,shape为  #[batch_size,512,14,14]左右
        batch_size, c_L28, h_L28, w_L28 = feature_maps_L28.size()
        feature_maps_L31 = features[1]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
        batch_size, c_L31, h_L31, w_L31 = feature_maps_L31.size()
        batch_size, num_classes = output.size()

        if index == None:
            index = np.argmax(
                output.cpu().data.numpy())  # 比方说output是：【1,200】，那么np.argmax返回的是单单一个scalar,output[0][index]便能找到这个max

        # one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        # one_hot[0][index] = 1  # 生成了一个[1,200]维的one_hot向量，1所在的位置就是最大值所属类别或者指定类别对应的位置，就是一个数字到类别的对应关系嘛
        # one_hot = torch.from_numpy(one_hot).requires_grad_(
        #     True)  # 我们将该onehot向量转化为一个gpu tensor,与输出的类别想乘，然后再求和，得到一个标量，损失函数就必须是
        # 标量的形式，这也是我们引入one_hot的原因，只考虑我们想要关注的那个类别的预测值；损失函数的这个标量在数值上与最大的类别预测值是相等的，反向传播时别的类别的预测值也就完全不起作用
        # one_hot = torch.sum(one_hot.cuda() * output)
        # self.net.zero_grad()
        # one_hot.backward(retain_graph=True)  # 这里为什么要保留计算图啊？搞不清楚
    ###################
        cam_L28 = torch.zeros(feature_maps_L28.shape[2:])  # [2h,2w]左右
        cam_L31 = torch.zeros(feature_maps_L31.shape[2:])  # [h,w]
        for i in range(c_L28):
            saliency_map_tmp=feature_maps_L28[0,i,0,0]
            saliency_map = interpolate(feature_maps_L28[0,i,:,:].view(1, 1, h_L28, w_L28).float(),
                                                    size=[h, w], mode="bilinear",align_corners=False)
            norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
            _,output = self.net(lisalisa * norm_saliency_map)
            output = F.softmax(output,dim=1)
            score = output[0][index].cpu().data
            cam_L28=cam_L28 + saliency_map_tmp * score

        for i in range(c_L31):
            saliency_map_tmp=feature_maps_L31[0,i,0,0]
            # print(h,w)
            saliency_map = interpolate(feature_maps_L31[0,i,:,:].view(1, 1, h_L31, w_L31).float(),
                                                    size=[h, w],mode="bilinear",align_corners=False)
            norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
            _,output = self.net(lisalisa * norm_saliency_map)
            output = F.softmax(output,dim=1)
            score = output[0][index].cpu().data
            cam_L31=cam_L31 + saliency_map_tmp * score

        cam_L28 = F.relu(cam_L28)
        cam_L28_min, cam_L28_max = cam_L28.min(), cam_L28.max()
        cam_L28_norm = ((cam_L28 - cam_L28_min).div(
            cam_L28_max - cam_L28_min)).cpu().data.numpy()

        cam_L31 = F.relu(cam_L31)
        cam_L31_min, cam_L31_max = cam_L31.min(), cam_L31.max()
        cam_L31_norm = ((cam_L31 - cam_L31_min).div(
            cam_L31_max - cam_L31_min)).cpu().data.numpy()

        return cam_L28_norm , cam_L31_norm ,feature_maps_L28[0].cpu().data.numpy(), feature_maps_L31[0].cpu().data.numpy()







