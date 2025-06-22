import os
print(os.getcwd())
import sys
sys.path.append("./")
print(sys.path)

from about_pretraineds.net_res50 import FGIAnet100_metric5_1, \
    FGIAnet100_metric5, FGIAnet100_metric5_2, FGIAnet100_R18, FGIAnet100_R34, FGIAnet100_R101, FGIAnet100_R152
import dataloader
import dataloader_test  # DGCRL test  无监督test
import dataloader_DGPCRL_test  # DGPCRL test
from cfga_core.bwconncomp import largestConnectComponent
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
from cfga_core.grad_cam import PoolCam_yuhan_kernel_scda_origin


parser = argparse.ArgumentParser()
parser.add_argument('--img_size', type=int, default = 700,help='进行特征提取的图片的尺寸的上界所对应的数量级')
PARA=[]
parser.add_argument('--dataset_name',default='./models/', help="当前所处理的数据集的名称")
parser.add_argument("--model_name", type=str, nargs='+', help="The name of the model to be loaded.")
args = parser.parse_args()

#一般模型只指定一个
if 'resnet50' in args.model_name[0]:
    net = FGIAnet100_metric5_1()
elif 'resnet101' in args.model_name[0]:
    net = FGIAnet100_R101()
elif 'resnet152' in args.model_name[0]:
    net = FGIAnet100_R152()

     

for index in range(len(args.model_name)):
    print("正在加载imagenet预训练模型:{}".format(args.model_name[index]))
    save_model = torch.load(args.model_name[index])
    model_dict =  net.state_dict()
    state_dict = {k:v for k,v in save_model.items() if k in model_dict.keys()}
    print(state_dict.keys())  # dict_keys(['w', 'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias'])
    model_dict.update(state_dict)#这样就对我们自定义网络的cnn部分的参数进行了更新，更新为vgg16网络中cnn部分的参数值
    net.load_state_dict(model_dict)

    net.eval()
    net.cuda()

    train_paths_name='./datafile/{}/train_paths.json'.format(args.dataset_name)
    test_paths_name='./datafile/{}/test_paths.json'.format(args.dataset_name)
    train_labels_name='./datafile/{}/train_labels.json'.format(args.dataset_name)
    test_labels_name='./datafile/{}/test_labels.json'.format(args.dataset_name)
    with open(train_paths_name) as miki:
            train_paths=json.load(miki)
    with open(test_paths_name) as miki:
            test_paths=json.load(miki)
    with open(train_labels_name) as miki:
            train_labels=json.load(miki)
    with open(test_labels_name) as miki:
            test_labels=json.load(miki)
    loaders = dataloader_test.get_dataloaders(train_paths, test_paths,train_labels,test_labels,args.img_size, 1,1,1,SCDA_flag=1)#返回值为由可迭代DataLoader对象所组成的字典


    net_dict = net.state_dict()
    for key,value in net_dict.items():
            print(key,'\t',net.state_dict()[key].size())

    print("cnn model is ready.")
    num_tr = len(loaders['train'].dataset)#%num_tr便是训练集所对应的图片的个数
    num_te = len(loaders['test'].dataset)#%num_te便是测试集所对应的图片的个数

    tr_L28_mean = []
    te_L28_mean = []
    tr_L31_mean = []
    te_L31_mean = []

    tr4_1_1=[]
    tr4_1_3=[]
    tr4_2_1=[]
    tr4_2_3=[]
    te4_1_1=[]
    te4_1_3=[]
    te4_2_1=[]
    te4_2_3=[]


    tr_L28_mean_2 = []
    te_L28_mean_2 = []
    tr_L31_mean_2 = []
    te_L31_mean_2 = []



    tr_L31_mean_3 = []
    te_L31_mean_3 = []

    tr_L31_mean_3_1 = []
    te_L31_mean_3_1 = []



    ii=0
    for phase in ['test']:
    # for phase in ['train','test']:
        for images, labels in loaders[phase]:#一个迭代器，迭代数据集中的每一batch_size图片;迭代的返回值dataset的子类中的__getitem__()方法是如何进行重写的；
            print(ii)
            ii=ii+1
            # print(images.size())
            for flip in range(1):
                if flip==0:
                        pass
                else:
                        images=images[:,:,:,torch.arange(images.size(3)-1,-1,-1).long()]  #整个batch_size的所有图像水平翻转
                # print(images[0].size())
                # image=images[0]#去除了batch_size那一维度，反正batch_size都是1，有没有无所谓
                batch_size,c,h,w=images.size()
                if min(h,w) > args.img_size:
                        images= interpolate(images,size=[int(h * (args.img_size / min(h, w))),int(w * (args.img_size / min(h, w)))],mode="bilinear",align_corners=True)
                        # %我就打个比方吧，min(h,w)=h的话  h*(700/min(h,w)=700   w*(700/min(h,w)=w*(700/h)=700*(w/h) 图像的size由[h,w]变为[700,700*(w/h)]
                        #%由此可见，在min(h,w) > 700的前提下，图像被适当的进行分辨率的缩小，到700这一级，但是长宽比是没有改变的，图像没有变形
                        # %这一步操作只是为了对于图像的分辨率的上限进行一个限制
                batch_size, c, h, w = images.size()
                #matlab版本的实现中这里是对可能出现的灰度图像进行通道数扩充，并减去图像在各个通道上的均值，以上过程我在dataloader.py中已经实现了，以上。
                images,labels=images.cuda(),labels.cuda()
                labels=torch.zeros_like(labels).cuda()
                # labels=labels.cpu()

                grad_cam_scda = PoolCam_yuhan_kernel_scda_origin(net)   #有最大连通域
                
                output_mean,output_mean_2,output_mean_3=grad_cam_scda(images,ii=ii,flip=flip)
                #output_mean_26 与output_mean_28 分别存储着来自,'layer4.2.conv2.weight','layer4.1.conv2.weight'的卷积核梯度特征
                f4_1_1,output_mean_28 ,f4_1_3,f4_2_1, output_mean_26,f4_2_3=output_mean[0],output_mean[1],output_mean[2],output_mean[3],output_mean[4],output_mean[5]
                #output_mean_26_2 , output_mean_28_2是resnet50的backzone输出的特征图所得到的scda特征，
                # 分别是
                # #先norm再拼接
                # 与
                # #先拼接再norm
                # 所对应的特征
                output_mean_26_2 , output_mean_28_2=output_mean_2[0],output_mean_2[1]
                embedding_mean_max ,embedding_mean_max_1= output_mean_3[0],output_mean_3[1]

                feature_maps_L28_mean_norm,feature_maps_L31_mean_norm = output_mean_26,output_mean_28
                feature_maps_L28_2_mean_norm,feature_maps_L31_2_mean_norm = output_mean_26_2,output_mean_28_2



                if phase == 'train':
                    if flip == 0:
                        tr_L31_mean.append(feature_maps_L31_mean_norm.tolist())
                        tr_L28_mean.append(feature_maps_L28_mean_norm.tolist())
                        tr_L31_mean_2.append(feature_maps_L31_2_mean_norm.tolist())
                        tr_L28_mean_2.append(feature_maps_L28_2_mean_norm.tolist())
                        tr4_1_1.append(f4_1_1.tolist())
                        tr4_1_3.append(f4_1_3.tolist())
                        tr4_2_1.append(f4_2_1.tolist())
                        tr4_2_3.append(f4_2_3.tolist())
                        tr_L31_mean_3.append(embedding_mean_max.tolist())
                        tr_L31_mean_3_1.append(embedding_mean_max_1.tolist())

                    else:
                        pass
                else:
                    if flip == 0:
                        te_L31_mean.append(feature_maps_L31_mean_norm.tolist())
                        te_L28_mean.append(feature_maps_L28_mean_norm.tolist())
                        te_L31_mean_2.append(feature_maps_L31_2_mean_norm.tolist())
                        te_L28_mean_2.append(feature_maps_L28_2_mean_norm.tolist())
                        te4_1_1.append(f4_1_1.tolist())
                        te4_1_3.append(f4_1_3.tolist())
                        te4_2_1.append(f4_2_1.tolist())
                        te4_2_3.append(f4_2_3.tolist())
                        te_L31_mean_3.append(embedding_mean_max.tolist())
                        te_L31_mean_3_1.append(embedding_mean_max_1.tolist())

                    else:
                        pass
    print("save starting..........................................")



    print('SCDA avgPool and maxpool for trainset and dataset is done................................')
    print('stacking starting...............................................')

    tr_L31_mean = np.array(tr_L31_mean)
    te_L31_mean = np.array(te_L31_mean)
    tr_L28_mean = np.array(tr_L28_mean)
    te_L28_mean = np.array(te_L28_mean)




    tr_L31_mean_2 = np.array(tr_L31_mean_2)
    te_L31_mean_2 = np.array(te_L31_mean_2)
    tr_L28_mean_2 = np.array(tr_L28_mean_2)
    te_L28_mean_2 = np.array(te_L28_mean_2)



    tr_L31_mean_3 = np.array(tr_L31_mean_3)
    te_L31_mean_3 = np.array(te_L31_mean_3)

    tr_L31_mean_3_1 = np.array(tr_L31_mean_3_1)
    te_L31_mean_3_1 = np.array(te_L31_mean_3_1)


    tr4_1_1=np.array(tr4_1_1)
    tr4_1_3=np.array(tr4_1_3)
    tr4_2_1=np.array(tr4_2_1)
    tr4_2_3=np.array(tr4_2_3)
    te4_1_1=np.array(te4_1_1)
    te4_1_3=np.array(te4_1_3)
    te4_2_1=np.array(te4_2_1)
    te4_2_3=np.array(te4_2_3)

    model_I = args.model_name[index].replace('.', '_')  # 避免路径出错
    target_path = join('datafile',args.dataset_name, model_I)

    os.makedirs(target_path, exist_ok=True)
    print(target_path)

    train_data=tr4_1_1.tolist()
    print('train_data.shape:',np.array(train_data).shape)
    test_data=te4_1_1.tolist()
    print('test_data.shape:',np.array(test_data).shape)
    final_features={}
    final_features['train']=train_data
    final_features['test']=test_data
    filename=join(target_path,'layer4_1_conv1_weight.json')#'layer4.1.conv1.weight'

    with open(filename,'w') as f_obj:
        json.dump(final_features,f_obj)



    train_data=tr4_1_3.tolist()
    print('train_data.shape:',np.array(train_data).shape)
    test_data=te4_1_3.tolist()
    print('test_data.shape:',np.array(test_data).shape)
    final_features={}
    final_features['train']=train_data
    final_features['test']=test_data
    filename=join(target_path,'layer4_1_conv3_weight.json')#'layer4.1.conv3.weight'

    with open(filename,'w') as f_obj:
        json.dump(final_features,f_obj)



    train_data=tr4_2_3.tolist()
    print('train_data.shape:',np.array(train_data).shape)
    test_data=te4_2_3.tolist()
    print('test_data.shape:',np.array(test_data).shape)
    final_features={}
    final_features['train']=train_data
    final_features['test']=test_data
    filename=join(target_path,'layer4_2_conv3_weight.json')#'layer4.1.conv3.weight'
    with open(filename,'w') as f_obj:
        json.dump(final_features,f_obj)





    train_data=tr4_2_1.tolist()
    print('train_data.shape:',np.array(train_data).shape)
    test_data=te4_2_1.tolist()
    print('test_data.shape:',np.array(test_data).shape)
    final_features={}
    final_features['train']=train_data
    final_features['test']=test_data
    filename=join(target_path,'layer4_2_conv1_weight.json')#'layer4.1.conv3.weight'
    with open(filename,'w') as f_obj:
        json.dump(final_features,f_obj)







    train_data=tr_L31_mean.tolist()
    print('train_data.shape:',np.array(train_data).shape)
    test_data=te_L31_mean.tolist()
    print('test_data.shape:',np.array(test_data).shape)
    final_features={}
    final_features['train']=train_data
    final_features['test']=test_data
    filename=join(target_path,'layer4_1_conv2_weight.json')#'layer4.1.conv3.weight'

    with open(filename,'w') as f_obj:
        json.dump(final_features,f_obj)




    train_data=tr_L28_mean.tolist()
    print('train_data.shape:',np.array(train_data).shape)
    test_data=te_L28_mean.tolist()
    print('test_data.shape:',np.array(test_data).shape)
    final_features={}
    final_features['train']=train_data
    final_features['test']=test_data
    filename=join(target_path,'layer4_2_conv2_weight.json')#'layer4.1.conv3.weight'

    with open(filename,'w') as f_obj:
        json.dump(final_features,f_obj)










    train_data=np.hstack([tr_L31_mean,
                          tr_L28_mean,
                          ]).tolist()
    print('train_data.shape:',np.array(train_data).shape)
    test_data=np.hstack([te_L31_mean,
                         te_L28_mean,
                         ]).tolist()
    print('test_data.shape:',np.array(test_data).shape)

    final_features={}
    final_features['train']=train_data
    final_features['test']=test_data
    filename=join(target_path,'layer4_conv2_weight.json')#'layer4.1.conv3.weight'

    with open(filename,'w') as f_obj:
        json.dump(final_features,f_obj)





    train_data=tr_L31_mean_2.tolist()
    print('train_data.shape:',np.array(train_data).shape)
    test_data=te_L31_mean_2.tolist()
    print('test_data.shape:',np.array(test_data).shape)

    final_features={}
    final_features['train']=train_data
    final_features['test']=test_data
    filename=join(target_path,'scda_max_avg_norm.json')#'layer4.1.conv3.weight'

    with open(filename,'w') as f_obj:
        json.dump(final_features,f_obj)







    train_data=tr_L28_mean_2.tolist()
    print('train_data.shape:',np.array(train_data).shape)
    test_data=te_L28_mean_2.tolist()
    print('test_data.shape:',np.array(test_data).shape)

    final_features={}
    final_features['train']=train_data
    final_features['test']=test_data
    filename=join(target_path,'scda_norm_max_avg.json')#'layer4.1.conv3.weight'

    with open(filename,'w') as f_obj:
        json.dump(final_features,f_obj)







    train_data=tr_L31_mean_3.tolist()
    print('train_data.shape:',np.array(train_data).shape)
    test_data=te_L31_mean_3.tolist()
    print('test_data.shape:',np.array(test_data).shape)

    final_features={}
    final_features['train']=train_data
    final_features['test']=test_data
    filename=join(target_path,'embedding_max_avg.json')#'layer4.1.conv3.weight'

    with open(filename,'w') as f_obj:
        json.dump(final_features,f_obj)



    train_data=tr_L31_mean_3_1.tolist()
    print('train_data.shape:',np.array(train_data).shape)
    test_data=te_L31_mean_3_1.tolist()
    print('test_data.shape:',np.array(test_data).shape)

    final_features={}
    final_features['train']=train_data
    final_features['test']=test_data
    filename=join(target_path,'embedding_norm_max_avg.json')#'layer4.1.conv3.weight'

    with open(filename,'w') as f_obj:
        json.dump(final_features,f_obj)