from SCDA_cub_resnet50.bwconncomp import largestConnectComponent
from SCDA_cub_resnet50.about_msloss.net_bninception import FGIAnet100_metric5_ms
from SCDA_cub_resnet50.about_msloss import dataloader_bn
from SCDA_cub_resnet50.about_msloss.ms_loss import MultiSimilarityLoss
from SCDA_cub_resnet50.about_msloss.lr_schedule import WarmupMultiStepLR


import argparse
from os.path import join
import uuid
import time
import json
import pickle
import sys,os
sys.path.append(os.pardir)
import torch
from torch.nn.functional import interpolate
import torch.nn as nn
from torchvision import transforms
import numpy as np
from torch.nn import functional as F
import cv2
from SCDA_cub_resnet50.about_msloss.grad_cam import  PoolCam_yuhan_kernel_embedding,PoolCam_yuhan_kernel_embedding_1


parser = argparse.ArgumentParser()
parser.add_argument('--img_size', type=int, default = 700,help='进行特征提取的图片的尺寸的上界所对应的数量级')





# PARA=[
#     'checkpoint_2800_0.3054.pth',
#     'checkpoint_2800_0.3413.pth',
#     'checkpoint_2800_0.3645.pth',
#     'checkpoint_2999_0.2703.pth',
#     'checkpoint_2999_0.3285.pth',
#     'checkpoint_3000_0.3531.pth',
#     'checkpoint_3200_0.3674.pth',
#     'checkpoint_3400_0.3181.pth',
#     'checkpoint_3600_0.2753.pth',
#     'checkpoint_3800_0.2408.pth',
#     'checkpoint_3999_0.1732.pth',
#
# ]

PARA=[
    'checkpoint_2800_0.4553.pth',
    'checkpoint_2999_0.3860.pth',
    'checkpoint_2800_0.4602.pth',
    'checkpoint_2999_0.4015.pth',
    'checkpoint_2800_0.5468.pth',
    'checkpoint_2999_0.5321.pth',

    #
    # 'checkpoint_3000_0.2741.pth',
    # 'checkpoint_3200_0.3272.pth',
    # 'checkpoint_3399_0.2182.pth',


]

#
#
# PARA=['bn_inception-52deb4733.pth']
#


parser.add_argument('--savedir',default='./models/', help="Path to save weigths and logs")
parser.add_argument("--model_name", type=str, dest="model_name", help="The name of the model to be loaded.",
                        default=PARA)
args = parser.parse_args()

# args = parser.parse_args()


# net = FGIAnet100_metric5_1()
net = FGIAnet100_metric5_ms()
# net = FGIAnet100_metric5_2()

for index in range(len(PARA)):

    #SCDA_fine_tuning
    # checkpoint = torch.load(join(args.savedir,args.model_name))

    checkpoint = torch.load(join(os.path.abspath(os.path.dirname(os.getcwd())),args.savedir,args.model_name[index]))
    print(args.model_name[index])
    net.load_state_dict(checkpoint['model'])
    print(checkpoint['model'].keys())
    print(net.state_dict().keys())


    # # SCDA_pretrained_model
    # save_model = torch.load(join(os.path.abspath(os.path.dirname(os.getcwd())),args.savedir,'bn_inception-52deb4733.pth'))
    # model_dict =  net.state_dict()
    # state_dict = {k:v for k,v in save_model.items() if k in model_dict.keys()}
    # print(state_dict.keys())  # dict_keys(['w', 'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias'])
    # model_dict.update(state_dict)#这样就对我们自定义网络的cnn部分的参数进行了更新，更新为vgg16网络中cnn部分的参数值
    # net.load_state_dict(model_dict)
    # print(args.model_name[index])


    ####################################################################################
    net.eval()
    net.cuda()




    train_paths_name=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/train_paths.json')
    test_paths_name=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/test_paths.json')
    train_labels_name=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/train_labels.json')
    test_labels_name=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/test_labels.json')
    # train_paths_name='./datafile/train_paths.json'
    # test_paths_name='./datafile/test_paths.json'
    # train_labels_name='./datafile/train_labels.json'
    # test_labels_name='./datafile/test_labels.json'
    with open(train_paths_name) as miki:
            train_paths=json.load(miki)
    with open(test_paths_name) as miki:
            test_paths=json.load(miki)
    with open(train_labels_name) as miki:
            train_labels=json.load(miki)
    with open(test_labels_name) as miki:
            test_labels=json.load(miki)
    loaders = dataloader_bn.get_dataloaders(train_paths, test_paths,train_labels,test_labels,args.img_size, 1,1,1,SCDA_flag=1)#返回值为由可迭代DataLoader对象所组成的字典


    net_dict = net.state_dict()
    for key,value in net_dict.items():
            print(key,'\t',net.state_dict()[key].size())

    print("cnn model is ready.")
    num_tr = len(loaders['train'].dataset)#%num_tr便是训练集所对应的图片的个数
    num_te = len(loaders['test'].dataset)#%num_te便是测试集所对应的图片的个数









    tr4_1_1=[]
    tr4_1_2=[]
    tr4_1_3=[]
    tr4_1_4=[]
    te4_1_1=[]
    te4_1_2=[]
    te4_1_3=[]
    te4_1_4=[]
    tr_L31_mean_3 = []
    te_L31_mean_3 = []

    tr2 = []
    tr4 = []
    te2 = []
    te4 = []


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

                            grad_cam_scda = PoolCam_yuhan_kernel_embedding_1(net)#2021.1.13 论文

                            output_mean,output_mean_3,output_mean_no_norm=grad_cam_scda(images,ii=ii,flip=flip)
                            f4_1_1,f4_1_2 ,f4_1_3,f4_1_4=output_mean[0],output_mean[1],output_mean[2],output_mean[3]
                            # print(f4_1_1.shape)
                            # print(f4_1_2.shape)
                            # print(f4_1_3.shape)
                            # print(f4_1_4.shape)
                            f4_1_1no, f4_1_2no, f4_1_3no, f4_1_4no = output_mean_no_norm[0], output_mean_no_norm[1],\
                                                                     output_mean_no_norm[2], output_mean_no_norm[3]

                            f2 = np.hstack([
                                f4_1_2no,
                                f4_1_3no,
                            ])
                            f4 = np.hstack([
                                f4_1_1no,
                                f4_1_2no,
                                f4_1_3no,
                                f4_1_4no,
                            ])
                            aa = np.linalg.norm(f2)
                            if aa != 0:
                                f2 = f2 / aa
                            else:
                                f2 = np.zeros_like(f2)

                            aa = np.linalg.norm(f4)
                            if aa != 0:
                                f4 = f4 / aa
                            else:
                                f4 = np.zeros_like(f4)

                            # print(f2.shape)
                            # print(f4.shape)

                            embedding_mean_max = output_mean_3[0]






                            if phase == 'train':
                                if flip == 0:
                                    tr2.append(f2.tolist())
                                    tr4.append(f4.tolist())


                                    tr4_1_1.append(f4_1_1.tolist())
                                    tr4_1_2.append(f4_1_2.tolist())

                                    tr4_1_3.append(f4_1_3.tolist())
                                    tr4_1_4.append(f4_1_4.tolist())

                                    tr_L31_mean_3.append(embedding_mean_max.tolist())



                                else:
                                    pass
                            else:
                                if flip == 0:
                                    te2.append(f2.tolist())
                                    te4.append(f4.tolist())

                                    te4_1_1.append(f4_1_1.tolist())
                                    te4_1_2.append(f4_1_2.tolist())

                                    te4_1_3.append(f4_1_3.tolist())
                                    te4_1_4.append(f4_1_4.tolist())

                                    te_L31_mean_3.append(embedding_mean_max.tolist())



                                else:
                                    pass
    print("save starting..........................................")



    print('SCDA avgPool and maxpool for trainset and dataset is done................................')
    print('stacking starting...............................................')








    tr_L31_mean_3 = np.array(tr_L31_mean_3)
    te_L31_mean_3 = np.array(te_L31_mean_3)





    tr4_1_1=np.array(tr4_1_1)
    tr4_1_3=np.array(tr4_1_3)
    tr4_1_2=np.array(tr4_1_2)
    tr4_1_4=np.array(tr4_1_4)
    te4_1_1=np.array(te4_1_1)
    te4_1_3=np.array(te4_1_3)
    te4_1_2=np.array(te4_1_2)
    te4_1_4=np.array(te4_1_4)

    tr4=np.array(tr4)
    tr2=np.array(tr2)
    te4=np.array(te4)
    te2=np.array(te2)

    PARA_I = PARA[index].replace('.', '_')  # 避免路径出错
    # target_path = os.path.join(target, str(clas) + class_name)
    target_path = join(os.path.abspath(os.path.dirname(os.getcwd())), 'datafile', PARA_I)

    if not os.path.isdir(target_path):
        os.mkdir(target_path)
    print(target_path)



    train_data=tr4_1_1.tolist()
    print('train_data.shape:',np.array(train_data).shape)
    test_data=te4_1_1.tolist()
    print('test_data.shape:',np.array(test_data).shape)
    final_features={}
    final_features['train']=train_data
    final_features['test']=test_data
   # filename=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/layer4_1_conv1_weight.json')#'layer4.1.conv1.weight'
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
    # filename=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/layer4_1_conv3_weight.json')#'layer4.1.conv3.weight'
    filename=join(target_path,'layer4_1_conv3_weight.json')#'layer4.1.conv3.weight'

    with open(filename,'w') as f_obj:
        json.dump(final_features,f_obj)



    train_data=tr4_1_4.tolist()
    print('train_data.shape:',np.array(train_data).shape)
    test_data=te4_1_4.tolist()
    print('test_data.shape:',np.array(test_data).shape)
    final_features={}
    final_features['train']=train_data
    final_features['test']=test_data
    # filename=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/layer4_2_conv3_weight.json')#'layer4.2.conv3.weight'
    filename=join(target_path,'layer4_1_conv4_weight.json')#'layer4.1.conv3.weight'

    with open(filename,'w') as f_obj:
        json.dump(final_features,f_obj)





    train_data=tr4_1_2.tolist()
    print('train_data.shape:',np.array(train_data).shape)
    test_data=te4_1_2.tolist()
    print('test_data.shape:',np.array(test_data).shape)
    final_features={}
    final_features['train']=train_data
    final_features['test']=test_data
    # filename=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/layer4_2_conv1_weight.json')#'layer4.2.conv1.weight'
    filename=join(target_path,'layer4_1_conv2_weight.json')#'layer4.1.conv3.weight'

    with open(filename,'w') as f_obj:
        json.dump(final_features,f_obj)





    train_data=np.hstack([tr4_1_1,
                          tr4_1_2,
                          tr4_1_3,
                          tr4_1_4
                          ]).tolist()
    print('train_data.shape:',np.array(train_data).shape)
    test_data=np.hstack([te4_1_1,
                         te4_1_2,
                         te4_1_3,
                         te4_1_4
                         ]).tolist()
    print('test_data.shape:',np.array(test_data).shape)

    final_features={}
    final_features['train']=train_data
    final_features['test']=test_data
    # filename=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/layer4_conv2_weight.json')##'layer4.1.conv2.weight' + #'layer4.2.conv2.weight'
    filename=join(target_path,'layer4_conv4_weight.json')#'layer4.1.conv3.weight'

    with open(filename,'w') as f_obj:
        json.dump(final_features,f_obj)



    train_data=tr4.tolist()
    print('train_data.shape:',np.array(train_data).shape)
    test_data=te4.tolist()
    print('test_data.shape:',np.array(test_data).shape)

    final_features={}
    final_features['train']=train_data
    final_features['test']=test_data
    # filename=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/layer4_conv2_weight.json')##'layer4.1.conv2.weight' + #'layer4.2.conv2.weight'
    filename=join(target_path,'layer4_conv4_weight_1.json')#'layer4.1.conv3.weight'

    with open(filename,'w') as f_obj:
        json.dump(final_features,f_obj)



    train_data=np.hstack([
                          tr4_1_2,
                          tr4_1_3,

                          ]).tolist()
    print('train_data.shape:',np.array(train_data).shape)
    test_data=np.hstack([
                         te4_1_2,
                         te4_1_3,

                         ]).tolist()
    print('test_data.shape:',np.array(test_data).shape)

    final_features={}
    final_features['train']=train_data
    final_features['test']=test_data
    # filename=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/layer4_conv2_weight.json')##'layer4.1.conv2.weight' + #'layer4.2.conv2.weight'
    filename=join(target_path,'layer4_conv2_weight.json')#'layer4.1.conv3.weight'

    with open(filename,'w') as f_obj:
        json.dump(final_features,f_obj)



    train_data=tr2.tolist()
    print('train_data.shape:',np.array(train_data).shape)
    test_data=te2.tolist()
    print('test_data.shape:',np.array(test_data).shape)

    final_features={}
    final_features['train']=train_data
    final_features['test']=test_data
    # filename=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/layer4_conv2_weight.json')##'layer4.1.conv2.weight' + #'layer4.2.conv2.weight'
    filename=join(target_path,'layer4_conv2_weight_1.json')#'layer4.1.conv3.weight'

    with open(filename,'w') as f_obj:
        json.dump(final_features,f_obj)





    train_data=tr_L31_mean_3.tolist()
    print('train_data.shape:',np.array(train_data).shape)
    test_data=te_L31_mean_3.tolist()
    print('test_data.shape:',np.array(test_data).shape)

    final_features={}
    final_features['train']=train_data
    final_features['test']=test_data
    # filename=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/embedding_max_avg.json')
    filename=join(target_path,'embedding_max_avg.json')#'layer4.1.conv3.weight'

    with open(filename,'w') as f_obj:
        json.dump(final_features,f_obj)







