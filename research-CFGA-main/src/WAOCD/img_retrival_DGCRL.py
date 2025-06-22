import numpy as np
from sklearn.neighbors import KDTree
import json
import matplotlib.pyplot as plt
from SCDA_cub_resnet50.net_res50 import FGIAnet100_metric5_1,FGIAnet100_metric5,FGIAnet100_metric5_2
from SCDA_cub_resnet50.bwconncomp import largestConnectComponent
import os
import argparse
from os.path import join
import uuid
import json
from SCDA_cub_resnet50 import dataloader_DGPCRL_test
from torch.nn.functional import interpolate
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from SCDA_cub_resnet50.grad_cam import GradCam_yuhan_kernel_version30_DGCRL_scda_many_for_retrivial,\
    GradCam_pp,GradCam,GradCam_new,GradCam_yuhan_kernel_version30_amsoftmax_scda,\
    GradCam_yuhan_kernel_version30_amsoftmax_scda_for_retrivial,\
    GradCam_yuhan_kernel_version30_DGPCRL_scda_many_4_for_retrieval,PoolCam_yuhan_kernel_for_retrieval,\
    PoolCam_yuhan_kernel_scda_origin_no_lcc_for_visualize
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--datasetdir', default=r'C:\Users\于涵\Desktop\Caltech-UCSD Birds-200 2011\Caltech-UCSD Birds-200-2011\CUB_200_2011',  help="path to cub200_2011 dir")
parser.add_argument('--imgdir', default='images',  help="path to train img dir")
parser.add_argument('--tr_te_split_txt', default=r'train_test_split.txt',  help="关于训练集与测试集的划分，0代表测试集，1代表训练集")
parser.add_argument('--tr_te_image_name_txt', default=r'images.txt',  help="关于训练集与测试集的图片的相对路径名字")
parser.add_argument('--image_labels_txt', default=r'image_class_labels.txt',  help="图像的类别标签标记")
parser.add_argument('--class_name_txt', default=r'classes.txt',  help="图像的200个类别名称")
parser.add_argument("--num_classes", type=int, dest="num_classes", help="Total number of epochs to train", default=200)
parser.add_argument('--img_size', type=int, default = 700,help='进行特征提取的图片的尺寸的上界所对应的数量级')
# parser.add_argument("--model_name", type=str, dest="model_name", help="The name of the model to be loaded.", default='checkpoint_11_0.6863.pth')
# parser.add_argument("--model_name", type=str, dest="model_name", help="The name of the model to be loaded.", default='checkpoint_15_1.0.pth')
# parser.add_argument("--model_name", type=str, dest="model_name", help="The name of the model to be loaded.", default='checkpoint_120_0.9943.pth')
# parser.add_argument("--model_name", type=str, dest="model_name", help="The name of the model to be loaded.", default='checkpoint_112_0.9433.pth')
parser.add_argument("--model_name", type=str, dest="model_name", help="The name of the model to be loaded.", default='checkpoint_100_0.9981.pth')

parser.add_argument('--savedir',default='./models/', help="Path to save weigths and logs")
args = parser.parse_args()

# net = FGIAnet100_metric5()
# net = FGIAnet100_metric5_1()
net = FGIAnet100_metric5_2(scale=100)
# net = FGIAnet100_metric5_2(scale=128)

#SCDA_CAM
checkpoint = torch.load(join(os.path.abspath(os.path.dirname(os.getcwd())),args.savedir,args.model_name))
net.load_state_dict(checkpoint['model'])


#
# save_model = torch.load(join(os.path.abspath(os.path.dirname(os.getcwd())),args.savedir,'vgg16-397923af.pth'))
# model_dict =  net.state_dict()
# state_dict = {k:v for k,v in save_model.items() if k in model_dict.keys()}
# print(state_dict.keys())  # dict_keys(['w', 'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias'])
# model_dict.update(state_dict)#这样就对我们自定义网络的cnn部分的参数进行了更新，更新为vgg16网络中cnn部分的参数值
# net.load_state_dict(model_dict)



net.eval()
net.cuda()
# img_name=''
# image = Image.open(img_name).convert('RGB')
# print(image.shape)
# totensor=transforms.ToTensor()
# image=totensor(image)
# batch_size, c, h, w = image.size()
# image = image.cuda()
# _, cnnFeature_maps = net(image)
# feature_maps_L28=cnnFeature_maps[0] #如果输入图像的shape为[batch_size,3,224,224]的tensor的话，关于输出的feature map,shape为  #[batch_size,512,14,14]左右
# batch_size, c_L28, h_L28, w_L28=feature_maps_L28.size()
# feature_maps_L31=cnnFeature_maps[1]##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
# batch_size, c_L31, h_L31, w_L31=feature_maps_L31.size()


train_paths_name=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/train_paths.json')
test_paths_name=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/test_paths.json')
train_labels_name=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/train_labels.json')
test_labels_name=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/test_labels.json')
with open(train_paths_name) as miki:
        train_paths=json.load(miki)
with open(test_paths_name) as miki:
        test_paths=json.load(miki)
with open(train_labels_name) as miki:
        train_labels=json.load(miki)
with open(test_labels_name) as miki:
        test_labels=json.load(miki)












#上面的这些行都没用，主要是指定test_paths,这里面指定这我们所要检索的query所在的路径
# img_name='001.Black_footed_Albatross/Black_Footed_Albatross_0085_92.jpg'
# img_name='001.Black_footed_Albatross/Black_Footed_Albatross_0053_796109.jpg'
# img_name='001.Black_footed_Albatross/Black_Footed_Albatross_0025_796057.jpg'
# img_name='019.Gray_Catbird/Gray_Catbird_0032_21551.jpg'
# img_name='155.Warbling_Vireo/Warbling_Vireo_0086_158564.jpg'
# img_name='091.Mockingbird/Mockingbird_0095_81177.jpg'
# img_name='098.Scott_Oriole/Scott_Oriole_0052_92440.jpg'
# img_name='036.Northern_Flicker/Northern_Flicker_0021_28741.jpg'
# img_name='048.European_Goldfinch/European_Goldfinch_0053_794639.jpg'
# img_name = '073.Blue_Jay/Blue_Jay_0031_62913.jpg'
# img_name = '087.Mallard/Mallard_0123_76653.jpg'
# img_name='200.Common_Yellowthroat/Common_Yellowthroat_0055_190967.jpg'
# img_name='181.Worm_eating_Warbler/Worm_Eating_Warbler_0072_795559.jpg'
# img_name='199.Winter_Wren/Winter_Wren_0047_190390.jpg'
# img_name='001.Black_footed_Albatross/Black_Footed_Albatross_0046_18.jpg'
# img_name='011.Rusty_Blackbird/Rusty_Blackbird_0120_6762.jpg'
# img_name='025.Pelagic_Cormorant/Pelagic_Cormorant_0002_23680.jpg'
# img_name='186.Cedar_Waxwing/Cedar_Waxwing_0043_178321.jpg'
# img_name='199.Winter_Wren/Winter_Wren_0082_189549.jpg'
# img_name='060.Glaucous_winged_Gull/Glaucous_Winged_Gull_0035_44576.jpg'
# img_name='180.Wilson_Warbler/Wilson_Warbler_0060_175420.jpg'
# img_name='180.Wilson_Warbler/Wilson_Warbler_0060_175420.jpg'
# img_name='120.Fox_Sparrow/Fox_Sparrow_0118_114884.jpg'
# img_name='145.Elegant_Tern/Elegant_Tern_0079_150953.jpg'
# img_name='145.Elegant_Tern/Elegant_Tern_0067_151185.jpg'
# img_name='169.Magnolia_Warbler/Magnolia_Warbler_0114_165467.jpg'
# img_name='190.Red_cockaded_Woodpecker/Red_Cockaded_Woodpecker_0022_794700.jpg'
# img_name='115.Brewer_Sparrow/Brewer_Sparrow_0014_107435.jpg'
# img_name='112.Great_Grey_Shrike/Great_Grey_Shrike_0062_106628.jpg'
# img_name='105.Whip_poor_Will/Whip_Poor_Will_0046_796440.jpg'
#
# img_name='136.Barn_Swallow/Barn_Swallow_0056_132916.jpg'
# img_name='136.Barn_Swallow/Barn_Swallow_0060_130110.jpg'
# img_name='152.Blue_headed_Vireo/Blue_Headed_Vireo_0095_156092.jpg'
# img_name='152.Blue_headed_Vireo/Blue_Headed_Vireo_0111_156258.jpg'
# img_name='191.Red_headed_Woodpecker/Red_Headed_Woodpecker_0094_183401.jpg'
# img_name='189.Red_bellied_Woodpecker/Red_Bellied_Woodpecker_0039_180814.jpg'
# img_name='187.American_Three_toed_Woodpecker/American_Three_Toed_Woodpecker_0009_179919.jpg'
# img_name='176.Prairie_Warbler/Prairie_Warbler_0115_172689.jpg'
# img_name='172.Nashville_Warbler/Nashville_Warbler_0115_167039.jpg'
# img_name='172.Nashville_Warbler/Nashville_Warbler_0006_167497.jpg'
# img_name='167.Hooded_Warbler/Hooded_Warbler_0070_164930.jpg'
# img_name='166.Golden_winged_Warbler/Golden_Winged_Warbler_0004_164470.jpg'
# img_name='166.Golden_winged_Warbler/Golden_Winged_Warbler_0035_164362.jpg'
# img_name='159.Black_and_white_Warbler/Black_And_White_Warbler_0122_160106.jpg'
# img_name='158.Bay_breasted_Warbler/Bay_Breasted_Warbler_0080_159749.jpg'
# img_name='156.White_eyed_Vireo/White_Eyed_Vireo_0001_159237.jpg'
# img_name='156.White_eyed_Vireo/White_Eyed_Vireo_0050_158829.jpg'
# img_name='154.Red_eyed_Vireo/Red_Eyed_Vireo_0049_156785.jpg'
# img_name='154.Red_eyed_Vireo/Red_Eyed_Vireo_0101_156988.jpg'
# img_name='152.Blue_headed_Vireo/Blue_Headed_Vireo_0097_156272.jpg'
# img_name='148.Green_tailed_Towhee/Green_Tailed_Towhee_0082_797395.jpg'
# img_name='100.Brown_Pelican/Brown_Pelican_0039_95216.jpg'
# img_name='096.Hooded_Oriole/Hooded_Oriole_0130_90422.jpg'
# img_name='093.Clark_Nutcracker/Clark_Nutcracker_0047_85630.jpg'
# img_name='079.Belted_Kingfisher/Belted_Kingfisher_0013_70753.jpg'
# img_name='188.Pileated_Woodpecker/Pileated_Woodpecker_0080_180589.jpg'
# img_name='188.Pileated_Woodpecker/Pileated_Woodpecker_0086_180096.jpg'
# img_name='141.Artic_Tern/Artic_Tern_0023_140898.jpg'
# img_name='108.White_necked_Raven/White_Necked_Raven_0005_102653.jpg'
# img_name='159.Black_and_white_Warbler/Black_And_White_Warbler_0095_160406.jpg'
# img_name='159.Black_and_white_Warbler/Black_And_White_Warbler_0135_160334.jpg'
# img_name = '159.Black_and_white_Warbler/Black_And_White_Warbler_0002_160376.jpg'
# img_name = '159.Black_and_white_Warbler/Black_And_White_Warbler_0051_160603.jpg'
# img_name = '159.Black_and_white_Warbler/Black_And_White_Warbler_0047_160547.jpg'
# img_name = '159.Black_and_white_Warbler/Black_And_White_Warbler_0102_160073.jpg'
# img_name='073.Blue_Jay/Blue_Jay_0024_63167.jpg'
# img_name='070.Green_Violetear/Green_Violetear_0086_795639.jpg'
# img_name='056.Pine_Grosbeak/Pine_Grosbeak_0088_38830.jpg'
# img_name='056.Pine_Grosbeak/Pine_Grosbeak_0115_38330.jpg'
# img_name='056.Pine_Grosbeak/Pine_Grosbeak_0094_38912.jpg'  #
# img_name='056.Pine_Grosbeak/Pine_Grosbeak_0032_38473.jpg'
# img_name='056.Pine_Grosbeak/Pine_Grosbeak_0115_38330.jpg'  #ok
# img_name='056.Pine_Grosbeak/Pine_Grosbeak_0025_38443.jpg'
# img_name='054.Blue_Grosbeak/Blue_Grosbeak_0095_37007.jpg'
# img_name='054.Blue_Grosbeak/Blue_Grosbeak_0022_37082.jpg'
# img_name='054.Blue_Grosbeak/Blue_Grosbeak_0057_37116.jpg'
# img_name='054.Blue_Grosbeak/Blue_Grosbeak_0078_36655.jpg'
# img_name='054.Blue_Grosbeak/Blue_Grosbeak_0027_36703.jpg'




# img_name='097.Orchard_Oriole/Orchard_Oriole_0030_91612.jpg'
# img_name = '097.Orchard_Oriole/Orchard_Oriole_0106_91830.jpg'
# img_name='035.Purple_Finch/Purple_Finch_0124_27567.jpg'

# img_name='186.Cedar_Waxwing/Cedar_Waxwing_0034_179715.jpg'
# img_name='186.Cedar_Waxwing/Cedar_Waxwing_0061_179305.jpg'
# img_name='186.Cedar_Waxwing/Cedar_Waxwing_0093_178139.jpg'
# img_name='186.Cedar_Waxwing/Cedar_Waxwing_0037_179710.jpg'
# img_name='188.Pileated_Woodpecker/Pileated_Woodpecker_0080_180589.jpg'
# img_name='192.Downy_Woodpecker/Downy_Woodpecker_0058_184520.jpg'
# img_name='194.Cactus_Wren/Cactus_Wren_0080_185901.jpg'
# img_name='196.House_Wren/House_Wren_0083_187406.jpg'
# img_name='199.Winter_Wren/Winter_Wren_0095_189985.jpg'
# img_name='134.Cape_Glossy_Starling/Cape_Glossy_Starling_0047_129348.jpg'
# img_name='170.Mourning_Warbler/Mourning_Warbler_0029_166530.jpg'
# img_name='115.Brewer_Sparrow/Brewer_Sparrow_0014_107435.jpg'
# img_name='101.White_Pelican/White_Pelican_0031_97064.jpg'
# img_name='101.White_Pelican/White_Pelican_0080_95721.jpg'
# img_name='101.White_Pelican/White_Pelican_0013_96901.jpg'
# img_name='101.White_Pelican/White_Pelican_0086_95538.jpg'
# img_name='101.White_Pelican/White_Pelican_0034_97466.jpg'
# img_name='101.White_Pelican/White_Pelican_0024_96554.jpg'
# img_name='101.White_Pelican/White_Pelican_0073_96260.jpg'
# img_name='106.Horned_Puffin/Horned_Puffin_0039_100890.jpg'
# img_name='106.Horned_Puffin/Horned_Puffin_0033_100731.jpg'
# img_name='106.Horned_Puffin/Horned_Puffin_0079_100847.jpg'
# img_name='106.Horned_Puffin/Horned_Puffin_0065_100625.jpg'
# img_name='106.Horned_Puffin/Horned_Puffin_0074_100886.jpg'
# img_name='114.Black_throated_Sparrow/Black_Throated_Sparrow_0011_107115.jpg'
# img_name='114.Black_throated_Sparrow/Black_Throated_Sparrow_0055_107213.jpg'
# img_name='114.Black_throated_Sparrow/Black_Throated_Sparrow_0068_106960.jpg'
# img_name='114.Black_throated_Sparrow/Black_Throated_Sparrow_0088_107220.jpg'
# img_name='114.Black_throated_Sparrow/Black_Throated_Sparrow_0019_107192.jpg'
# img_name='186.Cedar_Waxwing/Cedar_Waxwing_0103_179559.jpg'
# img_name='186.Cedar_Waxwing/Cedar_Waxwing_0002_179071.jpg'
# img_name='186.Cedar_Waxwing/Cedar_Waxwing_0003_178570.jpg'
# img_name='186.Cedar_Waxwing/Cedar_Waxwing_0048_178960.jpg'
# img_name='186.Cedar_Waxwing/Cedar_Waxwing_0059_178500.jpg'
# img_name='200.Common_Yellowthroat/Common_Yellowthroat_0099_190531.jpg'
# img_name='200.Common_Yellowthroat/Common_Yellowthroat_0004_190606.jpg'
# img_name='200.Common_Yellowthroat/Common_Yellowthroat_0090_190503.jpg'
# img_name='200.Common_Yellowthroat/Common_Yellowthroat_0081_190525.jpg'
# img_name='200.Common_Yellowthroat/Common_Yellowthroat_0006_190576.jpg'

img_name='169.Magnolia_Warbler/Magnolia_Warbler_0114_165467.jpg'

# img_name = '016.Painted_Bunting/Painted_Bunting_0091_15198.jpg'
# img_name='095.Baltimore_Oriole/Baltimore_Oriole_0073_87187.jpg'
# img_name='054.Blue_Grosbeak/Blue_Grosbeak_0082_36991.jpg'
# img_name='068.Ruby_throated_Hummingbird/Ruby_Throated_Hummingbird_0090_57411.jpg'
# img_name='068.Ruby_throated_Hummingbird/Ruby_Throated_Hummingbird_0110_57851.jpg'
# img_name='075.Green_Jay/Green_Jay_0023_65898.jpg'
# img_name='075.Green_Jay/Green_Jay_0100_65786.jpg'
# img_name='075.Green_Jay/Green_Jay_0074_65889.jpg'
# img_name='063.Ivory_Gull/Ivory_Gull_0007_49364.jpg'
# img_name='063.Ivory_Gull/Ivory_Gull_0043_49755.jpg'
# img_name='063.Ivory_Gull/Ivory_Gull_0114_49535.jpg'
# img_name='060.Glaucous_winged_Gull/Glaucous_Winged_Gull_0118_2081.jpg'
# img_name='060.Glaucous_winged_Gull/Glaucous_Winged_Gull_0023_45090.jpg'
# img_name='060.Glaucous_winged_Gull/Glaucous_Winged_Gull_0007_44575.jpg'
# img_name='066.Western_Gull/Western_Gull_0065_55728.jpg'
# img_name='066.Western_Gull/Western_Gull_0050_54425.jpg'
# img_name='066.Western_Gull/Western_Gull_0036_54329.jpg'
# img_name='066.Western_Gull/Western_Gull_0045_54735.jpg'
# img_name='065.Slaty_backed_Gull/Slaty_Backed_Gull_0086_786387.jpg'
# img_name='071.Long_tailed_Jaeger/Long_Tailed_Jaeger_0001_797061.jpg'
# img_name='094.White_breasted_Nuthatch/White_Breasted_Nuthatch_0115_86760.jpg'
# img_name='094.White_breasted_Nuthatch/White_Breasted_Nuthatch_0096_86140.jpg'
# img_name='079.Belted_Kingfisher/Belted_Kingfisher_0082_70711.jpg'
# img_name='079.Belted_Kingfisher/Belted_Kingfisher_0048_70532.jpg'
# img_name='079.Belted_Kingfisher/Belted_Kingfisher_0080_70725.jpg'
# img_name='079.Belted_Kingfisher/Belted_Kingfisher_0034_70329.jpg'
# img_name='196.House_Wren/House_Wren_0083_187406.jpg'










# img_name='170.Mourning_Warbler/Mourning_Warbler_0029_166530.jpg'

# img_name=r'D:\python_work\SCDA\SCDA_pro\retrivial_visualize\croped_images\422_1.jpg'


# img_name='165.Chestnut_sided_Warbler/Chestnut_Sided_Warbler_0105_163996.jpg'
# img_name='165.Chestnut_sided_Warbler/Chestnut_Sided_Warbler_0090_163629.jpg'
# img_name='165.Chestnut_sided_Warbler/Chestnut_Sided_Warbler_0033_163607.jpg'
# img_name='165.Chestnut_sided_Warbler/Chestnut_Sided_Warbler_0058_163990.jpg'
# img_name = '165.Chestnut_sided_Warbler/Chestnut_Sided_Warbler_0068_164184.jpg'
# img_name = '165.Chestnut_sided_Warbler/Chestnut_Sided_Warbler_0094_164152.jpg'
# img_name = '165.Chestnut_sided_Warbler/Chestnut_Sided_Warbler_0052_163728.jpg'
# img_name = '165.Chestnut_sided_Warbler/Chestnut_Sided_Warbler_0016_164060.jpg'

# img_name = '151.Black_capped_Vireo/Black_Capped_Vireo_0037_797495.jpg'
# img_name = '133.White_throated_Sparrow/White_Throated_Sparrow_0023_129179.jpg'
# img_name='063.Ivory_Gull/Ivory_Gull_0037_49068.jpg'
# img_name='063.Ivory_Gull/Ivory_Gull_0007_49364.jpg'
# img_name = '063.Ivory_Gull/Ivory_Gull_0043_49755.jpg'
# img_name = '060.Glaucous_winged_Gull/Glaucous_Winged_Gull_0023_45090.jpg'
# img_name = '060.Glaucous_winged_Gull/Glaucous_Winged_Gull_0116_45236.jpg'






test_path=[]
test_path.append(join(args.datasetdir,args.imgdir,img_name))
# test_path.append(img_name)

loaders = dataloader_DGPCRL_test.get_dataloaders(train_paths, test_path,train_labels,test_labels,args.img_size, 1,1,1,SCDA_flag=1)#返回值为由可迭代DataLoader对象所组成的字典



print("cnn model is ready.")

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
yuri=2040


ii=0
for phase in ['test']:
# for phase in ['train','test']:
        for images, labels in loaders[phase]:#一个迭代器，迭代数据集中的每一batch_size图片;迭代的返回值dataset的子类中的__getitem__()方法是如何进行重写的；
                print(ii)
                ii=ii+1
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

                        #传统的SCDA
                        # img_preds, cnnFeature_maps = net(images)  #img_preds:[batch_size=1,200]

                        #Grad_CAM_based_SCDA
                        # grad_cam_scda = GradCam_yuhan_kernel_version30_DGCRL_scda_many_for_retrivial(
                        #     net)  #####!!!!!!!!!!!!!!!!!!!!!最重要的

                        grad_cam_scda =GradCam_yuhan_kernel_version30_DGPCRL_scda_many_4_for_retrieval(net)
                        # grad_cam_scda = PoolCam_yuhan_kernel_for_retrieval(net)

                        ##########################

                        # cam_L28, cam_L31, feature_maps_L28, feature_maps_L31=grad_cam_scda(images)
                        output_mean, output_mean_2,output_mean_3, out_for_visualize,out_for_visualize_kernel = grad_cam_scda(images, ii=ii, flip=flip)
                        # output_mean_26 与output_mean_28 分别存储着来自,'layer4.2.conv2.weight','layer4.1.conv2.weight'的卷积核梯度特征
                        f4_1_1, output_mean_28, f4_1_3, f4_2_1, output_mean_26, f4_2_3 = output_mean[0], output_mean[1], \
                                                                                         output_mean[2], output_mean[3], \
                                                                                         output_mean[4], output_mean[5]
                        # output_mean_26_2 , output_mean_28_2是resnet50的backzone输出的特征图所得到的scda特征，分别是avg_pool与max_pool所对应的特征
                        output_mean_26_2, output_mean_28_2 = output_mean_2[0], output_mean_2[1]
                        embedding_mean_max, embedding_mean_max_1 = output_mean_3[0], output_mean_3[1]

                        feature_maps_L28_mean_norm, feature_maps_L31_mean_norm = output_mean_26, output_mean_28
                        feature_maps_L28_2_mean_norm, feature_maps_L31_2_mean_norm = output_mean_26_2, output_mean_28_2

                        if flip==0:
                            feature_maps=out_for_visualize[0]
                            feature_maps_L31 = feature_maps[
                                0]  # 如果输入图像的shape为[batch_size,3,224,224]的tensor的话，关于输出的feature map,shape为  #[batch_size,512,14,14]左右
                            #
                            feature_maps_L31_sum = torch.sum(feature_maps_L31[0],
                                                             0)
                            #
                            highlight_conn_L31_beifen=out_for_visualize[1]
                            #
                            feature_maps_L31_sum_noflip=feature_maps_L31_sum
                            #    #这是为最后的可视化准备的
                            #######
                            #######
                            # feature_maps_L31_sum_noflip = feature_maps[0][0][yuri]
                            #

                            highlight_conn_L31_noflip=highlight_conn_L31_beifen



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



tr4_1_1=np.array(tr4_1_1)
tr4_1_3=np.array(tr4_1_3)
tr4_2_1=np.array(tr4_2_1)
tr4_2_3=np.array(tr4_2_3)
te4_1_1=np.array(te4_1_1)
te4_1_3=np.array(te4_1_3)
te4_2_1=np.array(te4_2_1)
te4_2_3=np.array(te4_2_3)

tr_L31_mean_3 = np.array(tr_L31_mean_3)
te_L31_mean_3 = np.array(te_L31_mean_3)

tr_L31_mean_3_1 = np.array(tr_L31_mean_3_1)
te_L31_mean_3_1 = np.array(te_L31_mean_3_1)




# train_data=tr4_1_1.tolist()
# print('train_data.shape:',np.array(train_data).shape)
test_data=te4_1_1.tolist()
print('test_data.shape:',np.array(test_data).shape)
final_features={}
# final_features['train']=train_data
final_features['test']=test_data
layer4_1_conv1_weight=final_features
# filename=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/layer4_1_conv1_weight.json')#'layer4.1.conv1.weight'
# with open(filename,'w') as f_obj:
#     json.dump(final_features,f_obj)



# train_data=tr4_1_3.tolist()
# print('train_data.shape:',np.array(train_data).shape)
test_data=te4_1_3.tolist()
print('test_data.shape:',np.array(test_data).shape)
final_features={}
# final_features['train']=train_data
final_features['test']=test_data
layer4_1_conv3_weight=final_features
# filename=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/layer4_1_conv3_weight.json')#'layer4.1.conv3.weight'
# with open(filename,'w') as f_obj:
#     json.dump(final_features,f_obj)



# train_data=tr4_2_3.tolist()
# print('train_data.shape:',np.array(train_data).shape)
test_data=te4_2_3.tolist()
print('test_data.shape:',np.array(test_data).shape)
final_features={}
# final_features['train']=train_data
final_features['test']=test_data
layer4_2_conv3_weight=final_features
# filename=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/layer4_2_conv3_weight.json')#'layer4.2.conv3.weight'
# with open(filename,'w') as f_obj:
#     json.dump(final_features,f_obj)





# train_data=tr4_2_1.tolist()
# print('train_data.shape:',np.array(train_data).shape)
test_data=te4_2_1.tolist()
print('test_data.shape:',np.array(test_data).shape)
final_features={}
# final_features['train']=train_data
final_features['test']=test_data
layer4_2_conv1_weight=final_features
# filename=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/layer4_2_conv1_weight.json')#'layer4.2.conv1.weight'
# with open(filename,'w') as f_obj:
#     json.dump(final_features,f_obj)







# train_data=tr_L31_mean.tolist()
# print('train_data.shape:',np.array(train_data).shape)
test_data=te_L31_mean.tolist()
print('test_data.shape:',np.array(test_data).shape)
final_features={}
# final_features['train']=train_data
final_features['test']=test_data
layer4_1_conv2_weight=final_features
# filename=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/layer4_1_conv2_weight.json')#'layer4.1.conv2.weight'
# with open(filename,'w') as f_obj:
#     json.dump(final_features,f_obj)




# train_data=tr_L28_mean.tolist()
# print('train_data.shape:',np.array(train_data).shape)
test_data=te_L28_mean.tolist()
print('test_data.shape:',np.array(test_data).shape)
final_features={}
# final_features['train']=train_data
final_features['test']=test_data
layer4_2_conv2_weight=final_features
# filename=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/layer4_2_conv2_weight.json')#'layer4.2.conv2.weight'
# with open(filename,'w') as f_obj:
#     json.dump(final_features,f_obj)













# train_data=np.hstack([tr_L31_mean,
#                       tr_L28_mean,
#                       ]).tolist()
# print('train_data.shape:',np.array(train_data).shape)
test_data=np.hstack([te_L31_mean,
                     te_L28_mean,
                     ]).tolist()
print('test_data.shape:',np.array(test_data).shape)

final_features={}
# final_features['train']=train_data
final_features['test']=test_data
layer4_conv2_weight=final_features
# filename=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/layer4_conv2_weight.json')##'layer4.1.conv2.weight' + #'layer4.2.conv2.weight'
# with open(filename,'w') as f_obj:
#     json.dump(final_features,f_obj)






#
# train_data = tr_L31_mean_2.tolist()
# print('train_data.shape:', np.array(train_data).shape)
test_data = te_L31_mean_2.tolist()
print('test_data.shape:', np.array(test_data).shape)
final_features = {}
# final_features['train'] = train_data
final_features['test'] = test_data
scda_max_avg_norm=final_features
# filename=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/embedding_max_avg.json')
# filename = join(target_path, 'scda_max_avg_norm.json')  # 'layer4.1.conv3.weight'
# with open(filename, 'w') as f_obj:
#     json.dump(final_features, f_obj)



#
# train_data = tr_L28_mean_2.tolist()
# print('train_data.shape:', np.array(train_data).shape)
test_data = te_L28_mean_2.tolist()
print('test_data.shape:', np.array(test_data).shape)
final_features = {}
# final_features['train'] = train_data
final_features['test'] = test_data
scda_norm_max_avg=final_features
# filename=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/embedding_max_avg.json')
# filename = join(target_path, 'scda_norm_max_avg.json')  # 'layer4.1.conv3.weight'
# with open(filename, 'w') as f_obj:
#     json.dump(final_features, f_obj)


#
#
# train_data = tr_L31_mean_3.tolist()
# print('train_data.shape:', np.array(train_data).shape)
test_data = te_L31_mean_3.tolist()
print('test_data.shape:', np.array(test_data).shape)
final_features = {}
# final_features['train'] = train_data
final_features['test'] = test_data
embedding_max_avg=final_features
# filename=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/embedding_max_avg.json')
# filename = join(target_path, 'embedding_max_avg.json')  # 'layer4.1.conv3.weight'
# with open(filename, 'w') as f_obj:
#     json.dump(final_features, f_obj)





# train_data = tr_L31_mean_3_1.tolist()
# print('train_data.shape:', np.array(train_data).shape)
test_data = te_L31_mean_3_1.tolist()
print('test_data.shape:', np.array(test_data).shape)
final_features = {}
# final_features['train'] = train_data
final_features['test'] = test_data
embedding_norm_max_avg=final_features
# filename=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/embedding_max_avg.json')
# filename = join(target_path, 'embedding_norm_max_avg.json')  # 'layer4.1.conv3.weight'
#
# with open(filename, 'w') as f_obj:
#     json.dump(final_features, f_obj)







#以下12组任选即可
# test_data=layer4_1_conv2_weight['test']
# filename=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/layer4_1_conv2_weight.json')


# test_data=scda_max_avg_norm['test']
# filename=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/scda_max_avg_norm.json')




# test_data=scda_norm_max_avg['test']
# filename=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/scda_norm_max_avg.json')


# test_data=embeddinng_max_avg['test']
# filename=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/embedding_max_avg.json')




# test_data=embedding_norm_max_avg['test']
# filename=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/embeddinng_norm_max_avg.json')



# test_data=layer4_2_conv2_weight['test']
# filename=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/layer4_2_conv2_weight.json')


test_data=layer4_conv2_weight['test']
filename=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/layer4_conv2_weight.json')




with open(filename) as f_obj:
    final_features=json.load(f_obj)
train_data=final_features['test']



train_paths_name=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/train_paths.json')
test_paths_name=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/test_paths.json')
train_labels_name=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/train_labels.json')
test_labels_name=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/test_labels.json')
with open(train_paths_name) as miki:
        train_paths=json.load(miki)
with open(test_paths_name) as miki:
        test_paths=json.load(miki)
with open(train_labels_name) as miki:
        train_labels=json.load(miki)
with open(test_labels_name) as miki:
        test_labels=json.load(miki)


train_paths=test_paths
train_labels=test_labels


print(KDTree.valid_metrics)
X = np.array(train_data)
print(X.shape)
query=np.array(test_data)
print(query.shape)
kdt = KDTree(X, leaf_size=40, metric='euclidean')
# s = pickle.dumps(kdt)                     # doctest: +SKIP
# kdt = pickle.loads(s)
dis,result=kdt.query(query, k=49, return_distance=True)
#大概就是这样子的，有K个query,最终返回的a矩阵就有K行，每一行都有九个元素，代表
#我们以第一行为例吧，第一行的九个元素。就代表距离第一个query距离最近的九张图片的索引（这九张图片来自训练集），我们通过这个索引
#就可以获知这九张图片的具体的类别（如果我们已知了query的类别，既可以据此来计算mAP），
# 以及对应的存储路径（在图像检索系统中进行最后的检索结果的输出）
print(dis)
print('检索结果的在训练集中的标号索引：')
print(result)
print(result.shape)
query_result_labels=np.zeros(result.shape)
h,w=result.shape
for i in range(h):
    for j in range(w):
        query_result_labels[i,j]=train_labels[result[i,j]]
print('检索结果的类别索引：')
print(query_result_labels)




query_result_paths=[]
h,w=result.shape
for i in range(h):
    query_result_paths.append([])
    for j in range(w):
        query_result_paths[i].append(train_paths[result[i,j]])



# miki_1=cv2.imread(str(test_paths[0]))
miki_1 = Image.open(test_path[0]).convert('RGB')
width,height = miki_1.size
print(miki_1.size)
# print(cam_L31)

print(len(out_for_visualize_kernel))
kernel_0=out_for_visualize_kernel[0]
kernel_1=out_for_visualize_kernel[1]
kernel_2=out_for_visualize_kernel[2]
# kernel_2=torch.from_numpy(output_mean_28_2)
print(torch.min(kernel_0))
print(torch.max(kernel_0))
kernel_0 = kernel_0 - torch.min(kernel_0)
kernel_0 = kernel_0 / torch.max(kernel_0)
kernel_0 = kernel_0.cpu().data.numpy()*255*5
kernel_1 = kernel_1 - torch.min(kernel_1)
kernel_1 = kernel_1 / torch.max(kernel_1)
kernel_1 = kernel_1.cpu().data.numpy()*255
kernel_2 = kernel_2 - torch.min(kernel_2)
kernel_2 = kernel_2 / torch.max(kernel_2)
kernel_2 = kernel_2.cpu().data.numpy().reshape(1,kernel_2.shape[0])*255
kernel_2 = kernel_2 * np.ones([120,kernel_2.shape[1]])
cam_L31 = Image.fromarray(kernel_0).convert('L')
print(cam_L31.size)
cam_L31 = cv2.cvtColor(np.asarray(cam_L31),cv2.COLOR_RGB2BGR)
# cam_L31 = cv2.applyColorMap(cam_L31, cv2.COLORMAP_JET)
result = cam_L31*1 #+ miki_1 * 0.5
print(result.shape)
cv2.imwrite(join(os.path.abspath(os.path.dirname(os.getcwd())),'./retrivial_visualize/kernel0.jpg'),result)
cv2.waitKey(0)


cam_L31 = Image.fromarray(kernel_1).convert('L')
print(cam_L31.size)
cam_L31 = cv2.cvtColor(np.asarray(cam_L31),cv2.COLOR_RGB2BGR)
# cam_L31 = cv2.applyColorMap(cam_L31, cv2.COLORMAP_JET)
result = cam_L31*1 #+ miki_1 * 0.5
print(result.shape)
cv2.imwrite(join(os.path.abspath(os.path.dirname(os.getcwd())),'./retrivial_visualize/kernel1.jpg'),result)
cv2.waitKey(0)


cam_L31 = Image.fromarray(kernel_2).convert('L')
print(cam_L31.size)
cam_L31 = cv2.cvtColor(np.asarray(cam_L31),cv2.COLOR_RGB2BGR)
# cam_L31 = cv2.applyColorMap(cam_L31, cv2.COLORMAP_JET)
result = cam_L31*1 #+ miki_1 * 0.5
print(result.shape)
cv2.imwrite(join(os.path.abspath(os.path.dirname(os.getcwd())),'./retrivial_visualize/kernel2.jpg'),result)
cv2.waitKey(0)



highlight_L31=highlight_conn_L31_noflip * 255
# highlight_L28=highlight_conn_L28_noflip * 255
# feature_maps_L28_sum=feature_maps_L28_sum_noflip
feature_maps_L31_sum=feature_maps_L31_sum_noflip
# feature_maps_L28_sum = feature_maps_L28_sum - torch.min(feature_maps_L28_sum)
# feature_maps_L28_sum = feature_maps_L28_sum / torch.max(feature_maps_L28_sum)
feature_maps_L31_sum = feature_maps_L31_sum - torch.min(feature_maps_L31_sum)
feature_maps_L31_sum = feature_maps_L31_sum / torch.max(feature_maps_L31_sum)
feature_maps_L31_sum=feature_maps_L31_sum.cpu().data.numpy()*255
# feature_maps_L28_sum=feature_maps_L28_sum.cpu().data.numpy()*255








cam_L31 = Image.fromarray(highlight_L31).convert('L')
cam_L31 = cam_L31.resize((width, height),Image.ANTIALIAS)
print(cam_L31.size)
# Image._show(cam_L31)
cam_L31 = cv2.cvtColor(np.asarray(cam_L31),cv2.COLOR_RGB2BGR)
# cam_L31 = cv2.cvtColor(np.asarray(cam_L31))
cam_L31 = cv2.applyColorMap(cam_L31, cv2.COLORMAP_JET)
miki_1=cv2.cvtColor(np.asarray(miki_1),cv2.COLOR_RGB2BGR)
# cam_L31 = Image.fromarray(cv2.cvtColor(cam_L31,cv2.COLOR_BGR2RGB))
# heatmap_L31,miki_1=np.array(heatmap_L31),np.array(miki_1)
result = cam_L31 + miki_1 * 0.5
print(result.shape)
# plt.imsave('1.jpg',heatmap_L31)
cv2.imwrite(join(os.path.abspath(os.path.dirname(os.getcwd())),'./retrivial_visualize/highlight_L31_original_scda.jpg'),result)
cv2.waitKey(0)

# cam_L31 = Image.fromarray(highlight_L28).convert('L')
# cam_L31 = cam_L31.resize((width, height),Image.ANTIALIAS)
# print(cam_L31.size)
# # Image._show(cam_L31)
# cam_L31 = cv2.cvtColor(np.asarray(cam_L31),cv2.COLOR_RGB2BGR)
# # cam_L31 = cv2.cvtColor(np.asarray(cam_L31))
# cam_L31 = cv2.applyColorMap(cam_L31, cv2.COLORMAP_JET)
# miki_1=cv2.cvtColor(np.asarray(miki_1),cv2.COLOR_RGB2BGR)
# # cam_L31 = Image.fromarray(cv2.cvtColor(cam_L31,cv2.COLOR_BGR2RGB))
# # heatmap_L31,miki_1=np.array(heatmap_L31),np.array(miki_1)
# result = cam_L31*0.9 + miki_1 * 0.2
# print(result.shape)
# # plt.imsave('1.jpg',heatmap_L31)
# cv2.imwrite(join(os.path.abspath(os.path.dirname(os.getcwd())),'./retrivial_visualize/highlight_L28_original_scda.jpg'),result)
# cv2.waitKey(0)


cam_L31 = Image.fromarray(feature_maps_L31_sum).convert('L')
cam_L31 = cam_L31.resize((width, height),Image.ANTIALIAS)
print(cam_L31.size)
# Image._show(cam_L31)
cam_L31 = cv2.cvtColor(np.asarray(cam_L31),cv2.COLOR_RGB2BGR)
# cam_L31 = cv2.cvtColor(np.asarray(cam_L31))
cam_L31 = cv2.applyColorMap(cam_L31, cv2.COLORMAP_JET)
miki_1=cv2.cvtColor(np.asarray(miki_1),cv2.COLOR_RGB2BGR)
# cam_L31 = Image.fromarray(cv2.cvtColor(cam_L31,cv2.COLOR_BGR2RGB))
# heatmap_L31,miki_1=np.array(heatmap_L31),np.array(miki_1)
result = cam_L31*1 + miki_1 * 0.5
print(result.shape)
# plt.imsave('1.jpg',heatmap_L31)
cv2.imwrite(join(os.path.abspath(os.path.dirname(os.getcwd())),'./retrivial_visualize/feature_maps_L31_sum.jpg'),result)
cv2.waitKey(0)

# cam_L31 = Image.fromarray(feature_maps_L28_sum).convert('L')
# cam_L31 = cam_L31.resize((width, height),Image.ANTIALIAS)
# print(cam_L31.size)
# # Image._show(cam_L31)
# cam_L31 = cv2.cvtColor(np.asarray(cam_L31),cv2.COLOR_RGB2BGR)
# # cam_L31 = cv2.cvtColor(np.asarray(cam_L31))
# cam_L31 = cv2.applyColorMap(cam_L31, cv2.COLORMAP_JET)
# miki_1=cv2.cvtColor(np.asarray(miki_1),cv2.COLOR_RGB2BGR)
# # cam_L31 = Image.fromarray(cv2.cvtColor(cam_L31,cv2.COLOR_BGR2RGB))
# # heatmap_L31,miki_1=np.array(heatmap_L31),np.array(miki_1)
# result = cam_L31*0.9 + miki_1 * 0.2
# print(result.shape)
# # plt.imsave('1.jpg',heatmap_L31)
# cv2.imwrite(join(os.path.abspath(os.path.dirname(os.getcwd())),'./retrivial_visualize/feature_maps_L28_sum.jpg'),result)
# cv2.waitKey(0)



plt.figure()
miki=plt.imread(os.path.join(join(args.datasetdir,args.imgdir,img_name)))
plt.imshow(miki)





plt.figure()
for i in range(49):
    plt.subplot(7,7,i+1)
    # print(query_result_paths[0])
    miki=plt.imread(str(query_result_paths[0][i]))
    plt.imshow(miki)
    plt.xticks([])
    plt.yticks([])
plt.show()