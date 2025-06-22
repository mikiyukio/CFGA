import numpy as np
from sklearn.neighbors import KDTree
import json
import matplotlib.pyplot as plt
from SCDA_cub_resnet50.net_res50 import FGIAnet100_metric5_1_for_cam,FGIAnet100_metric6_1_for_cam,\
    FGIAnet100_metric5_for_cam,FGIAnet100_for_cam, FGIAnet100_metric5_2_for_cam
from SCDA_cub_resnet50.bwconncomp import largestConnectComponent
import os
import argparse
from os.path import join
import uuid
import json
from  SCDA_cub_resnet50 import dataloader
from torch.nn.functional import interpolate
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from SCDA_cub_resnet50.grad_cam import GradCam_pp,GradCam,GradCam_new,GradCam_new_many_layers,GradCam_new_many_layers_yu,\
    GradCam_pp_many_layers,GradCam_many_layers,GradCam_new_many_layers_3out,GradCam_new_many_layers_version_2_2out_tmp,\
    GradCam_new_many_layers_version_2,GradCam_new_many_layers_version_2_3out,GradCam_new_many_layers_version_2_3out_tmp,\
    GradCam_new_many_layers_version_2_3out_tmp1,GradCam_new_many_layers_3out0, PoolCam_new_many_layers_3out0,PoolCam_new_many_layers_3out0_saliency_map
import cv2
import random


def random_int(length,a,b):
    list=[]
    count=0
    while (count<length):
        number=random.randint(a,b)
        list.append(number)
        count=count+1
    return list






parser = argparse.ArgumentParser()
parser.add_argument('--datasetdir', default=r'C:\Users\于涵\Desktop\Caltech-UCSD Birds-200 2011\Caltech-UCSD Birds-200-2011\CUB_200_2011',  help="path to cub200_2011 dir")
parser.add_argument('--imgdir', default='images',  help="path to train img dir")
parser.add_argument('--tr_te_split_txt', default=r'train_test_split.txt',  help="关于训练集与测试集的划分，0代表测试集，1代表训练集")
parser.add_argument('--tr_te_image_name_txt', default=r'images.txt',  help="关于训练集与测试集的图片的相对路径名字")
parser.add_argument('--image_labels_txt', default=r'image_class_labels.txt',  help="图像的类别标签标记")
parser.add_argument('--class_name_txt', default=r'classes.txt',  help="图像的200个类别名称")
parser.add_argument("--num_classes", type=int, dest="num_classes", help="Total number of epochs to train", default=200)
parser.add_argument('--img_size', type=int, default = 700,help='进行特征提取的图片的尺寸的上界所对应的数量级')
# parser.add_argument("--model_name", type=str, dest="model_name", help="The name of the model to be loaded.", default='checkpoint_15_1.0.pth')
# parser.add_argument("--model_name", type=str, dest="model_name", help="The name of the model to be loaded.", default='checkpoint_30_0.9928.pth')
parser.add_argument("--model_name", type=str, dest="model_name", help="The name of the model to be loaded.", default='checkpoint_200_1.0.pth')

# parser.add_argument("--model_name", type=str, dest="model_name", help="The name of the model to be loaded.", default='vgg16-397923af.pth')

parser.add_argument('--savedir',default='./models/', help="Path to save weigths and logs")
args = parser.parse_args()






# net = FGIAnet_3layers_1000()
# net = FGIAnet_vgg_100()#创建该神经网络的一个对象，并将其cuda化    #弱监督检索


# net = FGIAnet100_metric5_1_for_cam()
net = FGIAnet100_metric5_2_for_cam(scale=100)

# net = FGIAnet100_metric6_for_cam()
# net = FGIAnet100_metric5_for_cam()
# net = FGIAnet_2kernels_2layers_for_cam()

# net = FGIAnet_GARD_CAM100_metric()

#SCDA_fine_tuning
# checkpoint = torch.load(join(args.savedir,args.model_name))
checkpoint = torch.load(join(os.path.abspath(os.path.dirname(os.getcwd())),args.savedir,args.model_name))

net.load_state_dict(checkpoint['model'])



#
# # net = FGIAnet_GARD_CAM()
# net = FGIAnet_manylayers()
#
# #SCDA_CAM
# checkpoint = torch.load(join(os.path.abspath(os.path.dirname(os.getcwd())),args.savedir,args.model_name))
# net.load_state_dict(checkpoint['model'])





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
# img_name='133.White_throated_Sparrow/White_Throated_Sparrow_0102_128911.jpg'
# img_name='019.Gray_Catbird/Gray_Catbird_0032_21551.jpg'
# img_name='155.Warbling_Vireo/Warbling_Vireo_0086_158564.jpg'
# img_name='091.Mockingbird/Mockingbird_0095_81177.jpg'
# img_name='098.Scott_Oriole/Scott_Oriole_0052_92440.jpg'
# img_name='036.Northern_Flicker/Northern_Flicker_0021_28741.jpg'
# img_name='048.European_Goldfinch/European_Goldfinch_0053_794639.jpg'
# img_name='200.Common_Yellowthroat/Common_Yellowthroat_0055_190967.jpg'
# img_name='181.Worm_eating_Warbler/Worm_Eating_Warbler_0072_795559.jpg'
# img_name='199.Winter_Wren/Winter_Wren_0047_190390.jpg'
# img_name='001.Black_footed_Albatross/Black_Footed_Albatross_0046_18.jpg'
# img_name='011.Rusty_Blackbird/Rusty_Blackbird_0120_6762.jpg'
# img_name='025.Pelagic_Cormorant/Pelagic_Cormorant_0002_23680.jpg'
# img_name = '087.Mallard/Mallard_0123_76653.jpg'
# img_name = '073.Blue_Jay/Blue_Jay_0031_62913.jpg'
# img_name='186.Cedar_Waxwing/Cedar_Waxwing_0043_178321.jpg'
# img_name='199.Winter_Wren/Winter_Wren_0082_189549.jpg'
# img_name='060.Glaucous_winged_Gull/Glaucous_Winged_Gull_0035_44576.jpg'
#  200.Common_Yellowthroat/Common_Yellowthroat_0094_190690.jpg
# 200.Common_Yellowthroat/Common_Yellowthroat_0063_190440.jpg
#  200.Common_Yellowthroat/Common_Yellowthroat_0037_190698.jpg
#  200.Common_Yellowthroat/Common_Yellowthroat_0058_190958.jpg
#  200.Common_Yellowthroat/Common_Yellowthroat_0008_190703.jpg
#  200.Common_Yellowthroat/Common_Yellowthroat_0049_190708.jpg
#  200.Common_Yellowthroat/Common_Yellowthroat_0055_190967.jpg
# img_name = '097.Orchard_Oriole/Orchard_Oriole_0106_91830.jpg'
# img_name='035.Purple_Finch/Purple_Finch_0124_27567.jpg'
# img_name='200.Common_Yellowthroat/Common_Yellowthroat_0055_190967.jpg'
# img_name='200.Common_Yellowthroat/Common_Yellowthroat_0049_190708.jpg'
# img_name='200.Common_Yellowthroat/Common_Yellowthroat_0058_190958.jpg'
# img_name='200.Common_Yellowthroat/Common_Yellowthroat_0037_190698.jpg'
# img_name='200.Common_Yellowthroat/Common_Yellowthroat_0094_190690.jpg'
# img_name='084.Red_legged_Kittiwake/Red_Legged_Kittiwake_0036_73814.jpg'
# img_name = '016.Painted_Bunting/Painted_Bunting_0091_15198.jpg'
# 5676 097.Orchard_Oriole/Orchard_Oriole_0095_91345.jpg
# 5677 097.Orchard_Oriole/Orchard_Oriole_0030_91612.jpg
# 5678 097.Orchard_Oriole/Orchard_Oriole_0044_91360.jpg
# 5679 097.Orchard_Oriole/Orchard_Oriole_0018_91601.jpg
# 5680 097.Orchard_Oriole/Orchard_Oriole_0116_91645.jpg
# 5681 097.Orchard_Oriole/Orchard_Oriole_0046_91646.jpg
# 5682 097.Orchard_Oriole/Orchard_Oriole_0070_91383.jpg
# img_name='097.Orchard_Oriole/Orchard_Oriole_0030_91612.jpg'
# 5524 095.Baltimore_Oriole/Baltimore_Oriole_0019_88186.jpg
# 5525 095.Baltimore_Oriole/Baltimore_Oriole_0087_89726.jpg
# 5526 095.Baltimore_Oriole/Baltimore_Oriole_0014_87690.jpg
# 5527 095.Baltimore_Oriole/Baltimore_Oriole_0073_87187.jpg
# 5528 095.Baltimore_Oriole/Baltimore_Oriole_0101_87207.jpg
# img_name='095.Baltimore_Oriole/Baltimore_Oriole_0019_88186.jpg'
# img_name='095.Baltimore_Oriole/Baltimore_Oriole_0073_87187.jpg'
# img_name='054.Blue_Grosbeak/Blue_Grosbeak_0082_36991.jpg'
# img_name='068.Ruby_throated_Hummingbird/Ruby_Throated_Hummingbird_0090_57411.jpg'
# img_name='068.Ruby_throated_Hummingbird/Ruby_Throated_Hummingbird_0110_57851.jpg'
# img_name='075.Green_Jay/Green_Jay_0023_65898.jpg'
# img_name='075.Green_Jay/Green_Jay_0100_65786.jpg'
# img_name='075.Green_Jay/Green_Jay_0074_65889.jpg'
# img_name='063.Ivory_Gull/Ivory_Gull_0007_49364.jpg'
# img_name='063.Ivory_Gull/Ivory_Gull_0043_49755.jpg'
# img_name='169.Magnolia_Warbler/Magnolia_Warbler_0114_165467.jpg'
# img_name='112.Great_Grey_Shrike/Great_Grey_Shrike_0062_106628.jpg'
# img_name='105.Whip_poor_Will/Whip_Poor_Will_0046_796440.jpg'
# img_name='186.Cedar_Waxwing/Cedar_Waxwing_0034_179715.jpg'   #优秀
# img_name='188.Pileated_Woodpecker/Pileated_Woodpecker_0080_180589.jpg'  #可
# img_name='199.Winter_Wren/Winter_Wren_0095_189985.jpg'#ok
# img_name='134.Cape_Glossy_Starling/Cape_Glossy_Starling_0047_129348.jpg'  #可
# img_name='060.Glaucous_winged_Gull/Glaucous_Winged_Gull_0023_45090.jpg'#ok
# img_name='060.Glaucous_winged_Gull/Glaucous_Winged_Gull_0007_44575.jpg'
# img_name='063.Ivory_Gull/Ivory_Gull_0114_49535.jpg'#ok
# img_name='063.Ivory_Gull/Ivory_Gull_0007_49364.jpg'#ok
# img_name='079.Belted_Kingfisher/Belted_Kingfisher_0034_70329.jpg'  #ok
# img_name='079.Belted_Kingfisher/Belted_Kingfisher_0082_70711.jpg'
# img_name='079.Belted_Kingfisher/Belted_Kingfisher_0048_70532.jpg'
# img_name='079.Belted_Kingfisher/Belted_Kingfisher_0080_70725.jpg'
# img_name='094.White_breasted_Nuthatch/White_Breasted_Nuthatch_0115_86760.jpg'
# img_name='094.White_breasted_Nuthatch/White_Breasted_Nuthatch_0096_86140.jpg'
# img_name='065.Slaty_backed_Gull/Slaty_Backed_Gull_0086_786387.jpg'
# img_name='066.Western_Gull/Western_Gull_0065_55728.jpg'
# img_name='066.Western_Gull/Western_Gull_0050_54425.jpg'
# img_name='066.Western_Gull/Western_Gull_0036_54329.jpg'
# img_name = '073.Blue_Jay/Blue_Jay_0031_62913.jpg'
# img_name='025.Pelagic_Cormorant/Pelagic_Cormorant_0002_23680.jpg'
# img_name='011.Rusty_Blackbird/Rusty_Blackbird_0120_6762.jpg'
# img_name='019.Gray_Catbird/Gray_Catbird_0032_21551.jpg'
# img_name='035.Purple_Finch/Purple_Finch_0124_27567.jpg'
# img_name='054.Blue_Grosbeak/Blue_Grosbeak_0082_36991.jpg'
# img_name='068.Ruby_throated_Hummingbird/Ruby_Throated_Hummingbird_0090_57411.jpg'
# img_name = '097.Orchard_Oriole/Orchard_Oriole_0106_91830.jpg'
# img_name='181.Worm_eating_Warbler/Worm_Eating_Warbler_0072_795559.jpg'
# img_name='120.Fox_Sparrow/Fox_Sparrow_0118_114884.jpg'
# img_name='180.Wilson_Warbler/Wilson_Warbler_0060_175420.jpg'
# img_name='105.Whip_poor_Will/Whip_Poor_Will_0046_796440.jpg'
# img_name='169.Magnolia_Warbler/Magnolia_Warbler_0114_165467.jpg'
# img_name='112.Great_Grey_Shrike/Great_Grey_Shrike_0062_106628.jpg'
# img_name='134.Cape_Glossy_Starling/Cape_Glossy_Starling_0047_129348.jpg'
# img_name='192.Downy_Woodpecker/Downy_Woodpecker_0058_184520.jpg'
# img_name='200.Common_Yellowthroat/Common_Yellowthroat_0055_190967.jpg'




# img_name='063.Ivory_Gull/Ivory_Gull_0114_49535.jpg'
# img_name='060.Glaucous_winged_Gull/Glaucous_Winged_Gull_0118_2081.jpg'
# img_name='060.Glaucous_winged_Gull/Glaucous_Winged_Gull_0023_45090.jpg'
# img_name='060.Glaucous_winged_Gull/Glaucous_Winged_Gull_0007_44575.jpg'
# img_name='066.Western_Gull/Western_Gull_0065_55728.jpg'
# img_name='066.Western_Gull/Western_Gull_0050_54425.jpg'
# img_name='066.Western_Gull/Western_Gull_0036_54329.jpg'
# img_name='066.Western_Gull/Western_Gull_0045_54735.jpg'
# img_name='065.Slaty_backed_Gull/Slaty_Backed_Gull_0086_786387.jpg'
# img_name='065.Slaty_backed_Gull/Slaty_Backed_Gull_0078_796042.jpg'
# img_name='065.Slaty_backed_Gull/Slaty_Backed_Gull_0086_786387.jpg'
# img_name='071.Long_tailed_Jaeger/Long_Tailed_Jaeger_0001_797061.jpg'

# img_name='094.White_breasted_Nuthatch/White_Breasted_Nuthatch_0115_86760.jpg'
# img_name='094.White_breasted_Nuthatch/White_Breasted_Nuthatch_0096_86140.jpg'
# img_name='079.Belted_Kingfisher/Belted_Kingfisher_0082_70711.jpg'
# img_name='079.Belted_Kingfisher/Belted_Kingfisher_0048_70532.jpg'
# img_name='079.Belted_Kingfisher/Belted_Kingfisher_0080_70725.jpg'
# img_name='079.Belted_Kingfisher/Belted_Kingfisher_0034_70329.jpg'
# img_name='196.House_Wren/House_Wren_0083_187406.jpg'
# img_name='134.Cape_Glossy_Starling/Cape_Glossy_Starling_0047_129348.jpg'










# img_name='170.Mourning_Warbler/Mourning_Warbler_0029_166530.jpg'
# img_name='189.Red_bellied_Woodpecker/Red_Bellied_Woodpecker_0022_181969.jpg'
#上面的这些行都没用，主要是指定test_paths,这里面指定这我们所要检索的query所在的路径
# img_name='001.Black_footed_Albatross/Black_Footed_Albatross_0085_92.jpg'
# img_name='001.Black_footed_Albatross/Black_Footed_Albatross_0053_796109.jpg'
# img_name='001.Black_footed_Albatross/Black_Footed_Albatross_0025_796057.jpg'
# img_name='019.Gray_Catbird/Gray_Catbird_0032_21551.jpg'
# img_name='155.Warbling_Vireo/Warbling_Vireo_0086_158564.jpg'
# img_name='091.Mockingbird/Mockingbird_0095_81177.jpg'
# img_name='098.Scott_Oriole/Scott_Oriole_0052_92440.jpg'
# img_name='036.Northern_Flicker/Northern_Flicker_0021_28741.jpg'
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
# img_name='169.Magnolia_Warbler/Magnolia_Warbler_0114_165467.jpg'
# img_name='194.Cactus_Wren/Cactus_Wren_0080_185901.jpg'

# img_name='190.Red_cockaded_Woodpecker/Red_Cockaded_Woodpecker_0022_794700.jpg'
# img_name='115.Brewer_Sparrow/Brewer_Sparrow_0014_107435.jpg'
# img_name='112.Great_Grey_Shrike/Great_Grey_Shrike_0062_106628.jpg'
# img_name='105.Whip_poor_Will/Whip_Poor_Will_0046_796440.jpg'
#
# img_name='192.Downy_Woodpecker/Downy_Woodpecker_0058_184520.jpg'

# img_name='186.Cedar_Waxwing/Cedar_Waxwing_0034_179715.jpg'
# img_name='186.Cedar_Waxwing/Cedar_Waxwing_0093_178139.jpg'
# img_name='186.Cedar_Waxwing/Cedar_Waxwing_0037_179710.jpg'
# img_name='188.Pileated_Woodpecker/Pileated_Woodpecker_0080_180589.jpg'
# img_name='192.Downy_Woodpecker/Downy_Woodpecker_0058_184520.jpg'
# img_name='194.Cactus_Wren/Cactus_Wren_0080_185901.jpg'
# img_name='196.House_Wren/House_Wren_0083_187406.jpg'
# img_name='199.Winter_Wren/Winter_Wren_0095_189985.jpg'
# img_name='134.Cape_Glossy_Starling/Cape_Glossy_Starling_0047_129348.jpg'
# img_name=r'D:\python_work\SCDA\SCDA_pro\retrivial_visualize\croped_images\422_1.jpg'
# img_name='105.Whip_poor_Will/Whip_Poor_Will_0046_796440.jpg'
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
# img_name='073.Blue_Jay/Blue_Jay_0024_63167.jpg'
# img_name='070.Green_Violetear/Green_Violetear_0086_795639.jpg'
# img_name='056.Pine_Grosbeak/Pine_Grosbeak_0088_38830.jpg'
# img_name='056.Pine_Grosbeak/Pine_Grosbeak_0115_38330.jpg'  #ok
# img_name='056.Pine_Grosbeak/Pine_Grosbeak_0094_38912.jpg'  #ok
# img_name='056.Pine_Grosbeak/Pine_Grosbeak_0032_38473.jpg'
# img_name='056.Pine_Grosbeak/Pine_Grosbeak_0025_38443.jpg'
img_name='054.Blue_Grosbeak/Blue_Grosbeak_0095_37007.jpg'
# img_name='054.Blue_Grosbeak/Blue_Grosbeak_0022_37082.jpg'
# img_name='054.Blue_Grosbeak/Blue_Grosbeak_0057_37116.jpg'
# img_name='054.Blue_Grosbeak/Blue_Grosbeak_0078_36655.jpg'
# img_name='054.Blue_Grosbeak/Blue_Grosbeak_0027_36703.jpg'



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
# img_name = '063.Ivory_Gull/Ivory_Gull_0007_49364.jpg'
# img_name = '063.Ivory_Gull/Ivory_Gull_0043_49755.jpg'
# img_name = '060.Glaucous_winged_Gull/Glaucous_Winged_Gull_0023_45090.jpg'
# img_name = '060.Glaucous_winged_Gull/Glaucous_Winged_Gull_0116_45236.jpg'
mikimiki=53

test_path=[]
test_path.append(join(args.datasetdir,args.imgdir,img_name))
loaders = dataloader.get_dataloaders(train_paths, test_path,train_labels,test_labels,args.img_size, 1,1,1,SCDA_flag=1)#返回值为由可迭代DataLoader对象所组成的字典



print("cnn model is ready.")



# miki_1=cv2.imread(str(test_paths[0]))
miki = Image.open(test_path[0]).convert('RGB')
width,height = miki.size
print(miki.size)
# print(cam_L31)


ii=0
for phase in ['test']:
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
                        # labels=labels.cpu()

                        #proxy_based
                        # grad_cam_scda = GradCam_new_many_layers(net)#两个输出，softmax时用，乘上了-1的loss
                        grad_cam_scda = GradCam_new_many_layers_3out(net)#三个输出，度量学习时用,乘上了-1的loss

                        # grad_cam_scda = GradCam_new_many_layers_version_2(net)##两个输出，softmax时用，没有乘上-1的loss
                        grad_cam_scda_2 = GradCam_new_many_layers_version_2_3out(net)##三个输出，度量学习时用,没有乘上-1的loss,特征图点乘梯度
                        grad_cam_scda_2_1 =GradCam_new_many_layers_3out0(net)###三个输出，度量学习时用,没有乘上-1的loss,通道级均值作为权重

                        grad_cam_scda_3 = GradCam_new_many_layers_version_2_3out_tmp(net)##三个输出，度量学习时用，特征图点乘梯度
                        # grad_cam_scda = GradCam_new_many_layers_version_2_2out_tmp(net)##两个输出，softmax时用

                        grad_cam_scda_4 = GradCam_new_many_layers_version_2_3out_tmp1(net)##三个输出，度量学习时用，这是标准的grad_cam，通道级均值作为权重

                        cams ,cams_relu=grad_cam_scda(images)
                        cams_2, cams_relu_2 = grad_cam_scda_2(images)
                        cams_2_1, cams_relu_2_1 = grad_cam_scda_2_1(images)

                        cams_3, cams_relu_3 = grad_cam_scda_3(images,index=mikimiki)
                        cams_4, cams_relu_4 = grad_cam_scda_4(images,index=mikimiki)

                        #                        pooling_based_no_saliency_map

                        # grad_cam_scda_2_1 = PoolCam_new_many_layers_3out0(net)  ###三个输出，度量学习时用,没有乘上-1的loss,通道级均值作为权重
                        # grad_cam_scda_4 = GradCam_new_many_layers_version_2_3out_tmp1(
                        #     net)  ##三个输出，度量学习时用，这是标准的grad_cam，通道级均值作为权重
                        #
                        # cams_2_1, cams_relu_2_1 = grad_cam_scda_2_1(images)
                        # cams_4, cams_relu_4 = grad_cam_scda_4(images, index=mikimiki)

                                              # pooling_based_with_saliency_map

                        # grad_cam_scda_2_1 = PoolCam_new_many_layers_3out0_saliency_map(net)  ###三个输出，度量学习时用,没有乘上-1的loss,通道级均值作为权重
                        # grad_cam_scda_4 = GradCam_new_many_layers_version_2_3out_tmp1(
                        #      net)  ##三个输出，度量学习时用，这是标准的grad_cam，通道级均值作为权重
                        #
                        # cams_2_1, cams_relu_2_1 = grad_cam_scda_2_1(images)
                        # cams_4, cams_relu_4 = grad_cam_scda_4(images, index=mikimiki)

                # random_chanels=random_int(10,0,99)
                        # random_chanels=[iii for iii in range(100)]
                        # print(random_chanels)
                        # for j in range(len(random_chanels)):
                        #     cams_random,cams_relu_random = grad_cam_scda_3(images,index=random_chanels[j])
                        #     for i in range(len(cams_random)):
                        #         cam = cams_random[i] * 255
                        #         # print(np.max(cam))
                        #         cam_L31 = Image.fromarray(cam).convert('L')
                        #         cam_L31 = cam_L31.resize((width, height), Image.ANTIALIAS)
                        #         print(cam_L31.size)
                        #         # Image._show(cam_L31)
                        #         cam_L31 = cv2.cvtColor(np.asarray(cam_L31), cv2.COLOR_RGB2BGR)
                        #         # cam_L31 = cv2.cvtColor(np.asarray(cam_L31))
                        #         cam_L31 = cv2.applyColorMap(cam_L31, cv2.COLORMAP_JET)
                        #         miki_1 = cv2.cvtColor(np.asarray(miki), cv2.COLOR_RGB2BGR)
                        #         # cam_L31 = Image.fromarray(cv2.cvtColor(cam_L31,cv2.COLOR_BGR2RGB))
                        #         # heatmap_L31,miki_1=np.array(heatmap_L31),np.array(miki_1)
                        #         result = cam_L31 + miki_1 * 0.3
                        #         # result =  miki_1
                        #
                        #         # result = (cam_L31/255) *miki_1
                        #
                        #         print(result.shape)
                        #         # plt.imsave('1.jpg',heatmap_L31)
                        #         path = './retrivial_visualize/' + 'random_class_'+str(random_chanels[j]) + '_grad_cam.jpg'
                        #         cv2.imwrite(join(os.path.abspath(os.path.dirname(os.getcwd())), path), result)
                        #         cv2.waitKey(0)

                            # for i in range(len(cams_relu_random)):
                            #     cam = cams_relu_random[i] * 255
                            #     # print(np.max(cam))
                            #     cam_L31 = Image.fromarray(cam).convert('L')
                            #     cam_L31 = cam_L31.resize((width, height), Image.ANTIALIAS)
                            #     print(cam_L31.size)
                            #     # Image._show(cam_L31)
                            #     cam_L31 = cv2.cvtColor(np.asarray(cam_L31), cv2.COLOR_RGB2BGR)
                            #     # cam_L31 = cv2.cvtColor(np.asarray(cam_L31))
                            #     cam_L31 = cv2.applyColorMap(cam_L31, cv2.COLORMAP_JET)
                            #     miki_1 = cv2.cvtColor(np.asarray(miki), cv2.COLOR_RGB2BGR)
                            #     # cam_L31 = Image.fromarray(cv2.cvtColor(cam_L31,cv2.COLOR_BGR2RGB))
                            #     # heatmap_L31,miki_1=np.array(heatmap_L31),np.array(miki_1)
                            #     result = cam_L31 + miki_1 * 0.3
                            #     # result =  miki_1
                            #
                            #     # result = (cam_L31/255) *miki_1
                            #
                            #     print(result.shape)
                            #     # plt.imsave('1.jpg',heatmap_L31)
                            #     path = './retrivial_visualize/' + 'random_chanel_' + str(
                            #         random_chanels[j]) + '_grad_cam_with_relu.jpg'
                            #     cv2.imwrite(join(os.path.abspath(os.path.dirname(os.getcwd())), path), result)
                            #     cv2.waitKey(0)

# hina=[0,1,4,6,9,11,13,16,18,20,23,25,27,30]
hina=[29]

# cam_L31=cam_L31_noflip * 255  #因为grad_cam.py中最后是有对cam_L31进行0-1的限制的
# cam_L28=cam_L28_noflip * 255


#注释
# for i  in range(len(cams)):
#     cam=cams[i]*255
#     # print(np.max(cam))
#     cam_L31 = Image.fromarray(cam).convert('L')
#     cam_L31 = cam_L31.resize((width, height),Image.ANTIALIAS)
#     print(cam_L31.size)
#     # Image._show(cam_L31)
#     cam_L31 = cv2.cvtColor(np.asarray(cam_L31),cv2.COLOR_RGB2BGR)
#     # cam_L31 = cv2.cvtColor(np.asarray(cam_L31))
#     cam_L31 = cv2.applyColorMap(cam_L31, cv2.COLORMAP_JET)
#     miki_1=cv2.cvtColor(np.asarray(miki),cv2.COLOR_RGB2BGR)
#     # cam_L31 = Image.fromarray(cv2.cvtColor(cam_L31,cv2.COLOR_BGR2RGB))
#     # heatmap_L31,miki_1=np.array(heatmap_L31),np.array(miki_1)
#     result = cam_L31+ miki_1*0.3
#     # result =  miki_1
#
#     # result = (cam_L31/255) *miki_1
#
#     print(result.shape)
#     # plt.imsave('1.jpg',heatmap_L31)
#     path='./retrivial_visualize/' + str(hina[i]) + '_-1_loss.jpg'
#     cv2.imwrite(join(os.path.abspath(os.path.dirname(os.getcwd())) , path),result)
#     cv2.waitKey(0)









for i  in range(len(cams_2_1)):
    cam=cams_2_1[i]*255
    # print(np.max(cam))
    cam_L31 = Image.fromarray(cam).convert('L')
    cam_L31 = cam_L31.resize((width, height),Image.ANTIALIAS)
    print(cam_L31.size)
    # Image._show(cam_L31)
    cam_L31 = cv2.cvtColor(np.asarray(cam_L31),cv2.COLOR_RGB2BGR)
    # cam_L31 = cv2.cvtColor(np.asarray(cam_L31))
    cam_L31 = cv2.applyColorMap(cam_L31, cv2.COLORMAP_JET)
    miki_1=cv2.cvtColor(np.asarray(miki),cv2.COLOR_RGB2BGR)
    # cam_L31 = Image.fromarray(cv2.cvtColor(cam_L31,cv2.COLOR_BGR2RGB))
    # heatmap_L31,miki_1=np.array(heatmap_L31),np.array(miki_1)
    result = cam_L31+ miki_1*0.5
    # result =  miki_1

    # result = (cam_L31/255) *miki_1

    print(result.shape)
    # plt.imsave('1.jpg',heatmap_L31)
    path='./retrivial_visualize/' + str(hina[i]) + '_+1_loss_channel_mean.jpg'
    cv2.imwrite(join(os.path.abspath(os.path.dirname(os.getcwd())) , path),result)
    cv2.waitKey(0)
















# for i  in range(len(cams_2)):
#     cam=cams_2[i]*255
#     # print(np.max(cam))
#     cam_L31 = Image.fromarray(cam).convert('L')
#     cam_L31 = cam_L31.resize((width, height),Image.ANTIALIAS)
#     print(cam_L31.size)
#     # Image._show(cam_L31)
#     cam_L31 = cv2.cvtColor(np.asarray(cam_L31),cv2.COLOR_RGB2BGR)
#     # cam_L31 = cv2.cvtColor(np.asarray(cam_L31))
#     cam_L31 = cv2.applyColorMap(cam_L31, cv2.COLORMAP_JET)
#     miki_1=cv2.cvtColor(np.asarray(miki),cv2.COLOR_RGB2BGR)
#     # cam_L31 = Image.fromarray(cv2.cvtColor(cam_L31,cv2.COLOR_BGR2RGB))
#     # heatmap_L31,miki_1=np.array(heatmap_L31),np.array(miki_1)
#     result = cam_L31+ miki_1*0.5
#     # result =  miki_1
#
#     # result = (cam_L31/255) *miki_1
#
#     print(result.shape)
#     # plt.imsave('1.jpg',heatmap_L31)
#     path='./retrivial_visualize/' + str(hina[i]) + '_+1_loss.jpg'
#     cv2.imwrite(join(os.path.abspath(os.path.dirname(os.getcwd())) , path),result)
#     cv2.waitKey(0)
#
#     # L31_sum_mean = np.mean(cams_2[i])  # 返回的是scalar tensor ,这是与matlab的mean不同的地方
#     # # largestConnectComponent将最大连通区域所对应的像素点置为true
#     # jj = (cams_2[i] > L31_sum_mean)+0
#     #
#     # cam = jj * 255
#     # # print(np.max(cam))
#     # cam_L31 = Image.fromarray(cam).convert('L')
#     # cam_L31 = cam_L31.resize((width, height), Image.ANTIALIAS)
#     # print(cam_L31.size)
#     # # Image._show(cam_L31)
#     # cam_L31 = cv2.cvtColor(np.asarray(cam_L31), cv2.COLOR_RGB2BGR)
#     # # cam_L31 = cv2.cvtColor(np.asarray(cam_L31))
#     # cam_L31 = cv2.applyColorMap(cam_L31, cv2.COLORMAP_JET)
#     # miki_1 = cv2.cvtColor(np.asarray(miki), cv2.COLOR_RGB2BGR)
#     # # cam_L31 = Image.fromarray(cv2.cvtColor(cam_L31,cv2.COLOR_BGR2RGB))
#     # # heatmap_L31,miki_1=np.array(heatmap_L31),np.array(miki_1)
#     # result = cam_L31 + miki_1 * 0.3
#     # # result =  miki_1
#     #
#     # # result = (cam_L31/255) *miki_1
#     #
#     # print(result.shape)
#     # # plt.imsave('1.jpg',heatmap_L31)
#     # path = './retrivial_visualize/' + str(hina[i]) + '_+1_loss_highlight.jpg'
#     # cv2.imwrite(join(os.path.abspath(os.path.dirname(os.getcwd())), path), result)
#     # cv2.waitKey(0)




# for i  in range(len(cams_3)):
#     cam=cams_3[i]*255
#     # print(np.max(cam))
#     cam_L31 = Image.fromarray(cam).convert('L')
#     cam_L31 = cam_L31.resize((width, height),Image.ANTIALIAS)
#     print(cam_L31.size)
#     # Image._show(cam_L31)
#     cam_L31 = cv2.cvtColor(np.asarray(cam_L31),cv2.COLOR_RGB2BGR)
#     # cam_L31 = cv2.cvtColor(np.asarray(cam_L31))
#     cam_L31 = cv2.applyColorMap(cam_L31, cv2.COLORMAP_JET)
#     miki_1=cv2.cvtColor(np.asarray(miki),cv2.COLOR_RGB2BGR)
#     # cam_L31 = Image.fromarray(cv2.cvtColor(cam_L31,cv2.COLOR_BGR2RGB))
#     # heatmap_L31,miki_1=np.array(heatmap_L31),np.array(miki_1)
#     result = cam_L31+ miki_1*0.5
#     # result =  miki_1
#
#     # result = (cam_L31/255) *miki_1
#
#     print(result.shape)
#     # plt.imsave('1.jpg',heatmap_L31)
#     path='./retrivial_visualize/' + str(hina[i]) + '_true_grad_cam.jpg'
#     cv2.imwrite(join(os.path.abspath(os.path.dirname(os.getcwd())) , path),result)
#     cv2.waitKey(0)
#
#     # L31_sum_mean = np.mean(cams_3[i])  # 返回的是scalar tensor ,这是与matlab的mean不同的地方
#     # # largestConnectComponent将最大连通区域所对应的像素点置为true
#     # jj = (cams_3[i] > L31_sum_mean) + 0
#     #
#     # cam = jj * 255
#     # # print(np.max(cam))
#     # cam_L31 = Image.fromarray(cam).convert('L')
#     # cam_L31 = cam_L31.resize((width, height), Image.ANTIALIAS)
#     # print(cam_L31.size)
#     # # Image._show(cam_L31)
#     # cam_L31 = cv2.cvtColor(np.asarray(cam_L31), cv2.COLOR_RGB2BGR)
#     # # cam_L31 = cv2.cvtColor(np.asarray(cam_L31))
#     # cam_L31 = cv2.applyColorMap(cam_L31, cv2.COLORMAP_JET)
#     # miki_1 = cv2.cvtColor(np.asarray(miki), cv2.COLOR_RGB2BGR)
#     # # cam_L31 = Image.fromarray(cv2.cvtColor(cam_L31,cv2.COLOR_BGR2RGB))
#     # # heatmap_L31,miki_1=np.array(heatmap_L31),np.array(miki_1)
#     # result = cam_L31 + miki_1 * 0.3
#     # # result =  miki_1
#     #
#     # # result = (cam_L31/255) *miki_1
#     #
#     # print(result.shape)
#     # # plt.imsave('1.jpg',heatmap_L31)
#     # path = './retrivial_visualize/' + str(hina[i]) + '_true_grad_cam_highlight.jpg'
#     # cv2.imwrite(join(os.path.abspath(os.path.dirname(os.getcwd())), path), result)
#     # cv2.waitKey(0)








for i  in range(len(cams_4)):
    cam=cams_4[i]*255
    # print(np.max(cam))
    cam_L31 = Image.fromarray(cam).convert('L')
    cam_L31 = cam_L31.resize((width, height),Image.ANTIALIAS)
    print(cam_L31.size)
    # Image._show(cam_L31)
    cam_L31 = cv2.cvtColor(np.asarray(cam_L31),cv2.COLOR_RGB2BGR)
    # cam_L31 = cv2.cvtColor(np.asarray(cam_L31))
    cam_L31 = cv2.applyColorMap(cam_L31, cv2.COLORMAP_JET)
    miki_1=cv2.cvtColor(np.asarray(miki),cv2.COLOR_RGB2BGR)
    # cam_L31 = Image.fromarray(cv2.cvtColor(cam_L31,cv2.COLOR_BGR2RGB))
    # heatmap_L31,miki_1=np.array(heatmap_L31),np.array(miki_1)
    result = cam_L31+ miki_1*0.5
    # result =  miki_1

    # result = (cam_L31/255) *miki_1

    print(result.shape)
    # plt.imsave('1.jpg',heatmap_L31)
    path='./retrivial_visualize/' + str(hina[i]) + '_grad_cam_channel_mean.jpg'
    cv2.imwrite(join(os.path.abspath(os.path.dirname(os.getcwd())) , path),result)
    cv2.waitKey(0)












# for i  in range(len(cams_relu)):
#     cam=cams_relu[i]*255
#     # print(np.max(cam))
#     cam_L31 = Image.fromarray(cam).convert('L')
#     cam_L31 = cam_L31.resize((width, height),Image.ANTIALIAS)
#     print(cam_L31.size)
#     # Image._show(cam_L31)
#     cam_L31 = cv2.cvtColor(np.asarray(cam_L31),cv2.COLOR_RGB2BGR)
#     # cam_L31 = cv2.cvtColor(np.asarray(cam_L31))
#     cam_L31 = cv2.applyColorMap(cam_L31, cv2.COLORMAP_JET)
#     miki_1=cv2.cvtColor(np.asarray(miki),cv2.COLOR_RGB2BGR)
#     # cam_L31 = Image.fromarray(cv2.cvtColor(cam_L31,cv2.COLOR_BGR2RGB))
#     # heatmap_L31,miki_1=np.array(heatmap_L31),np.array(miki_1)
#     result = cam_L31+ miki_1*0.3
#     # result =  miki_1
#
#     # result = (cam_L31/255) *miki_1
#
#     print(result.shape)
#     # plt.imsave('1.jpg',heatmap_L31)
#     path='./retrivial_visualize/' + str(hina[i]) + '_-1_loss_with_relu.jpg'
#     cv2.imwrite(join(os.path.abspath(os.path.dirname(os.getcwd())) , path),result)
#     cv2.waitKey(0)






for i  in range(len(cams_relu_2_1)):
    cam=cams_relu_2_1[i]*255
    # print(np.max(cam))
    cam_L31 = Image.fromarray(cam).convert('L')
    cam_L31 = cam_L31.resize((width, height),Image.ANTIALIAS)
    print(cam_L31.size)
    # Image._show(cam_L31)
    cam_L31 = cv2.cvtColor(np.asarray(cam_L31),cv2.COLOR_RGB2BGR)
    # cam_L31 = cv2.cvtColor(np.asarray(cam_L31))
    cam_L31 = cv2.applyColorMap(cam_L31, cv2.COLORMAP_JET)
    miki_1=cv2.cvtColor(np.asarray(miki),cv2.COLOR_RGB2BGR)
    # cam_L31 = Image.fromarray(cv2.cvtColor(cam_L31,cv2.COLOR_BGR2RGB))
    # heatmap_L31,miki_1=np.array(heatmap_L31),np.array(miki_1)
    result = cam_L31+ miki_1*0.5
    # result =  miki_1

    # result = (cam_L31/255) *miki_1

    print(result.shape)
    # plt.imsave('1.jpg',heatmap_L31)
    path='./retrivial_visualize/' + str(hina[i]) + '_+1_loss_with_relu_channel_mean.jpg'
    cv2.imwrite(join(os.path.abspath(os.path.dirname(os.getcwd())) , path),result)
    cv2.waitKey(0)






#
#
# for i  in range(len(cams_relu_2)):
#     cam=cams_relu_2[i]*255
#     # print(np.max(cam))
#     cam_L31 = Image.fromarray(cam).convert('L')
#     cam_L31 = cam_L31.resize((width, height),Image.ANTIALIAS)
#     print(cam_L31.size)
#     # Image._show(cam_L31)
#     cam_L31 = cv2.cvtColor(np.asarray(cam_L31),cv2.COLOR_RGB2BGR)
#     # cam_L31 = cv2.cvtColor(np.asarray(cam_L31))
#     cam_L31 = cv2.applyColorMap(cam_L31, cv2.COLORMAP_JET)
#     miki_1=cv2.cvtColor(np.asarray(miki),cv2.COLOR_RGB2BGR)
#     # cam_L31 = Image.fromarray(cv2.cvtColor(cam_L31,cv2.COLOR_BGR2RGB))
#     # heatmap_L31,miki_1=np.array(heatmap_L31),np.array(miki_1)
#     result = cam_L31+ miki_1*0.5
#     # result =  miki_1
#
#     # result = (cam_L31/255) *miki_1
#
#     print(result.shape)
#     # plt.imsave('1.jpg',heatmap_L31)
#     path='./retrivial_visualize/' + str(hina[i]) + '_+1_loss_with_relu.jpg'
#     cv2.imwrite(join(os.path.abspath(os.path.dirname(os.getcwd())) , path),result)
#     cv2.waitKey(0)
#
#     # L31_sum_mean = np.mean(cams_relu_2[i])  # 返回的是scalar tensor ,这是与matlab的mean不同的地方
#     # # largestConnectComponent将最大连通区域所对应的像素点置为true
#     # jj = (cams_relu_2[i] > L31_sum_mean) + 0
#     #
#     # cam = jj * 255
#     # # print(np.max(cam))
#     # cam_L31 = Image.fromarray(cam).convert('L')
#     # cam_L31 = cam_L31.resize((width, height), Image.ANTIALIAS)
#     # print(cam_L31.size)
#     # # Image._show(cam_L31)
#     # cam_L31 = cv2.cvtColor(np.asarray(cam_L31), cv2.COLOR_RGB2BGR)
#     # # cam_L31 = cv2.cvtColor(np.asarray(cam_L31))
#     # cam_L31 = cv2.applyColorMap(cam_L31, cv2.COLORMAP_JET)
#     # miki_1 = cv2.cvtColor(np.asarray(miki), cv2.COLOR_RGB2BGR)
#     # # cam_L31 = Image.fromarray(cv2.cvtColor(cam_L31,cv2.COLOR_BGR2RGB))
#     # # heatmap_L31,miki_1=np.array(heatmap_L31),np.array(miki_1)
#     # result = cam_L31 + miki_1 * 0.3
#     # # result =  miki_1
#     #
#     # # result = (cam_L31/255) *miki_1
#     #
#     # print(result.shape)
#     # # plt.imsave('1.jpg',heatmap_L31)
#     # path = './retrivial_visualize/' + str(hina[i]) + '_+1_loss_with_relu_highlight.jpg'
#     # cv2.imwrite(join(os.path.abspath(os.path.dirname(os.getcwd())), path), result)
#     # cv2.waitKey(0)

# for i  in range(len(cams_relu_3)):
#     cam=cams_relu_3[i]*255
#     # print(np.max(cam))
#     cam_L31 = Image.fromarray(cam).convert('L')
#     cam_L31 = cam_L31.resize((width, height),Image.ANTIALIAS)
#     print(cam_L31.size)
#     # Image._show(cam_L31)
#     cam_L31 = cv2.cvtColor(np.asarray(cam_L31),cv2.COLOR_RGB2BGR)
#     # cam_L31 = cv2.cvtColor(np.asarray(cam_L31))
#     cam_L31 = cv2.applyColorMap(cam_L31, cv2.COLORMAP_JET)
#     miki_1=cv2.cvtColor(np.asarray(miki),cv2.COLOR_RGB2BGR)
#     # cam_L31 = Image.fromarray(cv2.cvtColor(cam_L31,cv2.COLOR_BGR2RGB))
#     # heatmap_L31,miki_1=np.array(heatmap_L31),np.array(miki_1)
#     result = cam_L31+ miki_1*0.5
#     # result =  miki_1
#
#     # result = (cam_L31/255) *miki_1
#
#     print(result.shape)
#     # plt.imsave('1.jpg',heatmap_L31)
#     path='./retrivial_visualize/' + str(hina[i]) + '_true_grad_cam_with_relu.jpg'
#     cv2.imwrite(join(os.path.abspath(os.path.dirname(os.getcwd())) , path),result)
#     cv2.waitKey(0)
#
#     # L31_sum_mean = np.mean(cams_relu_3[i])  # 返回的是scalar tensor ,这是与matlab的mean不同的地方
#     # # largestConnectComponent将最大连通区域所对应的像素点置为true
#     # jj = (cams_relu_3[i] > L31_sum_mean) + 0
#     #
#     # cam = jj * 255
#     # # print(np.max(cam))
#     # cam_L31 = Image.fromarray(cam).convert('L')
#     # cam_L31 = cam_L31.resize((width, height), Image.ANTIALIAS)
#     # print(cam_L31.size)
#     # # Image._show(cam_L31)
#     # cam_L31 = cv2.cvtColor(np.asarray(cam_L31), cv2.COLOR_RGB2BGR)
#     # # cam_L31 = cv2.cvtColor(np.asarray(cam_L31))
#     # cam_L31 = cv2.applyColorMap(cam_L31, cv2.COLORMAP_JET)
#     # miki_1 = cv2.cvtColor(np.asarray(miki), cv2.COLOR_RGB2BGR)
#     # # cam_L31 = Image.fromarray(cv2.cvtColor(cam_L31,cv2.COLOR_BGR2RGB))
#     # # heatmap_L31,miki_1=np.array(heatmap_L31),np.array(miki_1)
#     # result = cam_L31 + miki_1 * 0.3
#     # # result =  miki_1
#     #
#     # # result = (cam_L31/255) *miki_1
#     #
#     # print(result.shape)
#     # # plt.imsave('1.jpg',heatmap_L31)
#     # path = './retrivial_visualize/' + str(hina[i]) + '_true_grad_cam_with_relu_highlight.jpg'
#     # cv2.imwrite(join(os.path.abspath(os.path.dirname(os.getcwd())), path), result)
#     # cv2.waitKey(0)





for i  in range(len(cams_relu_4)):
    cam=cams_relu_4[i]*255
    # print(np.max(cam))
    cam_L31 = Image.fromarray(cam).convert('L')
    cam_L31 = cam_L31.resize((width, height),Image.ANTIALIAS)
    print(cam_L31.size)
    # Image._show(cam_L31)
    cam_L31 = cv2.cvtColor(np.asarray(cam_L31),cv2.COLOR_RGB2BGR)
    # cam_L31 = cv2.cvtColor(np.asarray(cam_L31))
    cam_L31 = cv2.applyColorMap(cam_L31, cv2.COLORMAP_JET)
    miki_1=cv2.cvtColor(np.asarray(miki),cv2.COLOR_RGB2BGR)
    # cam_L31 = Image.fromarray(cv2.cvtColor(cam_L31,cv2.COLOR_BGR2RGB))
    # heatmap_L31,miki_1=np.array(heatmap_L31),np.array(miki_1)
    result = cam_L31+ miki_1*0.5
    # result =  miki_1

    # result = (cam_L31/255) *miki_1

    print(result.shape)
    # plt.imsave('1.jpg',heatmap_L31)
    path='./retrivial_visualize/' + str(hina[i]) + '_grad_cam_with_relu_channel_mean.jpg'
    cv2.imwrite(join(os.path.abspath(os.path.dirname(os.getcwd())) , path),result)
    cv2.waitKey(0)











for i  in range(len(cams_relu_2_1)):
    cam=cams_relu_2_1[i]*255
    # print(np.max(cam))
    cam_L31 = Image.fromarray(cam).convert('L')
    cam_L31 = cam_L31.resize((width, height),Image.ANTIALIAS)
    print(cam_L31.size)
    # Image._show(cam_L31)
    cam_L31 = cv2.cvtColor(np.asarray(cam_L31),cv2.COLOR_RGB2BGR)
    # cam_L31 = cv2.cvtColor(np.asarray(cam_L31))
    cam_L31 = cv2.applyColorMap(cam_L31, cv2.COLORMAP_JET)
    miki_1=cv2.cvtColor(np.asarray(miki),cv2.COLOR_RGB2BGR)
    # cam_L31 = Image.fromarray(cv2.cvtColor(cam_L31,cv2.COLOR_BGR2RGB))
    # heatmap_L31,miki_1=np.array(heatmap_L31),np.array(miki_1)
    result = cam_L31*0+ miki_1*1
    # result =  miki_1

    # result = (cam_L31/255) *miki_1

    print(result.shape)
    # plt.imsave('1.jpg',heatmap_L31)
    path='./retrivial_visualize/' + 'original_picture.jpg'
    cv2.imwrite(join(os.path.abspath(os.path.dirname(os.getcwd())) , path),result)
    cv2.waitKey(0)










#
# cam_L31 = Image.fromarray(cam_L28).convert('L')
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
# cv2.imwrite(join(os.path.abspath(os.path.dirname(os.getcwd())),'./retrivial_visualize/WACD_L28.jpg'),result)
# cv2.waitKey(0)








