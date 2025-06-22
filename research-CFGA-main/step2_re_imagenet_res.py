import os

pwd1 = 'cd src'

#reimplement_r50+CFGA@cub
# dataset_name='CUB_200_2011'
# model_pth=r'models\resnet50-19c8e357.pth'

#reimplement_r101+CFGA@cub
# dataset_name='CUB_200_2011'
# model_pth=r'models\resnet101-5d3b4d8f.pth'


#reimplement_r152+CFGA@cub
# dataset_name='CUB_200_2011'
# model_pth=r'models\resnet152-b121ed2d.pth'

#reimplement_r50+CFGA@cars196
# dataset_name='cars_196'
# model_pth=r'models\resnet50-19c8e357.pth'


# # #reimplement_r101+CFGA@cars196
# dataset_name='cars_196'
# model_pth=r'models\resnet101-5d3b4d8f.pth'


# #reimplement_r50+CFGA@cars196
dataset_name='cars_196'
model_pth=r'models\resnet152-b121ed2d.pth'

pwd2 = 'python about_pretraineds/WACD_Resnet50_101_152.py --model_name {} --dataset_name {}'.format(model_pth,dataset_name)
os.system("{}&&{}".format(pwd1,pwd2))