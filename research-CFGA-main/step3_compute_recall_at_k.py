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

# #reimplement_r50+CFGA@cars196
# dataset_name='cars_196'
# model_pth=r'models\resnet50-19c8e357.pth'

# #reimplement_r101+CFGA@cars196
# dataset_name='cars_196'
# model_pth=r'models\resnet101-5d3b4d8f.pth'

#reimplement_r152+CFGA@cars196
# dataset_name='cars_196'
# model_pth=r'models\resnet152-b121ed2d.pth'

###########################################

# #reimplement_msloss+CFGA@cub200
# dataset_name='CUB_200_2011'
# model_pth=r'models\cub200\msloss-512d\checkpoint_2999_0.2896.pth'


# #reimplement_arcface+CFGA@cub200
# dataset_name='CUB_200_2011'
# model_pth=r'models\cub200\arcface-512d\checkpoint_2800_6.9309_arc.pth'


# #reimplement_cosface+CFGA@cub200
# dataset_name='CUB_200_2011'
# model_pth=r'models\cub200\cosface-512d\checkpoint_2800_11.913_cos.pth'


# # reimplement_msloss+CFGA@cars196
# dataset_name='cars_196'
# dml_method_tag='ms-loss-512d'
# model_pth=r'models\cars196\{}\checkpoint_2800_0.5620.pth'.format(dml_method_tag)


# # reimplement_arcface+CFGA@cars196
# dataset_name='cars_196'
# dml_method_tag='arcface-512d'
# model_pth=r'models\cars196\{}\checkpoint_3800_7.2521_arc.pth'.format(dml_method_tag)


# # reimplement_cosface+CFGA@cars196
# dataset_name='cars_196'
# dml_method_tag='cosface-512d'
# model_pth=r'models\cars196\{}\checkpoint_3800_9.8567_cos.pth'.format(dml_method_tag)

# pwd2 = 'python WAOCD/compute_recall_as_ms.py {} {}'.format(model_pth, dataset_name)
# os.system("{}&&{}".format(pwd1,pwd2))

###################################################################

# # reimplement_proxy_anchor_loss+CFGA@cub
# dataset_name='CUB_200_2011'
# model_pth=r'models\cub200\proxy-anchor-provided-model\cub_resnet50_best.pth'

# pwd2 = 'python WAOCD/compute_recall_as_ms.py {} {}'.format(model_pth, dataset_name)
# os.system("{}&&{}".format(pwd1,pwd2))

# # reimplement_proxy_anchor_loss+CFGA@cars196
# dataset_name='cars_196'
# model_pth=r'models\cars196\proxy-anchor-provided-model\cars_resnet50_best.pth'

# pwd2 = 'python WAOCD/compute_recall_as_ms.py {} {}'.format(model_pth, dataset_name)
# os.system("{}&&{}".format(pwd1,pwd2))

######################################################################3

# reimplement_DGPCRL+CFGA@cub
dataset_name='CUB_200_2011'
model_pth=r'models\cub200\DGPCRL\checkpoint_100_0.9998.pth'

pwd2 = 'python WAOCD/compute_recall_as_ms.py {} {}'.format(model_pth, dataset_name)
os.system("{}&&{}".format(pwd1,pwd2))

# reimplement_DGPCRL+CFGA@cars196
dataset_name='cars_196'
model_pth=r'models\cars196\DGPCRL\1\checkpoint_100_0.9934.pth'

pwd2 = 'python WAOCD/compute_recall_as_ms.py {} {}'.format(model_pth, dataset_name)
os.system("{}&&{}".format(pwd1,pwd2))