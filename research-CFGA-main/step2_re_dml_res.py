import os
pwd1 = 'cd src'

# # reimplement_msloss+CFGA@cub
# dataset_name='CUB_200_2011'
# dml_method_tag='msloss-512d'
# model_pth=r'models\cub200\{}\checkpoint_2999_0.2896.pth'.format(dml_method_tag)

# # reimplement_arcface+CFGA@cub
# dataset_name='CUB_200_2011'
# dml_method_tag='arcface-512d'
# model_pth=r'models\cub200\{}\checkpoint_2800_6.9309_arc.pth'.format(dml_method_tag)

# # reimplement_cosface+CFGA@cub
# dataset_name='CUB_200_2011'
# dml_method_tag='cosface-512d'
# model_pth=r'models\cub200\{}\checkpoint_2800_11.913_cos.pth'.format(dml_method_tag)

# pwd2 = 'python ablation_study/cfga_with_dmls.py --model_name {} --dataset_name {}'.format(model_pth,dataset_name)
# os.system("{}&&{}".format(pwd1,pwd2))

################################################################

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

# pwd2 = 'python ablation_study/cfga_with_dmls.py --model_name {} --dataset_name {}'.format(model_pth,dataset_name)
# os.system("{}&&{}".format(pwd1,pwd2))


#################################################################3

# # reimplement_proxy_anchor_loss+CFGA@cub
# dataset_name='CUB_200_2011'
# model_pth=r'models\cub200\proxy-anchor-provided-model\cub_resnet50_best.pth'

# pwd2 = 'python WAOCD/WACD_proxy_anchor_trained.py --model_name {} --dataset_name {}'.format(model_pth,dataset_name)
# os.system("{}&&{}".format(pwd1,pwd2))

# # reimplement_proxy_loss+CFGA@cars196
# dataset_name='cars_196'
# model_pth=r'models\cars196\proxy-anchor-provided-model\cars_resnet50_best.pth'

# pwd2 = 'python WAOCD/WACD_proxy_anchor_trained.py --model_name {} --dataset_name {}'.format(model_pth,dataset_name)
# os.system("{}&&{}".format(pwd1,pwd2))

####################################################################3


# reimplement DGPCRL+CFGA@CUB
dataset_name='CUB_200_2011'
model_pth=r'models\cub200\DGPCRL\checkpoint_100_0.9998.pth'

pwd2 = 'python WAOCD\WACD_DGCRL_batch_2.py --model_name {} --dataset_name {}'.format(model_pth,dataset_name)
os.system("{}&&{}".format(pwd1,pwd2))

# reimplement DGPCRL+CFGA@CARS
dataset_name='cars_196'
model_pth=r'models\cars196\DGPCRL\1\checkpoint_100_0.9934.pth'

pwd2 = 'python WAOCD\WACD_DGCRL_batch_2.py --model_name {} --dataset_name {}'.format(model_pth,dataset_name)
os.system("{}&&{}".format(pwd1,pwd2))