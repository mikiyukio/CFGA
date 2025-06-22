import os

#CUB-200-2011
# pwd1 = 'cd src'
# dataset_pth=r"C:\Users\于涵\Desktop\Caltech-UCSD Birds-200 2011\Caltech-UCSD Birds-200-2011\CUB_200_2011"
# dataset_name='CUB_200_2011'
# pwd2 = 'python files.py --datasetdir "{}" --dataset_name {}'.format(dataset_pth, dataset_name)
# os.system("{}&&{}".format(pwd1,pwd2))


#cars196
pwd1 = 'cd src'
#原始图像路径
dataset_pth=r'C:\Users\于涵\Desktop\Stanford car dataset\car_ims'
#原始标注路径
annotationsdir=r'C:\Users\于涵\Desktop\Stanford car dataset\car_devkit\devkit'
#数据根据标注进行重构，图像按文件夹存储，重构数据的保存路径
targetdatasetdir = r'E:\dataset\cars196_rearrange_250617'
dataset_name='cars_196'
pwd2 = 'python files_cars196.py --datasetdir "{}" --annotationsdir "{}" --targetdatasetdir "{}" --dataset_name {}'.format(dataset_pth, annotationsdir, targetdatasetdir, dataset_name)
os.system("{}&&{}".format(pwd1,pwd2))