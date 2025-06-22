cd src

dataset_name=CUB_200_2011
model_pth=./models/resnet50-19c8e357.pth
python about_pretraineds/WACD_R50.py --model_name "${model_pth}" --dataset_name ${dataset_name}
