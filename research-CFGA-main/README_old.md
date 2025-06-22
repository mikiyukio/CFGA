# research-CFGA

## requirements
> python                3.6.5  
> numpy 1.16.3  
> pandas 0.24.2  
> torch 1.2.0 + cuda 9.2  
> torchvision 0.4.0 + cuda 9.2  
> scikit-image 0.17.2  
> scikit-learn 0.23.1  
> scipy 1.3.1  
> Pillow 6.1.0  
> matplotlib 3.0.3  
> opencv-contrib-python 4.1.0.25  
> opencv-python         4.1.0.25  
> NVIDIA Geforce GTX 1060

## 0.
> + Before debugging this work, make sure that you have prepared the dataset as [1](https://github.com/mikiyukio/CFGA_CUB200-2011_train/blob/main/README.md) and [2](https://github.com/mikiyukio/CFGA_CARS196_train/blob/main/README.md)
> + You need to do several slightly adjustments on ***.\SCDA\SCDA_cub_resnet50\files.py***, the detailed operations are very similar with the introduction in [here](https://github.com/mikiyukio/CFGA_CUB200-2011_train/blob/main/README.md)
> + You need to do several slightly adjustments on ***.\SCDA\SCDA_cars_resnet50\files_cars196.py***, the detailed operations are very similar with the introduction in [here](https://github.com/mikiyukio/CFGA_CARS196_train/blob/main/README.md)
> + Because the proposed CFGA method is not an optimization method and it can be combined with certain DML baselines or pertrain models, we provide some trained models to convenient readers (includes some models trained by us ,some [pytorch provided pre-trained models](https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html) and trained-model provided by [proxy-anchor loss](https://github.com/tjddus9597/Proxy-Anchor-CVPR2020)).
> > + CARS196 provided models:https://drive.google.com/drive/folders/17qfulk7GsDxmIb21Df_M-_pvWP1lwLTO
> > + CUB-200-2011 provided models: https://drive.google.com/drive/folders/1XVcAx69vr-irg65fAHJx9mQK89ExBlQL
> > + pytorch pretrained models: https://drive.google.com/drive/folders/1nE37By2q_ABNACaSwdehffgTzd3rL4Fv or https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html
>
> + for all the trained models belongs to CARS196 dataset (and pre-trained models), please save them in ***/SCDA/SCDA_cars_resnet50/models*** 
> + for all the trained models belongs to CUB-200-2011 dataset (and pre-trained models), please save them in ***/SCDA/SCDA_cub_resnet50/models*** 
> + all procedures related to CARS196 dataset is in ***/SCDA/SCDA_cars_resnet50/*** 
> + all procedures related to CUB-200-2011 dataset is in ***/SCDA/SCDA_cub_resnet50/*** 

## 1.
>1.1
>>1.1.1
>>> + ***.\SCDA\SCDA_cub_resnet50\WAOCD\WACD_DGCRL_batch_2.py*** is the test program of **DGPCRL** baseline (related to Tables 1, 2, 3, and 6),  
>>> + And the corresponding dataset is cub200-2011;   
>>> + And the dimension size of DGPCRL embedding is 4096.
>>> + You can use your own trained models, or just download our provided models from [here](https://drive.google.com/drive/folders/1bqRyOl4ohmtBkhwgFZXNLzyZsVU3nnpH)
>>> 
>>> + if you want to run this procedure, change the following code exists in it; 
>>>  `PARA=[
        'checkpoint_100_0.9934.pth',
        'checkpoint_105_0.9960.pth',
        'checkpoint_110_0.9945.pth',
]`  
>>>***'checkpoint_100_0.9934.pth'*** is the name of the model saved in ***/SCDA/SCDA_cub_resnet50/models*** which you want to evaluate.   
>>>And you can evaluate a series models belongs to a certain baseline in a evaluating run.  
>>>After the feature extract procedure(i.e., WACD_DGCRL_batch_2.py) finish, the extracted features will be saved in ***.\SCDA\SCDA_cub_resnet50\datafile***.   
>>>Specifically, for the extracted features correspondings to the model named as ***'checkpoint_100_0.9934.pth'***, the extraced features will be saved in the folder named  ***.\SCDA\SCDA_cub_resnet50\datafile\checkpoint_100_0_9934_pth***.  
>>>The extracted features can be used for image retrieval and the calculation of Recall@K/MAP@k. We will introduce this later.    
>>> + The above detailed statements will only be stated for just once, as the operations on other procedures is similar. 
>>
>>1.1.2
>>> + ***.\SCDA\SCDA_cars_resnet50\WAOCD\WACD_DGCRL_batch_2.py*** is the test program of **DGPCRL** baseline (related to Tables 1, 2, 3, and 6),
>>> + And the corresponding dataset is cars196;   
>>> + And the dimension size of DGPCRL embedding is 4096.
>>> + You can use your own trained models, or just download our provided models from [here](https://drive.google.com/drive/folders/1MVPiA95FcVT4Hn40799w6pYVY0F6yaVd)  
>
>1.2
>>1.2.1
>>> + ***.\SCDA\SCDA_cub_resnet50\WAOCD\WACD_proxy_anchor_trained.py*** is the test program of **Proxy anchor loss** baseline (related to Tables 1, 2, 3),  
>>> + And the corresponding dataset is cub200-2011;
>>> + And the dimension size of proxy anchor loss embedding is 512.
>>
>>1.2.2
>>> + ***.\SCDA\SCDA_cars_resnet50\WAOCD\WACD_proxy_anchor_trained.py*** is the test program of **Proxy anchor loss** baseline (related to Tables 1, 2, 3),
>>> + And the corresponding dataset is cars196; 
>>> + And the dimension size of proxy anchor loss embedding is 512.
>
>1.3
>>1.3.1
>>> + ***./SCDA/SCDA_cub_resnet50/ablation_study/WACD_CFGA_SCDA.py*** is the test program of **MS loss/cosface/arcface** baseline (related to Tables 1, 2, 3, 6),  
>>> + And the corresponding dataset is cub200-2011;
>>> + And the dimension size of **MS loss/cosface/arcface** embedding is 512.
>>
>>1.3.2
>>> + ***/SCDA/SCDA_cars_resnet50/ablation_study/WACD_CFGA_SCDA.py*** is the test program of **MS loss/cosface/arcface** baseline (related to Tables 1, 2, 3, 6),
>>> + And the corresponding dataset is cars196; 
>>>  + And the dimension size of **MS loss/cosface/arcface** embedding is 512.
>
>1.4
>>1.4.1
>>> + ***./SCDA/SCDA_cub_resnet50/WAOCD/WACD_1024.py*** is the test program of **MS loss/cosface/arcface** baseline (related to Tables 4),  
>>> + And the corresponding dataset is cub200-2011;
>>> + And the dimension size of **MS loss/cosface/arcface** embedding is 1024.
>>
>>1.4.2
>>> + ***./SCDA/SCDA_cars_resnet50/WAOCD/WACD_1024.py*** is the test program of **MS loss/cosface/arcface** baseline (related to Tables 4),
>>> + And the corresponding dataset is cars196; 
>>>  + And the dimension size of **MS loss/cosface/arcface** embedding is 1024.
>
>1.5
>>1.5.1
>>> + ***./SCDA/SCDA_cub_resnet50/ablation_study/WACD_direction.py*** corresponds to the ablation study of *table 8* (**MS loss/cosface** parts),  
>>> + And the corresponding dataset is cub200-2011;
>>> + The corresponding equation of this ablation study is Eq.6
>>
>>1.5.2
>>> + ***/SCDA/SCDA_cars_resnet50/ablation_study/WACD_direction.py*** corresponds to the ablation study of *table 8* (**MS loss/cosface** parts),
>>> + And the corresponding dataset is cars196; 
>>> + The corresponding equation of this ablation study is Eq.6
>
>1.6
>>1.6.1
>>> + ***/SCDA/SCDA_cub_resnet50/about_pretraineds/WACD_ablation_eq6_direction_R50.py*** corresponds to the ablation study of *table 8* (**ResNet-50 pre-trained model** part),  
>>> + And the corresponding dataset is cub200-2011;
>>> + The corresponding equation of this ablation study is Eq.6
>>
>>1.6.2
>>> + ***/SCDA/SCDA_cars_resnet50/about_pretraineds/WACD_ablation_eq6_direction_R50.py*** corresponds to the ablation study of *table 8* (**ResNet-50 pre-trained model** part),
>>> + And the corresponding dataset is cars196; 
>>> + The corresponding equation of this ablation study is Eq.6
>
>1.7
>>1.7.1
>>> + ***/SCDA/SCDA_cub_resnet50/about_pretraineds/WACD_R50.py*** and ***/SCDA/SCDA_cub_resnet50/about_pretraineds/WACD_proxy_anchor_trained.py*** are actually equivalent, which all corresponds to the main experiments(not ablation study) on **ResNet-50 pre-trained model** (related with tables 1, 2, 3, 5)
>>> + And the corresponding dataset is cub200-2011;
>>
>>1.7.2
>>> + ***/SCDA/SCDA_cars_resnet50/about_pretraineds/WACD_R50.py*** and ***/SCDA/SCDA_cars_resnet50/about_pretraineds/WACD_proxy_anchor_trained.py*** are actually equivalent, which all  corresponds to the main experiments(not ablation study) on **ResNet-50 pre-trained model**(related with tables 1, 2, 3, 5)
>>> + And the corresponding dataset is cars196; 
>
>1.8
>>1.8.1
>>> + ***/SCDA/SCDA_cub_resnet50/about_pretraineds/WACD_R101.py*** corresponds to the main experiments(not ablation study) on **ResNet-101 pre-trained model** (related with tables 5)  
>>> + And the corresponding dataset is cub200-2011;
>>
>>1.8.2
>>> + ***/SCDA/SCDA_cars_resnet50/about_pretraineds/WACD_R101.py*** corresponds to the main experiments(not ablation study) on **ResNet-101 pre-trained model** (related with tables 5)
>>> + And the corresponding dataset is cars196; 
>
>1.9
>>1.9.1
>>> + ***/SCDA/SCDA_cub_resnet50/about_pretraineds/WACD_R152.py*** corresponds to the main experiments(not ablation study) on **ResNet-152 pre-trained model** (related with tables 5)  
>>> + And the corresponding dataset is cub200-2011;
>>
>>1.9.2
>>> + ***/SCDA/SCDA_cars_resnet50/about_pretraineds/WACD_R152.py*** corresponds to the main experiments(not ablation study) on **ResNet-152 pre-trained model** (related with tables 5)
>>> + And the corresponding dataset is cars196; 
>
>1.10
>>1.10.1
>>> + ***/SCDA/SCDA_cub_resnet50/about_pretraineds/WACD_ablation_eq5_R50.py*** corresponds to the ablation study of *table 7* (**ResNet-50 pre-trained model** part)
>>> + And the corresponding dataset is cub200-2011;
>>> + The corresponding equation of this ablation study is Eq.5
>>
>>1.10.2
>>> + ***/SCDA/SCDA_cars_resnet50/about_pretraineds/WACD_ablation_eq5_R50.py*** corresponds to the ablation study of *table 7* (**ResNet-50 pre-trained model** part)
>>> + And the corresponding dataset is cars196; 
>>> + The corresponding equation of this ablation study is Eq.5
>
>1.11
>>1.11.1
>>> + ***/SCDA/SCDA_cub_resnet50/about_pretraineds/WACD_ablation_eq5_R101.py*** corresponds to the ablation study of *table 7* (**ResNet-101 pre-trained model** part)
>>> + And the corresponding dataset is cub200-2011;
>>> + The corresponding equation of this ablation study is Eq.5
>>
>>1.11.2
>>> + ***/SCDA/SCDA_cars_resnet50/about_pretraineds/WACD_ablation_eq5_R101.py*** corresponds to the ablation study of *table 7* (**ResNet-101 pre-trained model** part)
>>> + And the corresponding dataset is cars196;
>>> + The corresponding equation of this ablation study is Eq.5 

## 2. calaulate Recall@K/MAP@K
(1).
-----------------------
>if you want to compute Recall@K:  
>(quick)    
>     ***\SCDA\SCDA_cub_resnet50\WAOCD\compute_recall_as_ms.py*** is a right implementation.   
>          ***\SCDA\SCDA_cars_resnet50\WAOCD\compute_recall_as_ms.py*** is a right implementation.   

(2).
-----------------------------
>if you want to compute Recall@K/MAP@k:    
>(slow, but the result is saved in ***\SCDA\SCDA_cub_resnet50\result***   folder and ***\SCDA\SCDA_cars_resnet50\result***   folder):  
>     ***\SCDA\SCDA_cub_resnet50\WAOCD\compute_map_test_BATCH.py*** is a right implementation.  
>          ***\SCDA\SCDA_cars_resnet50\WAOCD\compute_map_test_BATCH.py*** is a right implementation.   

## 3. email
hello_yuhan@163.com


-------
