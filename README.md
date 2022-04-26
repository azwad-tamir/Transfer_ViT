# Transfer_ViT
## Pretraining and finetuning different vision transformer models on the ImageNet and Ham10000 dataset.

###########################################################################################################
This repository constians implementation of 11 image classifier models.<br />
List of implemented models:<br />
1) VGGNet19bn
2) ResNet152
3) DenseNet
4) InceptionV3
5) ViT_base
6) DeepViT_base
7) CaiT_base
8) T2TViT_base
9) ViT_pretrained
10) DeiT_pretrained
11) BeiT_pretrained

###########################################################################################################
## Data Preprocessing:
Carry out the following steps to download and preprocess the dataset:<br />
1. Download the HAM10000 dataset from the following link: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000<br />
2. Extract the zip file and put all put the HAM10000 folder in the parent directory.<br />
3. Copy all the images from the HAM10000_images_part_2 folder and paste them into the HAM10000_images_part_1 folder
4. Run the data_preprocessing.py script

###########################################################################################################
## Implementing models:
To implement each model run the python script with the model name

