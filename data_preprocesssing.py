from PIL import Image
import numpy as np
import glob
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import os

file_list1 = glob.glob(r'./HAM10000/HAM10000_images_part_1/*.jpg')
file_list2 = glob.glob(r'./HAM10000/HAM10000_images_part_2/*.jpg')
metadata_df = pd.read_csv('./HAM10000/HAM10000_metadata.csv')
dx = metadata_df['dx'].to_list()
dx_type = metadata_df['dx_type'].to_list()
image_id = metadata_df['image_id'].to_list()
le = preprocessing.LabelEncoder()
dx_labels = list(le.fit_transform(dx))

X_train, X_temp, y_train, y_temp = train_test_split(image_id, dx_labels, stratify=dx_labels, test_size=0.15)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, stratify=y_temp, test_size=0.5)

sample_num = 0
splits = ['train', 'test', 'val']
X_data = [X_train, X_test, X_val]
y_data = [y_train, y_test, y_val]
for j in range(0, len(splits)):
    for i in range(0, len(X_data[j])):
        destination = './HAM10000/' + splits[j] + '/' + str(y_data[j][i]) + '/' + X_data[j][i] + '.jpg'
        source = './HAM10000/HAM10000_images_part_1/' + X_data[j][i] + '.jpg'
        os.system("cp " + " " + source + " " + destination)
        sample_num+=1





# dx_types_temp = []
# for label in dx_type:
#     if label not in dx_types_temp:
#         dx_types_temp.append(label)
#
#
# dx_types = []
# for label in dx:
#     if label not in dx_types:
#         dx_types.append(label)

