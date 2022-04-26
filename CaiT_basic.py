from __future__ import print_function
from PIL import Image
from vit_pytorch import ViT
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
import glob
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import os
import pickle
import collections
import torch
import seaborn as sn
import matplotlib.pyplot as plt
from vit_pytorch.deepvit import DeepViT
from vit_pytorch.cait import CaiT
from sklearn.metrics import confusion_matrix, mean_squared_error, classification_report
import numpy as np

# Setting CUDA Device:
torch.cuda.set_device(0)
# CUDA_VISIBLE_DEVICES= [1,2]

# Hyperparameters:
num_epochs = 50
# batch_size_list = [32, 64, 128]
batch_size = 64
# lr_list = [3e-5, 2e-4, 1e-3, 5e-6]
lr_list = [3e-5]
# gamma_list = [0.9, 0.7]  # for learning rate scheduler
gamma_list = [0.7]


# Importing metadata:
total_df = pd.read_csv('./HAM10000/HAM10000_metadata.csv')
file_list = glob.glob(r'./HAM10000/HAM10000_images_part_1/*.jpg')

# Generating labels:
y_all = []
for name in file_list:
    head, tail = os.path.split(name)
    tail = tail.replace('.jpg', '')
    row = ( total_df.loc [total_df['image_id'] == tail]['dx'] ).values.astype(str)
    y_all.append(str(row[0]))

# Encodeing lables:
le = preprocessing.LabelEncoder()
y_all1 = le.fit_transform(y_all)

# Applying test train split
X_train, X_temp, y_train, y_temp = train_test_split(file_list, y_all1, random_state=1, stratify=y_all1, test_size=0.15)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, random_state=1, stratify=y_temp, test_size=0.5)

# Creating Dataset:
data_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
    ]
)

class MyDataset(Dataset):
    def __init__(self, file_list, y_all, transform=None):
        self.file_list = file_list
        self.transform = transform
        self.y_all = y_all


    def __getitem__(self, index1):
        img_path = self.file_list[index1]
        img = Image.open(img_path)
        img_transformed = self.transform(img)
        label = self.y_all[index1]
        return img_transformed, label


train_data = MyDataset(X_train, y_train, transform=data_transforms)
valid_data = MyDataset(X_val, y_val, transform=data_transforms)
test_data = MyDataset(X_test, y_test, transform=data_transforms)

# Fixing Dataloader:
train_loader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True )
valid_loader = DataLoader(dataset = valid_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset = test_data, batch_size=batch_size, shuffle=True)


# Model Specifications
model = CaiT(
    image_size = 256,
    patch_size = 32,
    num_classes = 7,
    dim = 1024,
    depth = 12,             # depth of transformer for patch to patch attention only
    cls_depth = 2,          # depth of cross attention of CLS tokens to patch
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1,
    layer_dropout = 0.05    # randomly dropout 5% of the layers
).to(torch.device("cuda"))

#start training
train_loss_list_list = []
val_loss_list_list = []
train_acc_list_list = []
val_acc_list_list = []

for lr in lr_list:
    for gamma in gamma_list:
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

        train_loss_list = []
        val_loss_list = []
        train_acc_list = []
        val_acc_list = []
        print("lr:", lr, "    gamma:", gamma)
        for epoch in range(num_epochs):
            epoch_loss = 0
            epoch_accuracy = 0
            for data, label in train_loader:
                data = data.to(torch.device("cuda"))
                label = label.to(torch.device("cuda"))
                output = model(data)
                loss = criterion(output, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                accuracy = (output.argmax(dim=1) == label).float().mean()
                epoch_accuracy += accuracy / len(train_loader)
                epoch_loss += loss / len(train_loader)
            with torch.no_grad():
                epoch_val_accuracy = 0
                epoch_val_loss = 0
                for data, label in valid_loader:
                    data = data.to(torch.device("cuda"))
                    label = label.to(torch.device("cuda"))
                    val_output = model(data)
                    val_loss = criterion(val_output, label)
                    acc = (val_output.argmax(dim=1) == label).float().mean()
                    epoch_val_accuracy += acc / len(valid_loader)
                    epoch_val_loss += val_loss / len(valid_loader)
            train_loss_list.append(epoch_loss.detach().cpu().numpy().flatten()[0])
            train_acc_list.append(epoch_accuracy.detach().cpu().numpy().flatten()[0])
            val_loss_list.append(epoch_val_loss.detach().cpu().numpy().flatten()[0])
            val_acc_list.append(epoch_val_accuracy.detach().cpu().numpy().flatten()[0])
            print(
                f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
            )
        val_acc_list_list.append(val_acc_list)
        val_loss_list_list.append(val_loss_list)
        train_acc_list_list.append(train_acc_list)
        train_loss_list_list.append(train_loss_list)

torch.save(model.state_dict(), './final_models_simple/CaiT_model.pth')
metrices = [train_loss_list, val_loss_list, train_acc_list, val_acc_list]
with open("./final_models_simple/CaiT_metrices.pk", "wb") as fp:   #Pickling
    pickle.dump(metrices, fp)
#
# a = []
# for i in range(0, num_epochs):
#     a.append(val_acc_list[i].detach().cpu().numpy().flatten()[0])
# maxi = []
# for i in range(0, 8):
#     maxi.append(max(val_acc_list_list[i]))

# loading model:
# model.load_state_dict(torch.load("./final_models_simple/CaiT_model.pth"))
model.eval()

#loading dataset_locker
# with open('./final_models_simple/dataset_locker.pk', 'rb') as f:
#     x = pickle.load(f)
# X_train = x[0]
# y_train = x[1]
# X_val = x[2]
# y_val = x[3]
# X_test = x[4]
# y_test = x[5]

# Creating Validation predictions and metrics:
pred = []
pred_label = []
i=0
criterion = nn.CrossEntropyLoss()
for i in range(0,len(X_test)):
    test_data = MyDataset([X_test[i]], [y_test[i]], transform=transforms)
    test_loader = DataLoader(dataset = test_data, batch_size=1, shuffle=False)
    with torch.no_grad():
        epoch_val_accuracy = 0
        epoch_val_loss = 0

        for data, label in test_loader:
            if i%100==0:
                print("step: ", i)
            i +=1
            data = data.to(torch.device("cuda"))
            label = label.to(torch.device("cuda"))
            val_output = model(data)
            val_loss = criterion(val_output, label)
            pred.append(list(val_output.argmax(dim=1).detach().cpu().numpy()))
            pred_label.append(list(label.detach().cpu().numpy()))
            acc = (val_output.argmax(dim=1) == label).float().mean()
            epoch_val_accuracy += acc / len(valid_loader)
            epoch_val_loss += val_loss / len(valid_loader)

val_pred_flat = [item for sublist in pred for item in sublist]
val_pred_label_flat = [item for sublist in pred_label for item in sublist]

# Creating test set predictions and mertrices:
pred = []
pred_label = []
i=0
criterion = nn.CrossEntropyLoss()
for i in range(0,len(X_test)):
    test_data = MyDataset([X_test[i]], [y_test[i]], transform=transforms)
    test_loader = DataLoader(dataset = test_data, batch_size=1, shuffle=False)
    with torch.no_grad():
        epoch_val_accuracy = 0
        epoch_val_loss = 0

        for data, label in test_loader:
            if i%100==0:
                print("step: ", i)
            i +=1
            data = data.to(torch.device("cuda"))
            label = label.to(torch.device("cuda"))
            val_output = model(data)
            val_loss = criterion(val_output, label)
            pred.append(list(val_output.argmax(dim=1).detach().cpu().numpy()))
            pred_label.append(list(label.detach().cpu().numpy()))
            acc = (val_output.argmax(dim=1) == label).float().mean()
            epoch_val_accuracy += acc / len(valid_loader)
            epoch_val_loss += val_loss / len(valid_loader)

test_pred_flat = [item for sublist in pred for item in sublist]
test_pred_label_flat = [item for sublist in pred_label for item in sublist]

predictions = [val_pred_flat, val_pred_label_flat, test_pred_flat, test_pred_label_flat]
with open("./final_models_simple/CaiT_predictions.pk", "wb") as fp:   #Pickling
    pickle.dump(predictions, fp)


# sum = 0
# for i in range(0, len(pred_flat)):
#     if pred_flat[i] == pred_label_flat[i]:
#         sum+=1

# with open('./final_models_simple/CaiT_matrices.pk', 'rb') as f:
#     metrices = pickle.load(f)

# Plotting training and val accuracy with training steps:
train_acc_list = metrices[2]
val_acc_list = metrices[3]

plt.plot(range(0,len(val_acc_list)), val_acc_list, color='b', label='Validation accuracy')
plt.plot(range(0,len(train_acc_list)), train_acc_list, color='r', label='Training accuracy')
plt.title("Training and Validation accuracy")
plt.xlabel("Steps:")
plt.ylabel("Accuracy:")


# Precision and Recall:
# with open('./final_models_simple/CaiT_predictions.pk', 'rb') as f:
#     predictions = pickle.load(f)

val_pred_flat = predictions[0]
val_pred_label_flat = predictions[1]
test_pred_flat = predictions[2]
test_pred_label_flat = predictions[3]
print("\n\nCaiT MODEL:  ")
print(classification_report(test_pred_flat, test_pred_label_flat,digits=4))

# Confusion Matrix:
y_true = test_pred_label_flat
y_pred = test_pred_flat
data = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(data, columns=np.unique(y_true), index = np.unique(y_true))
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16}, fmt='d')# font size
plt.title("Confusion matrix for CaiT model")
plt.show()