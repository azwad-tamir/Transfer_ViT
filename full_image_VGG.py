# Importing Required libraries
from __future__ import print_function
from __future__ import division
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.backends import cudnn
import os
from PIL import Image
import glob
import numpy as np
from PIL import Image
import random as rn
print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

########################################################################################################################
## Input parameters:
########################################################################################################################
# Top level data directory. Here we assume the format of the directory conforms
# to the ImageFolder structure
# data_dir = "./resized_images_formatted_final"
data_dir = "./HAM10000"

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "vgg"

# Number of classes in the dataset
num_classes = 7

# Batch size for training (change depending on how much memory you have)
batch_size = 16

# Number of epochs to train for
num_epochs = 30

# Setting the learning rate for the SGD optimizer
learning_rate = 0.001

# path to store the models
result_path = './logs'

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = False

########################################################################################################################
## Formatting Dataset:
# Comment this part after the images have been saved in the desired format...
########################################################################################################################
#
# # Importing metadata:
# total_df = pd.read_csv('./HAM10000/HAM10000_metadata.csv')
#
# # # Get the filenames of the images
# # filelist_img_1 = glob.glob('./HAM10000/HAM10000_images_part_1/*.jpg')
# # filelist_img_2 = glob.glob('./HAM10000/HAM10000_images_part_2/*.jpg')
# # filelist_img = filelist_img_1 + filelist_img_2
# #
# # # Resizing image:
# # newsize = (224, 224)
# # for img in filelist_img:
# #     im = Image.open(img)
# #     im1 = im.resize(newsize)
# #     head, tail = os.path.split(img)
# #     im1.save("./resized_images_224/" + tail)
#
# filelist_resized = glob.glob('./resized_images/*.jpg')
#
# # Generating labels:
# y_all = []
# for name in filelist_resized:
#     head, tail = os.path.split(name)
#     tail = tail.replace('.jpg', '')
#     row = ( total_df.loc [total_df['image_id'] == tail]['dx'] ).values.astype(str)
#     y_all.append(str(row[0]))
#
# # Encodeing lables:
# le = LabelEncoder()
# y_all1 = le.fit_transform(y_all)
#
# # Bootstrapping:
# y_values = np.zeros((7))
# for i in range(0,7):
#     y_values[i] = np.count_nonzero(y_all1==i)
#
# # Loading images into arrays and resizing:
# X_all = [Image.open(fname) for fname in filelist_resized]
#
# # Generating Test Train split:
# X_train_array_raw, X_test_array, y_train_array_raw, y_test_array = train_test_split(X_all, y_all1, random_state=1, test_size=0.2)
#
# # Bootstrapping:
# y_values = np.zeros((7))
# for i in range(0,7):
#     y_values[i] = np.count_nonzero(y_train_array_raw==i)
#
# X_all_0 = []
# X_all_1 = []
# X_all_2 = []
# X_all_3 = []
# X_all_4 = []
# X_all_5 = []
# X_all_6 = []
#
# for i in range(0,len(y_train_array_raw)):
#     temp = [X_train_array_raw[i], y_train_array_raw[i]]
#     if y_train_array_raw[i]==0:
#         X_all_0.append(temp)
#     elif y_train_array_raw[i] == 1:
#         X_all_1.append(temp)
#     elif y_train_array_raw[i] == 2:
#         X_all_2.append(temp)
#     elif y_train_array_raw[i] == 3:
#         X_all_3.append(temp)
#     elif y_train_array_raw[i] == 4:
#         X_all_4.append(temp)
#     elif y_train_array_raw[i] == 5:
#         X_all_5.append(temp)
#     elif y_train_array_raw[i] == 6:
#         X_all_6.append(temp)
#     else:
#         print("Fetal error: 15!!!")
#
# X_all_boot = []
# X_all_boot = rn.sample(X_all_5, 3000)
# X_all_boot.extend(rn.choices(X_all_0, k=3000))
# X_all_boot.extend(rn.choices(X_all_1, k=3000))
# X_all_boot.extend(rn.choices(X_all_2, k=3000))
# X_all_boot.extend(rn.choices(X_all_3, k=3000))
# X_all_boot.extend(rn.choices(X_all_4, k=3000))
# X_all_boot.extend(rn.choices(X_all_6, k=3000))
#
# # Randomizing datapoints:
# rn.shuffle(X_all_boot)
#
# X_all_boot_formatted = []
# y_all_boot_formatted = []
#
# for i in range(0,len(X_all_boot)):
#     X_all_boot_formatted.append(X_all_boot[i][0])
#     y_all_boot_formatted.append(X_all_boot[i][1])
#
# X_train_array = X_all_boot_formatted
# y_train_array = y_all_boot_formatted
#
# ## Saving images in structured folders:
# # Creating folders:
# for i in range(0,7):
#     os.mkdir('./resized_images_formatted_450/train/'+str(i))
#
# for i in range(0, 7):
#     os.mkdir('./resized_images_formatted_450/val/' + str(i))
#
# # Saving training images:
# i=0
# for label in y_train_array:
#     img = X_train_array[i]
#     add = "./resized_images_formatted_450/train/" + str(label) + "/" + str(i) + ".jpg"
#     img.save(add, 'JPEG')
#     i+=1
#
# # Saving validation images:
# i=0
# for label in y_test_array:
#     img = X_test_array[i]
#     add = "./resized_images_formatted_450/val/" + str(label) + "/" + "v" + str(i) + ".jpg"
#     img.save(add, 'JPEG')
#     i+=1

########################################################################################################################
## Creating machine learning models:
########################################################################################################################

## This function takes the data and model architecture as the input and returns the best trained model:
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

## This function sets require.grad options of parameters to false for feature extraction option
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

## This function makes the model
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet152(pretrained=use_pretrained)
        #model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG19_bn
        """
        model_ft = models.vgg19_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

# Initialize the model for this run
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

# Print the model we just instantiated
print(model_ft)


# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("Initializing Datasets and Dataloaders...")

# Create training and validation datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Send the model to GPU
model_ft = model_ft.to(device)

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr=learning_rate, momentum=0.9)

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

# Train and evaluate
model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))

# Save the model that achieved the best validation accuracy:
save_file_path = "./Models/VGG19bn_224_new"

states = {
    'arch': "VGG19_bn",
    'state_dict': model_ft.state_dict(),
    'optimizer': optimizer_ft.state_dict(),
}

torch.save(states, save_file_path)