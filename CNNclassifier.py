############ IMPORTING NECESSARY LIBRARIES #########################
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import numpy as np

################### LOADING & PRE-PROCESSING CIFAR10 DATASET #########################

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

##################### DATASET NORMALIZATION ##############
train_images = train_images / 255.0
test_images = test_images / 255.0

############ DEFINING THE CLASSES ###########
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


############## Visualizing Some training images from the dataset ########
plt.figure(figsize=(8,8))
for i in range(10):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(True)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

####################### BUILDING A CUSTOMISED CNN MODEL #############
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)


    
    
    ])
            
    






    