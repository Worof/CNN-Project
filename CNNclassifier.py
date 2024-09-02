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
class_names = ['cat', 'dog', 'butterfly', 'balloon', 'car', 'tiger', 'lamp', 'rose']




