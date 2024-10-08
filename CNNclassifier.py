############ IMPORTING NECESSARY LIBRARIES #########################
from tabnanny import verbose
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import numpy as np
from kerastuner.tuners import RandomSearch
from tabnanny import verbose
from tensorflow.keras.utils import to_categorical



################### LOADING & PRE-PROCESSING CIFAR10 DATASET #########################

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

##################### DATASET NORMALIZATION ##############
train_images = train_images / 255.0
test_images = test_images / 255.0
################ONE HOT ENCODING##########
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

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
    plt.xlabel(class_names[np.argmax(train_labels[i])])
plt.show()

####################### BUILDING A CUSTOMISED CNN MODEL #############
def model_build(hp):
    model = models.Sequential([
        layers.Conv2D(hp.Int('conv1_units', min_value=32, max_value=128, step=16), (3,3), activation='relu', input_shape=(32,32,3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(hp.Int('conv2_units', min_value=32, max_value=128, step=16), (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(hp.Int('conv3_units', min_value=32, max_value=128, step=16), (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(hp.Int('dense_units', min_value=32, max_value=128, step=16), activation='relu'),
        layers.Dropout(hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

#############DEFINING THE TUNER ###############
tuner = RandomSearch(
    model_build,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=1,
    directory='my_dir',
    project_name='cifar10_tunning'
    
    )
tuner.search(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

##############TRAINING THE MODEL###############
model = tuner.hypermodel.build(best_hps)
history = model.fit(train_images, train_labels, epochs = 15, validation_data = (test_images, test_labels))

##################EVALUATING THE MODEL###################
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose =2)
print(f'\n Test accuracy is: {test_acc}')

##################### PLOTTING THE ACCURACY AND VALIDATION VALUES #################

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim([0,1])
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.grid(True)
plt.legend()


plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.ylim([0,1])
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.grid(True)
plt.legend()


plt.tight_layout()
plt.show()

    