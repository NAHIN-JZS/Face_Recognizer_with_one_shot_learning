import os
import cv2
import numpy as np
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, MaxPool2D, Conv2D, Flatten
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Get the data from the directory
img_width, img_height = 224, 224
labels = ["Not_Human", "Human"]
def get_data(data_dir):
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img))[...,::-1]
                resized_arr = cv2.resize(img_arr, (img_width, img_height))
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data)

train_data = get_data("D:\\Job Assignment\\human detection dataset\\train")
val_data = get_data("D:\\Job Assignment\\human detection dataset\\validation")

# print(*train_data)
# print(*train_data[1][0])
# print(labels[train_data[0][1]])

# plt.figure(figsize=(5,5))
# plt.imshow(val_data[1][0])
# plt.title(labels[val_data[0][1]])

# Separate Train and Validation Data

x_train = []
y_train = []
x_val = []
y_val = []
for feature, label in train_data:
    x_train.append(feature)
    y_train.append(label)
for feature, label in val_data:
    x_val.append(feature)
    y_val.append(label)

# Normalize and reshape the data
x_train = np.array(x_train)/255
x_train.reshape(-1, img_width, img_height, 1)
x_val = np.array(x_val)/255
x_val.reshape(-1, img_width, img_height, 1)

y_train = np.array(y_train)
y_val = np.array(y_val)

# Building model structure
input_shape = 0
if keras.backend.image_data_format() == "channels_first":
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
model = Sequential()
#Input Layer
model.add(Conv2D(32, (2,2), activation="relu", input_shape=input_shape))
model.add(MaxPool2D(pool_size=(2,2)))
#Hidden Layer 1
model.add(Conv2D(32, (2,2), activation="relu", input_shape=input_shape))
model.add(MaxPool2D(pool_size=(2,2)))
#Hidden Layer 2
model.add(Conv2D(64, (2,2), activation="relu", input_shape=input_shape))
model.add(MaxPool2D(pool_size=(2,2)))
#Hidden Layer 3
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
#Output Layer
model.add(Dense(2, activation="softmax"))

model.summary()

#Using Datageneratot to Augment the data
datagen = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(x_train)

#Setting up model
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=keras.optimizers.Adam(lr=0.000001),
              metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=500, validation_data=(x_val, y_val))
model.save_weights("Human_model_weights.h5")

#Showing results
predictions = model.predict_classes(x_val)
predictions = predictions.reshape(1,-1)[0]
print(classification_report(y_val, predictions, target_names=["Not_Human (0)", "Human (1)"]))

