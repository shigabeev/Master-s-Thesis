import os
import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np

from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from skimage.color import rgb2gray
from skimage import exposure
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import GlobalAveragePooling2D, Dropout, BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from keras.applications.densenet import DenseNet121
from keras.preprocessing.image import img_to_array
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD, Adam

class ImageToArrayPreprocessor:
    def __init__(self, dataFormat=None):
        # store the image data format
        self.dataFormat = dataFormat
        
    def preprocess(self, image):
        # apply the Keras utility function that correctly rearranges # the dimensions of the image
        return img_to_array(image, data_format=self.dataFormat)

class BWPreprocessor:
    def preprocess(self, image):
        return np.expand_dims(rgb2gray(image), 2)


class SimplePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        # store the target image width, height, and interpolation # method used when resizing
        self.width = width
        self.height = height
        self.inter = inter
        
    def preprocess(self, image):
        # resize the image to a fixed size, ignoring the aspect # ratio
        return cv2.resize(image, (self.width, self.height),
                                interpolation=self.inter)


class PictureControlProcessor:
    def preprocess(self, image):
        image = image.astype("float") / 255.0
        img_adapteq = exposure.equalize_hist(image)
        return img_adapteq

class BGR2RGBProcessor:
    def preprocess(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

class BlurProcessor:
    def preprocess(self, image, ksize=3):
        kernel = np.ones((3,3),np.float32)/(ksize**2)
        image = cv2.filter2D(image,-1,kernel)
        return image

class SimpleDatasetLoader:
    
    def __init__(self, preprocessors=None):
        # store the image preprocessor
        self.preprocessors = preprocessors

        # if the preprocessors are None, initialize them as an
        # empty list
        if self.preprocessors is None:
            self.preprocessors = []
            
    def load(self, imagePaths, verbose=-1):
        # initialize the list of features and labels
        data = []
        labels = []

        # loop over the input images
        for (i, imagePath) in enumerate(imagePaths):
            # load the image and extract the class label assuming
            # that our path has the following format:
            # /path/to/dataset/{class}/{image}.jpg
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]
            if self.preprocessors is not None:
                for p in self.preprocessors:
                    image = p.preprocess(image)
            
            # treat our processed image as a "feature vector" # by updating the data list followed by the labels 
            data.append(image)
            labels.append(label)
            # show an update every ‘verbose‘ images
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0: 
                print("[INFO] processed {}/{}".format(i + 1, len(imagePaths)))
            # return a tuple of the data and labels
        return (np.array(data), np.array(labels))


class ShallowNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model along with the input shape to be
        # "channels last"
        model = Sequential()
        inputShape = (height, width, depth)
        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
        
        # define the first (and only) CONV => RELU layer
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        # softmax classifier
        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model

class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model along with the input shape to be
        # "channels last"
        model = Sequential()
        inputShape = (height, width, depth)
        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
        
        # define the first (and only) CONV => RELU layer
        model.add(Conv2D(16, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D())
        model.add(Conv2D(32, (5, 5), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D())
        model.add(Flatten())
        model.add(Dense(240, activation='relu'))
        model.add(Dropout(rate = .2))
        model.add(BatchNormalization())
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        return model

class MobileNet:
    @staticmethod
    def build(width, height, depth, classes, trainable_conv=False):
        base_model = MobileNetV2(include_top=False, input_shape=(width, height, 3))
        for layer in base_model.layers:
            layer.trainable=trainable_conv
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(rate = .2)(x)
        x = BatchNormalization()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(rate = .2)(x)
        x = BatchNormalization()(x)
        predictions = Dense(classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        return model

def balance_classes(data, labels):
    class_names, counts = np.unique(labels, return_counts=True)
    amount = counts.min()
    new_data = []
    new_labels = []
    for c in class_names:
        new_data.append(data[labels == c][:amount])
        new_labels.append(labels[labels == c][:amount])
    new_labels = np.array(new_labels).reshape(-1)
    new_data = np.vstack(np.array(new_data))
    return new_data, new_labels

args = {'dataset':'data/Labrador_crop/'}
imagePaths = list(paths.list_images(args["dataset"]))
seed = 42
model_class = 'MobileNet'
imsize = 32
sp = SimplePreprocessor(imsize, imsize)
iap = ImageToArrayPreprocessor()
bw = BWPreprocessor()
pc = PictureControlProcessor()
bgr = BGR2RGBProcessor()
blur = BlurProcessor()

preprocessors = [sp, iap, bgr]
sdl = SimpleDatasetLoader(preprocessors=preprocessors)
(data, labels) = sdl.load(imagePaths, verbose=500)
data, labels = balance_classes(data, labels)
class_names = np.unique(labels)
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=100, random_state=seed)
lb = LabelBinarizer().fit(labels)
trainY = lb.transform(trainY)
testY = lb.transform(testY)

callbacks = [#EarlyStopping(monitor='val_loss', patience=20),
         ModelCheckpoint(filepath='best_model.h5', monitor='val_acc', save_best_only=True)]


print("[INFO] compiling model...", end='\r')
if model_class == "ShallowNet":
    epochs = 200
    opt = Adam(lr=0.03, decay=1e-3/epochs)
    model = ShallowNet.build(width=imsize, height=imsize, depth=trainX[0].shape[2], classes=3)
elif model_class == "LeNet":
    epochs = 200
    opt = Adam(lr=0.03, decay=1e-3/epochs)
    model = LeNet.build(width=imsize, height=imsize, depth=trainX[0].shape[2], classes=3)
elif model_class == "MobileNet":
    epochs = 20
    opt = Adam(lr=0.03, decay=1e-3/epochs)
    model = ShallowNet.build(width=imsize, height=imsize, depth=trainX[0].shape[2], classes=trainY[0].shape[0])


print("[INFO] compiling model... Done")
model.compile(loss="categorical_crossentropy" , optimizer= opt, metrics=['acc'])
print("[INFO] training network...")

np.random.seed(seed)
H = model.fit(trainX, trainY, 
              validation_data=(testX, testY), 
              batch_size=32, 
              epochs=epochs, 
              verbose=1,
              callbacks=callbacks
             )

history = H
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Точность модели')
plt.ylabel('Точность')
plt.xlabel('Эпоха')
plt.legend(['Обучающая выборка', 'Тестовая выборка'], loc='lower right')
plt.savefig('accuracy_over_epochs', dpi=300)
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Значение функции потерь')
plt.ylabel('Ошибка')
plt.xlabel('Эпоха')
plt.legend(['Обучающая выборка', 'Тестовая выборка'], loc='upper right')
plt.savefig('loss_over_epochs', dpi=300)
plt.show()

model.load_weights('best_model.h5')

predictions = model.predict(testX, batch_size=32) 
print(classification_report(testY.argmax(axis=1),
    predictions.argmax(axis=1), target_names=class_names))

confusion_matrix(testY.argmax(axis=1),
    predictions.argmax(axis=1), normalize='true')

print("[INFO] Complete! Weights are located in file best_model.h5")

