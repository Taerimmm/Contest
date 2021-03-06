import os
import numpy as np
import pandas as pd
import cv2

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import KFold

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# img = cv2.imread('../data/LPD_competition/train/0/0.jpg')
# cv2.imshow('img', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


labels = os.listdir('../data/LPD_competition/train')
print(labels)

for dir in os.scandir('../data/LPD_competition/train'):
    print(dir)
    for file in os.scandir(dir):
        print(file)
    break


# Found 39000 images belonging to 1000 classes.
train_generator = ImageDataGenerator(rescale=1./255, validation_split=0.2).flow_from_directory(
    '../data/LPD_competition/train',
    target_size=(128,128),
    # color_mode='grayscale',
    subset='training'
)
# Found 9000 images belonging to 1000 classes.
val_generator = ImageDataGenerator(rescale=1./255, validation_split=0.2).flow_from_directory(
    '../data/LPD_competition/train',
    target_size=(128,128),
    # color_mode='grayscale',
    subset='validation'
)

print(train_generator)
print(val_generator)

# Found 72000 images belonging to 1 classes.
test_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
    '../data/LPD_competition',
    target_size=(128,128),
    # color_mode='grayscale',
    classes=['test'],
    shuffle=False,
    class_mode=None
)

print(test_generator)

vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(128,128,3))

vgg16.trainable = False

input_tensor = Input(shape=(128,128,3))
layer = vgg16(input_tensor)
layer = Flatten()(layer)
layer = Dense(4096)(layer)
layer = Dense(4096)(layer)
output_tensor = Dense(1000, activation='softmax')(layer)

model = Model(inputs=input_tensor, outputs=output_tensor)

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
path = './Lotte/model.hdf5'
es = EarlyStopping(monitor='val_accuracy', patience=30)
cp = ModelCheckpoint(path, monitor='val_accuracy', save_best_only=True)
lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.8, patience=10)
model.fit(train_generator, epochs=2000, batch_size=32, validation_data=val_generator, callbacks=[es,cp,lr])

pred = model.predict(test_generator)
print(np.argmax(pred,1))

answer = pd.read_csv('./Lotte/sample.csv', header=0)

answer.iloc[:,1] = np.argmax(pred,1)
print(answer)
answer.to_csv('./Lotte/submission.csv', index=False)
