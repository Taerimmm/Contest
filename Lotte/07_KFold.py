import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Flatten, Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

x_train = np.load('../data/LPD_competition/npy/train_data.npy')
y_train = np.load('../data/LPD_competition/npy/label_data.npy')
print(x_train.shape)
print(y_train.shape)

y_train = to_categorical(y_train)

x_pred = np.load('../data/LPD_competition/npy/test_data.npy')
print(x_pred.shape)

x_train = preprocess_input(x_train)
x_pred = preprocess_input(x_pred)

# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
steps = 5
kfold = KFold(n_splits=steps, shuffle=True, random_state=42)

for i, (train_idx, val_idx) in enumerate(kfold.split(x_train, y_train)):
    x_train_ = x_train[train_idx]
    x_val_ = x_train[val_idx]
    y_train_ = y_train[train_idx]
    y_val_ = y_train[val_idx]

    print(x_train_.shape, x_val_.shape)
    print(y_train_.shape, y_val_.shape)

    train_generator = ImageDataGenerator(width_shift_range=(-1,1), height_shift_range=(-1,1), zoom_range=0.15).flow(x_train_, y_train_, batch_size=16)
    val_generator = ImageDataGenerator().flow(x_val_, y_val_)

    b7 = EfficientNetB7(weights='imagenet', include_top=False, input_shape=(128,128,3))
    # b7.summary()

    # b7.trainable = False

    layer = b7.output
    layer = GlobalAveragePooling2D()(layer)
    output_tensor = Dense(1000, activation='softmax')(layer)

    model = Model(inputs=b7.input, outputs=output_tensor)

    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=0.015, momentum=0.9), metrics=['accuracy'])
    path = './Lotte/b7_model_{}.hdf5'.format(i)
    es = EarlyStopping(monitor='val_accuracy', patience=20)
    cp = ModelCheckpoint(path, monitor='val_accuracy', save_best_only=True)
    lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=10)

    model.fit(train_generator, epochs=200, batch_size=32, validation_data=val_generator, callbacks=[es,cp,lr])


# Predict
test_generator = x_pred

result = 0
for i in range(steps):
    model = load_model('./Lotte/b7_model_{}.hdf5'.format(i))

    result += model.predict(test_generator) / steps


answer = pd.read_csv('./Lotte/sample.csv', header=0)
# print(answer.shape)

answer.iloc[:,1] = np.argmax(result,1)
print(answer)
answer.to_csv('./Lotte/kfold_submission.csv', index=False)

# 82.628
