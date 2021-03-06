import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from keras.optimizers import Adam

# 1. 데이터
train_data = pd.read_csv("./dacon2/data/train.csv", index_col=0, header=0)
print(train_data)

'''
# 그림 확인
idx = 999
img = train_data.loc[idx, '0':].values.reshape(28, 28).astype(int)
digit = train_data.loc[idx, 'digit']
letter = train_data.loc[idx, 'letter']
plt.title('Index: %i, Digit: %s, Letter: %s'%(idx, digit, letter))
plt.imshow(img)
plt.show()
'''

train_digit = train_data['digit'].values
train_letter = train_data['letter'].values

x_train = train_data.drop(['digit', 'letter'], axis=1).values
x_train = x_train.reshape(-1, 28, 28, 1)
x_train = x_train/255

y = train_data['digit']
y_train = np.zeros((len(y), len(y.unique())))
for i, digit in enumerate(y):
    y_train[i, digit] = 1

print(x_train.shape, y_train.shape)     # (2048, 28, 28, 1) (2048, 10)

# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, stratify=y_train)

datagen = ImageDataGenerator(
    width_shift_range=(-1,1),  
    height_shift_range=(-1,1))

datagen2 = ImageDataGenerator()

steps = 40
skfold = StratifiedKFold(n_splits=steps, random_state=42, shuffle=True)

# 2. 모델

def cnn_model(x_train):
    inputs = Input(shape=x_train.shape[1:])
    layer = Conv2D(16, 3, padding='same', strides=1, activation='relu')(inputs)
    layer = BatchNormalization()(layer)
    layer = Dropout(0.3)(layer)

    layer = Conv2D(32, 3, padding='same', strides=1, activation='relu')(layer)
    layer = BatchNormalization()(layer)
    layer = Conv2D(32, 5, padding='same', strides=1, activation='relu')(layer)
    layer = BatchNormalization()(layer)    
    layer = Conv2D(32, 5, padding='same', strides=1, activation='relu')(layer)
    layer = BatchNormalization()(layer)
    layer = Conv2D(32, 5, padding='same', strides=1, activation='relu')(layer)
    layer = BatchNormalization()(layer)
    layer = MaxPooling2D(3)(layer)
    layer = Dropout(0.3)(layer)

    layer = Conv2D(64, 3, padding='same', strides=1, activation='relu')(layer)
    layer = BatchNormalization()(layer)
    layer = Conv2D(64, 5, padding='same', strides=1, activation='relu')(layer)
    layer = BatchNormalization()(layer)
    layer = MaxPooling2D(3)(layer)
    layer = Dropout(0.3)(layer)

    layer = Flatten()(layer)

    layer = Dense(128, activation='relu')(layer)
    layer = BatchNormalization()(layer)
    layer = Dropout(0.3)(layer)
    layer = Dense(64, activation='relu')(layer)
    layer = BatchNormalization()(layer)
    layer = Dropout(0.3)(layer)

    outputs = Dense(10, activation='softmax')(layer)

    model = Model(inputs=inputs, outputs=outputs)

    return model

val_acc = []
for i, (train_idx, val_idx) in enumerate(skfold.split(x_train, y_train.argmax(1))):
    x_train_, x_val_ = x_train[train_idx], x_train[val_idx]
    y_train_, y_val_ = y_train[train_idx], y_train[val_idx]
    
    model = cnn_model(x_train)

    filepath = './dacon2/data/vision_model_{}.hdf5'.format(i)
    es = EarlyStopping(monitor='val_loss', patience=160, mode='auto')
    cp = ModelCheckpoint(filepath=filepath, monitor='val_loss', save_best_only=True, mode='auto')
    lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=100)

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.002, epsilon=None), metrics=['accuracy'])
    hist = model.fit_generator(datagen.flow(x_train_, y_train_, batch_size=32), epochs=2000,
               validation_data=(datagen.flow(x_val_, y_val_)), verbose=2, callbacks=[es, cp, lr])

    val_acc.append(max(hist.history['val_accuracy']))

    print('{}\'s CV End'.format(i+1))

# 3. 예측
test_data = pd.read_csv('./dacon2/data/test.csv', index_col=0, header=0)
x_test = test_data.drop(['letter'], axis=1).values
x_test = x_test.reshape(-1, 28, 28, 1)
x_test = x_test/255

# best model select
print(val_acc)

i_max = np.argmax(val_acc)

print('Best Model is {}\'s'.format(i_max))
model = load_model('./dacon2/data/vision_model_{}.hdf5'.format(i_max))

submission = pd.read_csv('./dacon2/data/submission.csv', index_col=0, header=0)

submission['digit'] = np.argmax(model.predict(x_test), axis=1)
print(submission)

submission.to_csv('./dacon2/data/submission_model_best.csv')


# KFold 값 평균내기
submission2 = pd.read_csv('./dacon2/data/submission.csv', index_col=0, header=0)

result = 0
for i in range(steps):
    model = load_model('./dacon2/data/vision_model_{}.hdf5'.format(i))

    result += model.predict_generator(datagen2.flow(x_test, shuffle=False)) / steps

submission2['digit'] = result.argmax(1)

print(submission2)
submission2.to_csv('./dacon2/data/submission_model_mean.csv')
