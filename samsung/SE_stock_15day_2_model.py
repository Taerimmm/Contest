import numpy as np
import pandas as pd

stock = pd.read_csv('./samsung/삼성전자.csv', index_col=0, header=0, encoding='cp949')
stock1 = pd.read_csv('./samsung/삼성전자2.csv', index_col=0, header=0, encoding='cp949')

stock.drop("2021-01-13", axis=0, inplace=True)
stock1 = stock1.iloc[:2,:]
stock = pd.concat([stock1, stock], join='inner')

print(stock)
print(stock.shape)      # (2401, 14)

stock.replace(',', '', inplace=True, regex=True)
stock = stock.astype('float32')

stock = stock.iloc[::-1] 
print(stock.shape)      # (2401, 14)
print(stock.iloc[-666:-663,:]) # <---- 제거
stock.dropna(inplace=True)
print(stock.shape)          # (2398, 14)
print(stock.iloc[-666:-661,:])

print(stock.iloc[:-663,:])

stock.iloc[:-663,0:4] = stock.iloc[:-663,0:4]/50.       # 시가, 고가, 저가, 종가 1/50 배 (액면분할)
stock.iloc[:-663,5] = stock.iloc[:-663,5]*50.           # 거래량 50배

print(stock.iloc[-666:-660,:])

print(stock.shape)      # (2398, 14)

print(stock.columns)    # ['시가', '고가', '저가', '종가', '등락률', '거래량', '금액(백만)', '신용비', '개인', '기관','외인(수량)', '외국계', '프로그램', '외인비']

'''
# 상관관계
print(stock.corr())

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.2)
sns.heatmap(data=stock.corr(), square=True, annot=True, cbar=True)
plt.show()
'''

# X, Y 나누기
x_data = stock.iloc[:,[0,1,2,3,4,8,9]]
y_data = stock['종가']

np.save('./samsung/etc/15day_data_2.npy', arr=x_data)

print(x_data.columns)         # ['시가', '고가', '저가', '종가', '등락률', '개인', '기관']

print(x_data.shape)         # (2398, 7)
print(y_data.shape)         # (2398,)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_data)
x_data = scaler.transform(x_data)

def split_x(data,size):
    a = []
    for i in range(data.shape[0] - size + 1):
        a.append(np.array(data.iloc[i:(i+size), 0:len(data.columns)]))
    return  np.array(a)

size = 10
x_data = split_x(pd.DataFrame(x_data),size)

y_target = y_data[size:]

print(x_data[:-1].shape)        # (2378, 20, 7)
print(y_target.shape)           # (2378,)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data[:-1], y_target, test_size=0.2)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

x_train = x_train.reshape(x_train.shape[0], x_data.shape[1], x_data.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_data.shape[1], x_data.shape[2])
x_val = x_val.reshape(x_val.shape[0], x_data.shape[1], x_data.shape[2])

print(x_train.shape)
print(x_test.shape)
print(x_val.shape)

print(x_train[-1])
print(x_test[0])

np.save('./samsung/etc/15day_x_train_2.npy', arr=x_train)
np.save('./samsung/etc/15day_x_test_2.npy', arr=x_test)
np.save('./samsung/etc/15day_x_val_2.npy', arr=x_val)
np.save('./samsung/etc/15day_y_train_2.npy', arr=y_train)
np.save('./samsung/etc/15day_y_test_2.npy', arr=y_test)

# 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout

model = Sequential()
model.add(Conv1D(256, 2, padding='same', activation='relu', input_shape=(x_train.shape[1],x_train.shape[2])))
model.add(MaxPooling1D(2))
model.add(Dropout(0.2))
model.add(Conv1D(64, 2, padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv1D(64, 2, padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath = './samsung/etc/15day_model_checkpoint_2.hdf5'
es = EarlyStopping(monitor='val_loss', patience=200, mode='auto')
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
model.fit(x_train, y_train, epochs=10000, batch_size=4, validation_data=(x_val, y_val), verbose=2, callbacks=[es,cp])

model.save('./samsung/etc/15day_stock_model_2.h5')

loss = model.evaluate(x_test, y_test, batch_size=8)
print('loss :', loss)

y_pred = model.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score
def rmse(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))

print('RMSE :', rmse(y_test, y_pred))
print('MSE :', mean_squared_error(y_test, y_pred))

print('R2 :', r2_score(y_test, y_pred))
print('\n', '===========================================','\n')

# Result
# loss : 1539390.125
# RMSE : 1240.7216
# MSE : 1539390.1
# R2 : 0.9913434158903249

# 2021-01-15 예측
last = x_data[-1].reshape(1,x_train.shape[1],x_train.shape[2])
a = model.predict(last)
print('\'2021-01-15\'의 종가는', a, '로 예측 됩니다.')

# 예측값
# 86592.586
