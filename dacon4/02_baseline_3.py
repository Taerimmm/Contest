import numpy as np
import pandas as pd
import math
import os
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Reshape, GRU, RNN

tf.keras.backend.set_floatx('float64')

train=pd.read_csv('./dacon4/data/train.csv', encoding='cp949')
test=pd.read_csv('./dacon4/data/test.csv', encoding='cp949')
submission=pd.read_csv('./dacon4/data/sample_submission.csv', encoding='cp949')

print(train)
print(test)
print(submission)