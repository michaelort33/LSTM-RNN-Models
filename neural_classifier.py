from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd

import math

from keras.models import load_model

import predictor_lstm
import my_plotter
import read_data
import my_backtester

train_data = read_data.train_data
test_data = read_data.test_data


def custom_scaler(df_fit, df_scale):
    my_scaler = MinMaxScaler()

    # fit to train
    my_scaler.fit(df_fit)

    # transform train or test
    my_scaled = my_scaler.transform(df_scale)

    return my_scaled


# Classify profitable hours
def classify_profitable(bin_train_y, bin_train_x):
    change_train = (bin_train_y.values - bin_train_x.price0)

    change_percent_train = change_train / bin_train_x.price0

    bins = [float('-inf'), -0.002, 0.002, float('inf')]

    names = [-1, 0, 1]

    bin_train_bins = pd.cut(change_percent_train, bins, labels=names)

    return bin_train_bins


def change_y_neural(y):
    y = np.empty((0, 3))
    for i in bin_train_bins:
        if i == -1:
            y = np.append(y, np.array([[1, 0, 0]]), axis=0)
        if i == 0:
            y = np.append(y, np.array([[0, 1, 0]]), axis=0)
        if i == 1:
            y = np.append(y, np.array([[0, 0, 1]]), axis=0)
    return y


def prep_x_and_y(train_data):
    # create test and train
    bin_train_x = predictor_lstm.create_features(train_data)

    bin_train_y = predictor_lstm.create_y(train_data.price)

    bin_train_bins = classify_profitable(bin_train_y, bin_train_x)

    return bin_train_x, bin_train_bins, bin_train_y


bin_train_x, bin_train_bins, bin_train_y = prep_x_and_y(train_data)

# prep x and y for neural net
bin_train_x_sc = custom_scaler(bin_train_x, bin_train_x)
bin_train_bins_neural = change_y_neural(bin_train_bins)


def train_model(x, y):
    nn = Sequential()
    nn.add(Dense(4, activation="relu", kernel_initializer='random_normal', input_dim=20))
    nn.add(Dense(3, activation="sigmoid"))
    nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    nn.fit(x, y, epochs=100)
    # Save the trained model to file
    nn.save('bin_predictor.h5')


# train_model(bin_train_x_sc, bin_train_bins_neural)

# bin classifier
bin_model = load_model('../trainer/bin_predictor.h5')

# get test data
bin_test_x, bin_test_bins, bin_test_y = prep_x_and_y(test_data)

# prep test x for neural
bin_test_x_sc = custom_scaler(bin_train_x, bin_test_x)


# predict the next hour's price
def get_bin_predictions(train_x, test_x, model):
    # scale test data
    test_x_sc = custom_scaler(train_x, test_x)

    # predictions
    my_predictions = model.predict(test_x_sc)

    buy_sell = np.array([])
    for i in my_predictions:
        if i[0] > 0.5:
            buy_sell = np.append(buy_sell, -1)
        elif i[1] > 0.5:
            buy_sell = np.append(buy_sell, 1)
        else:
            buy_sell = np.append(buy_sell, 0)

    return buy_sell


predictions = get_bin_predictions(bin_train_x, bin_test_x, bin_model)

eval_model = bin_model.evaluate(bin_train_x_sc, bin_train_bins_neural)

actual_rolled = np.roll(bin_test_y.values, 1)[1:]
predictions_truncated = predictions[1:]

trading_history = my_backtester.back_test_bins(predictions_truncated, actual_rolled)
