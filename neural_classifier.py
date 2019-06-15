from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

from sklearn.preprocessing import MinMaxScaler

import numpy as np
import pandas as pd

import prep_data_functions
import read_data
import my_backtester

seed = 7
np.random.seed(seed)

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

    names = [0, 1, 2]

    bin_train_bins = pd.cut(change_percent_train, bins, labels=names)

    return bin_train_bins


def prep_x_and_y(train_data):

    # create test and train
    bin_train_x = prep_data_functions.create_features(train_data, grouping_size=10, shifts=24)

    bin_train_y = prep_data_functions.create_y(train_data.price, grouping_size=10, shifts=24)

    return bin_train_x, bin_train_y


bin_train_x, bin_train_y = prep_x_and_y(train_data)

bin_train_bins = classify_profitable(bin_train_y, bin_train_x)

# prep x and y for neural net
bin_train_x_sc = custom_scaler(bin_train_x, bin_train_x)
bin_train_bins_neural = np_utils.to_categorical(bin_train_bins)


def train_model_1(x, y):
    nn = Sequential()
    nn.add(Dense(4, activation="relu", kernel_initializer='random_normal', input_dim=x.shape[1]))
    nn.add(Dense(3, activation="sigmoid"))
    nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    nn.fit(x, y, epochs=100)
    # Save the trained model to file
    nn.save('bin_predictor.h5')


# train_model_1(bin_train_x_sc, bin_train_bins_neural)

def train_model_2(x, y):
    model = Sequential()
    model.add(Dense(200, input_dim=120, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x, y, epochs=100)
    model.save('bin_predictor_2.h5')
    return model


# train_model_2(bin_train_x_sc, bin_train_bins_neural)

# bin classifier
model1 = load_model('../trainer/bin_predictor.h5')
model2 = load_model('../trainer/bin_predictor_2.h5')

# get test data
bin_test_x, bin_test_y = prep_x_and_y(test_data)

bin_test_bins = np_utils.to_categorical(bin_test_y)

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
        elif i[2] > 0.5:
            buy_sell = np.append(buy_sell, 1)
        else:
            buy_sell = np.append(buy_sell, 0)

    return buy_sell


def setup_for_back_test(bin_train_x, bin_test_x, model):

    # first model predictions and test
    predictions = get_bin_predictions(bin_train_x, bin_test_x, model)

    actual_rolled = np.roll(bin_test_y.values, 1)[1:]
    predictions_truncated = predictions[1:]

    trading_history = my_backtester.back_test_bins(predictions_truncated, actual_rolled)

    return trading_history


history_1 = setup_for_back_test(bin_train_x, bin_test_x, model1)

history_2 = setup_for_back_test(bin_train_x, bin_test_x, model2)
