from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd

import sys

sys.path.insert(0, '/home/michael/Documents/Bitcoin Trading/btc_trader')
import predictor_lstm

# reading the data with pandas
data = pd.read_csv('../input/jan_june_btc_minute.csv')

data = predictor_lstm.read(data)


def plot_2_y(x, y1, y2, df):
    fig, host = plt.subplots(num=1)
    fig.subplots_adjust(right=0.75)
    plt.xticks(rotation=60)
    par1 = host.twinx()

    p1, = host.plot(df[y1], "k-", label=y1)
    p2, = par1.plot(df[y2], "b-", label=y2, alpha=0.5)

    host.set_xlabel(x)
    host.set_ylabel(y1)
    par1.set_ylabel(y2)
    lines = [p1, p2]
    host.legend(lines, [l.get_label() for l in lines])

    plt.show()


# Plot the data
plot_2_y('Date', 'price', 'volume', data)

# Create features
data_features = predictor_lstm.create_features(data)

data_features_with_y = predictor_lstm.add_y_to_features(data, data_features)


# split train and test and x and y
def split_train_test(df, split=0.7):
    index = round(split * len(df))
    df_train = df.iloc[:index, :]
    df_test = df.iloc[index:, :]

    df_train_y = df_train.price
    df_train_x = df_train.loc[:, df_train.columns != 'price']

    df_test_y = df_test.price
    df_test_x = df_test.loc[:, df_test.columns != 'price']

    return df_train_x, df_train_y, df_test_x, df_test_y


# split the test and train
train_x, train_y, test_x, test_y = split_train_test(data_features_with_y)


def plot_2_x(my_train_y, my_test_y):
    # plt train and test
    plt.figure(1)
    plt.subplot(111)
    plt.xticks(rotation=60)
    plt.plot(my_train_y)
    plt.plot(my_test_y)
    plt.legend(['Train', 'Test'])
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.show()


plot_2_x(train_y, test_y)


def custom_scaler(df_fit, df_scale):
    my_scaler = MinMaxScaler()

    # fit to train
    my_scaler.fit(df_fit)

    # transform train or test
    my_scaled = my_scaler.transform(df_scale)

    return my_scaled


# Scale train to [0,1]
train_x_sc = custom_scaler(train_x, train_x)
train_y_sc = custom_scaler(train_y.values.reshape(-1, 1), train_y.values.reshape(-1, 1))

# Reshape train to 3d for neural net.
train_x_sc_neural = train_x_sc.reshape(train_x_sc.shape[0], train_x_sc.shape[1], 1)


def train_model(x, y):
    # Create LSTM model
    my_model = Sequential()
    my_model.add(LSTM(512, input_shape=(x.shape[1], x.shape[2]), activation="relu", kernel_initializer='lecun_uniform',
                      return_sequences=False))
    my_model.add(Dense(1))
    my_model.compile(loss="mean_squared_error", optimizer="adam")
    early_stop = EarlyStopping(monitor='loss', patience=20, verbose=1)
    my_model.fit(x, y, epochs=100, batch_size=1, verbose=1, shuffle=False, callbacks=[early_stop])
    # Save the trained model to file
    my_model.save('btc_predictor_5.h5')


# train the model
# train_model(train_x_sc_neural, train_y_sc)

predictions_train = predictor_lstm.get_predictions(train_x, train_x, train_y)
predictions_test = predictor_lstm.get_predictions(train_x, test_x, train_y)


# plot predictions and actual values
def plot_predictions(my_predictions, actual, fig_num):
    plt.figure(fig_num)
    plt.plot(my_predictions)
    plt.plot(actual)
    plt.legend(['predictions', 'actual'])
    plt.show()


# plot of fit on training data
plot_predictions(predictions_train, train_y.values, 2)

# plot of fit on test data
plot_predictions(predictions_test, test_y.values, 3)

# Print the MSE of the LSTM model
print(((predictions_test - test_y.values) ** 2).mean())

# Now compare this model to a baseline of a rolling average of 3 previous minutes
data['price_rolling'] = data.price.rolling(3).mean().shift()
data = data.dropna()
plot_predictions(data.price_rolling, data.price, 4)

# print the mse of the rolling mean model
print(((data.price_rolling.values - data.price.values) ** 2).mean())

# Check how often the direction was correct
change = test_y.values - test_x.price1
pred_change = predictions_test - np.roll(predictions_test, 1)

change_dir = change < 0
pred_change_dir = pred_change < 0
(change_dir.values.reshape(-1, 1) == pred_change_dir).sum() / pred_change_dir.shape[0]

# plot of errors
plt.plot(pred_change.reshape(-1, 1) - change.values.reshape(-1, 1))

# std of errors
(pred_change.reshape(-1, 1) - change.values.reshape(-1, 1)).std()

# mean error
(pred_change.reshape(-1, 1) - change.values.reshape(-1, 1)).mean()
