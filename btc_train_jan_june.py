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

train_data = read_data.train_data
test_data = read_data.test_data

# Create lstm features
lstm_train_x = predictor_lstm.create_features(train_data)

lstm_train_y = predictor_lstm.create_y(train_data.price)

lstm_test_x = predictor_lstm.create_features(test_data)

lstm_test_y = predictor_lstm.create_y(test_data.price)

# plot train and test y values
# plot_2_x(lstm_train_y, lstm_test_y)


def custom_scaler(df_fit, df_scale):
    my_scaler = MinMaxScaler()

    # fit to train
    my_scaler.fit(df_fit)

    # transform train or test
    my_scaled = my_scaler.transform(df_scale)

    return my_scaled


# Scale train to [0,1]
lstm_train_x_sc = custom_scaler(lstm_train_x, lstm_train_x)
lstm_train_y_sc = custom_scaler(lstm_train_y.values.reshape(-1, 1), lstm_train_y.values.reshape(-1, 1))

# Reshape train to 3d for neural net.
lstm_train_x_sc_neural = lstm_train_x_sc.reshape(lstm_train_x_sc.shape[0], lstm_train_x_sc.shape[1], 1)


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
# train_model(lstm_train_x_sc_neural, lstm_train_y_sc)


def get_predicted_comparisons(lstm_train_x, lstm_test_x, lstm_train_y, lstm_test_y):

    predictions_train = predictor_lstm.get_predictions(lstm_train_x, lstm_train_x, lstm_train_y)

    predictions_test = predictor_lstm.get_predictions(lstm_train_x, lstm_test_x, lstm_train_y)

    change = (lstm_test_y.values - lstm_test_x.price0)[1:]

    change_percent = change/lstm_test_x.price0[1:]

    pred_change = (predictions_test - np.roll(predictions_test, 1))[1:]

    pred_change_percent = pred_change/np.roll(predictions_test, 1)[1:]

    predicted_values = {'predictions_train': predictions_train, 'predictions_test': predictions_test,
                        'change': change, 'change_percent': change_percent, 'pred_change': pred_change,
                        'pred_change_percent': pred_change_percent}

    return predicted_values


predicted_values = get_predicted_comparisons(lstm_train_x, lstm_test_x, lstm_train_y, lstm_test_y)


def analyze_fit(lstm_train_y, lstm_test_y):

    predictions_train = predicted_values['predictions_train']
    predictions_test = predicted_values['predictions_test']
    change = predicted_values['change']
    pred_change = predicted_values['pred_change']

    fig, axs = plt.subplots(3, 1)

    # plot of fit on training data
    axs[0].plot(predictions_train)
    axs[0].plot(lstm_train_y.values)
    axs[0].set_title('fit of train')
    axs[0].legend(['predictions', 'actual'])

    # plot fit on test data
    axs[1].plot(predictions_test)
    axs[1].plot(lstm_test_y.values)
    axs[1].set_title('fit of test')
    axs[1].legend(['predictions', 'actual'])

    # plot of errors
    axs[2].plot(pred_change.reshape(-1, 1) - change.values.reshape(-1, 1))
    axs[2].set_title('errors')

    fig.tight_layout()

    plt.show()


analyze_fit(lstm_train_y, lstm_test_y)


def model_summary(lstm_test_y):

    predictions_test = predicted_values['predictions_test']
    change = predicted_values['change']
    pred_change = predicted_values['pred_change']

    summary_statistics = {}

    summary_statistics['MSE'] = ((predictions_test - lstm_test_y.values) ** 2).mean()

    # check how often change was down

    change_dir = change < 0
    pred_change_dir = pred_change < 0

    # direction rate
    summary_statistics['percent_direction'] = (change_dir.values.reshape(-1, 1) == pred_change_dir).sum() / pred_change_dir.shape[0]

    # std of errors
    summary_statistics['std_error'] = (pred_change.reshape(-1, 1) - change.values.reshape(-1, 1)).std()

    # mean error
    summary_statistics['mean_error'] = (pred_change.reshape(-1, 1) - change.values.reshape(-1, 1)).mean()

    return summary_statistics


summary_statistics = model_summary(lstm_test_y)


def back_test(lstm_test_y):

    # portfolio change
    percent_change = predicted_values['pred_change_percent']
    portfolio = []
    sale = []
    buy = []
    cash = 1000
    rolled_y = lstm_test_y[:-1]
    btc = 0
    fee = 0.998
    counter = 0
    bought_times = []
    sale_times = []

    for i, k in zip(percent_change, rolled_y):

        if i > 0.002 and cash > 0:
            btc = (cash*fee)/k
            cash = 0
            buy.append(k)
            bought_times.append(counter)

        if i < -0.002 and cash == 0:
            cash = (btc*k) * fee
            btc = 0
            sale.append(k)
            sale_times.append(counter)

        portfolio.append((btc * k) + cash)

        counter += 1

    fig, axs = plt.subplots(2, 1)

    axs[0].plot(portfolio)
    axs[0].set_title('portfolio_value')
    axs[1].scatter(bought_times, buy, color='blue')
    axs[1].scatter(sale_times, sale, color='red')
    axs[1].plot(rolled_y.values)
    axs[1].legend(['price', 'buy', 'sell'])
    plt.show()


back_test(lstm_test_y)
