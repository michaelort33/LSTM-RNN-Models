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

# Create lstm features
lstm_train_x = predictor_lstm.create_features(train_data)

lstm_train_y = predictor_lstm.create_y(train_data.price)


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

# test data
lstm_test_x = predictor_lstm.create_features(test_data)
lstm_test_y = predictor_lstm.create_y(test_data.price)

model = load_model('../trainer/btc_predictor_5.h5')


# predict the next hour's price
def get_predictions(train_x, test_x, train_y, model=model):
    # scale test data
    test_x_sc = custom_scaler(train_x, test_x)

    # reshape test data to 3D for prediction
    test_x_sc_neural = test_x_sc.reshape(test_x_sc.shape[0], test_x_sc.shape[1], 1)

    # inverse scale
    def inverse_scale(df1, df2):
        my_scaler = MinMaxScaler()
        my_scaler.fit(df1)
        unscaled_predictions = my_scaler.inverse_transform(df2)

        return unscaled_predictions

    # predictions
    predictions_sc = model.predict(test_x_sc_neural)

    # invert scaled predictions
    my_predictions = inverse_scale(train_y.values.reshape(-1, 1), predictions_sc)

    return my_predictions


def get_predicted_comparisons(lstm_train_x, lstm_test_x, lstm_train_y, lstm_test_y):
    predictions_train = get_predictions(lstm_train_x, lstm_train_x, lstm_train_y)

    predictions_test = get_predictions(lstm_train_x, lstm_test_x, lstm_train_y)

    change = (lstm_test_y.values - lstm_test_x.price0)[1:]

    change_percent = change / lstm_test_x.price0[1:]

    pred_change = (predictions_test - np.roll(predictions_test, 1))[1:]

    pred_change_percent = pred_change / np.roll(predictions_test, 1)[1:]

    buy_sell = np.array([])
    for i in pred_change_percent:
        if i > 0.01:
            buy_sell = np.append(buy_sell, 1)
        elif i < -0.01:
            buy_sell = np.append(buy_sell, -1)
        else:
            buy_sell = np.append(buy_sell, 0)

    predicted_values = {'predictions_train': predictions_train, 'predictions_test': predictions_test,
                        'change': change, 'change_percent': change_percent, 'pred_change': pred_change,
                        'pred_change_percent': pred_change_percent, 'buy_sell': buy_sell}

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
    summary_statistics['percent_direction'] = (change_dir.values.reshape(-1, 1) == pred_change_dir).sum() / \
                                              pred_change_dir.shape[0]

    # std of errors
    summary_statistics['std_error'] = (pred_change.reshape(-1, 1) - change.values.reshape(-1, 1)).std()

    # mean error
    summary_statistics['mean_error'] = (pred_change.reshape(-1, 1) - change.values.reshape(-1, 1)).mean()

    return summary_statistics


summary_statistics = model_summary(lstm_test_y)

actual_prices_rolled = lstm_test_y.shift(1).dropna()

truncated_predictions = predicted_values['predictions_test'][:-1]

trading_history = my_backtester.back_test_bins(predicted_values['buy_sell'], actual_prices_rolled.values,
                                                    truncated_predictions)
