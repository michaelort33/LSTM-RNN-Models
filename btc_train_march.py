from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping
from keras.models import load_model
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import pandas as pd

# reading the data with pandas
data = pd.read_csv('../input/march_april_btc_minute.csv')


def read(df, keep=1):
    # Use subset of data (my little gpu can't do much)
    cut = round(len(df) / keep)
    df = df.iloc[8*cut:9*cut, :]

    # convert to date type
    df.Date = df.Date.astype('datetime64[ns]')

    # Use Date column as index
    df = df.set_index('Date', drop=True)

    return df


data = read(data, keep=10)


def plot_2_y(x, y1, y2, df):
    fig, host = plt.subplots()
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


def create_features(df):
    # Feature engineering:
    # take the last three minutes as inputs for volume and price
    for i in list(range(1, 4)):
        df['price_shift' + str(i)] = df.price.shift(i)

    return df


# Create the features
data = create_features(data)


def clean(df):
    # Remove NA
    df = df.dropna()

    # drop volume
    df = df.drop(['volume'], axis=1)

    return df


data = clean(data)


def split_train_test(df, split=0.8):
    index = round(split * len(df))
    df_train = df.iloc[:index, :]
    df_test = df.iloc[index:, :]

    return df_train, df_test


# split the test and train
train, test = split_train_test(data)


# plt train and test
plt.figure(1)
plt.subplot(111)
plt.xticks(rotation=60)
plt.plot(train.price)
plt.plot(test.price)
plt.legend(['Train', 'Test'])
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()


def scaler(df1, df2):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(df1)
    scaled_test = scaler.transform(df2)

    return scaled_test


# Scale train to [0,1]
train = scaler(train, train)


def prep_neural_data_format(df):

    # split train to x and y
    train_x = df[:, 1:]
    train_y = df[:, 0]

    # reshape to 3d for lstm input
    train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], 1)

    return train_x, train_y


neural_x, neural_y = prep_neural_data_format(train)


def train_model(train_x, train_y):

    # Create LSTM model
    model = Sequential()
    model.add(LSTM(10, input_shape=(train_x.shape[1], 1), activation="relu", kernel_initializer='lecun_uniform',
                   return_sequences=False))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam")
    early_stop = EarlyStopping(monitor='loss', patience=20, verbose=1)
    model.fit(train_x, train_y, epochs=20, batch_size=1, verbose=1, shuffle=False, callbacks=[early_stop])
    # Save the trained model to file
    model.save('btc_predictor_5.h5')


# train the model
train_model(neural_x, neural_y)

# Load trained model from file
model_lstm = load_model('btc_predictor_5.h5')

test_sc = scaler(test, test)


def prep_test_data(df):

    # Select x matrix
    df_x = df[:, 1:]
    df_y = df[:, 0]

    # Reshape for prediction
    df_x = df_x.reshape(df_x.shape[0], df_x.shape[1], 1)

    return df_x, df_y


test_x_sc, test_y_sc = prep_test_data(test_sc)

# Measure error
lstm_test_mse = model_lstm.evaluate(test_x_sc, test_y_sc)
print('LSTM: %f' % lstm_test_mse)


# predictions
predictions_sc = model_lstm.predict(test_x_sc)

# reshape x to 2d so i can inverse transform
test_x_sc = test_x_sc.reshape(test_x_sc.shape[0], test_x_sc.shape[1])

# combine predictions with test_x
prediction_matrix_sc = np.hstack((predictions_sc, test_x_sc))


# inverse predictions back to
def inverse_scale(df1, df2):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(df1)
    unscaled_predictions = scaler.inverse_transform(df2)

    return unscaled_predictions


predictions = inverse_scale(test, prediction_matrix_sc)

# slice out of the matrix to get final predictions
final_predictions = predictions[:, 0]

# plot predictions and actual values
plt.cla()
plt.figure(2)
plt.plot(final_predictions)
plt.plot(test.price.values)
plt.legend(['predictions', 'actual'])

# Print the MSE of the LSTM model
print(((final_predictions[1:] - test.price.values)**2).mean(axis=0))

# Now compare this model to a baseline of a rolling average of 3 previous minutes
data['price_rolling'] = data.price.rolling(window=3).mean().shift()
data = data.dropna()
train, test = split_train_test(data)

plt.figure(3)
plt.plot(test.price)
plt.plot(test.price_rolling)
print(((test.price_rolling.values - test.price.values)**2).mean(axis=0))
