from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Flatten, Activation
from keras.callbacks import EarlyStopping

import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import load_model


# reading the data with pandas
data = pd.read_csv('../input/march_april_btc_minute.csv')

# convert to date type
data.Date = data.Date.astype('datetime64[ns]')
data = data.set_index('Date', drop=True)

# Plot the data
fig, host = plt.subplots()
fig.subplots_adjust(right=0.75)
plt.xticks(rotation=60)
par1 = host.twinx()

p1, = host.plot(data.price, "k-", label="price")
p2, = par1.plot(data.volume, "b-", label="volume", alpha=0.5)

host.set_xlabel("Date")
host.set_ylabel("price")
par1.set_ylabel("volume")
lines = [p1, p2]
host.legend(lines, [l.get_label() for l in lines])

plt.close()


def create_features(df):

    # Feature engineering:
    # take the last three minutes as inputs for volume and price

    for i in list(range(1,4)):
        df['price_shift'+str(i)] = df.price.shift(i)
        df['vol_shift'+str(i)] = df.volume.shift(i)

    # Get rolling means and std of 5, 20, 100, 200 for volume and price

    for i in [5, 20, 100, 200]:
        df['price_rolling_mean_'+ str(i)] = df.price.rolling(window=i).mean()
        df['price_rolling_std_' + str(i)] = df.price.rolling(window=i).std()

        df['vol_rolling_mean_'+str(i)] = df.volume.rolling(window=i).mean()
        df['vol_rolling_std_' + str(i)] = df.volume.rolling(window=i).std()

        # Remove the NAs this introduces in the first 200 rows

        df = df.dropna()

        return df


# Create the features
data = create_features(data)

# split the test and train
train_percent = 0.8
index = round(train_percent*len(data))
train = data.iloc[:index, :]
test = data.iloc[index:, :]

# plt train and test

plt.figure(1)
plt.subplot(111)
plt.xticks(rotation=60)
plt.plot(train.price)
plt.plot(test.price)

plt.legend(['Train', 'Test'])
plt.xlabel("Date")
plt.ylabel("Price")
plt.close()

# Scale it to [-1,1]
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(train)
train = scaler.transform(train)

# split train to x and y
train_x = train[:, 1:]

# reshape to 3d for lstm input
train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], 1)

train_y = train[:, 0]

# Create LSTM model

model = Sequential()

model.add(LSTM(10, input_shape=(train_x.shape[1], 1), activation="relu", kernel_initializer='lecun_uniform',return_sequences=False))

model.add(Dense(1))

model.compile(loss="mean_squared_error", optimizer="adam")

early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)

history = model.fit(train_x, train_y, epochs=100, batch_size=1, verbose=1, shuffle=False, callbacks=[early_stop])

model.save('btc_predictor.h5')

model = load_model('btc_predictor.h5')

# Predict the test
test = scaler.transform(test)

test_x = test[:, 1:]
test_x = test_x.reshape(test_x.shape[0], test_x.shape[1], 1)

test_y = test[:, 0]


preds = model.predict(test_x)

test_x = test_x.reshape(test_x.shape[0], test_x.shape[1])

predictions = np.hstack((preds, test_x))

predictions = scaler.inverse_transform(predictions)

final_preds = predictions[:, 0]

plt.plot(range(len(final_preds)), final_preds)
