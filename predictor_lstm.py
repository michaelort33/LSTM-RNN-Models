import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from sklearn.linear_model import LinearRegression
import math

model = load_model('../trainer/btc_predictor_5.h5')


def read(df):
    # add Date name
    df.columns = ['Date', 'price', 'volume']

    # convert to date type
    df.loc[:, 'Date'] = df.loc[:, 'Date'].astype('datetime64[ns]')

    # Use Date column as index
    df = df.set_index('Date', drop=True)

    return df


# create features from recent data
def create_features(df):
    chunk_size = 60
    num_chunks = math.floor(len(df) / chunk_size)

    df_features = pd.DataFrame(columns=['mean', 'std', 'slope', 'max_change', 'price'])

    for i in list(range(0, num_chunks)):
        chunk_indices = list(range(i * chunk_size, (i * chunk_size) + chunk_size))
        chunk = df.iloc[chunk_indices, :]

        x = list(range(0, chunk_size))
        lin_model = LinearRegression().fit(np.array(x).reshape(-1, 1), chunk.price.values)
        df_features.loc[i, 'slope'] = lin_model.coef_[0]
        df_features.loc[i, 'mean'] = chunk.price.mean()
        df_features.loc[i, 'std'] = chunk.price.std()
        df_features.loc[i, 'max_change'] = chunk.price.max() - chunk.price.min()
        df_features.loc[i, 'price'] = chunk.price[-1]

    df_shifted_features = pd.DataFrame()
    for i in list(range(0, 4)):
        df_shifted = df_features.shift(i)
        df_shifted.columns = df_shifted.columns.values + str(i)
        df_shifted_features = pd.concat([df_shifted_features, df_shifted], axis=1)

    # remove NAs created by the shift
    df = df_shifted_features.dropna()

    if df.shape[0] > 1:

        df = df.iloc[:-1,:]

    return df


# create a y price value from input data that matches features
def create_y(my_price):
    data_y = my_price[59::60]
    data_y = data_y.shift(-1)
    data_y = data_y[3:]
    data_y = data_y.dropna()

    return data_y


# combine features and y
def add_y_to_features(data, data_features):
    data_y = create_y(data.price)

    data_features = data_features.iloc[:-1, :]

    data_features.loc[:, 'price'] = data_y.values
    return data_features




