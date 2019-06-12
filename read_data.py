import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
import math

import predictor_lstm


# reading the data with pandas
data = pd.read_csv('../input/jan_june_btc_minute.csv')

data = predictor_lstm.read(data)

start_train_range = '2018-01-01'
end_train_range = '2018-05-01'
start_test_range = '2018-03-02'
end_test_range = '2018-03-30'

train_data = data[start_train_range:end_train_range]
test_data = data[start_test_range:end_test_range]