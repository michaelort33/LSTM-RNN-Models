import pandas as pd

# reading the data with pandas
data = pd.read_csv('../input/jan_june_btc_minute.csv')


def read(df):
    # add Date name
    df.columns = ['Date', 'price', 'volume']

    # convert to date type
    df.loc[:, 'Date'] = df.loc[:, 'Date'].astype('datetime64[ns]')

    # Use Date column as index
    df = df.set_index('Date', drop=True)

    return df


data = read(data)

start_train_range = '2018-01-01'
end_train_range = '2018-05-01'
start_test_range = '2018-01-01'
end_test_range = '2018-05-06'


def descriptive_stats(data, grouping_size=60):
    descriptives = {}

    # get average grouping change
    grouped_prices = data.price[grouping_size - 1::grouping_size]
    next_grouped_prices = grouped_prices.shift(-1).dropna()
    truncated_grouped_prices = grouped_prices[:-1]
    descriptives['grouping_pct_change'] = (truncated_grouped_prices - next_grouped_prices) / truncated_grouped_prices

    # get data that changed less than fee
    in_fee_range = (descriptives['grouping_pct_change'] > -0.002).values & (
                descriptives['grouping_pct_change'] < 0.002).values
    descriptives['in_fee_range'] = in_fee_range

    # pct that changed less than fee
    total_changes = next_grouped_prices.shape[0]
    descriptives['pct_greater_fee'] = in_fee_range.sum() / total_changes

    return descriptives


train_data = data[start_train_range:end_train_range]
test_data = data[start_test_range:end_test_range]
