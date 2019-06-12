import pandas as pd
import matplotlib.pyplot as plt


def back_test_bins(predictions, actual, predicted_values=None):

    trading_history = pd.DataFrame(columns=['portfolio', 'sale', 'buy', 'cash', 'fee'])
    cash = 1000
    btc = 0
    fee = 0.998
    counter = 0
    portfolio = 1000

    for i, k in zip(predictions, actual):
        trading_history.loc[counter, 'cash'] = cash
        trading_history.loc[counter, 'portfolio'] = portfolio

        if i == 1 and cash > 0:
            trading_history.loc[counter, 'buy'] = k
            trading_history.loc[counter, 'fee'] = cash*(1-fee)
            btc = (cash*fee)/k
            cash = 0
            portfolio = btc*k + cash

        if i == -1 and cash == 0:
            trading_history.loc[counter, 'sale'] = k
            trading_history.loc[counter, 'fee'] = btc*(1-fee)*k
            cash = (btc*k) * fee
            btc = 0
            portfolio = btc*k + cash

        counter += 1

    fig, axs = plt.subplots(4, 1)

    # portfolio
    axs[0].plot(trading_history.portfolio)
    axs[0].set_title('portfolio')

    # price buy and sell
    axs[1].scatter(list(trading_history.buy.dropna().index), trading_history.buy.dropna())
    axs[1].scatter(list(trading_history.sale.dropna().index), trading_history.sale.dropna())
    axs[1].plot(actual)
    axs[1].set_title('price, buy, and sell')
    axs[1].legend(['price', 'buy', 'sale'])

    if predicted_values is not None:
        # predicted price buy and sell
        axs[2].scatter(list(trading_history.buy.dropna().index), trading_history.buy.dropna())
        axs[2].scatter(list(trading_history.sale.dropna().index), trading_history.sale.dropna())
        axs[2].plot(predicted_values)
        axs[2].set_title('predicted_price, buy, and sell')
        axs[2].legend(['predicted_price', 'buy', 'sale'])

    # plot of cash
    axs[3].plot(trading_history.cash)
    axs[3].set_title('cash')

    fig.tight_layout()

    plt.show()

    return trading_history

