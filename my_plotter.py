import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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