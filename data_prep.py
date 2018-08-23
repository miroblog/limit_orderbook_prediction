import numpy as np
from keras.utils import np_utils
import pandas

class Data:
    def __init__(self, x, y=0):
        self.x = x
        self.y = y

class BookRecord(object):
    def __init__(self, data=None):
        if data:
            self.Price = float(data[0])
            self.Volume = float(data[1])
            self.Side = data[2]
    def __str__(self):
        result = ''
        result += '|{:^10}'.format(self.Price)
        result += '|{:^10}'.format(self.Volume)
        result += '|{:^10}'.format(self.Side)
        result += '|'
        return result

class TransactionRecord(object):
    def __init__(self, data=None):
        if data:
            self.Price = float(data[0])
            self.Volume = float(data[1])
    def __str__(self):
        result = ''
        result += '|{:^10}'.format(self.Price)
        result += '|{:^10}'.format(self.Volume)
        result += '|'
        return result
class OrderBook(object):
    buy_orders = []
    sell_orders = []
    current_trade = []

    X = []
    Y = []
    predict_step = 4
    transaction_price = 0.0
    prev_trade_prices = []

    def __init__(self, name='unknown'):
        self.name = name

    def load_orderbook_event(self, filename , predict_step):
        self.predict_step = predict_step
        df = pandas.read_pickle(filename)
        for index, row in df.iterrows():
            self.current_trade.append(TransactionRecord([row["TP"], row["TQ"]]))
            for i in range(6):
                # boolean bid(buy) is 1, ask(sell) is 0
                self.buy_orders.append(BookRecord([row["BP"+str(i+1)], row["BQ"+str(i+1)], 1]))
                self.sell_orders.append(BookRecord([row["AP"+str(i+1)], row["AQ"+str(i+1)], 0]))
            self.update_history()
            self.current_trade = []
            self.buy_orders = []
            self.sell_orders = []

    def update_history(self):
        levels = 5
        n = levels

        if n + 1 > len(self.sell_orders):
            n = len(self.sell_orders) - 1

        if n + 1 > len(self.buy_orders):
            n = len(self.buy_orders) - 1

        # current trade price, volume
        v0 = []
        v1 = []
        v2 = []
        v3 = []
        v4 = [0.0, 0.0, 0.0, 0.0]
        v5 = [0.0, 0.0]

        # addtional feature
        v0.append(self.current_trade[0].Price)
        v0.append(self.current_trade[0].Volume)

        for i in range(0, n):
            v1.append(self.sell_orders[i].Price)
            v1.append(self.sell_orders[i].Volume)
            v1.append(self.buy_orders[i].Price)
            v1.append(self.buy_orders[i].Volume)

            v2.append(self.sell_orders[i].Price - self.buy_orders[i].Price)
            v2.append((self.sell_orders[i].Price + self.buy_orders[i].Price) / 2.0)

            v3.append(abs(self.sell_orders[i + 1].Price - self.sell_orders[i].Price))
            v3.append(abs(self.buy_orders[i + 1].Price - self.buy_orders[i].Price))

        for i in range(0, n):
            v4[0] += self.sell_orders[i].Price
            v4[1] += self.buy_orders[i].Price
            v4[2] += self.sell_orders[i].Volume
            v4[3] += self.buy_orders[i].Volume
            v5[0] += self.sell_orders[i].Price - self.buy_orders[i].Price
            v5[1] += self.sell_orders[i].Volume - self.buy_orders[i].Volume

        if n > 0:
            v4[0] /= float(n)
            v4[1] /= float(n)
            v4[2] /= float(n)
            v4[3] /= float(n)

        X = self.getX(v0, v1, v2, v3, v4, v5)

        current_trade_price = self.current_trade[0].Price
        self.prev_trade_prices.append(self.current_trade[0].Price)
        prev_trade_price = self.get_n_prev_trade_price(self.predict_step+1)

        Y = self.getY(prev_trade_price, current_trade_price)
        self.X.append(X)
        self.Y.append(Y)

    def get_n_prev_trade_price(self, n):
        if(len(self.prev_trade_prices) >= n) :
            return self.prev_trade_prices[-n]
        else:
            return self.prev_trade_prices[-1]

    def getX(self, v0, v1, v2, v3, v4, v5):
        x = []
        x.extend(v0)
        x.extend(v1)
        x.extend(v2)
        x.extend(v3)
        x.extend(v4)
        x.extend(v5)
        return x

    def getY(self, prev_trade_price, curr_trade_price):
        y = 0 # same
        if prev_trade_price < curr_trade_price: # up
            y = 1
        elif prev_trade_price > curr_trade_price: # down
            y = 2
        return y
    def getXY(self):
        return self.X, self.Y


def get_balanced_subsample(x, y, subsample_size=1.0):
    class_xs = []
    min_elems = None

    for yi in np.unique(y):
        elems = x[(y == yi)]
        class_xs.append((yi, elems))
        if min_elems == None or elems.shape[0] < min_elems:
            min_elems = elems.shape[0]

    use_elems = min_elems
    if subsample_size < 1:
        use_elems = int(min_elems * subsample_size)

    xs = []
    ys = []

    for ci, this_xs in class_xs:
        if len(this_xs) > use_elems:
            np.random.shuffle(this_xs)

        x_ = this_xs[:use_elems]
        y_ = np.empty(use_elems)
        y_.fill(ci)

        xs.append(x_)
        ys.append(y_)

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)

    return xs, ys

def get_recent_sample(book, window_size):
    x, y = book.getXY()
    x_temp = []
    y_temp = []
    if(len(x) >= window_size):
        x_temp.append(x[-window_size:])
        x = np.array(x_temp)
        return True, Data(x,)
    else:
        return False, len(x)
def prepare_data(book, window_size, predict_step):
    x, y = book.getXY()
    x_temp = []
    y_temp = []
    for i in range(len(x) - window_size - predict_step):
        x_temp.append(x[i:(i + window_size)])
        y_temp.append(y[i + window_size + predict_step])
    print(len(x_temp))
    print(len(y_temp))
    x = np.array(x_temp)
    y = y_temp

    # comment out
    #x, y = get_balanced_subsample(x, y)

    xy = list(zip(x, y))
    x_, y_ = zip(*xy)
    x = np.array(x_)

    y = np_utils.to_categorical(y_, 3)
    # y = np_utils.to_categorical(y_, 3)
    print("{0} records with = (same)".format(sum(y[:, 0])))
    print("{0} records with < (plus profit) ".format(sum(y[:, 1])))
    print("{0} records with > (loss)".format(sum(y[:, 2])))
    print('x shape:', x.shape)
    print('y shape:', y.shape)

    return Data(x, y)

def get_test_data(window_size, predict_step, filename):
    xrp_orderbook = OrderBook(name=filename)
    xrp_orderbook.load_orderbook_event("./data/"+filename, predict_step)
    return prepare_data(xrp_orderbook, window_size, predict_step)