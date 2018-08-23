# Limit Order Book(L2) Prediction

*) Trys to predict price jumps from features derived from l2 order book information.   
*) price jumps : current bid price > previous bid price (within short period of time) 

order book information contains ask, bid prices and corresponding quantities at each level.

![features](https://github.com/miroblog/limit_orderbook_prediction/blob/master/features.png)  

Applies custom weights to overcome inbalance between classes.

[JuypterNotebook-Rnn](https://github.com/miroblog/limit_orderbook_prediction/blob/master/nn_example.ipynb)

![orderbook](https://github.com/miroblog/limit_orderbook_prediction/blob/master/l2_orderbook.png)

## Getting Started

```python
from nn import NeuralNetwork
from rnn import RNN

timestep = 50
n_cross_validation = 3
# for order book info only
data = data_prep.get_test_data(timestep, predict_step=5, filename="upbit_l2_orderbook_ADA")

# input_shape <- (timestep, n_features)
# output_shape <- n_classes
nn = NeuralNetwork(RNN(input_shape=data.x.shape[1:], output_dim=data.y.shape[1]), class_weight={0: 1., 1: 1., 2: 1.})

print("TRAIN")
nn.train(data)

print("TEST")
nn.test(data)

print("TRAIN WITH CROSS-VALIDATION")
nn.run_with_cross_validation(data, n_cross_validation)

```


### Prerequisites

common library such as BeautifulSoup, selenium, pyperclip

```python
pip install -r requirements.txt
```
### Impements LSTM / Conv2DLSTM 
![lstm](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
![convolutional](https://github.com/miroblog/limit_orderbook_prediction/blob/master/convolutional.png)

## Authors

* **Lee Hankyol** - *Initial work* - [Upbit_API_COLLECTION](https://github.com/miroblog/limit_orderbook_prediction)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
