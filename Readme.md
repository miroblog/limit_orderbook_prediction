# Upbit API Collections

Automates upbit exchange platform using selenium webdriver  
The code was written before the official rest-api for upbit exchange was released.  
The official API is released 2018. 06 https://docs.upbit.com/blog/  

It is for educational purposes only. This code may come in handy where official rest-api is not available.  


## Getting Started

```python

# upbit uses kakao id/pwd for login
kakao_id_secret = "ur_kakao_id"
kakao_pwd_secret = "ur_pass_wd"

trader = upbitTrader()
trader.set_up_trade("BTC")

# scrapes limit order book information directly from the web
ask_prices, bid_prices, ask_quantities, bid_quantities = trader.collector()

# buy 50% at the best bid price
print(bid_prices[0])
trader.put_buy_order(bid_prices[0], "HALF")

# sell all the coin at best ask price
print(bid_prices[0])
trader.put_sell_order(ask_prices[0], "ALL")

# cancel all order
trader.cancel_all_order()
```


### Prerequisites

common library such as BeautifulSoup, selenium, pyperclip

```python
pip install -r requirements.txt
```

![candle](https://github.com/miroblog/upbit_api_collection/blob/master/png/ohlc.png)

## Authors

* **Lee Hankyol** - *Initial work* - [Upbit_API_COLLECTION](https://github.com/miroblog/upbit_api_collection)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
