# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 12:04:38 2018

@author: Hashitha Kushayne
"""

import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import numpy as np
import requests
import json

all_cryptocurrency_list = {'litecoin':'LTC',
                           'zcash':'ZEC',
                           'monero':'XMR',
                           'dash':'DASH',
                           'bitcoin-gold':'BTG',
                           'bitcoin-cash':'BCH',
                           'vertcoin':'VTC',
                           'monacoin':'MONA',
                           'groestlcoin':'GRS',
                           'dogecoin':'DOGE',
                           'digibyte':'DGB',
                           'namecoin':'NMC',
                           'viacoin':'VIA',
                           'peercoin':'PPC',
                           'mooncoin':'MOON'}

cryptocurrency_list = ['litecoin', 'zcash']
market_info = {}
market = {'litecoin' : 'BITFINEX',
          'zcash' : 'BITFINEX',
          'vertcoin' : 'YOBIT'}
header = {'X-CoinAPI-Key': 'A781575A-514F-45D4-89E4-4718B44AC27B'} #{'X-CoinAPI-Key': '34BC0758-59E7-4D1C-8882-081607BD6728'}

for coin in cryptocurrency_list:
    r = requests.get('https://rest.coinapi.io/v1/ohlcv/{0}_SPOT_{1}_USD/latest?period_id=1HRS&limit=24'.format(market[coin], all_cryptocurrency_list[coin]), headers=header)
    data = r.json()
    market_info['{}'.format(coin)]  = pd.DataFrame(data)
    market_info['{}'.format(coin)].dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
    market_info['{}'.format(coin)].columns = ['{}_'.format(coin)+i for i in market_info['{}'.format(coin)].columns[0:]]
    market_info['{}'.format(coin)].columns = market_info['{}'.format(coin)].columns.str.replace('{}_time_period_start'.format(coin),'time_period_start')
    market_info['{}'.format(coin)].isnull().any()

g=0
for coin in cryptocurrency_list:
    g+=1
    if g == 1:
        all_market_info = market_info['{}'.format(coin)]
        continue                              
    all_market_info = pd.merge(all_market_info, market_info['{}'.format(coin)], on=['time_period_start'])
del g

market_info = all_market_info
del all_market_info

for coin in cryptocurrency_list:
    coin+='_'
    kwargs = {coin+'day_diff': lambda x: (x[coin+'price_close']-x[coin+'price_open'])/x[coin+'price_open']}
    market_info = market_info.assign(**kwargs)
    

for coin in cryptocurrency_list:
    coin+='_'
    kwargs = { coin+'close_off_high': lambda x: 2*(x[coin+'price_high']- x[coin+'price_close'])/(x[coin+'price_high']-x[coin+'price_low'])-1,
               coin+'volatility': lambda x: (x[coin+'price_high']- x[coin+'price_low'])/(x[coin+'price_open'])}
    market_info = market_info.assign(**kwargs)

model_data = market_info[['time_period_start']+[coin+'_'+metric for coin in cryptocurrency_list 
                          for metric in ['price_close','volume_traded','close_off_high','volatility','day_diff']]]

# need to reverse the data frame so that subsequent rows represent later timepoints
model_data = model_data.sort_values(by='time_period_start')

for coin in cryptocurrency_list:
    model_data['{}_close_off_high'.format(coin)]=model_data['{}_close_off_high'.format(coin)].fillna(0)
    model_data['{}_close_off_high'.format(coin)] = model_data['{}_close_off_high'.format(coin)].astype('int64')

test_set = {}
for coin in cryptocurrency_list:
    test_set['{}'.format(coin)] = model_data[['time_period_start']+[coin+'_'+metric
                          for metric in ['price_close','volume_traded','close_off_high','volatility','day_diff']]]
    test_set['{}'.format(coin)]  = test_set['{}'.format(coin)] .drop('time_period_start'.format(coin), 1)

window_len = 24
norm_cols = {}
for coin in cryptocurrency_list:
    norm_cols['{}'.format(coin)] = [coin+'_'+metric for metric in ['price_close','volume_traded']]

LSTM_test_inputs = {}
for coin in cryptocurrency_list:
    LSTM_test_inputs[coin] = []
    for i in range(len(test_set['{}'.format(coin)])):
        temp_set = test_set['{}'.format(coin)][i:(i+window_len)].copy()
        for col in norm_cols['{}'.format(coin)]:
            temp_set.loc[:, col] = temp_set[col]/temp_set[col].iloc[0] - 1
        LSTM_test_inputs['{}'.format(coin)].append(temp_set)

LSTM_test_outputs = {}    
for coin in cryptocurrency_list:
    LSTM_test_outputs['{}'.format(coin)] = (test_set['{}'.format(coin)]['{}_price_close'.format(coin)][window_len:].values/test_set['{}'.format(coin)]['{}_price_close'.format(coin)][:-window_len].values)-1

for coin in cryptocurrency_list:
    LSTM_test_inputs['{}'.format(coin)]  = [np.array(LSTM_test_inputs['{}'.format(coin)] ) for LSTM_test_inputs['{}'.format(coin)]  in LSTM_test_inputs['{}'.format(coin)]]
    LSTM_test_inputs['{}'.format(coin)]  = np.array(LSTM_test_inputs['{}'.format(coin)] )  

from numpy import zeros, newaxis
for coin in cryptocurrency_list:
    LSTM_test_inputs['{}'.format(coin)]  = np.array(test_set['{}'.format(coin)])
    LSTM_test_inputs['{}'.format(coin)] = LSTM_test_inputs['{}'.format(coin)][newaxis, :, :]

predictions = {}
price = {}
loaded_model = {}

from keras.models import model_from_yaml

for coin in cryptocurrency_list:
    
    # load YAML and create model
    yaml_file = open("{}_model.yaml".format(coin), 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model['{}'.format(coin)] = model_from_yaml(loaded_model_yaml)
    # load weights into new model
    loaded_model['{}'.format(coin)].load_weights("{}_model.h5".format(coin))
    print("Loaded model from disk")
    
    loaded_model['{}'.format(coin)].compile(loss='mae', optimizer='adam', metrics=['mse', 'mape', 'msle'])
    predictions['{}'.format(coin)] = loaded_model['{}'.format(coin)].predict(LSTM_test_inputs['{}'.format(coin)])
    price['{}'.format(coin)] = ((np.transpose(predictions['{}'.format(coin)])+1) * test_set['{}'.format(coin)]['{}_price_close'.format(coin)].values[:])[0]
        
##################################COMPUTE
##################################PROPOTIONS

#1. determine how many blocks were generated over the last 24 hours because advertised block time are not always true and for most coins the number of blocks per days is fairly consistent except for those with screwed up difficulty adjustment algorithms.
#2. then calculate what the btc or dollar volume of the coin on a daily basis. (How much wealth does the coin produce in 24 hours)
#3. Add your hashing power to the current network hash for the coin
#4. divide your hashing power by (network_hash + your hash) and you'll get a %
#5. take that % and multiply the daily coin yield by your percentage

url = {}
page = {}
data = {}
blocktime = {}
blockreward = {}
difficulty = {}
blocks_24h = {}
volume_in_24h = {}

for coin in cryptocurrency_list:
    url['{}'.format(coin)] = "http://www.coinwarz.com/v1/api/coininformation/?apikey=da171d9261fe441f989a7305c6d5c754&cointag={}".format(all_cryptocurrency_list[coin])
    page['{}'.format(coin)] = requests.get(url['{}'.format(coin)])
    data['{}'.format(coin)] = page['{}'.format(coin)].json()
    
    blocktime['{}'.format(coin)] = data['{}'.format(coin)]['Data']['BlockTimeInSeconds']
    blockreward['{}'.format(coin)] = data['{}'.format(coin)]['Data']['BlockReward']
    difficulty['{}'.format(coin)] = data['{}'.format(coin)]['Data']['Difficulty']
    blocks_24h['{}'.format(coin)] = 86400/blocktime['{}'.format(coin)]
    volume_in_24h['{}'.format(coin)] = blocks_24h['{}'.format(coin)] * blockreward['{}'.format(coin)]*price['{}'.format(coin)][len(price['{}'.format(coin)])-1]

total_difficulty = 0
for coin in cryptocurrency_list:
    total_difficulty+=difficulty['{}'.format(coin)]

relative_difficulty = {}
for coin in cryptocurrency_list:
    relative_difficulty['{}'.format(coin)] = difficulty['{}'.format(coin)]/total_difficulty

difficulty_of_volume = {}
total_difficulty_of_volume = 0
for coin in cryptocurrency_list:
    difficulty_of_volume['{}'.format(coin)] = relative_difficulty['{}'.format(coin)]*volume_in_24h['{}'.format(coin)]
    total_difficulty_of_volume+=difficulty_of_volume['{}'.format(coin)]

proportion = {}    
for coin in cryptocurrency_list:
    proportion['{}'.format(coin)] = difficulty_of_volume['{}'.format(coin)]*100/total_difficulty_of_volume
    print('{}:'.format(coin)+str(proportion['{}'.format(coin)]))

with open('Output.json', 'w') as outfile:
    json.dump(proportion, outfile)

########################### THE END##################################































