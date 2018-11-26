# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 11:38:30 2018

@author: Hash Kushayne
"""

import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import numpy as np
import urllib3,certifi
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
                           
cryptocurrency_list = ['litecoin', 'zcash', 'dash']
https = urllib3.PoolManager( cert_reqs='CERT_REQUIRED', ca_certs=certifi.where(),) 
market_info = {}
for coin in cryptocurrency_list:
    #url['{}'.format('litecoin')] = https.urlopen('GET',"https://coinmarketcap.com/currencies/{}/historical-data/?start=20130428&end=".format('litecoin')+time.strftime("%Y%m%d"))
    # get market info for ethereum from the start of 2016 to the current day
    market_info['{}'.format(coin)]  = pd.read_html(https.urlopen('GET',"https://coinmarketcap.com/currencies/{}/historical-data/?start=20130428&end=".format(coin)+time.strftime("%Y%m%d")).data)[0]
    # convert the date string to the correct date format
    market_info['{}'.format(coin)]  = market_info['{}'.format(coin)].assign(Date=pd.to_datetime(market_info['{}'.format(coin)]['Date']))
    if coin == 'litecoin':
        # when Volume is equal to '-' convert it to 0
        market_info['{}'.format(coin)].loc[market_info['{}'.format(coin)]['Volume']=="-",'Volume']=0
    # convert to int
    market_info['{}'.format(coin)]['Volume'] = market_info['{}'.format(coin)]['Volume'].astype('int64')
    market_info['{}'.format(coin)].columns =[market_info['{}'.format(coin)].columns[0]]+['{}_'.format(coin)+i for i in market_info['{}'.format(coin)].columns[1:]]
               
g=0
for coin in cryptocurrency_list:
    g+=1
    if g == 1:
        all_market_info = market_info['{}'.format(coin)]
        continue                              
    all_market_info = pd.merge(all_market_info, market_info['{}'.format(coin)], on=['Date'])
del g
all_market_info = all_market_info[all_market_info['Date']>='2016-01-01']
market_info = all_market_info
del all_market_info

for coin in cryptocurrency_list:
    coin+='_'
    kwargs = {coin+'day_diff': lambda x: (x[coin+'Close**']-x[coin+'Open*'])/x[coin+'Open*']}
    market_info = market_info.assign(**kwargs)

for coin in cryptocurrency_list:
    coin+='_'
    kwargs = { coin+'close_off_high': lambda x: 2*(x[coin+'High']- x[coin+'Close**'])/(x[coin+'High']-x[coin+'Low'])-1,
               coin+'volatility': lambda x: (x[coin+'High']- x[coin+'Low'])/(x[coin+'Open*'])}
    market_info = market_info.assign(**kwargs)

model_data = market_info[['Date']+[coin+'_'+metric for coin in cryptocurrency_list 
                          for metric in ['Close**','Volume','close_off_high','volatility']]]
# need to reverse the data frame so that subsequent rows represent later timepoints
model_data = model_data.sort_values(by='Date')

# we don't need the date columns anymore
split_date = '2018-01-01'
training_set, test_set = model_data[model_data['Date']<split_date], model_data[model_data['Date']>=split_date]
training_set = training_set.drop('Date', 1)
test_set = test_set.drop('Date', 1)

window_len = 10
norm_cols = [coin+'_'+metric for coin in cryptocurrency_list for metric in ['Close**','Volume']]

LSTM_training_inputs = []
for i in range(len(training_set)-window_len):
    temp_set = training_set[i:(i+window_len)].copy()
    for col in norm_cols:
        temp_set.loc[:, col] = temp_set[col]/temp_set[col].iloc[0] - 1
    LSTM_training_inputs.append(temp_set)

LSTM_test_inputs = []
for i in range(len(test_set)-window_len):
    temp_set = test_set[i:(i+window_len)].copy()
    for col in norm_cols:
        temp_set.loc[:, col] = temp_set[col]/temp_set[col].iloc[0] - 1
    LSTM_test_inputs.append(temp_set)

LSTM_training_outputs = {}    
for coin in cryptocurrency_list:
    LSTM_training_outputs['{}'.format(coin)] = (training_set['{}_Close**'.format(coin)][window_len:].values/training_set['{}_Close**'.format(coin)][:-window_len].values)-1

LSTM_test_outputs = {}    
for coin in cryptocurrency_list:
    LSTM_test_outputs['{}'.format(coin)] = (test_set['{}_Close**'.format(coin)][window_len:].values/test_set['{}_Close**'.format(coin)][:-window_len].values)-1

LSTM_training_inputs = [np.array(LSTM_training_input) for LSTM_training_input in LSTM_training_inputs]
LSTM_training_inputs = np.array(LSTM_training_inputs)

LSTM_test_inputs = [np.array(LSTM_test_inputs) for LSTM_test_inputs in LSTM_test_inputs]
LSTM_test_inputs = np.array(LSTM_test_inputs)                

# import the relevant Keras modules
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout

def build_model(inputs, output_size, neurons, activ_func="linear",
                dropout=0.25, loss="mae", optimizer="adam"):
    model = Sequential()

    model.add(LSTM(neurons, input_shape=(inputs.shape[1], inputs.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))

    model.compile(loss=loss, optimizer=optimizer)
    return model
    
# random seed for reproducibility
np.random.seed(202)
model = {}
history = {}
predictions = {}
price = {}

for coin in cryptocurrency_list:
    # initialise model architecture
    model['{}'.format(coin)] = build_model(LSTM_training_inputs, output_size=1, neurons = 20)
    # train model on data
    # note: eth_history contains information on the training error per epoch
    history['{}'.format(coin)] = model['{}'.format(coin)].fit(LSTM_training_inputs, LSTM_training_outputs['{}'.format(coin)], 
                              epochs=50, batch_size=1, verbose=2, shuffle=True)
    predictions['{}'.format(coin)] = model['{}'.format(coin)].predict(LSTM_test_inputs)
    price['{}'.format(coin)] = ((np.transpose(predictions['{}'.format(coin)])+1) * test_set['{}_Close**'.format(coin)].values[:-window_len])[0]

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
    data['{}'.format(coin)] = page.json()
    
    blocktime['{}'.format(coin)] = data['{}'.format(coin)]['Data']['BlockTimeInSeconds']
    blockreward['{}'.format(coin)] = data['{}'.format(coin)]['Data']['BlockReward']
    difficulty['{}'.format(coin)] = data['{}'.format(coin)]['Data']['Difficulty']
    blocks_24h['{}'.format(coin)] = 86400/blocktime['{}'.format(coin)]
    volume_in_24h['{}'.format(coin)] = blocks_24h['{}'.format(coin)] * blockreward['{}'.format(coin)]*price[len(price['{}'.format(coin)])-1]

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

with open('C:\Xampp_2\htdocs\Output.json', 'w') as outfile:
    json.dump(proportion, outfile)

########################### THE END##################################