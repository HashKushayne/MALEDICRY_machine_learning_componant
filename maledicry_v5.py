# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 11:26:13 2018

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
                           
cryptocurrency_list = ['zcash']
market_info = {}
for coin in cryptocurrency_list:
    market_info['{}'.format(coin)]  = pd.read_csv('{}.csv'.format(all_cryptocurrency_list[coin]))
    market_info['{}'.format(coin)].columns = ['{}_'.format(coin)+i for i in market_info['{}'.format(coin)].columns[0:]]
    print(market_info['{}'.format(coin)].isnull().any())
                
g=0
for coin in cryptocurrency_list:
    g+=1
    if g == 1:
        all_market_info = market_info['{}'.format(coin)]
        continue                              
    all_market_info = pd.merge(all_market_info, market_info['{}'.format(coin)], on=['Date'])
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

for coin in cryptocurrency_list:
    model_data = market_info[[coin+'_'+metric for coin in cryptocurrency_list 
                          for metric in ['time_period_start','price_close','volume_traded','close_off_high','volatility','day_diff']]]
# need to reverse the data frame so that subsequent rows represent later timepoints
for coin in cryptocurrency_list:
    model_data = model_data.sort_values(by='{}_time_period_start'.format(coin))
    print(model_data.isnull().any())
    model_data['{}_close_off_high'.format(coin)]=model_data['{}_close_off_high'.format(coin)].fillna(0)
    model_data['{}_close_off_high'.format(coin)] = model_data['{}_close_off_high'.format(coin)].astype('int64')
    

training_set = model_data
training_set = training_set.drop('{}_time_period_start'.format(coin), 1)

window_len = 24
norm_cols = [coin+'_'+metric for coin in cryptocurrency_list for metric in ['price_close','volume_traded']]

LSTM_training_inputs = []
for i in range(len(training_set)-window_len):
    temp_set = training_set[i:(i+window_len)].copy()
    for col in norm_cols:
        temp_set.loc[:, col] = temp_set[col]/temp_set[col].iloc[0] - 1
    LSTM_training_inputs.append(temp_set)

LSTM_training_outputs = {}    
for coin in cryptocurrency_list:
    LSTM_training_outputs['{}'.format(coin)] = (training_set['{}_price_close'.format(coin)][window_len:].values/training_set['{}_price_close'.format(coin)][:-window_len].values)-1

LSTM_training_inputs = [np.array(LSTM_training_input) for LSTM_training_input in LSTM_training_inputs]
LSTM_training_inputs = np.array(LSTM_training_inputs)



# import the relevant Keras modules
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import model_from_yaml
import os



def build_model(inputs, units, output_size, neurons, activ_func="linear",
                dropout=0.25, loss="mae", optimizer="adam"):
    model = Sequential()

    model.add(LSTM(neurons, return_sequences = True, input_shape=(inputs.shape[1], inputs.shape[2])))
    model.add(Dropout(dropout))
    
    model.add(LSTM(units = units, return_sequences = True))
    model.add(Dropout(dropout))
    
    model.add(LSTM(units = units))
    model.add(Dropout(dropout))
    
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))

    model.compile(loss=loss, optimizer=optimizer, metrics=['mse', 'mape', 'msle'])
    return model

# random seed for reproducibility
np.random.seed(202)
model = {}
history = {}
predictions = {}
price = {}
crypto_neurons = {'litecoin' : 80,
                  'vertcoin' : 50,
                  'zcash' : 75}
crypto_epoch = {'litecoin' : 110,
                'vertcoin' : 110,
                'zcash' : 110}
crypto_batch_size = {'litecoin' : 100,
                     'vertcoin' : 100,
                     'zcash' : 100}
loaded_model = {}
for coin in cryptocurrency_list:
    # initialise model architecture
    model['{}'.format(coin)] = build_model(LSTM_training_inputs, output_size=1, neurons = 20, units = crypto_neurons[coin])
    # train model on data
    # note: eth_history contains information on the training error per epoch
    history['{}'.format(coin)] = model['{}'.format(coin)].fit(LSTM_training_inputs, LSTM_training_outputs['{}'.format(coin)], 
                              epochs=crypto_epoch[coin], batch_size=crypto_batch_size[coin], verbose=2, shuffle=True)
   
    # serialize model to YAML
    model_yaml = model['{}'.format(coin)].to_yaml()
    with open("{}_model.yaml".format(coin), "w") as yaml_file:
        yaml_file.write(model_yaml)
    # serialize weights to HDF5
    model['{}'.format(coin)].save_weights("{}_model.h5".format(coin))
    print("Saved model to disk")
      


