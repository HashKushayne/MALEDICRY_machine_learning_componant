# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 00:48:08 2018

@author: Hashitha Kushayne
"""

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
    market_info['{}'.format(coin)].dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
    market_info['{}'.format(coin)].columns = ['{}_'.format(coin)+i for i in market_info['{}'.format(coin)].columns[0:]]
    market_info['{}'.format(coin)].isnull().any()
                
g=0
for coin in cryptocurrency_list:
    g+=1
    if g == 1:
        all_market_info = market_info['{}'.format(coin)]
        continue                              
    all_market_info = pd.merge(all_market_info, market_info['{}'.format(coin)], on=['{}_time_period_start'.format(coin)])
del g

market_info = all_market_info
del all_market_info


plt.plot(market_info['zcash_price_open'][:10000])
plt.ylabel('zcash price open')
plt.show()

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
    
    model_data['{}_close_off_high'.format(coin)]=model_data['{}_close_off_high'.format(coin)].fillna(0)
    model_data['{}_close_off_high'.format(coin)] = model_data['{}_close_off_high'.format(coin)].astype('int64')
# we don't need the date columns anymore
split_date = '2018-05-01T00:00:00.0000000Z'

training_set, test_set = model_data[model_data['{}_time_period_start'.format(coin)]<split_date], model_data[model_data['{}_time_period_start'.format(coin)]>=split_date]
training_set = training_set.drop('{}_time_period_start'.format(coin), 1)
test_set = test_set.drop('{}_time_period_start'.format(coin), 1)

window_len = 24
norm_cols = [coin+'_'+metric for coin in cryptocurrency_list for metric in ['price_close','volume_traded']]

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
    LSTM_training_outputs['{}'.format(coin)] = (training_set['{}_price_close'.format(coin)][window_len:].values/training_set['{}_price_close'.format(coin)][:-window_len].values)-1

LSTM_test_outputs = {}    
for coin in cryptocurrency_list:
    LSTM_test_outputs['{}'.format(coin)] = (test_set['{}_price_close'.format(coin)][window_len:].values/test_set['{}_price_close'.format(coin)][:-window_len].values)-1

LSTM_training_inputs = [np.array(LSTM_training_input) for LSTM_training_input in LSTM_training_inputs]
LSTM_training_inputs = np.array(LSTM_training_inputs)

LSTM_test_inputs = [np.array(LSTM_test_inputs) for LSTM_test_inputs in LSTM_test_inputs]
LSTM_test_inputs = np.array(LSTM_test_inputs)                

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
                              epochs=crypto_epoch[coin], batch_size=crypto_batch_size[coin], validation_split=0.20, verbose=2, shuffle=True)
    
    plt.plot(history['{}'.format(coin)].history['loss'])
    plt.plot(history['{}'.format(coin)].history['val_loss'])
    plt.title('model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()
    plt.close()
    plt.gcf().clear()

    predictions['{}'.format(coin)] = model['{}'.format(coin)].predict(LSTM_test_inputs)
    price['{}'.format(coin)] = ((np.transpose(predictions['{}'.format(coin)])+1) * test_set['{}_price_close'.format(coin)].values[:-window_len])[0]
    
    fig, ax1 = plt.subplots(1,1)
    ax1.set_xticks([datetime.date(2018,i+1,1) for i in range(12)])
    ax1.set_xticklabels([datetime.date(2018,i+1,1).strftime('%b %d %Y')  for i in range(12)])
    ax1.plot(model_data[model_data['{}_time_period_start'.format(coin)]>= split_date]['{}_time_period_start'.format(coin)][window_len:].astype(datetime.datetime),
             test_set['{}_price_close'.format(coin)][window_len:], label='Actual')
    ax1.plot(model_data[model_data['{}_time_period_start'.format(coin)]>= split_date]['{}_time_period_start'.format(coin)][window_len:].astype(datetime.datetime),
             ((np.transpose(predictions['{}'.format(coin)])+1) * test_set['{}_price_close'.format(coin)].values[:-window_len])[0], 
             label='Predicted')
    ax1.annotate('MAE: %.4f'%np.mean(np.abs((np.transpose(predictions['{}'.format(coin)])+1)-\
                                            (test_set['{}_price_close'.format(coin)].values[window_len:])/(test_set['{}_price_close'.format(coin)].values[:-window_len]))), 
                 xy=(0.75, 0.9),  xycoords='axes fraction',
                 xytext=(0.75, 0.9), textcoords='axes fraction')
    ax1.set_title('Test Set: Single Timepoint Prediction',fontsize=13)
    ax1.set_ylabel('{} Price ($)'.format(coin),fontsize=12)
    ax1.legend(bbox_to_anchor=(0.1, 1), loc=2, borderaxespad=0., prop={'size': 14})
    plt.show()
    plt.close()
    plt.gcf().clear()
    
    # serialize model to YAML
    model_yaml = model['{}'.format(coin)].to_yaml()
    with open("{}_model.yaml".format(coin), "w") as yaml_file:
        yaml_file.write(model_yaml)
    # serialize weights to HDF5
    model['{}'.format(coin)].save_weights("{}_model.h5".format(coin))
    print("Saved model to disk")
            
    # later...
                
    # load YAML and create model
    yaml_file = open("{}_model.yaml".format(coin), 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model['{}'.format(coin)] = model_from_yaml(loaded_model_yaml)
    # load weights into new model
    loaded_model['{}'.format(coin)].load_weights("{}_model.h5".format(coin))
    print("Loaded model from disk")
        
    # evaluate loaded model on test data
    loaded_model['{}'.format(coin)].compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    #score = loaded_model['{}'.format(coin)].evaluate(X, Y, verbose=0)
    #print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

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