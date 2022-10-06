from operator import length_hint
from matplotlib.pyplot import close
from sklearn.model_selection import PredefinedSplit
from transformers import Data2VecAudioConfig, TFBlenderbotSmallPreTrainedModel
import yfinance as yf
import random
import matplotlib.pyplot as plt
from statistics import mean
import os
import numpy as np
import pandas as pd 
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from datetime import date,timedelta
import time
#spy_ticker = yf.ticker("SPY")

def Ticker(user):
   
    data = yf.download (tickers = user, start='2021-07-08',end='2022-09-30',interval='1d')
    
    Close =  data['Close']

    Percentage = [] 
    for item in range(1,len(Close)):
        previous = Close[item-1]
        current = Close[item]
        percentage_value = ((current-previous)/previous)
        Percentage.append(percentage_value)

    Percentage_length = len(Percentage)

    #lists for probability storage
    prob_positive_half = []
    prob_negative_half = []
    prob_half_pone = []
    prob_half_none = []
    prob_pone = []
    prob_none = []

    for index in range(Percentage_length):
        if index <= Percentage_length-2:
            if 0<Percentage[index]<=0.01:
                prob_positive_half.append(Percentage[index+1])
            elif 0.01<Percentage[index]<=0.02:
                prob_half_pone.append(Percentage[index+1])
            elif 0.02<Percentage[index]:
                prob_pone.append(Percentage[index+1])
            elif -0.01<Percentage[index]<=0:
                prob_negative_half.append(Percentage[index+1])
            elif -0.02<Percentage[index]<=-0.01:
                prob_half_none.append(Percentage[index+1])
            else:
                prob_none.append(Percentage[index+1])


    
    average_point = []
    average = {}
    for a in range(10):
        last_value = Percentage[-1]
        close_value = Close[-1]
        projected = 0
        random_roll = 0
        while projected<100:
            if 0<last_value<=0.01:
                random_roll = random.choice(prob_positive_half)
            elif 0.01<last_value<=0.02:
                random_roll = random.choice(prob_half_pone)
            elif 0.01<last_value:
                random_roll = random.choice(prob_pone)
            elif -0.01<last_value<=0:
                random_roll = random.choice(prob_negative_half)
            elif -0.02<last_value<=0.01:
                random_roll = random.choice(prob_half_none)
            else:
                random_roll = random.choice(prob_none)
            last_value = random_roll
            close_value = (close_value*last_value)+close_value
        
            if projected in average:
                average[projected].append(close_value)
            else:
                average[projected] = [close_value]
            projected +=1
        
        

    for item in average:
        average_point.append(mean(average[item]))
    return average_point
    

def LSTM_Module(user):
    data = yf.download(tickers = user, start='2000-01-01',end='2022-06-30',interval='1d')
    data_actual = yf.download(tickers = user, start='2022-07-01',end='2022-10-05',interval='1d')
    data_forecast = yf.download(tickers = user, start='2000-01-01',end=date.today(),interval='1d')


    Close = data['Close'].values
    length = len(Close)
    Close = Close.reshape(-1,1)
    

    Close_actual = data_actual['Close'].values
    Close_actual = Close_actual.reshape(-1,1)
    
    Close_forecast = data_forecast['Close'].values
    lastsixty = Close_forecast[-100:]
    Close_forecast = Close_forecast.reshape(-1,1)

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_trained = scaler.fit_transform(Close)
    Close_forecast = scaler.fit_transform(Close_forecast)
    #Close_forecast = Close_forecast.reshape(Close_forecast.shape[0],)
    

    x_train = []
    y_train = []

    #prediction and comparison for current data sets
    for i in range (100, len(data_forecast)-5,1):
        x_train.append(Close_forecast[i-100:i,0])
        y_train.append(Close_forecast[i:i+5,0])
    x_train=np.array(x_train)
    y_train= np.array(y_train)
    x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
    model = Sequential()
    model.add(LSTM(units=100,return_sequences=True,input_shape=(x_train.shape[1],1)))
    model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train.shape[1],1)))
    model.add(LSTM(units=50,return_sequences=False))
    model.add(Dense(5))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
    StartTime= time.time()
    model.fit(x_train,y_train,epochs=100,batch_size=32)
    EndTime = time.time()
    total = round((EndTime-StartTime)/60)
    """
    This code below can be used to train and test within historical data prices

    dataset_total = pd.concat((data['Close'],data_actual['Close']),axis=0)
    inputs = dataset_total[len(dataset_total)-len(data_actual)-60:].values
    inputs = inputs.reshape(-1,1)
    inputs = scaler.transform(inputs)
    x_test = []
    for i in range(60,len(inputs)):
        x_test.append(inputs[i-60:i,0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
    prediction = model.predict(x_test)
    prediction = scaler.inverse_transform(prediction) 
    """
    
    #multi-step LSTM for forescasting future values
    lastsixty = lastsixty.reshape(-1,1)
    lastsixty = scaler.transform(lastsixty)
    lastsixty = lastsixty.reshape(1,lastsixty.shape[0],lastsixty.shape[1])
    prediction_future = model.predict(lastsixty)
    prediction_future = scaler.inverse_transform(prediction_future)
    

    return prediction_future, total

    
def test_prob(user):
    test_value = 0
    while test_value<100:
        plt.plot(Ticker(user))
        test_value +=1
    plt.xlabel('Days(100)')
    plt.ylabel('SPY Price')
    plt.show()
    
def LSTM_test(user):
    dataset,timetotal= LSTM_Module(user)
    #plt.plot(dataset)
    #plt.plot(correct)
    #plt.show()
    print(dataset)
    print(timetotal,'Minutes')
    
LSTM_test('SPY')

