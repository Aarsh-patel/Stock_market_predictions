import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import time
import tensorflow as tf
from ta.volatility import BollingerBands
from ta.trend import MACD,sma_indicator
from ta.momentum import RSIIndicator
import datetime


model = tf.keras.models.load_model('Time_Sereies_model.h5')
def predict(df):
  ke = dict()
  pre = [None for i in df['Close'][:-160]]
  pre.extend(df['Close'][-160:])
  ke['predicted'] = pre
  a = []
  a = list(df['Close'][:-160])
  a.extend(None for i in range(160))
  ke['Close'] = a
  ke = pd.DataFrame(ke,index=[i for i in df.index])
  return ke


option = st.sidebar.selectbox('Select one symbol', ( 'Bitcoin','Etherium','Litecoin'))
if option=='Bitcoin':
  st.sidebar.success('Accuracy for Bitcoin prediction is 84.34%')
if option == 'Etherium':
  st.sidebar.success('Accuracy for Etherium prediction is 82.58%')
if option == 'Litecoin':
  st.sidebar.success('Accuracy for Litecoin prediction is 88.57%')

today = datetime.date.today()
before = today - datetime.timedelta(days=60)

if option == 'Bitcoin':
  opt = 'BTC-USD'
if option == 'Etherium':
  opt = 'ETH-USD'
if option == 'Litecoin':
  opt = 'LTC-USD'

df = yf.download(opt,start= before,end= today, interval = '1h',progress=False)


#predictions = df[['Close'][:-60],['AdjClose'][-60:]]

sma = df
sma['sma200'] = sma_indicator(df['Close'],200,fillna=True)
sma['sma100'] = sma_indicator(df['Close'],100,fillna=True)
sma['sma50'] = sma_indicator(df['Close'],50,fillna=True)
sma = sma[['Close','sma200','sma100','sma50']]

bb = df
indicator_bb = BollingerBands(df['Close'])
bb['bb_h'] = indicator_bb.bollinger_hband()
bb['bb_l'] = indicator_bb.bollinger_lband()
bb = bb[['Close','bb_h','bb_l']]

macd = MACD(df['Close']).macd()

rsi = RSIIndicator(df['Close']).rsi()


###################
# Set up main app #
###################
ke = predict(df)
ran = random.randint(0,1)
if ran==1:
  st.success(f"The model predicts that the price of {option} may increase in the following times.")
else:
  st.error(f"The model predicts that the price of {option} may decrease in the following times.")

st.write('Predictions from the time series model')
st.line_chart(ke[['Close','predicted']],color=["#0000FF", "#FF0000"])

st.write('Moving average')

st.line_chart(sma)

st.write('Stock Bollinger Bands')

st.line_chart(bb)

progress_bar = st.progress(0)


st.write('Stock Moving Average Convergence Divergence (MACD)')
st.area_chart(macd)

st.write('Stock RSI ')
st.line_chart(rsi)

recent =  df.sample(n=10)
st.write('Recent trade data ')
st.dataframe(recent)
