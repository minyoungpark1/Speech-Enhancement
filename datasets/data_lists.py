#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 14:14:54 2023

@author: minyoungpark
"""


INDICATORS = [
    "macd",
    "atr", # average true range
    'cci', # commodity channel index
    'wr', # WR - Williams Overbought/Oversold Index
    'rsi', # Relative Strength Index
    'adx', # average directional index (ADX)
    "close_30_sma",
    "close_60_sma",
]

TICKERS = [
    'AAPL',
    'MSFT',
    'AMZN',
    'GOOGL',
    'BRK-B',
    # 'GOOG',
    'UNH',
    'TSLA', # missing
    'JNJ',
    'XOM',
    'NVDA',
    'JPM',
    'PG',
    'V', # missing
    'HD',
    'META']

INDEXES = ['FRED:PCE,CPIAUCSL,ICSA,UMCSENT,HSN1F,UNRATE,M2SL,BAMLH0A0HYM2,DFII10', 
              'US500', 
              'DJI', 
              'VIX', 
              # 'BTC/USD',
              'USD/EUR',
              'USD/CNY',
              'NG=F', 
              'GC=F', 
              'SI=F', 
              'HG=F', 
              'CL=F',
              '^IRX', # 13 Week Treasury Bill
              '^FVX', # Treasury Yield 5 Years
              '^TNX', # Treasury Yield 10 Years
              '^TYX', # Treasury Yield 30 Years
              ]
