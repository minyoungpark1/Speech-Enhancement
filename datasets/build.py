#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 14:22:30 2023

@author: minyoungpark
"""

import holidays
import pandas as pd
import numpy as np

from finai.datasets import build
from .data_lists import INDICATORS, TICKERS, INDEXES
from dateutil.relativedelta import relativedelta
import datetime as dt


def get_dataframe(dates):
    
    if not isinstance(dates[0],dt.datetime):
        dates = [dt.datetime.strptime(dates[0], '%Y-%m-%d'), 
                 dt.datetime.strptime(dates[1], '%Y-%m-%d')]
        
    expanded_dates = [dates[0]-relativedelta(days=365), dates[1]]
    df_asset = build.build_asset_dataframe(INDICATORS, TICKERS, expanded_dates)
    
    df_asset = build.process_asset_dataframe(df_asset, INDICATORS, normalize=False, filter=False, 
                                percent_change=True, use_turbulence=True)
    
    df_fred, df_index, neg_balance = build.build_index_dataframe(INDEXES, expanded_dates)
    df_index = build.process_index_dataframe(df_fred, df_index, df_asset)

    df = df_asset.copy()
    df = df.reset_index().rename(columns={'index':'time_idx'})
    df_index_copy = df_index.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    start_date = df['date'].iloc[0]
    end_date = df['date'].iloc[-1]
    
    date_array = pd.date_range(start=start_date,end=end_date)
    nyse_holidays = holidays.NYSE()
    
    near_holiday_dates = np.zeros((len(date_array), ), dtype=int)
    
    for i in range(5, len(date_array)-5):
        if date_array[i] in nyse_holidays:
            near_holiday_dates[-5+i:5+i] = np.arange(0, 10, dtype=int)
        
    df_holiday = pd.DataFrame(data={'date': date_array,
                                    'holiday': near_holiday_dates})
    df_holiday = df_holiday.set_index('date')
    df_index_copy = df_index_copy.set_index('date')
    df_index_copy = df_index_copy.drop('tic', axis=1)
    
    df['holiday'] = 0
    df['holiday'] = df_holiday.loc[df['date']]['holiday'].values
    df['holiday'] = df['holiday'].astype(str).astype("category")
    df['day'] = df['day'].astype(int).astype(str).astype("category")
    
    
    df["month"] = df.date.dt.month.map("{:02d}".format).astype("category")  # categories have be strings
    df["day_of_month"] = df.date.dt.day.map("{:02d}".format).astype("category")  # categories have be strings
    for col in df_index_copy.columns:
        df[col] = df_index_copy.loc[df['date']][col].values
    
    df = df[df.date >= dates[0]]
    return df