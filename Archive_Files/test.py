# import pandas as pd
# import numpy as np

# df = pd.read_csv('./data/Coinbase_BTCUSD_1h.csv',skiprows=1)
# df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d %I-%p")
# df = df.sort_values('Date')
# print(df)
# df.info()
# print(df['Date'].iloc[1:10])
# obs = df.iloc[1:10].drop(['Date','Symbol'], axis=1).values
# print(obs)


# df = pd.read_csv('./data/Coinbase_BTCUSD_1h.csv',skiprows=1)
# df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d %I-%p")
# df = df.sort_values('Date')
# df.rename(columns={'Volume USD': 'Volume'}, inplace=True)
# import matplotlib.dates as mdates

# def date2num(date):
#     converter = mdates.strpdate2num('%Y-%m-%d-%H')
#     return converter(date)
# print(df['Date'].iloc[range(1,10)].apply(lambda x: x.strftime('%Y-%m-%d-%H') ))
# dates = np.array([date2num(x)
#                           for x in  df['Date'].iloc[range(1,10)].apply(lambda x: x.strftime('%Y-%m-%d-%H') )])
                          
                          
#                           df = pd.read_csv('./data/MSFT.csv')
# df = df.sort_values('Date')
from gym import spaces
import numpy as np

print(spaces.Box(low=np.array([0,0]), high=np.array([3,1]), dtype=np.float16))