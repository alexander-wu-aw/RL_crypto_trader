### This was my own re-writing of the tutorial code

import gym
from gym import spaces
import numpy as np
import pandas as pd
import random

OBSERVATION_SIZE = 40
INITIAL_BALANCE = 10000
MAX_STEPS = 20000

class TradingEnv(gym.Env):
    metadata = {'render.modes': ['live', 'file', 'none']}

    def __init__(self, df):
        self.df = df

        self.action_space = spaces.Box(low=np.array([0,0]), high=np.array([3,1]), dtype=np.float16)
        self.observation_space = spaces.Box(low=0, high=1, shape=(5, OBSERVATION_SIZE +2), dtype=np.float16)
        
        self.visualization = None

    def _take_action(self, action):
        current_price = random.uniform(self.df.loc[self.current_step, "Open"], self.df.loc[self.current_step,"Close"])

        action_type = action[0]
        perc_amount = action[1]

        # Buy
        if action_type < 1:
            total_possible = int(self.balance/current_price)
            number_purchase = int(total_possible*perc_amount)
            total_purchase = number_purchase * current_price

            self.held += number_purchase
            self.balance -= total_purchase

            if total_purchase > 0:
                self.trades.append({'step': self.current_step,
                                    'shares': number_purchase,
                                    'total': total_purchase,
                                    'type':'buy' })
        # Sell
        elif action_type < 2:
            number_sell = int(self.held * perc_amount)
            total_sell = current_price * number_sell
            self.held -= number_sell
            self.balance += total_sell

            if total_sell > 0:
                self.trades.append({'step': self.current_step,
                                    'shares': number_sell,
                                    'total': total_sell,
                                    'type':'sell' })

        self.net_worth = self.balance + self.held * current_price


    def _next_observation(self):
        # obs = self.df.iloc[self.current_step:self.current_step +OBSERVATION_SIZE].drop(['Date','Symbol'], axis=1).values
        # return obs

        frame = np.zeros((5, OBSERVATION_SIZE +1))

        # Get the stock data points for the last 5 days and scale to between 0-1
        np.put(frame, [0, 4], [
            self.df.loc[self.current_step: self.current_step +
                        OBSERVATION_SIZE, 'Open'].values,
            self.df.loc[self.current_step: self.current_step +
                        OBSERVATION_SIZE, 'High'].values,
            self.df.loc[self.current_step: self.current_step +
                        OBSERVATION_SIZE, 'Low'].values ,
            self.df.loc[self.current_step: self.current_step +
                        OBSERVATION_SIZE, 'Close'].values,
            self.df.loc[self.current_step: self.current_step +
                        OBSERVATION_SIZE, 'Volume'].values
        ])

        # Append additional data and scale each value to between 0-1
        obs = np.append(frame, [
            [self.balance ],
            [0 ],
            [self.held ],
            [0],
            [0],
        ], axis=1)

        print(obs)
        return obs

    def step(self, action):
        self._take_action(action)
        self.current_step += 1

        reward = self.net_worth + self.current_step
        done = self.net_worth <= 0 or self.current_step >= MAX_STEPS
        obs = self._next_observation()

        #print(self.net_worth)
        return obs, reward, done, {}


    def reset(self):
        self.current_step=0

        self.balance = INITIAL_BALANCE
        self.net_worth = INITIAL_BALANCE
        self.held = 0
        self.trades = []

        return self._next_observation()

    def render(self, mode='live', **kwargs):
        if self.visualization == None:
            self.visualization = TradingGraph(self.df, kwargs.get('title', None))
        
        #Try removing if statement
        if self.current_step > OBSERVATION_SIZE:
            self.visualization.render(self.current_step,
                                self.net_worth,
                                self.trades,
                                observation_size=OBSERVATION_SIZE)

    def close(self):
        if self.visualization != None:
            self.visualization.close()
            self.visualization = None

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import style

# finance module is no longer part of matplotlib
# see: https://github.com/matplotlib/mpl_finance
from mpl_finance import candlestick_ochl as candlestick

style.use('dark_background')

VOLUME_CHART_HEIGHT = 0.33

UP_COLOR = '#27A59A'
DOWN_COLOR = '#EF534F'
UP_TEXT_COLOR = '#73D3CC'
DOWN_TEXT_COLOR = '#DC2C27'


def date2num(date):
    converter = mdates.strpdate2num('%Y-%m-%d-%H')
    return converter(date)


class TradingGraph:
    """A stock trading visualization using matplotlib made to render OpenAI gym environments"""

    def __init__(self, df, title=None):
        self.df = df
        self.net_worths = np.zeros(len(df['Date']))

        # Create a figure on screen and set the title
        fig = plt.figure()
        fig.suptitle(title)

        # Create top subplot for net worth axis
        self.net_worth_ax = plt.subplot2grid(
            (6, 1), (0, 0), rowspan=2, colspan=1)

        # Create bottom subplot for shared price/volume axis
        self.price_ax = plt.subplot2grid(
            (6, 1), (2, 0), rowspan=8, colspan=1, sharex=self.net_worth_ax)

        # Create a new axis for volume which shares its x-axis with price
        self.volume_ax = self.price_ax.twinx()

        # Add padding to make graph easier to view
        plt.subplots_adjust(left=0.11, bottom=0.24,
                            right=0.90, top=0.90, wspace=0.2, hspace=0)

        # Show the graph without blocking the rest of the program
        plt.show(block=False)

    def _render_net_worth(self, current_step, net_worth, step_range, dates):
        # Clear the frame rendered last step
        self.net_worth_ax.clear()

        # Plot net worths
        self.net_worth_ax.plot_date(
            dates, self.net_worths[step_range], '-', label='Net Worth')

        # Show legend, which uses the label we defined for the plot above
        self.net_worth_ax.legend()
        legend = self.net_worth_ax.legend(loc=2, ncol=2, prop={'size': 8})
        legend.get_frame().set_alpha(0.4)

        last_date = date2num(self.df['Date'].iloc[current_step].strftime('%Y-%m-%d-%H'))
        last_net_worth = self.net_worths[current_step]

        # Annotate the current net worth on the net worth graph
        self.net_worth_ax.annotate('{0:.2f}'.format(net_worth), (last_date, last_net_worth),
                                   xytext=(last_date, last_net_worth),
                                   bbox=dict(boxstyle='round',
                                             fc='w', ec='k', lw=1),
                                   color="black",
                                   fontsize="small")

        # Add space above and below min/max net worth
        self.net_worth_ax.set_ylim(
            min(self.net_worths[np.nonzero(self.net_worths)]) / 1.25, max(self.net_worths) * 1.25)

    def _render_price(self, current_step, net_worth, dates, step_range):
        self.price_ax.clear()

        # Format data for OHCL candlestick graph
        candlesticks = zip(dates,
                           self.df['Open'].values[step_range], self.df['Close'].values[step_range],
                           self.df['High'].values[step_range], self.df['Low'].values[step_range])

        # Plot price using candlestick graph from mpl_finance
        candlestick(self.price_ax, candlesticks, width=1/24,
                    colorup=UP_COLOR, colordown=DOWN_COLOR)

        last_date = date2num(self.df['Date'].iloc[current_step].strftime('%Y-%m-%d-%H'))
        last_close = self.df['Close'].values[current_step]
        last_high = self.df['High'].values[current_step]

        # Print the current price to the price axis
        self.price_ax.annotate('{0:.2f}'.format(last_close), (last_date, last_close),
                               xytext=(last_date, last_high),
                               bbox=dict(boxstyle='round',
                                         fc='w', ec='k', lw=1),
                               color="black",
                               fontsize="small")

        # Shift price axis up to give volume chart space
        ylim = self.price_ax.get_ylim()
        self.price_ax.set_ylim(ylim[0] - (ylim[1] - ylim[0])
                               * VOLUME_CHART_HEIGHT, ylim[1])

    def _render_volume(self, current_step, net_worth, dates, step_range):
        self.volume_ax.clear()

        volume = np.array(self.df['Volume'].values[step_range])

        pos = self.df['Open'].values[step_range] - \
            self.df['Close'].values[step_range] < 0
        neg = self.df['Open'].values[step_range] - \
            self.df['Close'].values[step_range] > 0

        # Color volume bars based on price direction on that date
        self.volume_ax.bar(dates[pos], volume[pos], color=UP_COLOR,
                           alpha=0.4, width=1/24, align='center')
        self.volume_ax.bar(dates[neg], volume[neg], color=DOWN_COLOR,
                           alpha=0.4, width=1/24, align='center')

        # Cap volume axis height below price chart and hide ticks
        self.volume_ax.set_ylim(0, max(volume) / VOLUME_CHART_HEIGHT)
        self.volume_ax.yaxis.set_ticks([])

    def _render_trades(self, current_step, trades, step_range):
        for trade in trades:
            if trade['step'] in step_range:
                date = date2num(self.df['Date'].iloc[trade['step']].strftime('%Y-%m-%d-%H'))
                high = self.df['High'].values[trade['step']]
                low = self.df['Low'].values[trade['step']]

                if trade['type'] == 'buy':
                    high_low = low
                    color = UP_TEXT_COLOR
                else:
                    high_low = high
                    color = DOWN_TEXT_COLOR

                total = '{0:.2f}'.format(trade['total'])

                # Print the current price to the price axis
                self.price_ax.annotate(f'${total}', (date, high_low),
                                       xytext=(date, high_low),
                                       color=color,
                                       fontsize=8,
                                       arrowprops=(dict(color=color)))

    def render(self, current_step, net_worth, trades, observation_size=40):
        self.net_worths[current_step] = net_worth

        window_start = max(current_step - observation_size, 0)
        step_range = range(window_start, current_step + 1)

        # Format dates as timestamps, necessary for candlestick graph
        dates = np.array([date2num(x)
                          for x in  self.df['Date'].iloc[step_range].apply(lambda x: x.strftime('%Y-%m-%d-%H') )])

        self._render_net_worth(current_step, net_worth, step_range, dates)
        self._render_price(current_step, net_worth, dates, step_range)
        self._render_volume(current_step, net_worth, dates, step_range)
        self._render_trades(current_step, trades, step_range)

        # Format the date ticks to be more easily read
        self.price_ax.set_xticklabels(self.df['Date'].iloc[step_range].apply(lambda x: x.strftime('%Y-%m-%d %H:00') ), rotation=45,
                                      horizontalalignment='right')

        # Hide duplicate net worth date labels
        plt.setp(self.net_worth_ax.get_xticklabels(), visible=False)

        # Necessary to view frames before they are unrendered
        plt.pause(0.001)

    def close(self):
        plt.close()


import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

# from env.StockTradingEnv import StockTradingEnv

import pandas as pd

# df = pd.read_csv('./data/Coinbase_BTCUSD_1h.csv',skiprows=1)
# df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d %I-%p")
# df = df.sort_values('Date')
# df.rename(columns={'Volume USD': 'Volume'}, inplace=True)
df = pd.read_csv('./data/Coinbase_BTCUSD_1h.csv',skiprows=1)
df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d %I-%p")
df = df.sort_values('Date')
df.rename(columns={'Volume USD': 'Volume'}, inplace=True)


# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: TradingEnv(df)])

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=50)

obs = env.reset()
for i in range(len(df['Date'])):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render(title="Bitcoin on Coinbase")