import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from collections import namedtuple
import random
import math


OBSERVATION_SIZE = 96
INITIAL_BALANCE = 10000

#
MAX_STEPS = 20000
#

## Environment
class TradingEnv(gym.Env):
    def __init__(self, df):
        self.df = df
        self.action_space = spaces.Box(low=np.array([0,0]), high=np.array([3,1]), dtype=np.float16)
        self.observation_space = spaces.Box(low=0, high=1, shape=(5, OBSERVATION_SIZE +2), dtype=np.float16)
        self.visualization = None
        self.total_steps = 0
    
    def _set_total_steps(self, steps):
        self.total_steps = steps

    def _next_observation(self):
        obs = np.array([
            self.df.iloc[self.current_step: self.current_step +
                        OBSERVATION_SIZE,].loc[:,'Open'].values,
            self.df.iloc[self.current_step: self.current_step +
                        OBSERVATION_SIZE,].loc[:,'High'].values,
            self.df.iloc[self.current_step: self.current_step +
                        OBSERVATION_SIZE,].loc[:,'Low'].values ,
            self.df.iloc[self.current_step: self.current_step +
                        OBSERVATION_SIZE,].loc[:,'Close'].values,
            self.df.iloc[self.current_step: self.current_step +
                        OBSERVATION_SIZE,].loc[:,'Volume'].values
        ])
        return torch.from_numpy(obs)[None,None,...]

    def _take_action(self, action_type, action_percentage):
        current_price = random.uniform(df.iloc[self.current_step, ]['Open'], df.iloc[self.current_step, ]["Close"])
        # print("ACTION 0:",action_type ==0)
        # print("ACTION 1:",action_type ==1)
        # print("ACTION 2:",action_type ==2)
        # print("Step:", self.current_step, "action: ",action_type )
        # Buy
        if action_type == 0:
            # print("BUY")
            total_possible = self.balance/current_price
            number_purchase = int(total_possible*action_percentage)
            amount_purchase = number_purchase * current_price

            self.hold += number_purchase
            self.balance -= amount_purchase

            if amount_purchase > 0:
                print("BUY TRADE",{'step': self.current_step,
                                    'price': current_price,
                                    'shares': number_purchase,
                                    'amount': amount_purchase,
                                    'type':'buy' } )
                self.trades.append({'step': self.current_step,
                                    'price': current_price,
                                    'shares': number_purchase,
                                    'amount': amount_purchase,
                                    'type':'buy' })
        # Sell
        elif action_type == 1:
            # print("SELL")
            number_sell = int(self.hold * action_percentage)
            amount_sell = current_price * number_sell
            self.hold -= number_sell
            self.balance += amount_sell

            if amount_sell > 0:
                print("SELL TRADE", {'step': self.current_step,
                                    'price': current_price,
                                    'shares': number_sell,
                                    'amount': amount_sell,
                                    'type':'sell' })
                self.trades.append({'step': self.current_step,
                                    'price': current_price,
                                    'shares': number_sell,
                                    'amount': amount_sell,
                                    'type':'sell' })

        self.net_worth.append(self.balance + self.hold * current_price)

    def get_status(self):
        return torch.DoubleTensor([self.balance, self.hold])[None,...]

    def _get_reward(self):
        new_worth = self.net_worth[-1]
        last_worth = self.net_worth[-2]

        return torch.DoubleTensor([new_worth - last_worth - self.total_steps])

    def step(self, action_type, action_percentage):
        self._take_action(action_type, action_percentage)
        if self.current_step > TOTAL_NUM_OBSERVATION:
            self.current_step = 0
        else:
            self.current_step += 1
        self.total_steps += 1

        reward = self._get_reward()
        done = self.net_worth[-1] <= 0 or self.current_step >= MAX_STEPS
        obs = self._next_observation()

        return obs, reward, done, {}

    def reset(self):
        self.current_step = 0

        self.balance = INITIAL_BALANCE
        self.net_worth = [INITIAL_BALANCE]
        self.hold = 0
        self.trades = []

        return self._next_observation()

    def print_status(self):
        profit = self.net_worth[-1] - INITIAL_BALANCE
        print(f'Step: {self.current_step}\n')
        print(f'Balance: {self.balance}\n')
        print(
            f'Shares held: {self.hold} \n')
        print(
            f'Net worth: {self.net_worth[-1]}\n')
        print(f'Profit: {profit}\n\n')
        return self.trades

    def render(self, **kwargs):
        # Render the environment to the screen
        if self.visualization == None:
            self.visualization = RenderGraph(self.df, kwargs.get('title', None))
        
        #Try removing if statement
        if self.current_step > OBSERVATION_SIZE:
            self.visualization.render(self.current_step,
                                self.net_worth[-1],
                                self.trades,
                                observation_size=OBSERVATION_SIZE)
    def close(self):
        if self.visualization != None:
            self.visualization.close()
            self.visualization = None

Experience = namedtuple('Experience', ('state', 'status', 'action', 'next_state', 'next_status','reward' ))
# Replay Memory
class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity # Set the max capacity of memory
        self.memory = [] # Array to store experiences
        self.push_count = 0 # track how many experiences we pushed
        
    def push(self,experience):
        # If memory still not at capacity
        if len(self.memory) < self.capacity:
            # Put experience in memory
            self.memory.append(experience)
        else:
            # If memory already full
            # Replace the oldest experience with the experience
            self.memory[self.push_count % self.capacity] = experience
        # Increment push count
        self.push_count += 1
    
    # Get random batch of memories
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    # Make sure there are enough experiences in memory
    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size
# Extract Tensors from Experiences
# Input - batch of experiences
def extract_tensors(experiences):
    
    # Transpose to batch of experience
    # EG Instead of three seperate Experience instances, have one instance with 3 values in each tuple
    # This gives one experience instance, with each state, action, next_state, reward having a tuple of all values in batch
    batch = Experience(*zip(*experiences))
    
    # Turn each into their own tensor
    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.status)
    t3 = torch.cat(batch.action)
    t4 = torch.cat(batch.next_state)
    t5 = torch.cat(batch.next_status)
    t6 = torch.cat(batch.reward)
    
    return (t1, t2, t3, t4, t5, t6)


### RL Agent
class DQN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 8, (1,24))
        self.conv2 = nn.Conv2d(8, 56, (5,24))
        self.fc1 = nn.Linear(in_features = 56*1*50 +2, out_features=512)
        self.fc2 = nn.Linear(in_features = 512, out_features=1024)
        self.fc3 = nn.Linear(in_features = 1024, out_features=512)
        self.fc4 = nn.Linear(in_features = 512, out_features=256)
        self.fc5 = nn.Linear(in_features = 256, out_features=128)
        self.fc6 = nn.Linear(in_features = 128, out_features=32)
        self.out = nn.Linear(in_features = 32, out_features=4)

    def forward(self,t,status):
        # Convolutional steps
        if type(t) == 'numpy.ndarray':
            t = torch.from_numpy(t)
        t = F.relu(self.conv1(t))
        t = F.relu(self.conv2(t))
        t = t.view(-1, self.num_flat_features(t))
        # Find the input of linear layers
        # print("t.shape", t.shape)

        # Adding agent info to input
        if type(status) == 'list':
            status = torch.DoubleTensor(status)
        t = torch.cat((t,status),dim=1) 

        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = F.relu(self.fc3(t))
        t = F.relu(self.fc4(t))
        t = F.relu(self.fc5(t))
        t = F.relu(self.fc6(t))
        t = self.out(t)

        # actions = t[:,0:3]
        # amount = t[:,3:]
        actions = t[0:3]
        amount = t[3:]
        seperate_activations = (
            F.softmax(actions),
            F.sigmoid(amount)
        )
        # out = torch.cat(seperate_activations, dim=1) 
        t = torch.cat(seperate_activations) 

        return t

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

                

# PreProcess Dataset
df = pd.read_csv('./data/Coinbase_BTCUSD_1h.csv',skiprows=1)
df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d %I-%p")
df = df.sort_values('Date')
df.rename(columns={'Volume USD': 'Volume'}, inplace=True)

TOTAL_NUM_OBSERVATION = int(df.shape[0]/OBSERVATION_SIZE)

### Hyperparamters
GAMMA = 0.999
LEARNING_RATE = 0.01
TARGET_NET_UPDATE_FREQ = 10
MEMORY_SIZE = 100000
BATCH_SIZE = 10
NUM_EPISODES = 100

EPS_START = 1
EPS_END = 0.01
EPS_DECAY = 0.0001
# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up environment
env = TradingEnv(df)

# Set up policy and target net
policy_net = DQN().double().to(device)
target_net = DQN().double().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
###
# Load parameters that I trained before
state_dict = torch.load('./saved_model/model.pt')
policy_net.load_state_dict(state_dict)
env._set_total_steps(torch.load('./saved_model/total_steps.pt'))
###

# Set up optimizer
optimizer = optim.Adam(params=policy_net.parameters(), lr=LEARNING_RATE)

# Set up memory
memory = ReplayMemory(MEMORY_SIZE)


state = env.reset()

all_rewards = []
all_losses = []

for episode in range(NUM_EPISODES):
    
    eps = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * env.total_steps * EPS_DECAY)
    random_num = random.random()

    if random_num > eps:
        with torch.no_grad():
            current_q_values = policy_net(state,env.get_status()).to(device)
        action_type = current_q_values[0,0:3].argmax().to(device)
        action_amount = current_q_values[0,-1]
    else:
        action_type = torch.LongTensor([random.randrange(3)])[0]
        action_amount = torch.DoubleTensor([random.random()])[0]

    status = env.get_status()
    # print("ACTION AMOUNT:",action_amount)
    next_state, reward, done, info = env.step(action_type, action_amount)
    next_status = env.get_status()

    memory.push(Experience(state, status, action_type[None,...], next_state, next_status, reward[None,...]))

    state = next_state

    all_rewards.append(reward.numpy())

    # See if we can get a sample from replay memory to train policy net
    # If enough experiences - more than batch size
    if memory.can_provide_sample(BATCH_SIZE):
        # Get a sample
        experiences = memory.sample(BATCH_SIZE)

        # Extract state, actions, rewards and next state into their own tensors
        states, statuses, actions, next_states, next_statuses, rewards = extract_tensors(experiences)

        # Get Q-values for state-action pairs from batch as pytorch tensor
        current_all_q_values =  policy_net(states,statuses)
        current_action_q_values = current_all_q_values.gather(dim =1, index=actions.unsqueeze(-1))
        current_amount_q_values = current_all_q_values[:,-1][...,None]
        current_q_values = torch.cat((current_action_q_values,current_amount_q_values),dim=1)
        # Get Q-value for next state in batch - using target_net
        next_all_q_values = target_net(next_states,next_statuses)
        next_action_q_values = next_all_q_values[:,:3].max(dim=1)[0].detach()[...,None]
        next_amount_q_values = next_all_q_values[:,-1][...,None]
        next_q_values = torch.cat((next_action_q_values,next_amount_q_values),dim=1)

        # Calculate target Q value using formula
        target_q_values = (next_q_values * GAMMA) + rewards

        # Get loss between current and target Q values
        loss = F.mse_loss(current_q_values, target_q_values)
        # Set gradients of all wieghts and biases in policy net to 0
        optimizer.zero_grad()
        # Compute gradient of loss for policy net
        loss.backward()
        # Updates weights and loss using gradients calculated above
        optimizer.step()

        all_losses.append(loss)



    # Check if we should update target net
    if episode % TARGET_NET_UPDATE_FREQ ==0:
        target_net.load_state_dict(policy_net.state_dict())
    
    if episode % 50 == 0:
        # Save model
        torch.save(policy_net.state_dict(), './saved_model/model.pt')
        # Save Exploration rate (epsilon)
        torch.save(env.total_steps, './saved_model/total_steps.pt')

    if episode % 10 == 0:
        print("Episode:",episode)
        # print("Loss:", loss)
        print("Reward:", reward)
        plt.subplot(2, 1, 1)
        plt.plot(list(range(0,episode+1)), all_rewards, "b")
        plt.xlabel('episode')
        plt.ylabel('Reward')
        plt.pause(0.05)

        if all_losses:
            plt.subplot(2, 1, 2)
            plt.plot(list(range(0,len(all_losses))), all_losses, "b")
            plt.xlabel('episode')
            plt.ylabel('Loss')
            plt.pause(0.05)
            


trades = env.print_status()

print(trades)
