
#%%
''' DEPENDENCIES '''
''' Binance API to get the data from Binance '''
# pip install python-binance

#%%
''' Tensorflow and Keras to execute the neural network operations '''
# pip install tensorflow

#%%

# pip install keras

#%%
''' REQUIRED PACKAGES '''

import pandas as pd
import numpy as np
import time
import csv
from binance.client import Client
from datetime import datetime
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD


#%%
''' To connect the Binance API we need to have bottom two keys passed into Client. 
This is how we connect to the Binance database/server. I could have done this step in
create_df function, but I prefered this to show it and emphasize it. This account that 
I provided the key and secret is an account I created for this project purpose, 
thus I am OK to share them'''

client = Client(
    api_key = 'm8MugRybcSpdBq223NPLvSM3ksD9J8r8ATxnMnWCZCAGRyvGLLtQkmsaWZ5b8u38', 
    api_secret = 'JzFpaKywWsNiMc01b4F8TVVoqzyiCc74SMYY0mPdYKrYkhXtZGaVRjxKNmqQRRmu'
    )

#%%

''' I decided to work with candle stick data. In order to get candle stick data information, 
I have written two functions. "to_formated_df" function, gets the downloaded binance data 
from the csv file and reads it to a data frame in a format that I would like to see and use. 
Second function "create_df" defines timestamps for 500 hours chunks. Binance is allowing 
maximum 500 data points to download for this type of data. I used those chunk information 
in the for loop to get more data. Depending on the value "n" the function downloads so many
chunks and formats them. '''

def to_formated_df(binance_list):
    df = pd.DataFrame(binance_list)
    df = df[df.columns[:6]]
    df.columns =   [
                    'time',
                    'open',
                    'high',
                    'low',
                    'close',
                    'volume'
                    ]
    df = df.apply(pd.to_numeric)
    df['time'] = (df['time']/1000).apply(datetime.fromtimestamp)
    df.set_index(['time'], inplace=True)

    return df


def create_df(n=3):
    now = datetime.now()
    x = datetime.timestamp(datetime.now())
    y = x // 3600
    timestamp = int(y) * 3600
    _500hours = 3600 * 500

    df = pd.DataFrame()
    for i in range(n):
        times = i + 1
        start_time = (timestamp - _500hours * times) * 1000
        candles = client.get_klines(symbol='BTCUSDT', interval=Client.KLINE_INTERVAL_1HOUR, startTime=start_time)
        new_df = to_formated_df(candles)
        df = pd.concat([new_df, df])

    df.to_csv('./data/data.csv')

    return df


#%%

''' I have tried different settings. Eventually using 2500 data_points (rouhgly 5 hours of training) felt like the right decision
for the purpose of this project. Depending on the need, one of the first two lines can be comment-out'''
df = create_df(n=5) # for each n, system gets 500 data points. For instance, to get 10k of data points n=20; or n=5 for 2500 data points.
df = pd.read_csv('./data/data.csv', index_col='time') # In order to use the downloaded data, instead of constantly downloading
df = df[:200] # for Christy. To keep the run short and limited. Coding starts processes at 170th data point. I suppose 30 data points
# would be enough to see how the codes is running. This coding handles each chunk of 500 in about an hour. First chunk is usually faster.
OBS = df.values


#%%


#%%



#%%


class ExperienceReplay(object):
    '''This class gathers and delivers the experience'''
    # to initialize the class
    def __init__(self, MAX_MEMORY=100, discount=.9):
        self.MAX_MEMORY = MAX_MEMORY
        self.memory = list()
        self.discount = discount

    # to save the batch lists into a memory to be retrieved later
    def remember(self, states, trade_over):
        self.memory.append([states, trade_over])
        if len(self.memory) > self.MAX_MEMORY:
            del self.memory[0]

    # to get the batch in order to calculate q-values
    def get_batch(self, model, BATCH_SIZE=10):
        len_memory = len(self.memory)
        NUM_ACTIONS = model.output_shape[-1]
        env_dim = self.memory[0][0][0].shape[1]
        inputs = np.zeros((min(len_memory, BATCH_SIZE), env_dim))
        targets = np.zeros((inputs.shape[0], NUM_ACTIONS))
        # q-values and reinforcement learning step
        for i, idx in enumerate(np.random.randint(0, len_memory, size=inputs.shape[0])):
            state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]
            trade_over = self.memory[idx][1]
            inputs[i:i+1] = state_t
            targets[i] = model.predict(state_t)[0]
            Q_sa = np.max(model.predict(state_tp1)[0])
            if trade_over:
                targets[i, action_t] = reward_t

            else:
                # update rule by Suttons' book
                targets[i, action_t] = reward_t + self.discount * Q_sa

        return inputs, targets


class Trade(object):
    '''This is the trade class to define the environment. Trading starts with a new episode, 
    whenever the agent decides to get into a trade, it takes an action (buy or sell). With the exit_action, 
    trade ends. Reward is given at the end of the episode. The state has 5 features (OHLCV) for the last 24 
    observations and the sign (reward) of the profit_loss. It is a single vector of 121 elements (1, 121). 
    '''
    def __init__(self, df, max_trade_len=1000, init_idx=None):
        self.df = df
        self.max_trade_len = max_trade_len
        
        self.is_over = False
        self.reward = 0
        self.profit_loss_sum = 0
        self.epsilon = 0
        if init_idx == None:
            print('------No init_idx set: stopping------')
            return

        else:
            self.init_idx = init_idx

        self.reset()
        

    def _update_state(self, action):
        
        # Update state here
        self.curr_idx += 1
        # If the agent reaches last data point, repeats at the last state until finishes.
        if self.curr_idx == len(df):
            self.curr_idx = self.curr_idx - 1

        self.curr_time = self.df.index[self.curr_idx]
        self.curr_price = self.df['close'][self.curr_idx]
        self.profit_loss = (self.curr_price - self.entry) * self.position / self.entry
        self._assemble_state()
        
        
        # Update position here
        if action == 0:  
            pass
        
        elif action == 2:
            if self.position == -1:
                self.is_over = True
                self._get_reward()
                self.trade_len = self.curr_idx - self.start_idx
   
            elif self.position == 0:
                self.position = 1
                self.entry = self.curr_price
                self.start_idx = self.curr_idx

            else: 
                pass
            
        elif action == 1:
            if self.position == 1:
                self.is_over = True
                self._get_reward()
                self.trade_len = self.curr_idx - self.start_idx

            elif self.position == 0:
                self.position = -1
                self.entry = self.curr_price
                self.start_idx = self.curr_idx

            else:
                pass
        
    
    def _assemble_state(self):
        # Assemble the state (adding other state features possible)
        state = []
        for i in range(24):
            state += (list(OBS[i + self.curr_idx - 23]))
        
        self.state = np.array([])
        self.state = np.append(self.state,state)
        self.state = np.append(self.state,self.position)
        np.append(self.state, np.sign(self.profit_loss_sum))
        self.state = (np.array(self.state)-np.mean(self.state))/np.std(self.state)

    def _get_reward(self):
        if self.position == 1 and self.is_over:
            profit_loss = (self.curr_price - self.entry)/self.entry
            self.reward = np.sign(profit_loss)-(self.curr_idx - self.start_idx)/1000.

        elif self.position == -1 and self.is_over:
            profit_loss = (-self.curr_price + self.entry)/self.entry
            self.reward = np.sign(profit_loss)-(self.curr_idx - self.start_idx)/1000.

        return self.reward
            
    def observe(self):
        return np.array([self.state])

    def act(self, action):
        self._update_state(action)
        reward = self.reward
        trade_over = self.is_over
        return self.observe(), reward, trade_over

    def reset(self):
        self.profit_loss = 0
        self.entry = 0
        self.curr_idx = self.init_idx
        self.start_idx = self.curr_idx
        self.curr_time = self.df.index[self.curr_idx]
        self.state = []
        self.position = 0
        self._update_state(0)





#%%

def run(df):
    # Hyperparameters
    CAPITAL = 10_000
    START_IDX = 24*7
    EPSILON_START = .036
    EPSILON_MIN = 0.0009
    NUM_ACTIONS = 3 
    MAX_EPISODE = 3_000
    MAX_MEMORY = 30_000
    MAX_STEP = 6
    BATCH_SIZE = 24*7

    env = Trade(df, max_trade_len=1000, init_idx=START_IDX)
    hidden_size = len(env.state)*3
    model = Sequential()
    model.add(Dense(hidden_size, input_shape=(len(env.state),), activation='relu'))
    model.add(Dense(hidden_size, activation='tanh'))
    model.add(Dense(NUM_ACTIONS))
    model.compile(SGD(lr=.005), "mse")

    # To continue training from a previous model, below line should be uncommented
    # model.load_weights("indicator_model.h5")

    # Initialize ExperienceReplay
    exp_replay = ExperienceReplay(MAX_MEMORY=MAX_MEMORY)

    # Training
    win_cnt = 0
    loss_cnt = 0
    wins = []
    losses = []
    profits_losses = []
    outputs = []
    for episode in range(MAX_EPISODE):

        if episode >=1:
            output = [env.curr_time, env.curr_idx, round(duration, 2), round(epsilon, 4), round(CAPITAL, 2), episode, step, env.position, env.entry, env.curr_price, env.trade_len, round(env.profit_loss*100, 2)]
            outputs.append(output)

        if episode == MAX_EPISODE:
            outputs_df = pd.DataFrame(outputs, columns=['time', 'index', 'duration', 'epsilon', 'capital', 'episode', 'step', 'position', 'entry_price', 'exit_price', 'trade_len', 'pnl_ratio'])
            outputs_df.to_csv('./data/outputs.csv')

        start = time.time()
        epsilon = EPSILON_START**(np.log10(episode))
        if epsilon < EPSILON_MIN:
            epsilon = EPSILON_MIN

        if env.curr_idx >= len(df) - 1:
            outputs_df = pd.DataFrame(outputs, columns=['time', 'index', 'duration', 'epsilon', 'capital', 'episode', 'step', 'position', 'entry_price', 'exit_price', 'trade_len', 'pnl_ratio'])
            outputs_df.to_csv('./data/outputs.csv')
            break
        
        env = Trade(df, max_trade_len=1000, init_idx=env.curr_idx)
        loss = 0.
        env.reset()
        trade_over = False
        input_t = env.observe()

        step = 0
        while not trade_over:
            step += 1
            input_tm1 = input_t
            
            if np.random.rand() <= epsilon or env.curr_idx == (len(df)-1):
                action = np.random.randint(0, NUM_ACTIONS, size=1)[0]
                
                if env.position == 0:
                    
                    if action == 2:
                        exit_action = 1
                    
                    elif action == 1:
                        exit_action = 2

            elif env.position == 0:
                q = model.predict(input_tm1)
                action = np.argmax(q[0])
                
                if action:
                    exit_action = np.argmin(q[0][1:])+1
                
            elif step > MAX_STEP:
                action = exit_action
                
            elif env.position:
                q = model.predict(input_tm1)
                action = np.argmax(q[0])

            input_t, reward, trade_over = env.act(action)
            if reward > 0:
                win_cnt += 1
            
            elif reward < 0:
                loss_cnt += 1

            if action or len(exp_replay.memory) < 20 or np.random.rand() < 0.1:
                exp_replay.remember([input_tm1, action, reward, input_t], trade_over)

            inputs, targets = exp_replay.get_batch(model, BATCH_SIZE=BATCH_SIZE)
            env.profit_loss_sum = sum(profits_losses)
            zz = model.train_on_batch(inputs, targets)
            loss += zz
            follow = ('step:', step, 'idx:', env.curr_idx, 'price', env.curr_price)
            print(follow)

        CAPITAL = CAPITAL * (1 + env.profit_loss)
        end = time.time()
        duration = end - start
        # Follow the steps on terminal
        prt_str = (f"Episode {episode:03d}  |  position entry point: {env.entry}  |  Loss {loss:.2f}  |  pos {env.position}  |  len {env.trade_len}  |  profit_loss {(sum(profits_losses)+env.profit_loss)*100:.2f}% @ {env.profit_loss*100:.2f}%  |  CAPITAL {CAPITAL:.2f}  |  eps {epsilon:.4f}  |  {env.curr_time}  |  duration:{duration:.2f} seconds")
        print(prt_str)
        profits_losses.append(env.profit_loss)
        if not episode%10:
            print('----saving weights-----')
            model.save_weights("./data/indicator_model.h5", overwrite=True)
    
#%%

if __name__ == "__main__":
    run(df)

# %%

