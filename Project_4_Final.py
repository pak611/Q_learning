

'''
!pip install tensorflow==2.30
!pip install gym
!pip install keras
!pip install keras-r12
!pip install matplotlib
!apt-get install -y xvfb python-opengl ffmpeg > /dev/null 2>&1
!pip install colabgymrender


%%bash
# install required system dependencies
apt-get install -y xvfb x11-utils

# install required python dependencies (might need to install additional gym extras depending)
pip install gym[box2d]==0.17.* pyvirtualdisplay==0.2.* PyOpenGL==3.1.* PyOpenGL-accelerate==3.1.*



'''

import pyvirtualdisplay


_display = pyvirtualdisplay.Display(visible=False,  # use False with Xvfb
                                    size=(1400, 900))
_ = _display.start()



import gym
from gym.version import VERSION
import numpy as np 
import tensorflow as tf


import matplotlib.pyplot as plt
from IPython import display as ipythondisplay


!apt-get install -y xvfb python-opengl ffmpeg > /dev/null 2>&1
!pip install -U colabgymrender


class Agent:
  def __init__(self, obs_shape, act_size):
                self.obs_shape = obs_shape
                self.act_size = act_size
                


  def network(self, train=True):

    print('network called')
    print('input is', input)
    inputs = tf.keras.Input(shape=(self.obs_shape,), name="input")

    #x = tf.keras.layers.Conv1D(filters = 32,kernel_size = 3, activation='relu', input_shape=self.obs_shape[1:])(inputs)

    x = tf.keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU(), name="dense_1")(inputs)
    
    x = tf.keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU(), name="dense_2")(x)
    #x = tf.keras.layers.Conv1D(64, activation=tf.keras.layers.LeakyReLU(), name="dense_2")(x)
    #x = tf.keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU(), name="dense_3")(x)
    outputs = tf.keras.layers.Dense(self.act_size, name="output")(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name="nn",trainable=train) 
    return model



class Util:
  def __init__(self):
    self.optimizer = tf.keras.optimizers.Adam(learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(0.01, decay_steps=50, decay_rate=0.9))
    self.history = []


  def record_history(self, current_state, action, reward, next_state): 
    self.history.append([current_state, action, reward, next_state])

    
  def td_loss(self, nn, discount=2.0): 
    loss = []
    for current_state, action, reward, next_state in self.history: 
      binary_action = [0.0] * nn.output.shape[1]
      binary_action[action] = 1.0
      binary_action = tf.constant([binary_action])
      q_current = nn(tf.convert_to_tensor([current_state]))
      max_q_next = tf.math.reduce_max(nn(tf.convert_to_tensor([next_state])))
      loss.append(tf.math.square((reward + discount * max_q_next - q_current) * binary_action))
            
    return tf.math.reduce_mean(loss, axis=0)


  def update_model(self, nn):
    with tf.GradientTape() as tape:
        loss = self.td_loss(nn)
    grads = tape.gradient(loss, nn.trainable_variables) 
    self.optimizer.apply_gradients(zip(grads, nn.trainable_variables))
    self.history = []




env = gym.make('CartPole-v1')

#env = gym.make("CartPole-v0")
#env = Recorder(env, <directory>, <fps>)
agent = Agent(4, 2).network()
utility = Util()



reward_list = []
# trainf
epsilon = 0.01
i, early_stop = 0, 0
n_game = 1000
while i < n_game:
  cum_reward = 0
  print('i=', i)
  current_state = env.reset()
  step = 0
  #env.render()
  while True:
    #print('ITS TRUE')
    if np.random.uniform() < epsilon:
      action = env.action_space.sample()
      #print('action is', action)
    else:


      #print('calling agent')

      #current_state = np.asarray((1,2,3,4))

      #print('current_state', type(current_state))
      #print('current_state', current_state.shape)
      #print('current_state', current_state)

      action = tf.math.argmax(tf.reshape(agent(tf.convert_to_tensor([current_state])), [-1])).numpy()

    next_state, reward, done, info = env.step(action)
    step += 1
    utility.record_history(current_state, action, reward, next_state)
    current_state = next_state

    cum_reward += reward


    if done == True:
        #print('cum_reward=', cum_reward)
        reward_list.append([i,cum_reward])

    if len(utility.history) == 50:
        utility.update_model(agent) 
    epsilon = max(epsilon * 0.99, 0.05)

    if done:
      #print(i, step)
      i += 1
      if step >= 500:
        early_stop += 1 
      else:
        early_stop = 0 
      if early_stop >= 10:
        i = n_game
      break




reward_array = np.array(reward_list)
x = reward_array[:,0]
y = reward_array[:,1]


#m,b = np.polyfit(x, y, 1)
plt.ylim(0, 300)
plt.plot(x,y)
plt.xlabel('episodes')
plt.ylabel('reward')
#plt.plot(x, m*x + b)


#------------------------------------------------------------------------

import gym
import numpy as np
import matplotlib.pyplot as plt
from IPython import display as ipythondisplay
import pyvirtualdisplay
from pyvirtualdisplay import Display

import os
import matplotlib.pyplot as plt

import random
#import tflearn
#from tflearn.layers.core import input_data, dropout, fully_connected
#from tflearn.layers.estimator import regression
from statistics import median, mean
from collections import Counter

from sklearn.preprocessing import KBinsDiscretizer
import numpy as np 
import time, math, random
from typing import Tuple
import tensorflow as tf



n_bins = (12,12,12,12)


lower_bounds = [ env.observation_space.low[0], env.observation_space.low[1],env.observation_space.low[2], -math.radians(50) ]
upper_bounds = [ env.observation_space.high[0], env.observation_space.high[1],env.observation_space.high[2], +math.radians(50) ]

def discretizer( cart_position, cart_velocity , pole_angle, pole_velocity ) -> Tuple[int,...]:
    """Convert continues state intro a discrete state"""
    est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    est.fit([lower_bounds, upper_bounds ])
    return tuple(map(int,est.transform([[cart_position, cart_velocity , pole_angle, pole_velocity]])[0]))




def policy( state : tuple ):
    """Choosing action based on epsilon-greedy policy"""
    return np.argmax(Q_table[state])

def new_Q_value( reward : float ,  new_state : tuple , discount_factor=1 ) -> float:
    """Temperal diffrence for updating Q-value of state-action pair"""
    future_optimal_value = np.max(Q_table[new_state])
    learned_value = reward + discount_factor * future_optimal_value
    return learned_value


# Adaptive learning of Learning Rate
def learning_rate(n : int , min_rate=0.01 ) -> float  :
    """Decaying learning rate"""
    return max(min_rate, min(1.0, 1.0 - math.log10((n + 1) / 25)))


def exploration_rate(n : int, min_rate= 0.1 ) -> float :
    """Decaying exploration rate"""
    return max(min_rate, min(1, 1.0 - math.log10((n  + 1) / 25)))



Q_table = np.zeros(n_bins + (env.action_space.n,))
Q_table.shape



reward_list = []
n_episodes = 100
for e in range(n_episodes):

    cum_reward = 0


    current_state, done = discretizer(*env.reset()), False
    
    while True:




        current_state = np.asarray(current_state)
     
        # insert random action
       #if np.random.random() < exploration_rate(e) : 
        if np.random.random() < 0.3 : 
          print('EXPLORE')
          action = env.action_space.sample() # explore 
         


        action = tf.math.argmax(tf.reshape(agent(tf.convert_to_tensor([current_state])), [-1])).numpy()
        # increment enviroment
        #print('env.step(action)', env.step(action))
        obs, reward, done, info = env.step(action)

        print('obs.shape', obs.shape)


        '''

        screen = env.render(mode='rgb_array')
        plt.imshow(screen)
        ipythondisplay.clear_output(wait=True)
        ipythondisplay.display(plt.gcf())

        '''


        new_state = discretizer(*obs)

        #print('new_state.shape', new_state.shape)


        cum_reward += reward


        if done == True:
            print('cum_reward=', cum_reward)
            reward_list.append([e,cum_reward])


        '''
        screen = env.render(mode='rgb_array')
        plt.imshow(screen)
        ipythondisplay.clear_output(wait=True)
        ipythondisplay.display(plt.gcf())
        '''
            

        '''    
        # Update Q-Table
        lr = learning_rate(e)
        learnt_value = new_Q_value(reward , new_state )
        old_value = Q_table[current_state][action]
        Q_table[current_state][action] = (1-lr)*old_value + lr*learnt_value

        '''

    epsilon = 0.3

    utility.record_history(current_state, action, reward, new_state)
    current_state = new_state

    if len(utility.history) == 20:
        utility.update_model(agent) 
    epsilon = max(epsilon * 0.99, 0.05)
'''
    if len(utility.history) == 50:
        utility.update_model(agent) 
    epsilon = max(epsilon * 0.99, 0.05)
'''