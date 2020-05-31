#RL python code

import gym
import numpy as np
import random
import math
import sys
import matplotlib.pyplot as plt
from collections import deque

def epsilon_greedy(env, Q, state, ith_episode, decay_rate, episodes_before_decay, epsi=None):
    #take greedy with an experimental part of epsilon
    if epsi is not None:
        epsilon = epsi
    elif ith_episode < episodes_before_decay:
      epsilon = 1.0
    else:
      epsilon = (decay_rate**(ith_episode-episodes_before_decay)) #change way you choose epsilon
    randNum = np.random.random()
    if randNum < epsilon:
        policy = math.floor(random.uniform(1,6))
    else:
        policy = np.argmax(Q[state]) #dude is it argmax or max... if it doesnt work then change it
    return int(policy) #return the action based on policy

def upd_q(Q, Q_next, reward, alpha, gamma):
    #update Q based on given inputs
    return Q + (alpha*(reward + (gamma*Q_next) - Q))

def q_learning(env, num_episodes, alpha, gamma, decay_rate, episodes_before_decay, epsilon=None):
    #q table
    Q = np.zeros((500, 6))

    best_avg_reward = -math.inf
    avg_rewards = deque(maxlen=num_episodes)
    samp_rewards = deque(maxlen= 100)

    for i in range(1, num_episodes+1):
        samp_reward = 0
        state = env.reset()
        while True:
            #Take action based on epsilon greedy
            action = epsilon_greedy(env, Q, state, i, decay_rate, episodes_before_decay, epsilon)
            #get inputs after taking action
            next_state, reward, done, info = env.step(action)
            #take greedy action based on Q
            next_action = epsilon_greedy(env, Q, state, i, decay_rate, episodes_before_decay, 0)
            #update q based
            Q[state][action] = upd_q(Q[state][action], Q[next_state][next_action], reward, alpha, gamma)
            state = next_state

            #reward thing
            samp_reward += reward

            if done:
              samp_rewards.append(samp_reward)
              break

        if (i >= 100):
            # get average reward from last 100 episodes
            avg_reward = np.mean(samp_rewards)
            # append to deque
            avg_rewards.append(avg_reward)
            # update best average reward
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
        # monitor progress
        print("\rEpisode {}/{} || Best average reward {}".format(i, num_episodes, best_avg_reward), end="")
        sys.stdout.flush()
        # check if task is solved (according to OpenAI Gym)
        if best_avg_reward >= 9.7:
            print('\nEnvironment solved in {} episodes.'.format(i), end="")
            break
        if i == num_episodes: print('\n')

    plt.plot(avg_rewards)
    plt.ylabel('Avg_Reward')
    plt.show()
    print(Q)
    return Q

env = gym.make('Taxi-v3')
Q = q_learning(env, 15000, alpha = 1, gamma = 1, decay_rate = 0.94, episodes_before_decay = 6) 
