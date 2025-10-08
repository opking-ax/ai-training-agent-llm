import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from environment import TradingEnvironment
from agent import TradingAgent
#from utils import fetch_stock_data
#from src.utils import calculate_techincal_indicators

data = pd.read_csv("AAPL_prices.csv", skiprows=[1])
prices = data["Close"].values

print(prices)

env = TradingEnvironment(price=prices)
state_size = env.reset().shape[0]
agent = TradingAgent(state_size)

all_rewards = []
num_episodes = 3

for episode in range(num_episodes):
    state = env.reset()
    done = False
    episode_rewards = 0

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, value = env.step(action)
        agent.update(state, action, reward, next_state)
        state = next_state
        episode_rewards += reward


    agent.decay_epsilon()
    all_rewards.append(episode_rewards)
    print(f"Episode {episode+1}/{num_episodes} - Reward: {episode_rewards:.2f} - Epsilon: {agent.epsilon:.3f}")
