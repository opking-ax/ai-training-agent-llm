import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from environment import TradingEnvironment
from agent import TradingAgent
#from utils import fetch_stock_data
#from src.utils import calculate_techincal_indicators
from llm_advisor import get_sentiment
from utils import plot_trades


data = pd.read_csv("AAPL_prices.csv", skiprows=[1])
prices = data["Close"].values

sentiment_score = np.array([get_sentiment("There is a chance for a reset to happen") for _ in prices])

env = TradingEnvironment(price=prices, sentiment_score=sentiment_score)
agent = TradingAgent(env.reset().shape[0])

all_rewards = []
num_episodes = 1000

for episode in range(1):
    state = env.reset()
    done = False
    episode_rewards = 0

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, value = env.step(action)
        agent.update(state, action, reward, next_state)
        state = next_state
        all_rewards.append(action)
        #episode_rewards += reward


    #agent.decay_epsilon()
    #all_rewards.append(episode_rewards)
    #print(f"Episode {episode+1}/{num_episodes} - Reward: {episode_rewards:.2f} - Epsilon: {agent.epsilon:.3f}")

plot_trades(prices, all_rewards)