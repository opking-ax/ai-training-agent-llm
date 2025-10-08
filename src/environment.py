import numpy as np
from llm_advisor import get_sentiment
import pandas as pd

class TradingEnvironment:
    def __init__(self, price, window_size=10):
        self.price = np.array(price)
        self.window_size = window_size
        self.reset()

    def reset(self):
        self.current_steps = self.window_size
        self.balance = 1000000.0
        self.holdings = 0.0
        return self._get_state()
    
    def _get_state(self):
        price_history = self.price[self.current_steps - self.window_size:self.current_steps]
        price_history = price_history / max(price_history)

        state = np.concatenate([price_history])
        return state
    
    def step(self, action: int):
        """
        A function takes steps when doing an action
        Args:
            action (int): an integer that represents a given action. 
                        [i.e., 0 = Hold, 1 = Buy, 2 = Sell]
        Returns:
            next_state: the next state of the learning
            reward: the rewards after doing an action
            done: a flag to continue or not.
        """
        done = False
        price = self.price[self.current_steps]
        reward = 0.0

        if action == 1 and self.balance >= price:
            self.holdings += 1
            self.balance -= price
        elif action == 2 and self.holdings > 0:
            self.balance += price
            self.holdings -= 1

        portfolio_value = self.balance + self.holdings * price
        reward = portfolio_value - 1000000.0
        
        self.current_steps += 1
        if self.current_steps >= len(self.price):
            done = True

        next_state = self._get_state()

        return next_state, reward, done, portfolio_value
    



    """
    before that I want the following: 1. Ways to improve 2. the other advanced RL methods
    1. add other option that could be done
    """