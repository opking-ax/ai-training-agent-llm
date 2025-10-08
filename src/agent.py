import numpy as np

class TradingAgent:
    def __init__(self, n_actions: int = 3, epsilon: float = 0.1, min_epsilon: float = 0.01, decay_rate: float = 0.995):
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.Q_table = {}

    def get_state_key(self, state: np.ndarray) -> str:
        return tuple(state.round(4))

    def choose_action(self, state: np.ndarray) -> int:
        key = self.get_state_key(state)

        if np.random.rand() < self.epsilon or key not in self.Q_table:
           action = np.random.randint(self.n_actions)
        else:
            action = int(np.argmax(self.Q_table[key]))
        return action
            
    def update(self, state, action, reward, next_state, alpha=0.1, gamma=0.99):
        state_key = self.get_state_key(state)
        next_key = self.get_state_key(next_state)

        if state_key not in self.Q_table:
            self.Q_table[state_key] = np.zeros(self.n_actions)
        if next_key not in self.Q_table:
            self.Q_table[next_key] = np.zeros(self.n_actions)

        self.Q_table[state_key][action] += alpha * (reward + gamma * np.max(self.Q_table[next_key])) - self.Q_table[state_key][action]

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)