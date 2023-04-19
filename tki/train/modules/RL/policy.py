import numpy as np

class PolicySpace(object):
    def __init__(self, policy_args) -> None:
        self.base = policy_args.epsilon
        self.style = policy_args.style
       
    def __call__(self, values, id):
        if self.style == "epsilon_greedy":
            self.epsilon_aneal(id=id)
            return self.epsilon_greedy_policy(values)
        else:
            self.epsilon = 1.0
            return self.epsilon_greedy_policy(values)

    def epsilon_greedy_policy(self, values):
        roll = np.random.uniform()
        if roll < self.epsilon:
            return np.random.randint(values.shape[-1])
        else:
            return max(range(values.shape[-1]), key=values[0][:].__getitem__) 

    def epsilon_aneal(self, id):
        if id < 5:
            self.epsilon = self.base
        elif 5 <= id <= 10:
            self.epsilon = self.base / 2.0
        elif 10 <= id <= 20:
            self.epsilon = self.base / 4.0
        elif 20 <= id <= 35:
            self.epsilon = self.base / 8.0
        else:
            self.epsilon = self.base / 16.0