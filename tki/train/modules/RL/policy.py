import numpy as np

class PolicySpace(object):
    def __init__(self, policy_args) -> None:
        self.epsilon = policy_args.epsilon
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
        if id < 50:
            self.epsilon = 0.8
        if id >= 100:
            self.epsilon = 0.4
        if id >= 150:
            self.epsilon = 0.2
        if id >= 200:
            self.epsilon = 0.1
        if id >= 250:
            self.epsilon = 0.05