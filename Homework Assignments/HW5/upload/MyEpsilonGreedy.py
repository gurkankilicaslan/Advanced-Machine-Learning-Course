import numpy as np

class MyEpsilonGreedy:
    def __init__(self, num_arms, epsilon):
        number = num_arms
        self.num_arms = number
        self.epsilon = epsilon
        self.createEmpty = np.zeros(number)
        self.addMany = np.zeros(number)

    def pull_arm(self):
        import random
        
        comp = random.random()
        
        if comp < self.epsilon:
            pullarm = random.choice(range(self.num_arms))
        else:
            pullarm = 0
            for i in range(len(self.createEmpty)):
                if i>0:
                    if self.createEmpty[i] > self.createEmpty[i-1]:
                        pullarm = i
                        
        self.pullarm = pullarm
                    
        self.addMany[pullarm] = self.addMany[pullarm] + 1
        
        return self.pullarm

    def update_model(self, reward):
        a = self.pullarm
        self.createEmpty[self.pullarm] = (self.createEmpty[a]*self.addMany[a] + reward)/(self.addMany[a]+1)
        self.addMany[a] = self.addMany[a] + 1
        
        
        
        
        
        
        