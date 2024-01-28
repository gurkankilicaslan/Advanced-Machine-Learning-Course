################################
# DO NOT EDIT THE FOLLOWING CODE
################################
import numpy as np
from matplotlib import pyplot as plt

from Environment import Environment

from MyEpsilonGreedy import MyEpsilonGreedy
from MyUCB import MyUCB
from MyThompsonSampling import MyThompsonSampling

num_arms = 8 # Number of arms for each bandit
num_rounds = 500 # Variable 'T' in the writeup
num_repeats = 10 # Variable 'repetitions' in the writeup

# Gaussian environment parameters
means = [7.2, 20.8, 30.4, 10.3, 40.7, 50.1, 1.5, 45.3]
variances = [0.01, 0.02, 0.03, 0.02, 0.04, 0.001, 0.0007, 0.06]

if len(means) != len(variances):
    raise ValueError('Number of means and variances must be the same.')
if len(means) != num_arms or len(variances) != num_arms:
    raise ValueError('Number of means and variances must be equal to the number of arms.')

# Bernoulli environment parameters
p = [0.45, 0.13, 0.71, 0.63, 0.11, 0.06, 0.84, 0.43]

if len(p) != num_arms:
    raise ValueError('Number of Bernoulli probabily values p must be equal to the number of arms.')

# Epsilon-greedy parameter
epsilon = 0.1

if epsilon < 0:
    raise ValueError('Epsilon must be >= 0.')

gaussian_env_params = {'means':means, 'variances':variances}
bernoulli_env_params = {'p':p}

# Use these two objects to simulate the Gaussian and Bernoulli environments.
# In particular, you need to call get_reward() and pass in the arm pulled to receive a reward from the environment.
# Use the other functions to compute the regret.
# See Environment.py for more details. 
gaussian_env = Environment(name='Gaussian', env_params=gaussian_env_params)
bernoulli_env = Environment(name='Bernoulli', env_params=bernoulli_env_params)

#####################
# ADD YOUR CODE BELOW
#####################



class MyRandom_Agent:
    def __init__(self, num_arms):
        number = num_arms
        self.num_arms = number
        self.createEmpty = np.zeros(number)
        self.addMany = np.zeros(number)

    def pull_arm(self):
        self.pullarm = np.random.choice(num_arms)
        self.addMany[self.pullarm] += 1
        return self.pullarm

    def update_model(self, reward):
        a = self.pullarm
        self.createEmpty[a] = (self.createEmpty[a]*self.addMany[a] + reward)/(self.addMany[a]+1)
        self.addMany[a] = self.addMany[a] + 1



def cumulRegret(environmentt, algorithmm, num_rounds, num_repeats):
    
    cumulRegret = 0
    cumulRegret_list = []
    
    num_list = list(range(num_rounds))
    for i in num_list:
        pullarms = []
        rewards = []
        for j in range(num_repeats):
            pullarms.append(algorithmm.pull_arm())
            reward = environmentt.get_reward(algorithmm.pull_arm())
            reward = reward/environmentt.get_opt_reward()
            reward = np.round(reward, 1)
            
            # ts_reward = np.round(ts_reward, 1)
            
            rewards.append(reward)
            
            algorithmm.update_model(reward)
            
                
        optimalArm = environmentt.get_opt_arm()
        optimalMean = environmentt.get_mean_reward(optimalArm)
        
        
        meanatt = []
        for k in pullarms:
            meanatt.append(environmentt.get_mean_reward(k))
        
        meansum = 0
        for l in meanatt:
            meansum = meansum + optimalMean - l
        cumulRegret = cumulRegret + meansum/len(meanatt)
        
        
        cumulRegret_list.append(cumulRegret)
    
    return cumulRegret_list



EG = MyEpsilonGreedy(num_arms, epsilon)
EGgauss_out = cumulRegret(gaussian_env, EG, num_rounds, num_repeats)
EGbern_out = cumulRegret(bernoulli_env, EG, num_rounds, num_repeats)

UCB = MyUCB(num_arms)
UCBgauss_out = cumulRegret(gaussian_env, UCB, num_rounds, num_repeats)
UCBbern_out = cumulRegret(bernoulli_env, UCB, num_rounds, num_repeats)

TS = MyThompsonSampling(num_arms)
TSgauss_out = cumulRegret(gaussian_env, TS, num_rounds, num_repeats)
TSbern_out = cumulRegret(bernoulli_env, TS, num_rounds, num_repeats)

RA = MyRandom_Agent(num_arms)
RAgauss_out = cumulRegret(gaussian_env, RA, num_rounds, num_repeats)
RAbern_out = cumulRegret(bernoulli_env, RA, num_rounds, num_repeats)

# Plots

plt.figure()
plt.plot(EGgauss_out, label='Epsilon Greedy')
plt.plot(UCBgauss_out, label='UCB')
plt.plot(TSgauss_out, label='Thompson Sampling')
plt.plot(RAgauss_out, label='Random Agent')
plt.ylim(0, 1200)  # Set y-axis limits from 0 to 12
plt.xlabel('Rounds')
plt.ylabel('Cumulative Regret')
plt.title('Gaussian Environment')
plt.legend()



plt.figure()
plt.plot(EGbern_out, label='Epsilon Greedy')
plt.plot(UCBbern_out, label='UCB')
plt.plot(TSbern_out, label='Thompson Sampling')
plt.plot(RAbern_out, label='Random Agent')
plt.xlabel('Rounds')
plt.ylabel('Cumulative Regret')
plt.title('Bernoulli Environment')
plt.legend()
