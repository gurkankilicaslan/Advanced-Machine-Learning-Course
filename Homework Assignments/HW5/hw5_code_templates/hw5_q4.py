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
