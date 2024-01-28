################################
# DO NOT EDIT THE FOLLOWING CODE
################################
import numpy as np

from matplotlib import pyplot as plt

from MyPCA import MyPCA

from hw4_utils import load_MNIST, convert_data_to_numpy, plot_points

np.random.seed(2023)

normalize_vals = (0.1307, 0.3081)

batch_size = 100

# load MNIST dataset
train_dataset, test_dataset, train_loader, test_loader = load_MNIST(batch_size, normalize_vals)

# convert to numpy
X, y = convert_data_to_numpy(train_dataset)

#####################
# ADD YOUR CODE BELOW
#####################
from matplotlib import pyplot as plt


usePCA = MyPCA(2)
usePCA.fit(X)
coordinates = usePCA.project(X)

points_x = coordinates[:,1]
points_y = coordinates[:,0]

plot_points(points_x, -points_y, y, 'p1_plot.png')
# I first tried scipy for eigenvalues because it is generally more correct thant numpy,
#then I didn't want to use another library and used numpy.linealg.eigh(). That is why
#I put a "-" before points_y