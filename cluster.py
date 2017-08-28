import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import kneighbors_graph

csv_file = 'data/testData6.csv'
X = np.genfromtxt(csv_file, delimiter=";", skip_header=1, dtype=np.float32)
model = GaussianMixture(n_components=5,covariance_type='full').fit(X)
y = model.predict(X)
plt.scatter(X[:,1],X[:,2],c=y)

plt.show()
