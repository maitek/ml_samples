import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import kneighbors_graph
from dataset import CsvDataset
import argparse

parser = argparse.ArgumentParser(description='Cluster using gaussian mixture model')
parser.add_argument('csv_file')
args = parser.parse_args()

train_data = CsvDataset(csv_file=args.csv_file, train=True)
test_data = CsvDataset(csv_file=args.csv_file, train=True)

# fit GMM
X = train_data.data
model = GaussianMixture(n_components=5,covariance_type='full').fit(X)

# predict on test data
X_test = test_data.data
y = model.predict(X_test)
plt.scatter(X_test[:,1],X_test[:,2],c=y)

plt.show()
