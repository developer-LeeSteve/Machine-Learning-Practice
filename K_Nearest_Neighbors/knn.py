import numpy as np
from collections import Counter


def eucliedan_distance(x1, x2):
	return np.sqrt(np.sum((x1-x2)**2))

class KNN:
	def __init__(self, k):
		self.k = k

	def fit(self, X_train, y_train):
		self.X_train = X_train
		self.y_train = y_train

	def predict(self, X_test):
		predicted_labels = [self._predict(x) for x in X_test]
		return np.array(predicted_labels)

	def _predict(self, X_test):
		# compute distances
		distances = [eucliedan_distance(X_test, x_train) for x_train in self.X_train]
		# get the k nearest samples, labels # indexes = indices
		k_indices = np.argsort(distances)[:self.k]
		k_nearest_labels = [self.y_train[i] for i in k_indices]
		# majority vote, most common class label
		most_common = Counter(k_nearest_labels).most_common(1)
		return most_common[0][0]