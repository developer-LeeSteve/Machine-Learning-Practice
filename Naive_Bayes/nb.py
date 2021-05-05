# Train sample들로 평균값과 분산을 구하고, test sample들을 집어 넣어 이 값들이 일어날 확률을 구하는 것. 가장 높은 확률을 가지는 class를 뽑아내는 과정

import numpy as np

class NaiveBayes:

	def fit(self, X, y):
		n_samples, n_features = X.shape
		self._classes = np.unique(y)
		n_classes = len(self._classes)

		# init mean, var, priors
		self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
		self._var = np.zeros((n_classes, n_features), dtype=np.float64)
		self._priors = np.zeros(n_classes, dtype=np.float64)

		for idx, c in enumerate(self._classes):
			X_c = X[c==y]
			self._mean[c,:] = X_c.mean(axis=0)
			self._var[c,:] = X_c.var(axis=0)
			self._priors[c] = X_c.shape[0] / float(n_samples)


	def predict(self, X):
		y_pred = [self._predict(x) for x in X]
		return y_pred


	def _predict(self, x):
		posteriors = []

		for idx, c in enumerate(self._classes):
			prior = np.log(self._priors[idx])
			class_conditional = np.sum(np.log(self._pdf(idx, x)))
			posterior = prior + class_conditional
			posteriors.append(posterior)

		return self._classes[np.argmax(posteriors)]


	def _pdf(self, class_idx, x):
		mean = self._mean[class_idx]
		var = self._var[class_idx]
		numerator = np.exp(- (x-mean)**2 / (2 * var))
		denominator = np.sqrt(2 * np.pi * var)
		return numerator / denominator