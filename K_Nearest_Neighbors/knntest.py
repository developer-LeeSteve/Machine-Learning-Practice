import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# print(X_test.shape)
# print(X_test[0])
# print(y_test.shape)
# print(y_test)

# plt.figure()
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolor='k', s=20)
# plt.show()

# a = [1, 1, 1, 2, 2, 3, 4, 5, 6]
# from collections import Counter
# most_common = Counter(a).most_common(1)
# print(most_common[0][0])

from knn import KNN
classifier = KNN(k=7)
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
print(predictions)

#predictions == y_test 는 1을 더한다
# 즉 총 y_test의 개수 중에서 몇개의 prediction이 y_test와 같았는지 계산해서 정확도를 구하는 것
acc = np.sum(predictions == y_test) / len(y_test)
print(y_test)
print(acc)

# 즉 이 코드는 특정한 좌표를 가지는 X_train 값과 그를 label한 y_train 값이 매칭되어 있는데,
# 이를 가지고 X_test, y_test 값을 주어진 좌표계에 넣어서 KNN으로 계산을 하여 y_test의 값을
# 예측해보는 것. 이미 y_test의 값은 나와있는데, KNN 방법을 통해서 y_test의 값을 예측해보는 것이다
# 근데 k 값이 높아질 수록 어쩔때는 더욱 더 정교하고, 어쩔때는 덜 정교한 상황이 나온다
# k 값은 주어진 좌표에서 몇개의 근처의 point들을 가지고 y_test label을 예측할 것인지 말하는 것이다
# K nearest neighbors