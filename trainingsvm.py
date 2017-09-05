print(__doc__)


# Code source: Gaël Varoquaux
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm


# Our dataset and targets
# X = np.c_[(.4, -.7),
#           (-1.5, -1),
#           (-1.4, -.9),
#           (-1.3, -1.2),
#           (-1.1, -.2),
#           (-1.2, -.4),
#           (-.5, 1.2),
#           (-1.5, 2.1),
#           (1, 1),
          
#           (1.3, .8),
#           (1.2, .5),
#           (.2, -2),
#           (.5, -2.4),
#           (.2, -2.3),
#           (0, -2.7),
#           (1.3, 2.1)].T

X = np.c_[(0, 0, 0, 5, 5, 0, 0, 0, 2, 7, 5, 0, 1, 3, 4, 1, 0, 0, 0, 0),
		  (0, 0, 0, 5, 3, 0, 0, 0, 1, 13, 3, 0, 0, 2, 5, 1, 0, 0, 0, 0),
		  (0, 0, 0, 4, 5, 1, 0, 0, 3, 8, 3, 0, 0, 3, 3, 2, 0, 0, 1, 0),
		  (0, 1, 1, 0, 4, 0, 0, 0, 2, 11, 5, 0, 0, 4, 3, 1, 0, 1, 0, 0),
		  (0, 0, 0, 4, 5, 3, 0, 0, 5, 4, 1, 1, 1, 3, 5, 0, 0, 0, 1, 0),
		  (0, 0, 0, 2, 5, 1, 0, 0, 3, 15, 2, 0, 0, 2, 2, 1, 0, 0, 0, 0),
		  (1, 0, 0, 9, 2, 0, 0, 1, 1, 4, 5, 0, 0, 4, 6, 0, 0, 0, 0, 0),
		  (0, 0, 0, 2, 4, 0, 0, 0, 4, 14, 3, 0, 0, 2, 4, 0, 0, 0, 0, 0),
		  (0, 0, 0, 1, 4, 0, 0, 0, 2, 16, 4, 0, 1, 2, 3, 0, 0, 0, 0, 0),
		  (0, 0, 0, 6, 3, 1, 0, 0, 2, 8, 4, 0, 2, 4, 3, 0, 0, 0, 0, 0),
		  (1, 0, 1, 2, 2, 2, 0, 0, 1, 12, 3, 0, 0, 4, 5, 0, 0, 0, 0, 0),
		  (0, 0, 0, 4, 3, 1, 1, 0, 4, 12, 2, 0, 1, 2, 2, 0, 0, 0, 1, 0),
		  (0, 0, 0, 2, 3, 2, 0, 0, 5, 11, 2, 0, 1, 3, 2, 1, 0, 0, 1, 0),
		  (0, 0, 0, 3, 5, 0, 0, 0, 3, 13, 2, 1, 0, 1, 3, 0, 0, 0, 1, 1),
		  (0, 0, 0, 1, 3, 1, 0, 0, 4, 16, 2, 0, 0, 3, 3, 0, 0, 0, 0, 0),
		  (0, 0, 0, 2, 3, 0, 0, 0, 3, 14, 5, 0, 0, 4, 2, 0, 0, 0, 0, 0),
		  (0, 0, 0, 2, 4, 0, 0, 0, 2, 16, 4, 0, 1, 3, 1, 0, 0, 0, 0, 0),
		  (0, 0, 0, 4, 3, 0, 0, 0, 3, 15, 4, 0, 0, 2, 1, 1, 0, 0, 0, 0),
		  (0, 0, 0, 1, 2, 0, 0, 0, 3, 20, 4, 0, 0, 2, 0, 1, 0, 0, 0, 0),
		  (0, 0, 0, 0, 4, 1, 0, 0, 4, 15, 3, 0, 1, 3, 2, 0, 0, 0, 0, 0),
		  (0, 0, 0, 1, 5, 0, 0, 0, 3, 16, 3, 1, 0, 2, 1, 0, 0, 0, 0, 1),
		  (0, 0, 0, 0, 5, 1, 0, 0, 5, 12, 3, 0, 1, 3, 1, 1, 1, 0, 0, 0),
		  (0, 0, 0, 0, 3, 3, 0, 0, 4, 17, 2, 0, 1, 3, 0, 0, 0, 0, 0, 0),
		  (0, 0, 0, 0, 5, 0, 0, 0, 1, 18, 4, 1, 2, 2, 0, 0, 0, 0, 0, 0),
		  (1, 0, 0, 2, 5, 1, 1, 0, 5, 6, 4, 0, 2, 2, 3, 0, 0, 0, 0, 0),
		  (0, 0, 0, 0, 2, 2, 0, 0, 2, 21, 2, 0, 2, 2, 0, 0, 0, 0, 0, 0),
		  (0, 0, 0, 1, 4, 0, 0, 0, 3, 20, 2, 0, 0, 2, 0, 0, 0, 1, 0, 0),
		  (0, 0, 0, 0, 5, 0, 0, 0, 2, 22, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0),
		  (0, 0, 0, 0, 2, 1, 0, 0, 2, 22, 2, 0, 1, 2, 1, 0, 0, 0, 0, 0),
		  (0, 0, 0, 2, 4, 1, 0, 0, 3, 14, 3, 0, 1, 4, 0, 0, 1, 0, 0, 0),
		  (0, 0, 0, 3, 3, 3, 0, 0, 4, 7, 3, 0, 3, 4, 3, 0, 0, 0, 0, 0),
		  (1, 0, 0, 0, 5, 0, 0, 1, 2, 13, 5, 0, 1, 3, 0, 1, 0, 1, 0, 0),
		  (0, 0, 0, 4, 6, 0, 0, 0, 6, 12, 1, 2, 1, 1, 0, 0, 0, 0, 0, 0),
		  (0, 0, 0, 2, 7, 0, 0, 0, 6, 12, 3, 0, 0, 1, 0, 0, 2, 0, 0, 0),
		  (0, 0, 0, 0, 2, 2, 0, 0, 4, 23, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0)].T

# X = [[0], [1], [2], [3]]
# Y = [0, 1, 2, 3]
Y = [0] * 17 + [1] * 18
print("X = ")
print(X)
print("Y = ")
print(Y)
# PATHsaxch = 'rawSAX/CH/com.txt'
# with open(PATHsaxch, 'r') as f:
#     # content = [x.strip('\n') for x in f.readlines()]
#     chcontent = []
#     chcount_lines = 0
#     for i, line in enumerate(f):
#     	chcontent.append(line.split())
#     	chcount_lines += 1
# PATHsaxjb = 'rawSAX/JB/com.txt'
# with open(PATHsaxjb, 'r') as f:
#     # content = [x.strip('\n') for x in f.readlines()]
#     jbcontent = []
#     jbcount_lines = 0
#     for i, line in enumerate(f):
#     	jbcontent.append(line.split())
#     	jbcount_lines += 1

# # label = ['CH'] * chcount_lines + ['JB'] * jbcount_lines
# # Y = [0] * 17 + [1] * 18
# # content = chcontent + jbcontent
# # X = content
# # Y = label
# # print(content[0])

# lin_clf = svm.LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
#      intercept_scaling=1, loss='squared_hinge', max_iter=1000,
#      multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
#      verbose=0)
# lin_clf.fit(X, Y) 
# LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
#      intercept_scaling=1, loss='squared_hinge', max_iter=1000,
#      multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
#      verbose=0)
# dec = lin_clf.decision_function([[1]])
# print(dec.shape[1])
clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovo', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
clf.fit(X, Y)

dec = clf.decision_function([[1]])
print(dec.shape[1])

# # figure number
# fignum = 1

# # fit the model
# for kernel in ('linear', 'poly', 'rbf'):
#     clf = svm.SVC(kernel=kernel, gamma=2)
#     clf.fit(X, Y)

#     # plot the line, the points, and the nearest vectors to the plane
#     plt.figure(fignum, figsize=(4, 3))
#     plt.clf()

#     plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
#                 facecolors='none', zorder=10, edgecolors='k')
#     plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired,
#                 edgecolors='k')

#     plt.axis('tight')
#     x_min = -3
#     x_max = 3
#     y_min = -3
#     y_max = 3

#     XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
#     Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

#     # Put the result into a color plot
#     Z = Z.reshape(XX.shape)
#     plt.figure(fignum, figsize=(4, 3))
#     plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
#     plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
#                 levels=[-.5, 0, .5])

#     plt.xlim(x_min, x_max)
#     plt.ylim(y_min, y_max)

#     plt.xticks(())
#     plt.yticks(())
#     fignum = fignum + 1
# plt.show()

