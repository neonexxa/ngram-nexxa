import matplotlib.pyplot as plt 

from sklearn import datasets
from sklearn import svm
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.metrics import accuracy_score
import argparse
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import pickle

ap = argparse.ArgumentParser()
ap.add_argument("-ds", "--dataset", required = True, help = "copy.txt")
ap.add_argument("-ts", "--testset", required = True, help = "textcopy.txt")
ap.add_argument("-at", "--algotype", required = True, help = "dtc/dtr/svm/gnb/erfc/bagc/model")
ap.add_argument("-m", "--model", required = False, help = "dtc/dtr/svm/gnb/erfc/bagc")
args = vars(ap.parse_args())

lines_train = open(args["dataset"], 'r').readlines()
datasets = []
for n in lines_train:
	datasets.append(list(map(int, n.split(' '))))

lines_test = open(args["testset"], 'r').readlines()
testset = []
for m in lines_test:
	testset.append(list(map(int, m.split(' '))))

X = datasets[:-1]
target = datasets[-1]
test = testset[:-1]
test_target = testset[-1]

# X = np.c_[(0, 0, 0, 5, 5, 0, 0, 0, 2, 7, 5, 0, 1, 3, 4, 1, 0, 0, 0, 0),
# 		  (0, 0, 0, 5, 3, 0, 0, 0, 1, 13, 3, 0, 0, 2, 5, 1, 0, 0, 0, 0),
# 		  (0, 0, 0, 4, 5, 1, 0, 0, 3, 8, 3, 0, 0, 3, 3, 2, 0, 0, 1, 0),
# 		  (0, 1, 1, 0, 4, 0, 0, 0, 2, 11, 5, 0, 0, 4, 3, 1, 0, 1, 0, 0),
# 		  (0, 0, 0, 4, 5, 3, 0, 0, 5, 4, 1, 1, 1, 3, 5, 0, 0, 0, 1, 0),
# 		  (0, 0, 0, 2, 5, 1, 0, 0, 3, 15, 2, 0, 0, 2, 2, 1, 0, 0, 0, 0),
# 		  (1, 0, 0, 9, 2, 0, 0, 1, 1, 4, 5, 0, 0, 4, 6, 0, 0, 0, 0, 0),
# 		  (0, 0, 0, 2, 4, 0, 0, 0, 4, 14, 3, 0, 0, 2, 4, 0, 0, 0, 0, 0),
# 		  (0, 0, 0, 1, 4, 0, 0, 0, 2, 16, 4, 0, 1, 2, 3, 0, 0, 0, 0, 0),
# 		  (0, 0, 0, 6, 3, 1, 0, 0, 2, 8, 4, 0, 2, 4, 3, 0, 0, 0, 0, 0),
# 		  (1, 0, 1, 2, 2, 2, 0, 0, 1, 12, 3, 0, 0, 4, 5, 0, 0, 0, 0, 0),
# 		  (0, 0, 0, 4, 3, 1, 1, 0, 4, 12, 2, 0, 1, 2, 2, 0, 0, 0, 1, 0),
# 		  (0, 0, 0, 2, 3, 2, 0, 0, 5, 11, 2, 0, 1, 3, 2, 1, 0, 0, 1, 0),
# 		  (0, 0, 0, 3, 5, 0, 0, 0, 3, 13, 2, 1, 0, 1, 3, 0, 0, 0, 1, 1),
# 		  (0, 0, 0, 1, 3, 1, 0, 0, 4, 16, 2, 0, 0, 3, 3, 0, 0, 0, 0, 0),
# 		  (0, 0, 0, 2, 3, 0, 0, 0, 3, 14, 5, 0, 0, 4, 2, 0, 0, 0, 0, 0),
# 		  (0, 0, 0, 2, 4, 0, 0, 0, 2, 16, 4, 0, 1, 3, 1, 0, 0, 0, 0, 0),
# 		  (0, 0, 0, 4, 3, 0, 0, 0, 3, 15, 4, 0, 0, 2, 1, 1, 0, 0, 0, 0),
# 		  (0, 0, 0, 1, 2, 0, 0, 0, 3, 20, 4, 0, 0, 2, 0, 1, 0, 0, 0, 0),
# 		  (0, 0, 0, 0, 4, 1, 0, 0, 4, 15, 3, 0, 1, 3, 2, 0, 0, 0, 0, 0),
# 		  (0, 0, 0, 1, 5, 0, 0, 0, 3, 16, 3, 1, 0, 2, 1, 0, 0, 0, 0, 1),
# 		  (0, 0, 0, 0, 5, 1, 0, 0, 5, 12, 3, 0, 1, 3, 1, 1, 1, 0, 0, 0),
# 		  (0, 0, 0, 0, 3, 3, 0, 0, 4, 17, 2, 0, 1, 3, 0, 0, 0, 0, 0, 0),
# 		  (0, 0, 0, 0, 5, 0, 0, 0, 1, 18, 4, 1, 2, 2, 0, 0, 0, 0, 0, 0),
# 		  (1, 0, 0, 2, 5, 1, 1, 0, 5, 6, 4, 0, 2, 2, 3, 0, 0, 0, 0, 0),
# 		  (0, 0, 0, 0, 2, 2, 0, 0, 2, 21, 2, 0, 2, 2, 0, 0, 0, 0, 0, 0),
# 		  (0, 0, 0, 1, 4, 0, 0, 0, 3, 20, 2, 0, 0, 2, 0, 0, 0, 1, 0, 0),
# 		  (0, 0, 0, 0, 5, 0, 0, 0, 2, 22, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0),
# 		  (0, 0, 0, 0, 2, 1, 0, 0, 2, 22, 2, 0, 1, 2, 1, 0, 0, 0, 0, 0),
# 		  (0, 0, 0, 2, 4, 1, 0, 0, 3, 14, 3, 0, 1, 4, 0, 0, 1, 0, 0, 0),
# 		  (0, 0, 0, 3, 3, 3, 0, 0, 4, 7, 3, 0, 3, 4, 3, 0, 0, 0, 0, 0),
# 		  (1, 0, 0, 0, 5, 0, 0, 1, 2, 13, 5, 0, 1, 3, 0, 1, 0, 1, 0, 0),
# 		  (0, 0, 0, 4, 6, 0, 0, 0, 6, 12, 1, 2, 1, 1, 0, 0, 0, 0, 0, 0),
# 		  (0, 0, 0, 2, 7, 0, 0, 0, 6, 12, 3, 0, 0, 1, 0, 0, 2, 0, 0, 0),
# 		  (0, 0, 0, 0, 2, 2, 0, 0, 4, 23, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0),
# 		  (0, 0, 0, 6, 2, 1, 0, 0, 3, 13, 2, 0, 0, 2, 1, 1, 0, 1, 1, 0),
# 		  (0, 0, 0, 7, 5, 0, 0, 0, 5, 5, 4, 0, 0, 2, 2, 1, 0, 2, 0, 0),
# 		  (0, 0, 0, 3, 4, 2, 0, 0, 2, 14, 2, 1, 3, 2, 0, 0, 0, 0, 0, 0),
# 		  (0, 2, 0, 0, 4, 1, 0, 2, 3, 13, 2, 1, 1, 2, 0, 0, 0, 2, 0, 0),
# 		  (0, 0, 0, 4, 3, 2, 0, 0, 4, 10, 2, 0, 2, 3, 3, 0, 0, 0, 0, 0),
# 		  (0, 0, 0, 6, 5, 1, 0, 0, 2, 9, 3, 1, 1, 2, 0, 1, 2, 0, 0, 0),
# 		  (1, 0, 0, 2, 6, 0, 0, 1, 3, 9, 3, 1, 1, 3, 2, 0, 0, 0, 1, 0),
# 		  (0, 0, 0, 3, 4, 1, 0, 0, 5, 10, 3, 1, 1, 3, 1, 0, 0, 0, 1, 0),
# 		  (0, 0, 0, 3, 5, 0, 0, 0, 4, 12, 3, 0, 0, 4, 0, 0, 0, 1, 1, 0),
# 		  (1, 0, 0, 4, 4, 0, 0, 1, 3, 6, 4, 0, 0, 6, 2, 0, 0, 0, 1, 1),
# 		  (0, 0, 0, 11, 3, 1, 0, 0, 4, 5, 2, 0, 0, 3, 0, 0, 0, 2, 0, 2),
# 		  (1, 0, 1, 6, 2, 0, 0, 0, 3, 10, 3, 1, 0, 4, 1, 0, 0, 0, 1, 0),
# 		  (0, 2, 1, 3, 5, 0, 0, 1, 5, 6, 3, 1, 0, 3, 1, 0, 1, 0, 1, 0),
# 		  (0, 1, 1, 1, 2, 0, 0, 0, 4, 17, 2, 0, 0, 2, 2, 0, 0, 1, 0, 0),
# 		  (0, 0, 0, 3, 2, 1, 1, 0, 4, 15, 1, 0, 0, 2, 2, 1, 0, 1, 0, 0),
# 		  (0, 0, 0, 4, 5, 0, 0, 0, 4, 13, 3, 0, 0, 1, 0, 1, 0, 2, 0, 0),
# 		  (0, 0, 0, 7, 3, 1, 0, 0, 3, 10, 4, 0, 1, 2, 1, 1, 0, 0, 0, 0),
# 		  (0, 0, 0, 3, 2, 4, 0, 0, 2, 12, 2, 0, 3, 3, 2, 0, 0, 0, 0, 0),
# 		  (0, 0, 0, 8, 3, 0, 2, 0, 4, 10, 2, 0, 1, 1, 0, 0, 0, 2, 0, 0),
# 		  (0, 0, 0, 4, 3, 0, 1, 0, 4, 6, 4, 2, 1, 4, 1, 0, 0, 2, 1, 0),
# 		  (0, 0, 0, 9, 1, 1, 1, 0, 3, 3, 2, 1, 1, 3, 4, 1, 0, 2, 1, 0),
# 		  (0, 0, 0, 3, 3, 2, 1, 0, 4, 9, 2, 2, 1, 3, 0, 0, 2, 1, 0, 0),
# 		  (0, 0, 0, 3, 3, 3, 0, 0, 4, 12, 1, 1, 3, 2, 0, 0, 0, 1, 0, 0),
# 		  (0, 0, 0, 4, 3, 2, 0, 0, 3, 6, 4, 0, 2, 4, 3, 1, 0, 0, 1, 0),
# 		  (0, 0, 0, 5, 4, 1, 1, 0, 2, 6, 3, 1, 2, 2, 4, 0, 1, 1, 0, 0),
# 		  (0, 0, 0, 3, 4, 1, 0, 0, 5, 6, 4, 0, 1, 4, 5, 0, 0, 0, 0, 0),
# 		  (0, 0, 0, 6, 6, 0, 0, 0, 4, 8, 2, 0, 1, 1, 3, 1, 0, 1, 0, 0),
# 		  (0, 0, 0, 4, 4, 1, 1, 0, 5, 6, 3, 0, 0, 4, 4, 0, 1, 0, 0, 0),
# 		  (0, 0, 0, 5, 6, 2, 0, 0, 4, 5, 3, 0, 3, 1, 1, 1, 0, 1, 0, 1),
# 		  (0, 0, 0, 5, 2, 2, 1, 0, 3, 5, 4, 1, 1, 5, 2, 0, 0, 2, 0, 0),
# 		  (0, 0, 0, 4, 3, 4, 0, 0, 3, 5, 3, 1, 2, 4, 3, 0, 1, 0, 0, 0),
# 		  (0, 0, 0, 3, 5, 0, 0, 0, 3, 10, 6, 0, 2, 3, 0, 1, 0, 0, 0, 0),
# 		  (0, 0, 0, 0, 4, 0, 0, 0, 4, 15, 4, 1, 0, 4, 0, 0, 0, 0, 1, 0),
# 		  (0, 0, 0, 2, 4, 0, 0, 0, 3, 17, 3, 0, 0, 2, 2, 0, 0, 0, 0, 0),
# 		  (0, 0, 0, 0, 4, 2, 0, 0, 4, 13, 4, 0, 2, 2, 0, 1, 0, 1, 0, 0),
# 		  (0, 0, 0, 1, 2, 1, 0, 0, 4, 15, 4, 0, 0, 4, 2, 0, 0, 0, 0, 0),
# 		  (0, 0, 0, 1, 5, 2, 0, 0, 5, 8, 5, 0, 1, 5, 1, 0, 0, 0, 0, 0),
# 		  (0, 0, 0, 2, 4, 1, 0, 0, 4, 14, 2, 0, 1, 2, 3, 0, 0, 0, 0, 0),
# 		  (0, 0, 0, 1, 5, 3, 0, 0, 4, 5, 5, 0, 3, 5, 2, 0, 0, 0, 0, 0),
# 		  (0, 0, 0, 3, 5, 0, 0, 0, 4, 10, 4, 1, 0, 5, 0, 0, 0, 1, 0, 0),
# 		  (0, 0, 0, 0, 6, 1, 0, 0, 6, 10, 3, 0, 1, 5, 1, 0, 0, 0, 0, 0),
# 		  (0, 0, 0, 2, 6, 0, 0, 0, 3, 8, 5, 1, 2, 2, 3, 0, 0, 1, 0, 0),
# 		  (0, 0, 0, 1, 5, 1, 0, 0, 4, 12, 4, 0, 1, 2, 1, 1, 0, 0, 1, 0),
# 		  (0, 0, 0, 1, 2, 3, 0, 0, 3, 12, 4, 0, 3, 4, 1, 0, 0, 0, 0, 0),
# 		  (0, 0, 0, 3, 4, 0, 0, 0, 3, 12, 4, 1, 0, 3, 2, 0, 0, 0, 1, 0),
# 		  (0, 0, 0, 2, 3, 0, 0, 0, 3, 14, 5, 0, 1, 4, 1, 0, 0, 0, 0, 0),
# 		  (0, 0, 0, 2, 5, 1, 0, 0, 5, 10, 4, 0, 1, 4, 1, 0, 0, 0, 0, 0),
# 		  (0, 0, 0, 0, 5, 1, 0, 0, 4, 10, 5, 0, 2, 3, 1, 1, 0, 1, 0, 0),
# 		  (0, 0, 0, 3, 4, 0, 0, 0, 4, 13, 5, 0, 1, 2, 0, 1, 0, 0, 0, 0),
# 		  (0, 0, 0, 1, 4, 1, 0, 0, 4, 14, 4, 0, 0, 4, 1, 0, 0, 0, 0, 0),
# 		  (0, 0, 0, 1, 3, 0, 0, 0, 2, 19, 4, 0, 0, 3, 1, 0, 0, 0, 0, 0),
# 		  (0, 0, 0, 3, 3, 1, 0, 0, 3, 12, 4, 0, 1, 3, 1, 1, 0, 1, 0, 0),
# 		  (0, 0, 0, 1, 3, 1, 0, 0, 4, 14, 5, 0, 1, 3, 0, 1, 0, 0, 0, 0),
# 		  (0, 0, 0, 0, 3, 3, 0, 0, 5, 14, 3, 0, 1, 4, 0, 0, 0, 0, 0, 0),
# 		  (0, 0, 0, 1, 6, 1, 0, 0, 4, 8, 5, 0, 1, 5, 2, 0, 0, 0, 0, 0),
# 		  (0, 0, 0, 2, 3, 0, 0, 0, 4, 11, 6, 0, 0, 4, 3, 0, 0, 0, 0, 0),
# 		  (0, 0, 0, 2, 5, 1, 0, 0, 4, 8, 4, 1, 2, 4, 1, 0, 0, 0, 1, 0),
# 		  (0, 0, 0, 3, 4, 0, 0, 0, 3, 12, 5, 0, 0, 5, 1, 0, 0, 0, 0, 0),
# 		  (0, 0, 0, 4, 3, 1, 0, 0, 2, 14, 3, 0, 1, 4, 1, 0, 0, 0, 0, 0),
# 		  (0, 0, 0, 1, 6, 0, 0, 0, 3, 11, 5, 1, 2, 3, 0, 0, 0, 0, 1, 0),
# 		  (0, 0, 0, 5, 2, 1, 0, 0, 4, 15, 2, 0, 0, 2, 0, 1, 0, 0, 1, 0),
# 		  (0, 0, 0, 3, 4, 2, 0, 0, 5, 9, 3, 1, 2, 3, 0, 0, 0, 1, 0, 0),
# 		  (0, 0, 0, 3, 3, 1, 0, 0, 2, 13, 4, 0, 1, 3, 3, 0, 0, 0, 0, 0),
# 		  (0, 0, 0, 2, 3, 0, 0, 0, 3, 17, 4, 0, 0, 4, 0, 0, 0, 0, 0, 0),
# 		  (0, 0, 0, 1, 4, 1, 0, 0, 4, 14, 2, 1, 1, 2, 2, 0, 1, 0, 0, 0),
# 		  (0, 0, 0, 0, 5, 2, 0, 0, 6, 9, 3, 0, 1, 3, 2, 1, 0, 1, 0, 0)].T
# Y = [0] * 17 + [1] * 18 + [2] * 17 + [3] * 14 + [4] * 17 + [5] * 18
# test = np.c_[(1, 0, 1, 2, 2, 2, 0, 0, 1, 12, 3, 0, 0, 4, 5, 0, 0, 0, 0, 0),
# 			   (0, 0, 0, 4, 3, 1, 1, 0, 4, 12, 2, 0, 1, 2, 2, 0, 0, 0, 1, 0),
# 			   (0, 0, 0, 2, 3, 2, 0, 0, 5, 11, 2, 0, 1, 3, 2, 1, 0, 0, 1, 0),
# 			   (0, 0, 0, 3, 5, 0, 0, 0, 3, 13, 2, 1, 0, 1, 3, 0, 0, 0, 1, 1),
# 			   (0, 0, 0, 1, 3, 1, 0, 0, 4, 16, 2, 0, 0, 3, 3, 0, 0, 0, 0, 0),
# 			   (0, 0, 0, 2, 3, 0, 0, 0, 3, 14, 5, 0, 0, 4, 2, 0, 0, 0, 0, 0),
# 			   (0, 0, 0, 2, 4, 0, 0, 0, 2, 16, 4, 0, 1, 3, 1, 0, 0, 0, 0, 0),
# 			   (0, 0, 0, 0, 5, 0, 0, 0, 2, 22, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0),
# 			   (0, 0, 0, 0, 2, 1, 0, 0, 2, 22, 2, 0, 1, 2, 1, 0, 0, 0, 0, 0),
# 			   (0, 0, 0, 2, 4, 1, 0, 0, 3, 14, 3, 0, 1, 4, 0, 0, 1, 0, 0, 0),
# 			   (0, 0, 0, 3, 3, 3, 0, 0, 4, 7, 3, 0, 3, 4, 3, 0, 0, 0, 0, 0),
# 			   (1, 0, 0, 0, 5, 0, 0, 1, 2, 13, 5, 0, 1, 3, 0, 1, 0, 1, 0, 0),
# 			   (0, 0, 0, 4, 6, 0, 0, 0, 6, 12, 1, 2, 1, 1, 0, 0, 0, 0, 0, 0),
# 			   (0, 0, 0, 2, 7, 0, 0, 0, 6, 12, 3, 0, 0, 1, 0, 0, 2, 0, 0, 0),
# 			   (0, 0, 0, 0, 2, 2, 0, 0, 4, 23, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0),
# 			   (0, 0, 0, 11, 3, 1, 0, 0, 4, 5, 2, 0, 0, 3, 0, 0, 0, 2, 0, 2),
# 			   (1, 0, 1, 6, 2, 0, 0, 0, 3, 10, 3, 1, 0, 4, 1, 0, 0, 0, 1, 0),
# 			   (0, 2, 1, 3, 5, 0, 0, 1, 5, 6, 3, 1, 0, 3, 1, 0, 1, 0, 1, 0),
# 			   (0, 1, 1, 1, 2, 0, 0, 0, 4, 17, 2, 0, 0, 2, 2, 0, 0, 1, 0, 0),
# 			   (0, 0, 0, 3, 2, 1, 1, 0, 4, 15, 1, 0, 0, 2, 2, 1, 0, 1, 0, 0),
# 			   (0, 0, 0, 4, 5, 0, 0, 0, 4, 13, 3, 0, 0, 1, 0, 1, 0, 2, 0, 0),
# 			   (0, 0, 0, 7, 3, 1, 0, 0, 3, 10, 4, 0, 1, 2, 1, 1, 0, 0, 0, 0),
# 			   (0, 0, 0, 2, 4, 0, 0, 0, 5, 13, 2, 0, 0, 2, 3, 1, 0, 1, 0, 0),
# 			   (0, 0, 0, 4, 3, 0, 0, 0, 3, 13, 3, 0, 0, 3, 4, 0, 0, 0, 0, 0),
# 			   (0, 0, 0, 5, 4, 0, 0, 0, 4, 9, 4, 0, 0, 3, 1, 1, 0, 1, 0, 1),
# 			   (0, 0, 0, 6, 3, 1, 0, 0, 4, 5, 3, 1, 0, 4, 5, 0, 0, 1, 0, 0),
# 			   (0, 0, 0, 5, 3, 2, 0, 0, 4, 5, 3, 0, 0, 5, 6, 0, 0, 0, 0, 0),
# 			   (0, 0, 0, 0, 6, 1, 0, 0, 6, 10, 3, 0, 1, 5, 1, 0, 0, 0, 0, 0),
# 			   (0, 0, 0, 2, 6, 0, 0, 0, 3, 8, 5, 1, 2, 2, 3, 0, 0, 1, 0, 0),
# 			   (0, 0, 0, 1, 5, 1, 0, 0, 4, 12, 4, 0, 1, 2, 1, 1, 0, 0, 1, 0),
# 			   (0, 0, 0, 1, 2, 3, 0, 0, 3, 12, 4, 0, 3, 4, 1, 0, 0, 0, 0, 0),
# 			   (0, 0, 0, 3, 4, 0, 0, 0, 3, 12, 4, 1, 0, 3, 2, 0, 0, 0, 1, 0),
# 			   (0, 0, 0, 2, 3, 0, 0, 0, 3, 14, 5, 0, 1, 4, 1, 0, 0, 0, 0, 0),
# 			   (0, 0, 0, 2, 5, 1, 0, 0, 5, 10, 4, 0, 1, 4, 1, 0, 0, 0, 0, 0),
# 			   (0, 0, 0, 0, 5, 1, 0, 0, 4, 10, 5, 0, 2, 3, 1, 1, 0, 1, 0, 0),
# 			   (0, 0, 0, 4, 3, 1, 0, 0, 2, 14, 3, 0, 1, 4, 1, 0, 0, 0, 0, 0),
# 			   (0, 0, 0, 1, 6, 0, 0, 0, 3, 11, 5, 1, 2, 3, 0, 0, 0, 0, 1, 0),
# 			   (0, 0, 0, 5, 2, 1, 0, 0, 4, 15, 2, 0, 0, 2, 0, 1, 0, 0, 1, 0),
# 			   (0, 0, 0, 3, 4, 2, 0, 0, 5, 9, 3, 1, 2, 3, 0, 0, 0, 1, 0, 0),
# 			   (0, 0, 0, 3, 3, 1, 0, 0, 2, 13, 4, 0, 1, 3, 3, 0, 0, 0, 0, 0),
# 			   (0, 0, 0, 2, 3, 0, 0, 0, 3, 17, 4, 0, 0, 4, 0, 0, 0, 0, 0, 0),
# 			   (0, 0, 0, 1, 4, 1, 0, 0, 4, 14, 2, 1, 1, 2, 2, 0, 1, 0, 0, 0),
# 			   (0, 0, 0, 0, 5, 2, 0, 0, 6, 9, 3, 0, 1, 3, 2, 1, 0, 1, 0, 0)].T
# y_true = [0] * 7 + [1] * 8 + [2] * 7 + [3] * 5 + [4] * 8 + [5] * 8
Y = target
y_true = test_target
print('Train size :',len(X))
# print('Test size :',len(test))
# print(len(X),len(Y))
if(args["algotype"] == "svm"):
	clf = svm.SVC(gamma=0.001, C=100)
	clf.fit(X,Y)
	y_pred = clf.predict(test)
	print('Prediction : ',y_pred)
	print('Actual value : ',y_true)
	print('Testing Validation Accuracy : ',accuracy_score(y_true, y_pred))
	# Dump the trained decision tree classifier with Pickle
	model_name = 'svm.pkl'
	# Open the file to save as pkl file
	model_name_pkl = open(model_name, 'wb')
	pickle.dump(clf, model_name_pkl)
	# Close the pickle instances
	model_name_pkl.close()
elif(args["algotype"] == "dtc"):
	clf = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=3,max_features='log2', max_leaf_nodes=None, min_samples_leaf=5,min_samples_split=2, min_weight_fraction_leaf=0.0,presort=False, random_state=1000, splitter='best')
	# DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth=3, min_samples_leaf=5)
	clf = clf.fit(X, Y)
	# nt
	y_pred = clf.predict(test)
	print('Prediction : ',y_pred)
	print('Actual value : ',y_true)
	print('Testing Validation Accuracy : ',accuracy_score(y_true, y_pred))
	print('Decision Path:', clf.decision_path(X,check_input=True))
	dotfile = open("dtree2.dot", 'w')
	tree.export_graphviz(clf, out_file = dotfile)
	dotfile.close()
	# Dump the trained decision tree classifier with Pickle
	model_name = 'dtc.pkl'
	# Open the file to save as pkl file
	model_name_pkl = open(model_name, 'wb')
	pickle.dump(clf, model_name_pkl)
	# Close the pickle instances
	model_name_pkl.close()
elif(args["algotype"] == "dtr"):
	clf = tree.DecisionTreeRegressor()
	clf = clf.fit(X, Y)
	y_pred = clf.predict(test)
	print('Prediction : ',y_pred)
	print('Actual value : ',y_true)
	print('Testing Validation Accuracy : ',accuracy_score(y_true, y_pred))
	# Dump the trained decision tree classifier with Pickle
	model_name = 'dtr.pkl'
	# Open the file to save as pkl file
	model_name_pkl = open(model_name, 'wb')
	pickle.dump(clf, model_name_pkl)
	# Close the pickle instances
	model_name_pkl.close()
elif(args["algotype"] == "gnb"):
	clf = GaussianNB()
	clf.fit(X, Y)
	y_pred = clf.predict(test)
	print('Prediction : ',y_pred)
	print('Actual value : ',y_true)
	print('Testing Validation Accuracy : ',accuracy_score(y_true, y_pred))
	# Dump the trained decision tree classifier with Pickle
	model_name = 'gnb.pkl'
	# Open the file to save as pkl file
	model_name_pkl = open(model_name, 'wb')
	pickle.dump(clf, model_name_pkl)
	# Close the pickle instances
	model_name_pkl.close()
elif(args["algotype"] == "erfc"):
	clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',max_depth=None, max_features='log2', max_leaf_nodes=None,min_impurity_split=1e-07, min_samples_leaf=1,min_samples_split=2, min_weight_fraction_leaf=0.0,n_estimators=100, n_jobs=1, oob_score=False, random_state=None,verbose=0, warm_start=False)
	clf.fit(X, Y)
	y_pred = clf.predict(test)
	print('Prediction : ',y_pred)
	print('Actual value : ',y_true)
	print('Testing Validation Accuracy : ',accuracy_score(y_true, y_pred))
	print('Decision Path:', clf.decision_path(X))
	# dotfile = open("erfc2.dot", 'w')
	# tree.export_graphviz(clf, out_file = dotfile)
	# dotfile.close()
	# Dump the trained decision tree classifier with Pickle
	model_name = 'erfc.pkl'
	# Open the file to save as pkl file
	model_name_pkl = open(model_name, 'wb')
	pickle.dump(clf, model_name_pkl)
	# Close the pickle instances
	model_name_pkl.close()
elif(args["algotype"] == "bagc"):
	clf = BaggingClassifier(KNeighborsClassifier(),max_samples=0.5, max_features=0.5)
	clf.fit(X, Y)
	y_pred = clf.predict(test)
	print('Prediction : ',y_pred)
	print('Actual value : ',y_true)
	print('Testing Validation Accuracy : ',accuracy_score(y_true, y_pred))
	# Dump the trained decision tree classifier with Pickle
	model_name = 'bagc.pkl'
	# Open the file to save as pkl file
	model_name_pkl = open(model_name, 'wb')
	pickle.dump(clf, model_name_pkl)
	# Close the pickle instances
	model_name_pkl.close()
elif(args["algotype"] == "km"):
	clf = KMeans(n_clusters=6, random_state=0).fit(X)
	# clf_test = KMeans(n_clusters=6, random_state=0).fit(test)
	y_pred = clf.predict(test)
	centerc = clf.cluster_centers_
	labels = clf.labels_
	# labels_test = clf_test.labels_
	print(centerc,labels)
	print('Prediction : ',y_pred)
	def ClusterIndicesNumpy(clustNum, labels_array):
		return np.where(labels == clustNum)[0]
	clusterindex = [ClusterIndicesNumpy(v, clf.labels_) for v in range(6)]
	clustergrouping = [[scluster for scluster in clusterindex[iterclus]] for iterclus in range(6)]
	print(clustergrouping)
	# print('Actual value : ',accuracy_score(labels_test, y_pred))
	# y_pred = KMeans(n_clusters=2, random_state=random_state).fit_predict(X)
	# plt.subplot(221)
	# plt.scatter(X,X, c=y_pred)
	# plt.title("Incorrect Number of Blobs")
elif(args["algotype"] == "model"):
	# used the trained model
	pickle_name = args["model"]
	model_pkl = open(pickle_name+'.pkl', 'rb')
	model = pickle.load(model_pkl)
	y_pred = model.predict(test)
	print ("Loaded model : ", model)
	print('Prediction : ',y_pred)
	print('Actual value : ',y_true)
	print('Testing Validation Accuracy : ',accuracy_score(y_true, y_pred))