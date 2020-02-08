#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
import os, csv
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import math
import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

print(' /--------------------- Loading Training Data ---------------------/ ')
data = pd.read_csv('odisha.csv')
df = pd.DataFrame(data = data)
print(df)

print(' /--------------------- Normalizing Training Data ---------------------/ ')
df = df.loc[ :, 'Rainfall':'PET'].values.astype(float)
min_max_scaler = preprocessing.MinMaxScaler()
values = min_max_scaler.fit_transform(df)
df_normalized = pd.DataFrame(values)
print(df_normalized)

print(' /--------------------- Train - Test Split ---------------------/ ')
y = df_normalized[0]
X = df_normalized[[1,2,3,4,5,6]]
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
print('X_train', X_train)
print('X_test', X_test)
print('Y_train', Y_train)
print('Y_test', Y_test)

print('/--------------------- Initializing Neural Network ---------------------/ ')
# def init_weights(shape):
#   return tf.Variable(shape, stddev = 0.01)
#
# def model(x, w_h1, w_h2, w_h3, w_o, p_drop_input, p_drop_hidden):
#
#   x = tf.nn.dropout(x, p_drop_input)
#   h1 = tf.nn.relu(tf.matmul(x, w_h1))
#
#   h1 = tf.nn.dropout(h1, p_drop_hidden)
#   h2 = tf.nn.relu(tf.matmul(h1, w_h2))
#
#   h2 = tf.nn.dropout(h2, p_drop_hidden)
#   h3 = tf.nn.relu(tf.matmul(h2, w_h3))
#
#   h3 = tf.nn.dropout(h3, p_drop_hidden)
#   return tf.matmul(h3, w_o)
#
# x = tf.placeholder("float", [None, 6])
# y = tf.placeholder("float", [None, 2])
#
# w_h1 = init_weights([6, 450])
# w_h2 = init_weights([450, 450])
# w_h3 = init_weights([450, 450])
# w_o = init_weights([450, 2])
#
# p_keep_input = tf.placeholder("float")
# p_keep_hidden = tf.placeholder("float")
#
# dnn = model(x, w_h1, w_h2, w_h3, w_o, p_keep_input, p_keep_hidden)
#
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = dnn, labels = y))
# train_op = tf.train.RMSPropOptimizer(0.002, 0.9).minimize(cost)
# predict_op = tf.nn.softmax(dnn)
#
# with tf.Session() as sess:
#   sess.run(tf.global_variables_initializer())
#
# def compute_ams_score(prob_predict_train, prob_predict_valid, w_train, w_valid, pcut_percentile, cv_ratio):
#   pcut = np.percentile(prob_predict_train, pcut_percentile)
#   Yhat_train = prob_predict_train > pcut
#   Yhat_valid = prob_predict_valid > pcut
#   true_positive_train = W_train * (Y_train == 1.0) * (1.0/cv_ratio)
#   true_negative_train = W_train * (Y_train == 0.0) * (1.0/cv_ratio)
#   true_positive_valid = W_valid * (Y_valid == 1.0) * (1.0/(1-cv_ratio))
#   true_negative_valid = W_valid * (Y_valid == 0.0) * (1.0/(1-cv_ratio))
#   s_train = sum ( true_positive_train * (Yhat_train == 1.0) )
#   b_train = sum ( true_negative_train * (Yhat_train == 1.0) )
#   s_valid = sum ( true_positive_valid * (Yhat_valid == 1.0) )
#   b_valid = sum ( true_negative_valid * (Yhat_valid == 1.0) )
#
#   print('/-------------- Computing AMS Score --------------/')
#   def AMSScore(s,b):
#     return math.sqrt (2.*( (s + b + 10.) * math.log(1. + s / (b + 10.)) - s))
#   ams_train = AMSScore(s_train, b_train)
#   ams_valid = AMSScore(s_valid, b_valid)
#   return ams_train, ams_valid

# def mse(coef, x, y):
#     return np.mean((np.dot(x, coef) - y)**2)/2
#
# def gradients(coef, x, y):
#     return np.mean(x.transpose()*(np.dot(x, coef) - y), axis = 1)
#
# def multilinear_regression(coef, x, y, lr, b1 = 0.9, b2 = 0.999, epsilon = 1e-8):
#     prev_error = 0
#     m_coef = np.zeros(coef.shape)
#     v_coef = np.zeros(coef.shape)
#     moment_m_coef = np.zeros(coef.shape)
#     moment_v_coef = np.zeros(coef.shape)
#     t = 0
#
#     while True:
#         error = mse(coef, x, y)
#         if abs(error - prev_error) <= epsilon:
#             break
#         prev_error = error
#         grad = gradients(coef, x, y)
#         t += 1
#         m_coef = b1 * m_coef + (1-b1)*grad
#         v_coef = b2 * v_coef + (1-b2)*grad**2
#         moment_m_coef = m_coef / (1-b1**t)
#         moment_v_coef = v_coef / (1-b2**t)
#
#         delta = ((lr / moment_v_coef**0.5 + 1e-8) *
#                  (b1 * moment_m_coef + (1-b1)*grad/(1-b1**t)))
#
#         coef = np.subtract(coef, delta)
#     return coef
#
# coef = np.array([0, 0, 0])
# c = multilinear_regression(coef, X_train, Y_train, 1e-1)
# fig = plt.figure()
# ax = fig.gca(projection ='3d')
#
# ax.scatter(X_train[:, 1], X_train[:, 2], Y_train, label ='y',
#                 s = 5, color ="dodgerblue")
#
# ax.scatter(X_train[:, 1], X_train[:, 2], c[0] + c[1]*X_train[:, 1] + c[2]*X_train[:, 2],
#                     label ='regression', s = 5, color ="orange")
#
# ax.view_init(45, 0)
# ax.legend()
# plt.show()


# def generate_dataset(n):
# 	x = []
# 	y = []
# 	random_x1 = np.random.rand()
# 	random_x2 = np.random.rand()
# 	for i in range(n):
# 		x1 = i
# 		x2 = i/2 + np.random.rand()*n
# 		x.append([1, x1, x2])
# 		y.append(random_x1 * x1 + random_x2 * x2 + 1)
# 	return np.array(x), np.array(y)

x = np.asarray(X_train)
y = np.asarray(Y_train)
# # x, y = generate_dataset(200)
# print('X', x)
# print('////')
# print('Y', y)
mpl.rcParams['legend.fontsize'] = 12

fig = plt.figure()
ax = fig.gca(projection ='3d')

ax.scatter(x[:, 1], x[:, 2], y, label ='y', s = 5)
ax.legend()
ax.view_init(45, 0)

plt.show()


def mse(coef, x, y):
	return np.mean((np.dot(x, coef) - y)**2)/2

def gradients(coef, x, y):
	return np.mean(x.transpose()*(np.dot(x, coef) - y), axis = 1)

def multilinear_regression(coef, x, y, lr, b1 = 0.9, b2 = 0.999, epsilon = 1e-8):
	prev_error = 0
	m_coef = np.zeros(coef.shape)
	v_coef = np.zeros(coef.shape)
	moment_m_coef = np.zeros(coef.shape)
	moment_v_coef = np.zeros(coef.shape)
	t = 0

	while True:
		error = mse(coef, x, y)
		if abs(error - prev_error) <= epsilon:
			break
		prev_error = error
		grad = gradients(coef, x, y)
		t += 1
		m_coef = b1 * m_coef + (1-b1)*grad
		v_coef = b2 * v_coef + (1-b2)*grad**2
		moment_m_coef = m_coef / (1-b1**t)
		moment_v_coef = v_coef / (1-b2**t)

		delta = ((lr / moment_v_coef**0.5 + 1e-8) *
				(b1 * moment_m_coef + (1-b1)*grad/(1-b1**t)))

		coef = np.subtract(coef, delta)
	return coef

coef = np.array([0, 0, 0, 0, 0, 0])
c = multilinear_regression(coef, x, y, 1e-1)
fig = plt.figure()
ax = fig.gca(projection ='3d')

ax.scatter(x[:, 1], x[:, 2], y, label ='y',
				s = 5, color ="dodgerblue")

ax.scatter(x[:, 1], x[:, 2], c[0] + c[1]*x[:, 1] + c[2]*x[:, 2],
					label ='regression', s = 5, color ="orange")

ax.view_init(45, 0)
ax.legend()
plt.show()

