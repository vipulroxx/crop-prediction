#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
import os, csv
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import math
import tensorflow as tf
import sklearn
from sklearn import modelgithugigithub_selection
from matplotlib import pyplot as plt
print(' /--------------------- Loading Training Data ---------------------/ ')
with open('training.csv') as f:
  reader = csv.reader(f)
  row1 = next(reader)
  print(row1)
data_train = np.loadtxt('training.csv', delimiter=',', skiprows=1, converters={32: lambda x:int(x=='s'.encode('utf-8'))})
print('Data Train Shape {}'.format(data_train.shape))
print(data_train)
print(' /--------------------- Discarding all the phi features ---------------------/ ')
data_train = np.delete(data_train, 16, 1)
data_train = np.delete(data_train, 18, 1)
data_train = np.delete(data_train, 19, 1)
data_train = np.delete(data_train, 23, 1)
data_train = np.delete(data_train, 25, 1)
print('Data Train Shape {}'.format(data_train.shape))
print(' /----------------- Maximum index of the training data ------------------/ ')
max_idx = data_train.shape[1] - 1;
print(max_idx)
print(' /--------------------- Loading Testing Data ---------------------/ ')
with open('test.csv') as f:
  reader = csv.reader(f)
  row1 = next(reader)
  print(row1)
data_test = np.loadtxt('test.csv', delimiter=',', skiprows=1)
print('Data Test Shape {}'.format(data_test.shape))
print(data_test)
print(' /------------------ Discarding all the phi features ------------------/ ')
data_test = np.delete(data_test, 16, 1)
data_test = np.delete(data_test, 18, 1)
data_test = np.delete(data_test, 19, 1)
data_test = np.delete(data_test, 23, 1)
data_test = np.delete(data_test, 25, 1)
print('Data Test Shape {}'.format(data_test.shape))
print(' /----------------- Maximum index of the testing data ------------------/ ')
max_idx_test = data_test.shape[1] - 1;
print(max_idx_test)
print(' /----------------- X_test ------------------/ ')
X_test = data_test[:,1:max_idx_test]
print(X_test[0:5])
print(' /---------- ID of the testing data (First five) -----------/ ')
ID_test = list(data_test[:,0])
print(ID_test[0:5])
print('/-------------- ID Test length --------------/')
print(len(ID_test))

# Sorting data into Y(labels), X(input), W(weights)
print('/-------------- Assigning data to numpy arrays --------------/')
Y_data = data_train[:,max_idx] > 0
print("Y_Data {}".format(Y_data))
X_data = data_train[:,1:max_idx-2]
W_data = data_train[:,max_idx-1]
print(X_data.shape)
# Normalize Dataset
def find_mean_and_std(np_array_obj, n_rows, n_cols):
  columsum = np.zeros((1,n_cols))
  mean = np.zeros((1,n_cols))
  standard_dev = np.zeros((1,n_cols))

  try:
    columsum = sum(np_array_obj)
  except MemoryError:
    for k in range(n_rows):
      columsum += np_array_obj[k,:]

  mean = (1.0 / n_rows) * columsum

  for l in range(n_rows):
    standard_dev += (np_array_obj[l,:] - mean)**2

  standard_dev = (1.0 / n_rows) * standard_dev

  return mean, standard_dev

def normalize_features(np_array_obj, mean, standard_dev):
  n_rows, n_cols = np_array_obj.shape
  norm_array_obj = np.zeros((n_rows, n_cols))
  norm_row = np.zeros((1,n_rows))

  for i in range(n_rows):
    norm_row = np_array_obj[i]
    norm_row -= mean
    norm_row = norm_row / standard_dev
    norm_array_obj[i] = norm_row

  return norm_array_obj

print('/-------------- Normalizing --------------/')

mu, sigma = find_mean_and_std(X_data, X_data.shape[0], X_data.shape[1])
X_data = normalize_features(X_data, mu, sigma)
mu_feat_test, sigma_feat_test = find_mean_and_std(X_test, X_test.shape[0], X_test.shape[1])
X_test = normalize_features(X_test, mu_feat_test, sigma_feat_test)
skf = sklearn.model_selection.StratifiedKFold(n_splits=2, shuffle=True, random_state=None)
skf = skf.split(X_data, Y_data)


def init_weights(shape):
  return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w_h1, w_h2, w_h3, w_o, p_drop_input, p_drop_hidden):

  X = tf.nn.dropout(X, p_drop_input)
  h1 = tf.nn.relu(tf.matmul(X, w_h1))

  h1 = tf.nn.dropout(h1, p_drop_hidden)
  h2 = tf.nn.relu(tf.matmul(h1, w_h2))

  h2 = tf.nn.dropout(h2, p_drop_hidden)
  h3 = tf.nn.relu(tf.matmul(h2, w_h3))

  h3 = tf.nn.dropout(h3, p_drop_hidden)
  return tf.matmul(h3, w_o)


print('/-------------- Neural Network --------------/')

n_features = X_data.shape[1]

X = tf.placeholder("float", [None, n_features])
Y = tf.placeholder("float", [None, 2])

w_h1 = init_weights([n_features, 450])
w_h2 = init_weights([450, 450])
w_h3 = init_weights([450, 450])
w_o = init_weights([450, 2])

p_keep_input = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
dnn = model(X, w_h1, w_h2, w_h3, w_o, p_keep_input, p_keep_hidden)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=dnn, labels=Y))
train_op = tf.train.RMSPropOptimizer(0.002, 0.9).minimize(cost)
predict_op = tf.nn.softmax(dnn)

def convert_one_to_binary(np_array_obj):
  array_binary = np.zeros(shape=(np_array_obj.shape[0], 2))
  for k in range(np_array_obj.shape[0]):
    if (np_array_obj[k]):
      array_binary[k][1] = True
    else:
      array_binary[k][0] = False

  return array_binary


with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())


  def compute_ams_score(prob_predict_train, prob_predict_valid, W_train, W_valid, pcut_percentile, cv_ratio):
    pcut = np.percentile(prob_predict_train, pcut_percentile)
    Yhat_train = prob_predict_train > pcut
    Yhat_valid = prob_predict_valid > pcut
    true_positive_train = W_train * (Y_train == 1.0) * (1.0/cv_ratio)
    true_negative_train = W_train * (Y_train == 0.0) * (1.0/cv_ratio)
    true_positive_valid = W_valid * (Y_valid == 1.0) * (1.0/(1-cv_ratio))
    true_negative_valid = W_valid * (Y_valid == 0.0) * (1.0/(1-cv_ratio))
    s_train = sum ( true_positive_train * (Yhat_train == 1.0) )
    b_train = sum ( true_negative_train * (Yhat_train == 1.0) )
    s_valid = sum ( true_positive_valid * (Yhat_valid == 1.0) )
    b_valid = sum ( true_negative_valid * (Yhat_valid == 1.0) )

    print('/-------------- Computing AMS Score --------------/')
    def AMSScore(s,b):
      return math.sqrt (2.*( (s + b + 10.) * math.log(1. + s / (b + 10.)) - s))
    ams_train = AMSScore(s_train, b_train)
    ams_valid = AMSScore(s_valid, b_valid)
    return ams_train, ams_valid


  n_bags = 7;
  prob_bags = np.zeros(shape=(n_bags, X_test.shape[0]))

  for bag in range(n_bags):

    print('Neural Network Bag:', bag)


    w_h1 = init_weights([n_features, 450])
    w_h2 = init_weights([450, 450])
    w_h3 = init_weights([450, 450])
    w_o = init_weights([450, 2])

    sess.run(tf.global_variables_initializer())

    for i in range(10000):

      for train_index, valid_index in skf:

        X_train, X_valid = X_data[train_index], X_data[valid_index]
        Y_train, Y_valid = Y_data[train_index], Y_data[valid_index]
        W_train, W_valid = W_data[train_index], W_data[valid_index]

        Y_train_binary = convert_one_to_binary(Y_train)
        Y_valid_binary = convert_one_to_binary(Y_valid)

        sess.run(train_op, feed_dict={X: X_train, Y: Y_train_binary, p_keep_input: 0.8, p_keep_hidden: 0.5})
        prob_predict_train = sess.run(predict_op, feed_dict={X: X_train, Y: Y_train_binary, p_keep_input: 1.0, p_keep_hidden: 1.0})[:,1]
        prob_predict_valid = sess.run(predict_op, feed_dict={X: X_valid, Y: Y_valid_binary, p_keep_input: 1.0, p_keep_hidden: 1.0})[:,1]

        cv_ratio = (1.0 * X_train.shape[0]) / (X_train.shape[0] + X_valid.shape[0]);
        ams_train, ams_valid = compute_ams_score(prob_predict_train, prob_predict_valid, W_train, W_valid, 85, cv_ratio)
        print(i, '\t', ams_train, '\t', ams_valid)

    print('/-------------- Evaluating test data --------------/')
    prob_bags[bag,:] = sess.run(predict_op, feed_dict={X: X_test, p_keep_input: 1.0, p_keep_hidden: 1.0})[:,1]

  prob_predict_test = np.mean(prob_bags, axis=0)
  pcut = np.percentile(prob_predict_test, 0.85)
  Yhat_test = list(prob_predict_test > pcut)
  prob_predict_test = list(prob_predict_test)
  result_list = []
  for x in range(len(ID_test)):
    result_list.append([int(ID_test[x]), prob_predict_test[x], 's'*(Yhat_test[x]==1.0)+'b'*(Yhat_test[x]==0.0)])
  result_list = sorted(result_list, key=lambda a_entry: a_entry[1])
  for y in range(len(result_list)):
    result_list[y][1] = y+1
  result_list = sorted(result_list, key=lambda a_entry: a_entry[0])
  print('/-------------- Making a CSV file --------------/')
  fcsv = open('output.csv', 'w')
  fcsv.write('EventId,RankOrder,Class\n')
  for line in result_list:
    the_line = str(line[0]) + ',' + str(line[1]) + ',' + line[2] + '\n'
    fcsv.write(the_line);
  fcsv.close()
  classifier_s = sess.run(predict_op, feed_dict={X: X_train[Y_train>0.5], p_keep_input: 1.0, p_keep_hidden: 1.0})[:,1].ravel()
  classifier_b = sess.run(predict_op, feed_dict={X: X_train[Y_train<0.5], p_keep_input: 1.0, p_keep_hidden: 1.0})[:,1].ravel()
  classifier_testing_a = sess.run(predict_op, feed_dict={X: X_test, p_keep_input: 1.0, p_keep_hidden: 1.0})[:,1].ravel()

  c_max = max([classifier_s.max(), classifier_b.max(), classifier_testing_a.max()])
  c_min = max([classifier_s.min(), classifier_b.min(), classifier_testing_a.min()])

  print('/-------------- Creating plot --------------/')
  histo_training_s = np.histogram(classifier_s, bins=50, range=(c_min,c_max))
  histo_training_b = np.histogram(classifier_b, bins=50, range=(c_min,c_max))
  histo_training_a = np.histogram(classifier_testing_a, bins=50, range=(c_min,c_max))
  all_histograms = [histo_training_s, histo_training_b]
  h_max = max([histo[0].max() for histo in all_histograms]) * 1.2
  h_min = 1.0
  edges = histo_training_s[1]
  centers = (edges[:-1] + edges[1:]) / 2.
  widths = (edges[1:] - edges[:-1])
  errorbar_testing_a = np.sqrt(histo_training_a[0])
  ax1 = plt.subplot(111)
  ax1.bar(centers - widths/2., histo_training_b[0], facecolor='red', linewidth=0, width=widths, label='Background (Train)', alpha=0.5)
  ax1.bar(centers - widths/2., histo_training_s[0], bottom=histo_training_b[0], facecolor='blue', linewidth=0, width=widths, label='Signal (Train)', alpha=0.5)
  ff = (1.0 * (sum(histo_training_s[0]) + sum(histo_training_b[0]))) / (1.0 * sum(histo_training_a[0]))
  ax1.errorbar(centers, ff * histo_training_a[0], yerr=ff*errorbar_testing_a, xerr=None, ecolor='black', c='black', fmt='.', label='Test (reweighted)')
  ax1.axvspan(pcut, c_max, color='blue', alpha=0.08)
  ax1.axvspan(c_min, pcut, color='red', alpha=0.08)

  plt.title("Higgs Boson Signal-Background Distribution")
  plt.xlabel("Probability Output (Dropout Neural-Nets)")
  plt.ylabel("Counts/Bin")

  legend = ax1.legend(loc='upper center', shadow=True, ncol=2)
  for alabel in legend.get_texts():
    alabel.set_fontsize('small')

  print('/-------------- Saving Histogram as png --------------/')
  plt.savefig('higgs_boson.png')
  plt.show()


