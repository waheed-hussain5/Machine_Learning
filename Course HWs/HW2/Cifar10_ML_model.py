from save_csv import results_to_csv

import numpy as np
from sklearn import svm, metrics
import matplotlib.pyplot as plt


print("\nHomework #2, Q 2:")

cifar10_data = np.load(f"data/cifar10-data.npz")

cifar10_training=cifar10_data['training_data']
cifar10_training_label=cifar10_data['training_labels']

cifar10_test=cifar10_data['test_data']
#mnist_training_label=mnist_data['training_labels']


X_tr_data = cifar10_training#.reshape((len(mnist_training), -1)) 
y_tr_data = cifar10_training_label

X_test_data= cifar10_test#.reshape((len(mnist_test), -1)) 
#X_tr_data[:5000].shape
# Retrain the model with the best C value using all labeled data
final_clf = svm.SVC(kernel='linear')#, C=0.1)
final_clf.fit(X_tr_data[:25000], y_tr_data[:25000])
print("training is completed")
# Now, you can use this final_clf to generate predictions for the test set.
final_pred = final_clf.predict(X_test_data)

results_to_csv(final_pred)
