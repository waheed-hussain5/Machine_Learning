from save_csv import results_to_csv

import numpy as np
from sklearn import svm, metrics
import matplotlib.pyplot as plt

print("\nHomework #2, Q 4:")

mnist_data = np.load(f"C:/8th/ML/hw1/hw1/data/mnist-data.npz")

mnist_training=mnist_data['training_data']
mnist_training_label=mnist_data['training_labels']

mnist_test=mnist_data['test_data']


X_tr_data = mnist_training.reshape((len(mnist_training), -1)) 
y_tr_data = mnist_training_label

X_test_data= mnist_test.reshape((len(mnist_test), -1)) 

# Retrain the model with the best C value using all labeled data
final_clf = svm.SVC(kernel='linear', C=0.1)
final_clf.fit(X_tr_data, y_tr_data)

# Now, you can use this final_clf to generate predictions for the test set.
final_pred = final_clf.predict(X_test_data)




results_to_csv(final_pred)
