import numpy as np
from sklearn import svm, metrics
import matplotlib.pyplot as plt

print("\nHomework #1, Q 4:")
print("this can take upto 5 minutes for running the whole code")

mnist_data = np.load(f"data/mnist-data.npz")
np.random.seed(30)  # Setting seed for reproducibility
i= np.arange(len(mnist_data['training_data']))
np.random.shuffle(i)

mnist_validation=mnist_data['training_data'][i[:10000]]
mnist_validation_label=mnist_data['training_labels'][i[:10000]]

mnist_training=mnist_data['training_data'][i[10000:]]
mnist_training_label=mnist_data['training_labels'][i[10000:]]

X_tr_data = mnist_training.reshape((len(mnist_training), -1)) 
y_tr_data = mnist_training_label

#print("1st",X_tr_data.shape)
X_val_set=mnist_validation.reshape((len(mnist_validation), -1))
y_val_set=mnist_validation_label

X_val_tr = X_val_set[:5000]
y_val_tr = y_val_set[:5000]

X_val_test = X_val_set[5000:]
y_val_test = y_val_set[5000:]

# Create a geometric sequence of C values
C_values = [10 ** i for i in range(-3, 4)]

# Initialize lists to store accuracy values
accuracies = []

for C_value in C_values:

    # Create and train the SVM model
    clf = svm.SVC(kernel='linear', C=C_value)
    clf.fit(X_val_tr, y_val_tr)

    # Predict on the validation set
    y_val_pred = clf.predict(X_val_test)

    # Calculate accuracy for the validation set
    val_accuracy = metrics.accuracy_score(y_val_test, y_val_pred)

    # Append accuracy values to the list
    accuracies.append(val_accuracy)

# Find the best C value
best_C_index = np.argmax(accuracies)
best_C = C_values[best_C_index]
best_accuracy = accuracies[best_C_index]
print("the accuracies using validation set of 10000 datapoints are:")
# Print the results
for i in range(len(C_values)):
    print(f"C = {C_values[i]:.5f}, Accuracy = {accuracies[i]:.4f}")

print(f"\nBest C value: {best_C:.5f}, Best Accuracy: {best_accuracy:.4f}")

# Retrain the model with the best C value using all labeled data
final_clf = svm.SVC(kernel='linear', C=best_C)
final_clf.fit(X_tr_data[:15000], y_tr_data[:15000])

# Now, you can use this final_clf to generate predictions for the test set.
final_val_pred = clf.predict(X_val_set)

# Calculate accuracy for the validation set
final_accuracy = metrics.accuracy_score(y_val_set, final_val_pred)
print("\nNow below is the Accuracy of the training data using 15k datapoints in this case using best C:")
print(f"Final accuracy with best C {best_C} = {final_accuracy}")
