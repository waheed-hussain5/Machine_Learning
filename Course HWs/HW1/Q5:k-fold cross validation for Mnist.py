import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

print("Homework #1, Q5")
spam_data = np.load(f"data/spam-data.npz") 

spam_training = spam_data['training_data']
spam_training_label=spam_data['training_labels']
# Define the values of C to try
C_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]

# Set the number of folds for cross-validation
num_folds = 5

# Calculate the size of each fold
fold_size = len(spam_training) // num_folds

cv_accuracies = []

# Iterate over different C values
for C_value in C_values:
    fold_accuracies = []

    # Iterate over folds
    for fold in range(num_folds):
        # Define the validation set for this fold
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size
        val_indices = list(range(val_start, val_end))

        # Use the rest of the data for training
        train_indices = [i for i in range(len(spam_training)) if i not in val_indices]

        X_train, X_val = spam_training[train_indices], spam_training[val_indices]
        y_train, y_val = spam_training_label[train_indices], spam_training_label[val_indices]

        # Initialize and train the SVM model
        svm_model = SVC(kernel='linear', C=C_value)
        svm_model.fit(X_train, y_train)

        # Predict on the validation set
        y_val_pred = svm_model.predict(X_val)

        # Calculate accuracy for this fold
        fold_accuracy = accuracy_score(y_val, y_val_pred)
        fold_accuracies.append(fold_accuracy)

    # Calculate average accuracy for this C value
    avg_accuracy = np.mean(fold_accuracies)
    cv_accuracies.append(avg_accuracy)

# Find the best C value
best_C_index = np.argmax(cv_accuracies)
best_C = C_values[best_C_index]
best_accuracy = cv_accuracies[best_C_index]

# Print the results
for i in range(len(C_values)):
    print(f"C = {C_values[i]:.5f}, Average Accuracy = {cv_accuracies[i]:.4f}")

print(f"\nBest C value: {best_C:.5f}, Best Average Accuracy: {best_accuracy:.4f}")

# Retrain the model with the best C value using all labeled data
final_svm_model = SVC(kernel='linear', C=best_C)
final_svm_model.fit(spam_training, spam_training_label) #Now we can use this model for prediction of new data

