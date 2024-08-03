from save_csv import results_to_csv

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

print("\nHomework #2, Q 3:")

spam_data = np.load(f"data/spam-data.npz")

spam_training = spam_data['training_data']
spam_training_label=spam_data['training_labels']
spam_test= spam_data['test_data']

# Retrain the model with the best C value using all labeled data
final_svm_model = SVC(kernel='linear', C=1)
final_svm_model.fit(spam_training, spam_training_label) #Now we can use this model for prediction of new data

final_spam_pred= final_svm_model.predict(spam_test)

results_to_csv(final_spam_pred)
