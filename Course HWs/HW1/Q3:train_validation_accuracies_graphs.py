import numpy as np
from sklearn import svm, metrics
import matplotlib.pyplot as plt

print("\nHomework #1, Q3 All parts (a,b,c)")
print("This can take upto maximum of 8 minutes to run the whole code")
datta = ["mnist", 'spam',"cifar10"]
a=0
for x in datta:
    
    #this bunch of code load and shuffle the datasets using the indexes of training datapoints 
    load_data = np.load(f"data/{x}-data.npz")
    np.random.seed(30)  # Setting seed for reproducibility
    i = np.arange(len(load_data['training_data']))
    np.random.shuffle(i)
    
    spam_size=0
    size = [10000, int(0.2 * len(load_data['training_data'])) ,5000]

    # these lines split the data into training and validation set as required in homework
    vvalidation_data = load_data['training_data'][i[:size[a]]]
    vvalidation_label = load_data['training_labels'][i[:size[a]]]
    ttraining_data = load_data['training_data'][i[size[a]:]]
    ttraining_label = load_data['training_labels'][i[size[a]:]]
    a=a+1

    # for the mnist we had to flatten the shape from 10000x1x28x28 to 10000x784
    # where 10000 are the datapoints and 28x28 are pixel change into 784 (input features)
    
    if x == 'mnist':
        X_tr_data = ttraining_data.reshape((len(mnist_training), -1)) 
        X_val_set = vvalidation_data.reshape((len(mnist_validation), -1))
    elif x == 'cifar10' or x== 'spam' :
        X_tr_data = ttraining_data
        X_val_set = vvalidation_data
    
    y_tr_data = ttraining_label
    y_val_set=vvalidation_label


    # Define the number of training examples
    if x == 'mnist':
        training_sizes = [100, 200, 500, 1000, 2000, 3000,10000]
    elif x == 'cifar10':
        training_sizes = [100, 200, 500, 1000,2000, 5000]
    elif x == 'spam':
        training_sizes = [100, 200, 500, 1000, 2000,len(ttraining_data)]
    
    train_accuracies = []
    val_accuracies = []

    for size in training_sizes:

        # Split the training datapoints into number of required datapoint mentioned in assigment.
        X_train=X_tr_data[:size]
        y_train=y_tr_data[:size]
        
        X_val = X_val_set
        y_val = y_val_set

        # Create and train the SVM model
        clf = svm.SVC(kernel='linear')
        clf.fit(X_train, y_train)

        # Predict on training and validation sets
        y_train_pred = clf.predict(X_train)
        y_val_pred = clf.predict(X_val)

        # Calculate accuracy for training and validation sets
        train_accuracy = metrics.accuracy_score(y_train, y_train_pred)

        val_accuracy = metrics.accuracy_score(y_val, y_val_pred)

        # Append accuracy values to lists
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

    # Plot the results
    plt.plot(training_sizes, train_accuracies,marker ='o', label='Training Accuracy')
    plt.plot(training_sizes, val_accuracies,marker ='o', label='Validation Accuracy')
    plt.xlabel('Number of Training Examples')
    plt.ylabel('Accuracy')
    plt.title(f'Q3 part {a}: Linear SVM on {x} Dataset')
    plt.legend()
    plt.show()
