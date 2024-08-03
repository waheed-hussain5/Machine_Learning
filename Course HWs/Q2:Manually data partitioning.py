#HW1 Q2 all parts.
import numpy as np

dataa=["mnist", "spam", "cifar10"]

size = []

for x in range(len(dataa)):
    load_data = np.load(f"C:/8th/ML/hw1/hw1/data/{dataa[x]}-data.npz")
    np.random.seed(30)  # Setting seed for reproducibility
    i = np.arange(len(load_data['training_data']))
    np.random.shuffle(i)
    #np.random.shuffle(j)
    
    size=[10000, int(0.2*len(load_data['training_data'])) ,5000]
    Validation=load_data['training_data'][i[:size[x]]]
    Validation_label=load_data['training_labels'][i[:size[x]]]
    
    Training=load_data['training_data'][i[:size[x]]]
    Training_label=load_data['training_labels'][i[:size[x]]]
    
    print(f"shape of {dataa[x]}`s validation set:",Validation.shape)
    print(f"shape of {dataa[x]}`s labels of validation set:",Validation_label.shape)
   
    print(f"shape of {dataa[x]}`s training set:",Training.shape)
    print(f"shape of {dataa[x]}`s labels of training set:",Training_label.shape)
    print(' ')
