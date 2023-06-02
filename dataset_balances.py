import numpy as np
import medmnist
from medmnist import INFO, Evaluator
from medmnist import PneumoniaMNIST

data = np.load('/home/ubuntu/.medmnist/pneumoniamnist.npz', allow_pickle=True)
print(data.files)
train_labels = data['train_labels']
train_labels = train_labels.flatten()

# Count the number of samples in each class
class_counts_1 = np.bincount(train_labels)

# Print the class counts
print("Class 0 (Pneumonia-Negative):", class_counts_1[0])
print("Class 1 (Pneumonia-Positive):", class_counts_1[1])

#############################################################################

data_2 = np.load('/home/ubuntu/.medmnist/breastmnist.npz', allow_pickle=True)
print(data_2.files)
train_labels = data_2['train_labels']
train_labels = train_labels.flatten()

# Count the number of samples in each class
class_counts_2 = np.bincount(train_labels)

# Print the class counts
print("Class 0 (Breast-Negative):", class_counts_2[0])
print("Class 1 (Breast-Positive):", class_counts_2[1])

#############################################################################



import numpy as np

# Load the dataset
data = np.load('/home/ubuntu/.medmnist/dermamnist.npz', allow_pickle=True)


names_dict = {
    '0': 'actinic keratoses and intraepithelial carcinoma',
    '1': 'basal cell carcinoma',
    '2': 'benign keratosis-like lesions',
    '3': 'dermatofibroma',
    '4': 'melanoma',
    '5': 'melanocytic nevi',
    '6': 'vascular lesions'
}


# Access the labels
train_labels = data['train_labels']
val_labels = data['val_labels']
test_labels = data['test_labels']

# Count the number of examples in each class
class_count = {}

# Count examples in the training set
unique_train_labels, train_counts = np.unique(train_labels, return_counts=True)
for label, count in zip(unique_train_labels, train_counts):
    class_count[label] = count

# Count examples in the validation set
unique_val_labels, val_counts = np.unique(val_labels, return_counts=True)
for label, count in zip(unique_val_labels, val_counts):
    class_count[label] = class_count.get(label, 0) + count

# Count examples in the test set
unique_test_labels, test_counts = np.unique(test_labels, return_counts=True)
for label, count in zip(unique_test_labels, test_counts):
    class_count[label] = class_count.get(label, 0) + count

# Print the count of examples in each class
for label, count in class_count.items():
    class_name = names_dict[str(label)]
    print(f"Class {label}: {class_name} ({count} examples)")
    #print(f"Class {label}: {count}")



