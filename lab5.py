import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

data = np.random.rand(100)
labels = ["Class1" if x <= 0.5 else "Class2" for x in data[:50]]

def knn(train_data, train_labels, point, k):
    dists = sorted([(abs(point - train_data[i]), train_labels[i]) for i in range(len(train_data))])
    k_labels = [label for _, label in dists[:k]]
    return Counter(k_labels).most_common(1)[0][0]

train_data, train_labels = data[:50], labels
test_data = data[50:]
ks = [1, 2, 3, 4, 5, 20, 30]

print("k-NN Classification\nTraining: first 50 points\nTesting: last 50 points\n")
results = {}

for k in ks:
    preds = [knn(train_data, train_labels, pt, k) for pt in test_data]
    results[k] = preds
    print(f"Results for k={k}:")
    for i, label in enumerate(preds, 51):
        print(f"x{i} (value={test_data[i-51]:.4f}) → {label}")
    print()

for k in ks:
    preds = results[k]
    plt.figure(figsize=(10,6))
    plt.scatter(train_data, [0]*len(train_data), c=["blue" if l=="Class1" else "red" for l in train_labels], label="Train", marker='o')
    plt.scatter([test_data[i] for i,l in enumerate(preds) if l=="Class1"], [1]*preds.count("Class1"), c='blue', label="Class1 (Test)", marker='x')
    plt.scatter([test_data[i] for i,l in enumerate(preds) if l=="Class2"], [1]*preds.count("Class2"), c='red', label="Class2 (Test)", marker='x')
    plt.title(f"k-NN Results (k={k})")
    plt.xlabel("Data Points")
    plt.yticks([0,1], ['Train', 'Test'])
    plt.legend()
    plt.grid(True)
    plt.show()
