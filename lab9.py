import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

data = fetch_olivetti_faces(shuffle=True, random_state=42)
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
gnb = GaussianNB().fit(X_train, y_train)
y_pred = gnb.predict(X_test)

print(f'Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%')
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=1))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

cv_acc = cross_val_score(gnb, X, y, cv=5, scoring='accuracy').mean()
print(f'\nCross-validation accuracy: {cv_acc*100:.2f}%')

fig, axes = plt.subplots(3, 5, figsize=(12, 8))
for ax, img, lbl, pred in zip(axes.ravel(), X_test, y_test, y_pred):
    ax.imshow(img.reshape(64, 64), cmap='gray')
    ax.set_title(f"T:{lbl}, P:{pred}")
    ax.axis('off')
plt.show()
