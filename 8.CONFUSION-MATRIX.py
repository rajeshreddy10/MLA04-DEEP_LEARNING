from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt

# 9. General Confusion Matrix Demo (using Iris dataset)
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)
model_iris = RandomForestClassifier(random_state=42)
model_iris.fit(X_train, y_train)
y_pred_iris = model_iris.predict(X_test)

cm_iris = confusion_matrix(y_test, y_pred_iris)
acc_iris = accuracy_score(y_test, y_pred_iris)

print("=== General Confusion Matrix (Iris) ===")
print("Confusion Matrix:\n", cm_iris)
print("Accuracy: {:.2f}%".format(acc_iris * 100))

disp1 = ConfusionMatrixDisplay(confusion_matrix=cm_iris, display_labels=iris.target_names)
disp1.plot(cmap="Blues")
plt.title("General Confusion Matrix - Iris")
plt.show()

# 10. Two-Class Confusion Matrix (using Breast Cancer dataset)
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.3, random_state=42)
model_cancer = RandomForestClassifier(random_state=42)
model_cancer.fit(X_train, y_train)
y_pred_cancer = model_cancer.predict(X_test)

cm_cancer = confusion_matrix(y_test, y_pred_cancer)
acc_cancer = accuracy_score(y_test, y_pred_cancer)

print("\n=== Two-Class Confusion Matrix (Breast Cancer) ===")
print("Confusion Matrix:\n", cm_cancer)
print("Accuracy: {:.2f}%".format(acc_cancer * 100))

disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_cancer, display_labels=cancer.target_names)
disp2.plot(cmap="Greens")
plt.title("Two-Class Confusion Matrix - Breast Cancer")
plt.show()

# 11. Multi-Class Confusion Matrix (again Iris dataset with 3 classes)
print("\n=== Multi-Class Confusion Matrix (Iris - 3 Classes) ===")
print("Confusion Matrix:\n", cm_iris)
print("Accuracy: {:.2f}%".format(acc_iris * 100))

disp3 = ConfusionMatrixDisplay(confusion_matrix=cm_iris, display_labels=iris.target_names)
disp3.plot(cmap="Oranges")
plt.title("Multi-Class Confusion Matrix - Iris")
plt.show()
