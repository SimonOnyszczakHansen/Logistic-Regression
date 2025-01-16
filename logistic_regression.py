import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, 
    classification_report,
    roc_curve, 
    roc_auc_score
)
import matplotlib.pyplot as plt

#Load data from CSV
df = pd.read_csv("logistic_regression_data.csv")

#Separate features and target
X = df[["Alder", "Indkomst", "Kredit_score", "Antal_åbne_konti", "Gældsandel"]]
y = df["Godkendt"]

#Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=42
)

#Create logistic regression model
model = LogisticRegression(max_iter=200)

#Fit model on training data
model.fit(X_train, y_train)

#Predict on test data (class predictions)
y_pred = model.predict(X_test)

#Evaluate classification performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:")
print(report)

#Get predicted probabilities for the positive class
y_proba = model.predict_proba(X_test)[:, 1]

#Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
auc_score = roc_auc_score(y_test, y_proba)

#Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
plt.plot([0, 1], [0, 1], 'r--')  # Diagonal line for reference
plt.title("ROC Curve for Logistic Regression")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.show()
