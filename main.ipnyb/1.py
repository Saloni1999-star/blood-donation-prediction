# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from tpot import TPOTClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib # Added joblib for saving the model later

warnings.filterwarnings("ignore")

# %%
import os

file_path = "your_data.csv"  # Replace with your actual CSV file path

if os.path.exists(file_path):
	data = pd.read_csv(file_path)
	print(f"Data loaded successfully from: {file_path}")
else:
	print(f"File '{file_path}' not found. Creating a sample DataFrame for demonstration.")
	data = pd.DataFrame({
		'Recency': [2, 4, 5, 2, 7],
		'Frequency': [50, 13, 16, 20, 24],
		'Monetary': [12500, 3250, 4000, 5000, 6000],
		'Time': [98, 28, 35, 45, 77],
		'whether he/she donated blood in March 2007': [1, 0, 0, 1, 0]
	})

print("\nFirst 5 rows of the dataset:")
print(data.head())
print("\nDataset Info:")
data.info()


# %%
X = data.drop(columns=["whether he/she donated blood in March 2007"])
y = data["whether he/she donated blood in March 2007"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
# Order 2: Select features and target & Split into train and test sets
X = data.drop(columns=["whether he/she donated blood in March 2007"])
y = data["whether he/she donated blood in March 2007"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nData split into training and test sets.")
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# %%
# Order 3: Train a logistic regression model with class_weight balanced
print("\n--- Training Logistic Regression Model ---")
model = LogisticRegression(random_state=42, solver='liblinear', class_weight='balanced')
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate Logistic Regression Model Performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Logistic Regression Test set accuracy: {accuracy:.2f}")

# %%
# Confusion matrix for LR
cm_lr = confusion_matrix(y_test, y_pred)
print("\nLogistic Regression Confusion Matrix:")
print(cm_lr)
# Classification report for LR
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, y_pred))  # This was duplicated in your last chunk, keeping one.

# %%
# Visualize LR Confusion Matrix (from your snippet)
plt.figure(figsize=(6,5))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Logistic Regression Confusion Matrix')
plt.show()

# %%
# Feature Coefficients and Visualization for LR (from your snippet)
feature_names = X.columns
coefficients = model.coef_[0]
print("\nLogistic Regression Coefficients:")
for name, coef in zip(feature_names, coefficients):
    print(f"{name}: {coef:.4f}")

# %%
plt.figure(figsize=(7,5))
plt.bar(feature_names, coefficients, color='skyblue')
plt.xlabel('Feature')
plt.ylabel('Coefficient Value')
plt.title('Logistic Regression Coefficients')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# %%
# Train and Evaluate TPOT Classifier (from your snippet)
print("\n--- Training TPOT Classifier (This may take some time) ---")
tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2, random_state=42, scoring='f1_weighted', cv=5)
# Added scoring='f1_weighted' and cv=5 for better handling of imbalanced data
# Verbosity=2 will show progress during training
tpot.fit(X_train, y_train)

# Ensure TPOTClassifier is instantiated and fitted before scoring
tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2, random_state=42, scoring='f1_weighted', cv=5)
tpot.fit(X_train, y_train)
print(f"\nTPOT test accuracy: {tpot.score(X_test, y_test):.2f}")

# %%
# Evaluate TPOT's best model (from your snippet)
y_pred_tpot = tpot.predict(X_test)

# %%
print("\nTPOT Confusion Matrix:")
cm_tpot = confusion_matrix(y_test, y_pred_tpot)
print(cm_tpot)


# %%
print("\nTPOT Classification Report:")
print(classification_report(y_test, y_pred_tpot))

# %%
# Visualize TPOT Confusion Matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm_tpot, annot=True, fmt='d', cmap='Greens', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('TPOT Confusion Matrix')
plt.show()

# %%
# Print TPOT's best pipeline
print("\nBest pipeline found by TPOT:")
print(tpot.fitted_pipeline_)

# %%
# Save the Logistic Regression model (from your snippet)
# Consider saving TPOT's best pipeline too if it performs better
joblib.dump(model, 'logistic_regression_model.pkl')
print("\nLogistic Regression model saved as 'logistic_regression_model.pkl'")

# %%
# tpot.export('tpot_best_pipeline.py')
# print("TPOT best pipeline exported as 'tpot_best_pipeline.py'")

# %%
loaded_model = joblib.load('logistic_regression_model.pkl')
print("Model loaded successfully!")

# %%
# Example of new data (ensure the order of features matches training order)
# Assuming features are: 'Recency', 'Frequency', 'Monetary', 'Time'
# Example: a donor who donated 3 months ago, 10 times, 2500 c.c., and first donation was 60 months ago
new_donor_data = pd.DataFrame([[3, 10, 2500, 60]], columns=X.columns)  # Use X.columns to ensure correct feature names

# %%
# Make a prediction
prediction = loaded_model.predict(new_donor_data)
prediction_proba = loaded_model.predict_proba(new_donor_data)  # Get probabilities

print(f"\nPrediction for new donor (0=No, 1=Yes): {prediction[0]}")
print(f"Probability of not donating (0): {prediction_proba[0][0]:.4f}")
print(f"Probability of donating (1): {prediction_proba[0][1]:.4f}")


