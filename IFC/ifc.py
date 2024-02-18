import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the Iris dataset from a local CSV file
# Replace 'CODESOFT\IFC\dataset\IRIS.csv' with the actual path to your dataset file
iris_df = pd.read_csv('CODESOFT\IFC\dataset\IRIS.csv')

# Extract features (X) and target variable (y)
X = iris_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = iris_df['species']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (optional but recommended for some algorithms)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the K-Nearest Neighbors classifier with a different number of neighbors
knn_classifier = KNeighborsClassifier(n_neighbors=5)  # You can try different values

# Train the model
knn_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn_classifier.predict(X_test)

# Evaluate the model on the test set
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Display classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Use cross-validation to assess the model's performance
cross_val_scores = cross_val_score(knn_classifier, X, y, cv=5)
print(f"Cross-validated Accuracy: {cross_val_scores.mean() * 100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris_df['species'].unique(), yticklabels=iris_df['species'].unique())
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Scatter Plot for Training Data
plt.figure(figsize=(12, 6))
sns.scatterplot(x='sepal_length', y='sepal_width', hue='species', data=pd.concat([X_train, y_train], axis=1))
plt.title('Scatter Plot for Training Data')
plt.show()

# Scatter Plot for Testing Data
plt.figure(figsize=(12, 6))
sns.scatterplot(x='sepal_length', y='sepal_width', hue='species', data=pd.concat([X_test, y_test], axis=1))
plt.title('Scatter Plot for Testing Data')
plt.show()
