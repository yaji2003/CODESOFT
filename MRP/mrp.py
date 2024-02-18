import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer  # Add this import
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv('CODESOFT\MRP\dataset\data.csv')

# Extract numeric part from 'Year' column using regular expressions
df['Year'] = df['Year'].str.extract('(\d+)').astype(float)

# Extract numeric part from 'Duration' column
df['Duration'] = df['Duration'].str.extract('(\d+)').astype(float)

# Remove commas and non-numeric characters from 'Votes' and convert to float
df['Votes'] = df['Votes'].replace('[\$,M]', '', regex=True).astype(float)

# Data preprocessing and feature engineering
# You may need to handle missing values, convert categorical variables to numerical, etc.
# For simplicity, let's focus on numerical features for now
numerical_features = ['Year', 'Duration', 'Votes']

# Select relevant features
X = df[numerical_features]
y = df['Rating']

# Drop rows with missing values in both X and y
missing_rows = X.isnull() | y.isnull()
df = df.dropna(subset=numerical_features + ['Rating'])

# Impute missing values in X with the mean of each column
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Select X and y after dropping rows with NaN values
X = df[numerical_features]
y = df['Rating']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Visualize predictions vs actual values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.title('Actual Ratings vs Predicted Ratings')
plt.show()
