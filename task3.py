import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn import tree

# Load the dataset
data_path = '/content/bank-full.csv'
df = pd.read_csv(data_path, sep=';')  
print("Dataset Loaded Successfully.")

print("First 5 Rows of Data:")
print(df.head())
print("\nData Info:")
print(df.info())

# Target variable encoding
df['y'] = df['y'].map({'yes': 1, 'no': 0})  # 'y' in binary

# Encode categorical features
categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
label_encoders = {}
for col in categorical_columns:
    label_encoders[col] = LabelEncoder()
    df[col] = label_encoders[col].fit_transform(df[col])

# Check for missing values 
print("\nMissing Values:")
print(df.isnull().sum())

# Define features (X) and target (y)
X = df.drop(columns=['y'])
y = df['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"Training set size: {X_train.shape}, Test set size: {X_test.shape}")

# Initialize and train the model
clf = DecisionTreeClassifier(random_state=42, max_depth=5)  # Limit depth for interpretability
clf.fit(X_train, y_train)

plt.figure(figsize=(20, 10)) # Visualize Decision Tree
tree.plot_tree(clf, feature_names=X.columns, class_names=['No', 'Yes'], filled=True, rounded=True)
plt.title("Decision Tree Visualization")
plt.show()
# Predictions
y_pred = clf.predict(X_test)

# Metrics
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
import joblib

# Save the model
joblib.dump(clf, 'decision_tree_model.pkl')

# Save the label encoders
for col, le in label_encoders.items():
    joblib.dump(le, f'label_encoder_{col}.pkl')

print("Model and encoders saved successfully.")
