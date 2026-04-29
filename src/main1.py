import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Data Preparation
def generate_synthetic_data(samples=1000, features=20):
    np.random.seed(42)
    X = np.random.randn(samples, features)  # Randomly generated features
    y = np.random.choice([0, 1], size=(samples,))  # Binary labels: 0 or 1
    return pd.DataFrame(X), pd.Series(y)

features, labels = generate_synthetic_data()
labels = labels.rename("Label")

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Scaling the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Function to evaluate models
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)
    
    print(f"{model_name} Accuracy: {accuracy:.2f}")
    print(f"{model_name} Confusion Matrix:\n{conf_matrix}\n")
    
    return accuracy, conf_matrix

# Dictionary to store model performances
model_performance = {}

# Step 2: SVM Model
svm_model = svm.SVC(kernel='linear')
model_performance['SVM'] = evaluate_model(svm_model, X_train_scaled, y_train, X_test_scaled, y_test, 'SVM')

# Step 3: Random Forest Model
rf_model = RandomForestClassifier(random_state=42)
model_performance['Random Forest'] = evaluate_model(rf_model, X_train_scaled, y_train, X_test_scaled, y_test, 'Random Forest')

# Step 4: K-Nearest Neighbors Model
knn_model = KNeighborsClassifier(n_neighbors=5)
model_performance['KNN'] = evaluate_model(knn_model, X_train_scaled, y_train, X_test_scaled, y_test, 'KNN')

# Step 5: Logistic Regression Model
logreg_model = LogisticRegression(max_iter=200)
model_performance['Logistic Regression'] = evaluate_model(logreg_model, X_train_scaled, y_train, X_test_scaled, y_test, 'Logistic Regression')

# Step 6: XGBoost Model
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model_performance['XGBoost'] = evaluate_model(xgb_model, X_train_scaled, y_train, X_test_scaled, y_test, 'XGBoost')

# Step 7: CNN Model (reshaping data for CNN)
X_train_cnn = X_train_scaled.reshape(-1, X_train_scaled.shape[1], 1)
X_test_cnn = X_test_scaled.reshape(-1, X_test_scaled.shape[1], 1)

cnn_model = models.Sequential([
    layers.Conv1D(32, 3, activation='relu', input_shape=(X_train_scaled.shape[1], 1)),
    layers.MaxPooling1D(2),
    layers.Conv1D(64, 3, activation='relu'),
    layers.MaxPooling1D(2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = cnn_model.fit(X_train_cnn, y_train, epochs=10, validation_split=0.2, verbose=0)

cnn_loss, cnn_accuracy = cnn_model.evaluate(X_test_cnn, y_test, verbose=0)
print(f"CNN Accuracy: {cnn_accuracy:.2f}")

# Storing CNN performance
model_performance['CNN'] = (cnn_accuracy, None)

# Visualizing Model Comparisons
model_names = list(model_performance.keys())
accuracies = [performance[0] for performance in model_performance.values()]

plt.figure(figsize=(10, 6))
plt.barh(model_names, accuracies, color='coral')  # Changed color to coral
plt.xlabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.show()

# Visualizing the Confusion Matrix for SVM as an example
plt.figure(figsize=(6, 6))
sns.heatmap(model_performance['SVM'][1], annot=True, fmt="d", cmap="coolwarm")  # Changed color map to 'coolwarm'
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
