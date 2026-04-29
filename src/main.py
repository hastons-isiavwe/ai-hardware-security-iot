import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Data Loading and Preparation
file_path = 'C:\\Users\\14439\\OneDrive\\Desktop\\BSU\\CTEC\\CTEC 701_ Dr. Haydar\\Project\\Group Project\\pythonProject1\\data_1.xlsx'  # Correct Excel file path
data = pd.read_excel(file_path)  # Using read_excel for an Excel file

# Step 1.1: Encode the labels
label_encoder = LabelEncoder()
data['Label'] = label_encoder.fit_transform(data['Label'])  # 0 for 'Trojan Free', 1 for 'Trojan Infected'

# Step 1.2: Selecting features (dropping non-numeric and unnecessary columns)
features = data.drop(['Label', 'Circuit'], axis=1)  # 'Circuit' is unnecessary for training
labels = data['Label']

# Step 1.3: Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Step 1.4: Scaling the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 2: Implementing the SVM Model
svm_model = svm.SVC(kernel='linear')
svm_model.fit(X_train_scaled, y_train)
svm_predictions = svm_model.predict(X_test_scaled)

# Step 2.1: Evaluating the SVM Model
svm_accuracy = accuracy_score(y_test, svm_predictions)
svm_conf_matrix = confusion_matrix(y_test, svm_predictions)

print(f"SVM Accuracy: {svm_accuracy:.2f}")
print("Confusion Matrix:")
print(svm_conf_matrix)

# Visualizing the confusion matrix
plt.figure(figsize=(6, 6))
sns.heatmap(svm_conf_matrix, annot=True, fmt="d", cmap="YlGnBu")
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Step 3: Prepare Data for CNN (Converting to Image-Like Data)
# Reshape data for CNN; using 2D features directly since the dataset likely isn't image-based
X_train_cnn = X_train_scaled.reshape(-1, X_train_scaled.shape[1], 1)
X_test_cnn = X_test_scaled.reshape(-1, X_test_scaled.shape[1], 1)

# Step 4: Implementing the CNN Model
cnn_model = models.Sequential([
    layers.Input(shape=(X_train_scaled.shape[1], 1)),  # Define input layer explicitly
    layers.Conv1D(32, 3, activation='relu'),
    layers.MaxPooling1D(2),
    layers.Conv1D(64, 3, activation='relu'),
    layers.MaxPooling1D(2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the model
history = cnn_model.fit(X_train_cnn, y_train, epochs=10, validation_split=0.2)

# Step 5: Evaluating the CNN Model
cnn_loss, cnn_accuracy = cnn_model.evaluate(X_test_cnn, y_test)
print(f"CNN Accuracy: {cnn_accuracy:.2f}")

# Plotting the CNN accuracy and loss over epochs (visualization)
history_dict = history.history
plt.figure(figsize=(12, 4))

# Plotting accuracy
plt.subplot(1, 2, 1)
plt.plot(history_dict['accuracy'], label='Train Accuracy')
plt.plot(history_dict['val_accuracy'], label='Validation Accuracy')
plt.title('CNN Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plotting loss
plt.subplot(1, 2, 2)
plt.plot(history_dict['loss'], label='Train Loss')
plt.plot(history_dict['val_loss'], label='Validation Loss')
plt.title('CNN Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
