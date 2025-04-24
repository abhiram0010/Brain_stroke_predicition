import os
import pickle
import numpy as np
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report

# Load VGG16 model for feature extraction
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
vgg_model = Model(inputs=base_model.input, outputs=base_model.output)  # Extract features before FC layers

def extract_vgg16_features(folder_path, label):
    features, labels = [], []
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (224, 224))  # Resize
            img = preprocess_input(img)  # Normalize
            img = np.expand_dims(img, axis=0)  # Add batch dimension
            feature = vgg_model.predict(img)
            features.append(feature.flatten())  # Flatten the extracted features
            labels.append(label)
    return np.array(features), np.array(labels)

# Define dataset paths
train_normal_path = "/content/Brain_Stroke_CT-SCAN_image/Train/Normal"
train_stroke_path = "/content/Brain_Stroke_CT-SCAN_image/Train/Stroke"
test_normal_path = "/content/Brain_Stroke_CT-SCAN_image/Test/Normal"
test_stroke_path = "/content/Brain_Stroke_CT-SCAN_image/Test/Stroke"
val_normal_path = "/content/Brain_Stroke_CT-SCAN_image/Validation/Normal"
val_stroke_path = "/content/Brain_Stroke_CT-SCAN_image/Validation/Stroke"

# Extract VGG16 features
X_train_normal, y_train_normal = extract_vgg16_features(train_normal_path, 0)
X_train_stroke, y_train_stroke = extract_vgg16_features(train_stroke_path, 1)
X_test_normal, y_test_normal = extract_vgg16_features(test_normal_path, 0)
X_test_stroke, y_test_stroke = extract_vgg16_features(test_stroke_path, 1)
X_val_normal, y_val_normal = extract_vgg16_features(val_normal_path, 0)
X_val_stroke, y_val_stroke = extract_vgg16_features(val_stroke_path, 1)

# Combine Normal & Stroke features
X_train = np.vstack((X_train_normal, X_train_stroke))
y_train = np.hstack((y_train_normal, y_train_stroke))
X_test = np.vstack((X_test_normal, X_test_stroke))
y_test = np.hstack((y_test_normal, y_test_stroke))
X_val = np.vstack((X_val_normal, X_val_stroke))
y_val = np.hstack((y_val_normal, y_val_stroke))

# Apply SMOTE for balancing dataset
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
# Perform Grid Search for best parameters
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [10, 15],
    "min_samples_split": [5, 10]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=3, scoring="accuracy", n_jobs=-1)
grid_search.fit(X_train_bal, y_train_bal)
# Get the best model
best_rf = grid_search.best_estimator_
# Train the best model
best_rf.fit(X_train_bal, y_train_bal)

# Make predictions
y_train_pred = best_rf.predict(X_train)
y_val_pred = best_rf.predict(X_val)
y_test_pred = best_rf.predict(X_test)

# Calculate accuracy
train_acc = accuracy_score(y_train, y_train_pred)
val_acc = accuracy_score(y_val, y_val_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"✅ Training Accuracy: {train_acc * 100:.2f}%")
print(f"✅ Validation Accuracy: {val_acc * 100:.2f}%")
print(f"✅ Testing Accuracy: {test_acc * 100:.2f}%")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Stroke'], yticklabels=['Normal', 'Stroke'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Save Model
with open("random_forest_vgg16.pkl", "wb") as f:
    pickle.dump(best_rf, f)
print("✅ Model saved successfully!")
# Get detailed metrics
report = classification_report(y_test, y_test_pred, target_names=["Normal", "Stroke"])
print("✅ Classification Report:\n", report)
