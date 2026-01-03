import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

# File paths
TRAIN_PATH = r"C:\Users\harsh\Downloads\train.csv"
TEST_PATH = r"C:\Users\harsh\Downloads\test.csv"
OUTPUT_PATH = r"C:\Users\harsh\Downloads\submission.csv"
MODEL_PATH = r"C:\Users\harsh\Downloads\asteroid_hazard_model.h5"

print("Loading training data...")
# Load training data
train_data = pd.read_csv(TRAIN_PATH)

print(f"Training data shape: {train_data.shape}")
print(f"\nTarget variable (pha) distribution:")
print(train_data['pha'].value_counts())

# Store asteroid names for later
train_names = train_data['name'].copy()

# Check data types
print("\n=== Checking Data Types ===")
print(train_data.dtypes)

# Convert all numerical columns to numeric, coercing errors to NaN
print("\n=== Converting to Numeric ===")
numerical_columns = ['a', 'e', 'i', 'om', 'w', 'q', 'ad', 'per_y', 'data_arc', 
                     'condition_code', 'H', 'diameter', 'albedo', 'rot_per', 'moid']

for col in numerical_columns:
    if col in train_data.columns:
        train_data[col] = pd.to_numeric(train_data[col], errors='coerce')

# Feature Engineering AFTER converting to numeric
print("\n=== Feature Engineering ===")

# Create new features based on orbital mechanics
train_data['orbital_period'] = train_data['per_y']  # Already in years
train_data['orbit_range'] = train_data['ad'] - train_data['q']  # Difference between aphelion and perihelion
train_data['mean_distance'] = (train_data['ad'] + train_data['q']) / 2  # Mean orbital distance
train_data['eccentricity_factor'] = train_data['e'] * train_data['a']  # Eccentricity scaled by semi-major axis

# Handle missing values AFTER converting to numeric
print("\nHandling missing values...")
numerical_columns_extended = numerical_columns + ['orbital_period', 'orbit_range', 'mean_distance', 'eccentricity_factor']

for col in numerical_columns_extended:
    if col in train_data.columns:
        # Use median for better robustness to outliers
        median_val = train_data[col].median()
        if pd.isna(median_val):
            # If median is NaN, use 0
            train_data[col].fillna(0, inplace=True)
        else:
            train_data[col].fillna(median_val, inplace=True)

# Encode categorical variables
print("\nEncoding categorical variables...")
if 'neo' in train_data.columns:
    train_data['neo'] = train_data['neo'].map({'Y': 1, 'N': 0})
    train_data['neo'].fillna(0, inplace=True)  # Fill any remaining NaN

if 'pha' in train_data.columns:
    train_data['pha'] = train_data['pha'].map({'Y': 1, 'N': 0})
    train_data['pha'].fillna(0, inplace=True)  # Fill any remaining NaN

# Drop unnecessary columns
columns_to_drop = ['name']
train_data.drop(columns=[col for col in columns_to_drop if col in train_data.columns], inplace=True)

# Drop rows with any remaining NaN values
before_dropna = len(train_data)
train_data.dropna(inplace=True)
after_dropna = len(train_data)
print(f"\nRows dropped due to NaN: {before_dropna - after_dropna}")
print(f"Data shape after preprocessing: {train_data.shape}")

# Separate features and target
X = train_data.drop(columns=['pha'])
y = train_data['pha']

print(f"\nFeatures shape: {X.shape}")
print(f"Features: {list(X.columns)}")
print(f"\nTarget distribution after preprocessing:")
print(y.value_counts())

# Check for any remaining non-numeric values
print("\n=== Verifying All Columns are Numeric ===")
for col in X.columns:
    if X[col].dtype == 'object':
        print(f"WARNING: {col} is still object type")
        X[col] = pd.to_numeric(X[col], errors='coerce')
        X[col].fillna(0, inplace=True)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Validation set size: {X_val.shape[0]}")

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Calculate class weights for imbalanced data
class_counts = y_train.value_counts()
total = len(y_train)
class_weight = {
    0: total / (2 * class_counts[0]),
    1: total / (2 * class_counts[1])
}
print(f"\nClass weights: {class_weight}")

# Build an improved neural network model
print("\n=== Building Neural Network Model ===")
model = Sequential([
    Dense(128, input_dim=X_train_scaled.shape[1], activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(16, activation='relu'),
    Dropout(0.2),
    
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

print(model.summary())

# Define callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

# Train the model
print("\n=== Training Model ===")
history = model.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val_scaled, y_val),
    class_weight=class_weight,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Evaluate on validation set
print("\n=== Validation Results ===")
y_val_pred_proba = model.predict(X_val_scaled)
y_val_pred = (y_val_pred_proba > 0.5).astype(int).flatten()

val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"\nValidation Accuracy: {val_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_val, y_val_pred, target_names=['Non-Hazardous', 'Hazardous']))
print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_val_pred))

# Save the model
model.save(MODEL_PATH)
print(f"\nModel saved as '{MODEL_PATH}'")

# Save feature columns and scaler
import pickle
with open(r"C:\Users\harsh\Downloads\feature_columns.pkl", 'wb') as f:
    pickle.dump(list(X.columns), f)
with open(r"C:\Users\harsh\Downloads\scaler.pkl", 'wb') as f:
    pickle.dump(scaler, f)
print("Feature columns and scaler saved")

# Load and process test data
print("\n=== Processing Test Data ===")
test_data = pd.read_csv(TEST_PATH)
print(f"Test data shape: {test_data.shape}")

# Store asteroid names for submission
test_names = test_data['name'].copy()

# Convert numerical columns to numeric
for col in numerical_columns:
    if col in test_data.columns:
        test_data[col] = pd.to_numeric(test_data[col], errors='coerce')

# Apply the same feature engineering to test data
test_data['orbital_period'] = test_data['per_y']
test_data['orbit_range'] = test_data['ad'] - test_data['q']
test_data['mean_distance'] = (test_data['ad'] + test_data['q']) / 2
test_data['eccentricity_factor'] = test_data['e'] * test_data['a']

# Handle missing values
for col in numerical_columns_extended:
    if col in test_data.columns:
        median_val = test_data[col].median()
        if pd.isna(median_val):
            test_data[col].fillna(0, inplace=True)
        else:
            test_data[col].fillna(median_val, inplace=True)

# Encode categorical variables
if 'neo' in test_data.columns:
    test_data['neo'] = test_data['neo'].map({'Y': 1, 'N': 0})
    test_data['neo'].fillna(0, inplace=True)

if 'pha' in test_data.columns:
    test_data['pha'] = test_data['pha'].map({'Y': 1, 'N': 0})
    test_data['pha'].fillna(0, inplace=True)

# Drop name column
test_data.drop(columns=['name'], inplace=True)

# Remove pha column if it exists (we're predicting it)
if 'pha' in test_data.columns:
    test_data.drop(columns=['pha'], inplace=True)

# Ensure test data has the same columns as training data
X_test = test_data.reindex(columns=X.columns, fill_value=0)

print(f"Test data shape after preprocessing: {X_test.shape}")

# Check for any remaining non-numeric values in test data
for col in X_test.columns:
    if X_test[col].dtype == 'object':
        print(f"WARNING: {col} in test data is still object type")
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
        X_test[col].fillna(0, inplace=True)

# Fill any remaining NaN values
X_test.fillna(0, inplace=True)

# Scale test data using the same scaler
X_test_scaled = scaler.transform(X_test)

# Make predictions
print("\n=== Making Predictions ===")
y_test_pred_proba = model.predict(X_test_scaled)
y_test_pred = (y_test_pred_proba > 0.5).astype(int).flatten()

# Convert predictions back to Y/N format
y_test_pred_labels = ['Y' if pred == 1 else 'N' for pred in y_test_pred]

# Create submission dataframe
submission = pd.DataFrame({
    'name': test_names,
    'pha': y_test_pred_labels
})

# Save submission file
submission.to_csv(OUTPUT_PATH, index=False)
print(f"\nPredictions saved to '{OUTPUT_PATH}'")
print(f"\nPrediction distribution:")
print(submission['pha'].value_counts())
print(f"\nFirst few predictions:")
print(submission.head(10))

print("\n=== Done! ===")
print(f"Check your Downloads folder for:")
print(f"1. {MODEL_PATH} (trained model)")
print(f"2. {OUTPUT_PATH} (predictions)")
print(f"3. feature_columns.pkl (feature list)")
print(f"4. scaler.pkl (scaler for future use)")