import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns

# 📂 Paths
DATA_PATH = "logs/login_attempts.csv"
MODEL_PATH = "model/bot_detector_model.h5"
SCALER_PATH = "model/bot_scaler.pkl"
os.makedirs("model", exist_ok=True)

# 📥 Load dataset
df = pd.read_csv(DATA_PATH)
df['scroll_behavior'] = df['scroll_behavior'].astype('category').cat.codes

# ❗ Drop missing or bad rows
feature_cols = [
    "mouse_movement_units", "typing_speed_cpm", "click_pattern_score",
    "time_spent_on_page_sec", "scroll_behavior", "captcha_success",
    "form_fill_time_sec", "captcha_time_sec"
]

df = df.dropna(subset=feature_cols + ['label'])

# 🧼 Convert to float/int
X = df[feature_cols].astype(np.float32)
y = df["label"].astype(np.int32)

# 🧪 Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 🧪 Compute class weights (to handle imbalance)
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(weights))
print("📊 Class Weights:", class_weights)

# 🧪 Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 💾 Save scaler
joblib.dump(scaler, SCALER_PATH)
print(f"✅ Scaler saved at: {SCALER_PATH}")

# 🧠 Build model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],), kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.4),

    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(32, activation='relu'),
    Dropout(0.2),

    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 🏋️ Train with class weights
history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_test_scaled, y_test),
    epochs=100,
    batch_size=32,
    class_weight=class_weights,
    callbacks=[
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
    ],
    verbose=1
)

# 💾 Save model
model.save(MODEL_PATH)
print(f"✅ Model saved at: {MODEL_PATH}")

# 📋 Evaluation
y_pred_prob = model.predict(X_test_scaled).ravel()
y_pred = (y_pred_prob > 0.5).astype(int)

print("\n📜 Classification Report:\n", classification_report(y_test, y_pred))
print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 🔷 Confusion matrix heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
