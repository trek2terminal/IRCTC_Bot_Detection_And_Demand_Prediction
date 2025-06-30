# 📦 Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 📂 Load and preprocess dataset
df = pd.read_csv('D:\\Python programs\\Tatkal_Project\\synthetic_manual_dataset.csv')
df['scroll_behavior'] = df['scroll_behavior'].astype('category').cat.codes
X = df.drop("is_bot", axis=1).values
y = df["is_bot"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🧠 Best DNN Model for Tabular Data
model = Sequential([
    Dense(128, activation='relu', input_shape=(X.shape[1],), kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.4),

    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(32, activation='relu'),
    Dropout(0.2),

    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# ⏳ Train with Callbacks
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    callbacks=[
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
    ],
    verbose=1
)

# 📈 Loss & Accuracy Curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title("Accuracy Curve")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.show()

# 📋 Evaluation
y_pred_prob = model.predict(X_test).ravel()
y_pred = (y_pred_prob > 0.5).astype(int)
print("Classification Report:\n", classification_report(y_test, y_pred))

# 🔷 Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# 🟧 ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
print("ROC AUC Score:", roc_auc)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color='darkorange')
plt.plot([0, 1], [0, 1], linestyle='--', color='navy')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.show()

# 🟨 Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
avg_prec = average_precision_score(y_test, y_pred_prob)
plt.figure(figsize=(6, 4))
plt.step(recall, precision, where='post', color='green')
plt.fill_between(recall, precision, step='post', alpha=0.2, color='green')
plt.title(f'Precision-Recall Curve (AP = {avg_prec:.2f})')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.tight_layout()
plt.show()

# 🔍 Inference on Sample Input
sample = pd.DataFrame([{
    "mouse_movement_units": 100.0,
    "typing_speed_cpm": 9000,
    "click_pattern_score": 0.01,
    "time_spent_on_page_sec": 2,
    "scroll_behavior": 0,
    "captcha_success": 0,
    "form_fill_time_sec": 0.02
}])
print("Prediction (1=bot, 0=human):", int(model.predict(sample.values)[0] > 0.5))

model.save("model/bot_detector_model.h5")
