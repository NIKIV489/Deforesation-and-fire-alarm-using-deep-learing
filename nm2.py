import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ----------------------------
# Define Paths

# ----------------------------
image_dir = "data/images"
label_path = "data/labels.csv"
model_save_path = "models/best_model.h5"
os.makedirs("models", exist_ok=True)


# ----------------------------
# Load Dataset from Disk
# ----------------------------
def load_dataset(image_dir, label_path, img_size=(300, 300)):
    print("Loading dataset...")
    df = pd.read_csv(label_path)
    X = []
    y = []
    for _, row in df.iterrows():
        img_path = os.path.join(image_dir, row['filename'])
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for correct colors
        img = cv2.resize(img, img_size)
        X.append(img)
        y.append(row['label'])
    print("Dataset loaded.")
    return np.array(X), np.array(y)

# ----------------------------
# Prepare Data
# ----------------------------
img_size = (300, 300)
X, y = load_dataset(image_dir, label_path, img_size=img_size)
X = X.astype('float32') / 255.0
y_cat = to_categorical(y, num_classes=3)

X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# ----------------------------
# Data Augmentation
# ----------------------------
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)
datagen.fit(X_train)

# ----------------------------
# Build Model with EfficientNetB3 Base
# ----------------------------
def create_model(input_shape):
    inputs = Input(shape=input_shape)
    base_model = EfficientNetB3(include_top=False, input_tensor=inputs, weights='imagenet')
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(3, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

model = create_model((img_size[0], img_size[1], 3))
model.summary()

# ----------------------------
# Custom Safe CSVLogger to fix Tensor serialization issue
# ----------------------------
class SafeCSVLogger(CSVLogger):
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            for k, v in logs.items():
                if isinstance(v, tf.Tensor):
                    logs[k] = float(v.numpy())
        super().on_epoch_end(epoch, logs)
        

print("Classification Report:")
print(classification_report(y_true, y_pred_classes))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred_classes))

# ----------------------------
# Visualization of Predictions
# ----------------------------
def visualize_prediction(index):
    plt.imshow(X_test[index])
    plt.title(f"Predicted: {y_pred_classes[index]}, True: {y_true[index]}")
    plt.axis('off')
    plt.show()

for i in range(5):
    visualize_prediction(i)
