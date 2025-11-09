import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# --- 1. Constants and Configuration ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DATASET_PATH = 'Solar_Panel_Dataset' 
EPOCHS = 50 # Max epochs

DRIVE_SAVE_DIR = '/content/drive/MyDrive/SolarPanel_Project_Final/'
MODEL_SAVE_PATH = os.path.join(DRIVE_SAVE_DIR, 'solarguard_champion_model.h5')
PLOT_SAVE_PATH = os.path.join(DRIVE_SAVE_DIR, 'champion_training_plots.png')
MATRIX_SAVE_PATH = os.path.join(DRIVE_SAVE_DIR, 'champion_confusion_matrix.png')

# Create the save directory if it doesn't exist
os.makedirs(DRIVE_SAVE_DIR, exist_ok=True)

# --- 2. Load and Preprocess Data ---
datagen = ImageDataGenerator(
    rescale=1./255,          
    validation_split=0.2,    
    rotation_range=20,       
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

print(f"Loading data from: {DATASET_PATH}")
train_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    seed=42  
)

validation_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    seed=42, 
    shuffle=False
)

class_names = list(validation_generator.class_indices.keys())
print(f"Found {len(class_names)} classes: {class_names}")

# --- 3. Build the Model ---
base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')

# Freeze the base model layers
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x) # Adding this is good practice
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x) 
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# --- 4. Compile the Model ---
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# --- 5. Define Callbacks ---
early_stopping = EarlyStopping(
    monitor='val_accuracy', 
    mode='max', 
    patience=5, # Stop after 5 epochs of no improvement
    restore_best_weights=True, 
    verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.2, 
    patience=3, 
    verbose=1
)

# --- 6. Train the Model ---
print("\n--- Starting Model Training (with Callbacks) ---")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[early_stopping, reduce_lr] # Add the callbacks
)
print("--- Training Finished ---")

# --- 7. Save the Trained Model to Drive ---
print(f"\nSaving best model to Google Drive at: {MODEL_SAVE_PATH}")
model.save(MODEL_SAVE_PATH)
print("Model saved successfully.")

# --- 8. Plot and Save Training History to Drive ---
epochs_ran = len(history.history['loss'])
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(range(epochs_ran), acc, label='Training Accuracy')
plt.plot(range(epochs_ran), val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(range(epochs_ran), loss, label='Training Loss')
plt.plot(range(epochs_ran), val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')

print(f"Saving training plot to Google Drive at: {PLOT_SAVE_PATH}")
plt.savefig(PLOT_SAVE_PATH)
plt.show()

# --- 9. Final Evaluation (to confirm) ---
print("\n--- Final Model Evaluation ---")
validation_generator.reset() 
pred_probs = model.predict(validation_generator, verbose=1)
pred_labels = np.argmax(pred_probs, axis=1)
val_labels = validation_generator.classes

print("\n" + "="*30)
print("--- Final Champion Model Report ---")
print("="*30)
print(classification_report(val_labels, pred_labels, target_names=class_names, digits=4))

print("\n" + "="*30)
print("--- Final Champion Confusion Matrix ---")
print("="*30)
cm = confusion_matrix(val_labels, pred_labels)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Final Champion Model Confusion Matrix")
plt.savefig(MATRIX_SAVE_PATH)
plt.show()