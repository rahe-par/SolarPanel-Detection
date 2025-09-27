import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

#Constants and Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DATASET_PATH = r"C:\Users\rahen\Documents\GUVI\SolarPanel\Solar_Panel_Dataset"
EPOCHS = 50 # Start with 10 and increase if needed

#Load and Preprocess Data
# Use ImageDataGenerator for loading, splitting, and augmenting data
datagen = ImageDataGenerator(
    rescale=1./255,          # Normalize pixel values to 0-1
    validation_split=0.2,    # Split data: 80% training, 20% validation
    rotation_range=20,       # Simple data augmentation
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

train_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training' # Set as training data
)

validation_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation' # Set as validation data
)

# Load the pre-trained base model
base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')

# Freeze the base model layers
base_model.trainable = False

# Add our custom classification layers on top
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x) # Dropout to prevent overfitting
# The final layer has one neuron for each class
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

# Combine the base model and our custom layers
model = Model(inputs=base_model.input, outputs=predictions)

#Compile the Model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary() # Print a summary of the model architecture

#Train the Model
print("\n--- Starting Model Training ---")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator
)
print("--- Training Finished ---")

#Save the Trained Model
model.save("solarguard_classifier.h5")
print("\nModel saved as solarguard_classifier.h5")

#Plot Training History (Optional but Recommended
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()