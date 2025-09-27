import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

#Constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DATASET_PATH = r"C:\Users\rahen\Documents\GUVI\SolarPanel\Solar_Panel_Dataset"
MODEL_PATH = "solarguard_classifier.h5"

#Load trained model
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

#Recreate validation generator
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

validation_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False  # important for matching predictions with true labels
)

#Get predictions
val_labels = validation_generator.classes
class_names = list(validation_generator.class_indices.keys())

pred_probs = model.predict(validation_generator, verbose=1)
pred_labels = np.argmax(pred_probs, axis=1)

#Confusion matrix
cm = confusion_matrix(val_labels, pred_labels)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

#Classification report
print("\nClassification Report:\n")
print(classification_report(val_labels, pred_labels, target_names=class_names))

#Visualize misclassified images
validation_generator.reset()  # reset generator to start from beginning
file_paths = validation_generator.filepaths
misclassified_idx = np.where(val_labels != pred_labels)[0]

print(f"\nNumber of misclassified images: {len(misclassified_idx)}")

plt.figure(figsize=(12, 12))
for i, idx in enumerate(misclassified_idx[:9]):  # show first 9 mistakes
    img = plt.imread(file_paths[idx])
    true_label = class_names[val_labels[idx]]
    pred_label = class_names[pred_labels[idx]]
    
    plt.subplot(3, 3, i+1)
    plt.imshow(img)
    plt.title(f"True: {true_label}\nPred: {pred_label}")
    plt.axis("off")

plt.tight_layout()
plt.show()
