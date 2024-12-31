import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
# Set paths to your dataset directories
data_dir = "Test_Frames_Grayscale"
chd_dir = os.path.join(data_dir, 'With_CHDs')  # Images showing congenital heart defects
no_chd_dir = os.path.join(data_dir, 'Without_CHDs')  # Images showing normal hearts

# Define image size (e.g., 128x128)
img_size = 128
batch_size = 32

# ImageDataGenerator for data augmentation and loading images
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Create data generators
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary',  # binary classification (CHD vs No CHD)
    subset='training'
)

val_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary',  # binary classification
    subset='validation'
)
# Define a simple CNN model for binary classification
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()
# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=10,
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size
)
# Evaluate the model on validation data
val_loss, val_acc = model.evaluate(val_generator)
print(f"Validation accuracy: {val_acc * 100:.2f}%")
def classify_echocardiogram(image_path):
  # Preprocess the input image
  img = cv2.imread(image_path)
  img = cv2.resize(img, (img_size, img_size))  # Resize to match the input size
  img = np.expand_dims(img, axis=0)  # Add batch dimension
  img = img / 255.0  # Normalize the image

  # Predict using the trained model
  prediction = model.predict(img)
  result = "Congenital Heart Defect" if prediction[0] > 0.5 else "No Congenital Heart Defect"
  return result

# Example usage
result = classify_echocardiogram('/path/to/new/echocardiogram.jpg')
print(f"Prediction: {result}")
# Plot the training and validation accuracy/loss
plt.figure(figsize=(12, 6))

# Plot training & validation accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.legend()

# Plot training & validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.legend()

plt.show()

def classify_folder(folder_path):
    image_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(('.jpg', '.png'))]
    return classify_echocardiogram(image_paths)

# Example usage:
folder_predictions = classify_folder('InputPredictionsFrames/With_CHD')
for image, prediction in folder_predictions.items():
    print(f"{image}: {prediction}")
