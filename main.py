import numpy as np
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from huggingface_hub import login
from datasets import load_dataset
import matplotlib.pyplot as plt
from PIL import Image
import os
login("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", add_to_git_credential=True)

dataset = load_dataset("falah/eye-disease-dataset")

output_parent_dir = "C:\\Users\\Sayeed\\OneDrive\\Documents\\Python Scripts\\eyedisease\\data"


os.makedirs(output_parent_dir, exist_ok=True) 

target_width = 400
target_height = 400

for i in range(len(dataset["train"])):
    image = dataset["train"][i]["image"]
    label = dataset["train"][i]["label"]
    
    resized_img = image.resize((target_width, target_height))

    if resized_img.mode != 'RGB':
        resized_img = resized_img.convert('RGB')

    output_dir = os.path.join(output_parent_dir, str(label))
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"image_{i}.jpg" 
    output_path = os.path.join(output_dir, filename)
    resized_img.save(output_path)


original_dataset_dir = "C:\\Users\\Sayeed\\OneDrive\\Documents\\Python Scripts\\eyedisease\\data" 
base_dir = "C:\\Users\\Sayeed\\OneDrive\\Documents\\Python Scripts\\eyedisease\\base_dir"  
os.makedirs(base_dir, exist_ok=True)

train_dir = os.path.join(base_dir, 'train')
os.makedirs(train_dir, exist_ok=True)
validation_dir = os.path.join(base_dir, 'validation')
os.makedirs(validation_dir, exist_ok=True)

classes = os.listdir(original_dataset_dir)
for cls in classes:
    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(validation_dir, cls), exist_ok=True)

for cls in classes:
    cls_dir = os.path.join(original_dataset_dir, cls)
    filenames = os.listdir(cls_dir)
    np.random.shuffle(filenames)
    split = int(0.8 * len(filenames)) 

    train_files = filenames[:split]
    for file in train_files:
        src = os.path.join(cls_dir, file)
        dst = os.path.join(os.path.join(train_dir, cls), file)
        shutil.copyfile(src, dst)

    val_files = filenames[split:]
    for file in val_files:
        src = os.path.join(cls_dir, file)
        dst = os.path.join(os.path.join(validation_dir, cls), file)
        shutil.copyfile(src, dst)


img_width, img_height = 400, 400
batch_size = 16
num_classes = len(dataset["train"].features["label"].names)
channelsRGB = 3

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

train_dataset = tf.data.Dataset.from_generator(
    lambda: train_generator,
    output_types=(tf.float32, tf.float32),
    output_shapes=([None, img_height, img_width, channelsRGB], [None, num_classes])
).repeat()

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

val_generator = val_datagen.flow_from_directory(
    validation_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

val_dataset = tf.data.Dataset.from_generator(
    lambda: val_generator,
    output_types=(tf.float32, tf.float32),
    output_shapes=([None, img_height, img_width, channelsRGB], [None, num_classes])
).repeat()


steps_per_epoch = len(train_generator) // batch_size
validation_steps = len(val_generator) // batch_size

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, channelsRGB)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation='relu'),  
    MaxPooling2D((2, 2)),  
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

steps_per_epoch = len(train_generator) // batch_size

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=25, 
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size
)

loss, accuracy = model.evaluate(val_generator)
print(f"Validation Loss: {loss}, Validation Accuracy: {accuracy}")
