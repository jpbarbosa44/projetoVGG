import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

num_classes = 2
image_size = 224
batch_size_training = 100
batch_size_validation = 100

data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = data_generator.flow_from_directory(
    './concrete_data_week4/train',
    target_size=(image_size, image_size),
    batch_size=batch_size_training,
    class_mode='categorical'
)

valid_generator = data_generator.flow_from_directory(
    './concrete_data_week4/valid',
    target_size=(image_size, image_size),
    batch_size=batch_size_validation,
    class_mode='categorical'
)

model = Sequential()

model.add(
    VGG16(include_top=False, pooling='avg', weights='imagenet')
)

model.add(Dense(num_classes, activation='softmax'))
model.layers

model.layers[0].trainable = False
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
steps_per_epoch_training = len(train_generator)
steps_per_epoch_validation = len(valid_generator)
num_epochs = 2

fit_history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch_training,
    epochs=num_epochs,
    verbose=1,
    validation_data=valid_generator,
    validation_steps=steps_per_epoch_validation
)
