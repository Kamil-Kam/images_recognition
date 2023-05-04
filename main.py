import tensorflow as tf
from data_functions import clear_data
import cv2
from matplotlib import pyplot
import numpy as np
import os
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.metrics import Precision, Recall, BinaryAccuracy

data_dir = 'data'
img_extensions = ['jpg', 'jpeg', 'png']

clear_data(data_dir, img_extensions)

img_data = tf.keras.utils.image_dataset_from_directory('data')
img_data = img_data.map(lambda x, y: (x / 255, y))
img_data_iterator = img_data.as_numpy_iterator()
batch_img = img_data_iterator.next()


def show_images():
    figure, axes = pyplot.subplots(ncols=4, figsize=(16, 16))
    for idx, img in enumerate(batch_img[0][:4]):
        axes[idx].imshow(img.astype(int))
        axes[idx].title.set_text(batch_img[1][idx])

    pyplot.show()


batch_count = len(img_data)

train_size = int(batch_count * 0.7)
validation_size = int(batch_count * 0.2)
test_size = int(batch_count * 0.1) + 1


train = img_data.take(train_size)
validation = img_data.skip(train_size).take(validation_size)
test = img_data.skip(train_size + validation_size).take(test_size)


model = Sequential()

model.add(Conv2D(16, (3, 3), 1, activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3, 3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3, 3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])


logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
history = model.fit(train, epochs=20, validation_data=validation, callbacks=[tensorboard_callback])


def train():
    figure = pyplot.figure()
    pyplot.plot(history.history['loss'], color='green', label='loss')
    pyplot.plot(history.history['val_loss'], color='orange', label='val_loss')
    figure.suptitle('Loss', fontsize=20)
    pyplot.legend(loc='upper left')
    pyplot.show()

    figure = pyplot.figure()
    pyplot.plot(history.history['accuracy'], color='green', label='accuracy')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='val_accuracy')
    figure.suptitle('Accuracy', fontsize=20)
    pyplot.legend(loc="upper left")
    pyplot.show()


train()


pre = Precision()
re = Recall()
acc = BinaryAccuracy()

for batch in test.as_numpy_iterator():
    X, Y_true = batch
    Y_prediction = model.predict(X)
    pre.update_state(Y_true, Y_prediction)
    re.update_state(Y_true, Y_prediction)
    acc.update_state(Y_true, Y_prediction)

print(pre.result(), re.result(), acc.result())


image_path = os.path.join('test', 'test_dog.jpg')
img = cv2.imread(image_path)
pyplot.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
pyplot.show()

resize = tf.image.resize(img, (256, 256))
pyplot.imshow(resize.numpy().astype(int))
pyplot.show()


Y_prediction = model.predict(np.expand_dims(resize / 255, 0))
print(Y_prediction)

if Y_prediction > 0.5:
    print('dog')
else:
    print('cat')


model.save(os.path.join('models', 'images_recognition.h5'))
new_model = load_model('images_recognition.h5')
new_model.predict(np.expand_dims(resize/255, 0))






