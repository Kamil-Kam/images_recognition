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

# clear_data(data_dir, img_extensions)

img_data = tf.keras.utils.image_dataset_from_directory('data')
img_data = img_data.map(lambda x, y: (x / 255, y))
img_data_iterator = img_data.as_numpy_iterator()
batch_img = img_data_iterator.next()



# print(batch_img[0])
# print(batch_img[0].max())
# print(batch_img[1].shape)


def show_images():
    fig, ax = pyplot.subplots(ncols=4, figsize=(20, 20))
    for idx, img in enumerate(batch_img[0][:4]):
        ax[idx].imshow(img.astype(int))
        ax[idx].title.set_text(batch_img[1][idx])

    pyplot.show()


"""
1 = dog
0 = cat
"""


batch_count = len(img_data)

train_size = int(batch_count * 0.7)
validation_size = int(batch_count * 0.2)
test_size = int(batch_count * 0.1) + 1

print(train_size, validation_size, test_size)

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

# print(model.summary())


logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train, epochs=20, validation_data=validation, callbacks=[tensorboard_callback])

def train():
    fig = pyplot.figure()
    pyplot.plot(hist.history['loss'], color='teal', label='loss')
    pyplot.plot(hist.history['val_loss'], color='orange', label='val_loss')
    fig.suptitle('Loss', fontsize=20)
    pyplot.legend(loc='upper left')
    pyplot.show()


    fig = pyplot.figure()
    pyplot.plot(hist.history['accuracy'], color='teal', label='accuracy')
    pyplot.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
    fig.suptitle('Accuracy', fontsize=20)
    pyplot.legend(loc="upper left")
    pyplot.show()



pre = Precision()
re = Recall()
acc = BinaryAccuracy()

for batch in test.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)

print(pre.result(), re.result(), acc.result())


image_path = os.path.join('test', 'test_dog.jpg')
print(image_path)
img = cv2.imread(image_path)
pyplot.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
pyplot.show()

resize = tf.image.resize(img, (256,256))
pyplot.imshow(resize.numpy().astype(int))
pyplot.show()


yhat = model.predict(np.expand_dims(resize/255, 0))
print(yhat)

if yhat > 0.5:
    print('dog')
else:
    print('cat')



model.save(os.path.join('models','imagec_recognition.h5'))
new_model = load_model('iimagec_recognitionr.h5')
new_model.predict(np.expand_dims(resize/255, 0))


def main():
    clear_data(data_dir, img_extensions)


# if __name__ == '__main__':
#     main()



