import tensorflow as tf
import os
import cv2
import imghdr
from matplotlib import pyplot
import numpy as np


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# print(len(gpus))
# print(tf.config.list_physical_devices('GPU'))


data_dir = 'data'
img_extensions = ['jpg', 'jpeg', 'png']
data_folder = os.listdir(data_dir)
data_folder_1 = os.listdir(os.path.join(data_dir, 'cats'))


# img = cv2.imread(os.path.join(data_dir, 'dogs', 'black-lab-favorite-dog-main-220315-e8e0ee.jpg'))
#
# pyplot.imshow(img)
# pyplot.show()


def clear_data():
    for image_folder in os.listdir(data_dir):
        for image in os.listdir(os.path.join(data_dir, image_folder)):
            image_path = os.path.join(data_dir, image_folder, image)

            try:
                extension = imghdr.what(image_path)
                if extension not in img_extensions:
                    print(image_path)
                    os.remove(image_path)
            except:
                print(f'issue with {image_path}')
                os.remove(image_path)


img_data = tf.keras.utils.image_dataset_from_directory('data')
img_data_iterator = img_data.as_numpy_iterator()
single_img = img_data_iterator.next()

print(single_img[0].shape)
print(single_img[1])






