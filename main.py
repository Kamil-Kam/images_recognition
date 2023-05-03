import tensorflow as tf
from data_functions import clear_data
import cv2
from matplotlib import pyplot
import numpy as np
import os


data_dir = 'data'
img_extensions = ['jpg', 'jpeg', 'png']



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

train_size = batch_count * 0.7
validation_size = batch_count * 0.2
test_size = batch_count * 0.1

print(train_size, validation_size, test_size)


def main():
    clear_data(data_dir, img_extensions)


# if __name__ == '__main__':
#     main()



