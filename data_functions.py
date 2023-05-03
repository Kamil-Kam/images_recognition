import os
import imghdr


def clear_data(data_dir: str, img_extensions: list[str]) -> None:
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

