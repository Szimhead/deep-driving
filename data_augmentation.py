import os
import numpy as np
from PIL import Image
from jproperties import Properties
import matplotlib.pyplot as plt
import random

SEED = 42


def load_numpy_arrays(folder_path):
    black_cars = []
    orange_cars = []
    test = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.npy'):
            file_path = os.path.join(folder_path, file_name)
            array = np.load(file_path)
            if file_name.startswith('black'):
                black_cars.append(array)
            elif file_name.startswith('orange'):
                orange_cars.append(array)
            else:
                test.append(array)

    black_cars = np.array(black_cars)
    orange_cars = np.array(orange_cars)
    test = np.array(test)
    return black_cars, orange_cars, test


def process_image(mask_folder, image_folder, image_name, color):
    mask_path = os.path.join(mask_folder, f"{color}_{image_name[:4]}.npy")
    image_path = os.path.join(image_folder, f"{image_name}")

    mask = np.load(mask_path)[:,:,3]
    image = Image.open(image_path)

    # Resize the image to 256x256
    image = image.resize((256, 256))

    # Convert the images to NumPy arrays
    image = np.array(image)

    return image, mask


# all_landscape_arrays = []
# # Iterate over all files in the "landscapes" folder
# for file_name in os.listdir(landscapes_folder):
#     if file_name.endswith('.jpg'):
#         # Construct the full path to the image
#         image_path = os.path.join(landscapes_folder, file_name)
#         # Process the image and append the resulting array to the list
#         landscape_ar = process_image(image_path)
#         all_landscape_arrays.append(landscape_ar)

# Convert the list to a NumPy array if needed
# all_landscape_arrays = np.array(all_landscape_arrays)
# j = 0

# for landscape_array in all_landscape_arrays:
#     j += 1
#     for t in range(len(b_cars)):
#         new_bkg = np.zeros_like(landscape_array)
#         target_channel_array = info[t]
#         target_zero_mask = target_channel_array == 0
#         for k in range(2):
#             new_bkg[:, :, k] = np.where(target_zero_mask, landscape_array[:, :, k], b_cars[t, :, :, k])
#
#         new_bkg_uint8 = np.uint8(new_bkg)
#         new_image = Image.fromarray(new_bkg_uint8)
#         image_path = 'C:\\Users\\aless\\OneDrive - Danmarks Tekniske Universitet\\Period_13_1\\DL\\project\\carseg_data\\carseg_data\\images\\black_5_doors_landscape'
#         image_name = str(t + 1) + '_' + str(j) + '.jpg'
#         file_path = os.path.join(image_path, image_name)
#         # Save the image to a file
#         new_image.save(file_path)

def get_data():
    configs = Properties()
    with open('local.properties', 'rb') as read_prop:
        configs.load(read_prop)
    masks = []
    images = []

    for image_name in os.listdir(configs.get("BLACK_IMG").data):
        # check if the image ends with png
        if (image_name.endswith(".png")):
            image, mask = process_image(configs.get("FOLDER_PATH").data, configs.get("BLACK_IMG").data, image_name,'black_5_doors')
            masks.append(mask)
            images.append(image)

    for image_name in os.listdir(configs.get("ORANGE_IMG").data):
        # check if the image ends with png
        if (image_name.endswith(".png")):
            image, mask = process_image(configs.get("FOLDER_PATH").data, configs.get("ORANGE_IMG").data, image_name,'orange_3_doors')
            masks.append(mask)
            images.append(image)
    random.Random(SEED).shuffle(images)
    random.Random(SEED).shuffle(masks)
    return images, masks


if __name__ == '__main__':
    configs = Properties()
    with open('application.properties', 'rb') as read_prop:
        configs.load(read_prop)

    # Sample mapping dictionary
    class_mapping = {
        10: ("orange", "hood"),
        20: ("dark green", "front door"),
        30: ("yellow", "rear door"),
        40: ("cyan", "frame"),
        50: ("purple", "rear quater panel"),
        60: ("light green", "trunk lid"),
        70: ("blue", "fender"),
        80: ("pink", "bumper"),
        90: ("no color", "rest of car")
    }

    b_cars, o_cars, test = load_numpy_arrays(configs.get("FOLDER_PATH").data)
    masks = []
    images = []

    print(configs.get("BLACK_IMG"))
    print(configs.get("BLACK_IMG").data)
    for image_name in os.listdir(configs.get("BLACK_IMG").data):
        # check if the image ends with png
        if (image_name.endswith(".jpg")):
            image, mask = process_image(configs.get("FOLDER_PATH").data, configs.get("BLACK_IMG").data, image_name,'black_5_doors')
            masks.append(mask)
            images.append(image)

    for image_name in os.listdir(configs.get("ORANGE_IMG").data):
        # check if the image ends with png
        if (image_name.endswith(".jpg")):
            image, mask = process_image(configs.get("FOLDER_PATH").data, configs.get("ORANGE_IMG").data, image_name,'orange_3_doors')
            masks.append(mask)
            images.append(image)


