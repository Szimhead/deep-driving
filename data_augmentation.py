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


# def process_image(image_path):
#     img = Image.open(image_path)
#     # Resize the image to 256x256
#     img = img.resize((256, 256))
#     # Convert the image to a NumPy array
#     landscape_array = np.array(img)
#     return landscape_array

def process_image(mask_folder, image_folder, image_name,color):
    mask_path = os.path.join(mask_folder, f"{color}_{image_name[:4]}.npy")
    image_path = os.path.join(image_folder, f"{image_name}")

    #mask = Image.open(mask_path)
    mask = np.load(mask_path)
    image = Image.open(image_path)

    # Resize the image to 256x256
    #image = image.resize((256, 256))

    # Convert the images to NumPy arrays
    #mask_array = np.array(mask)
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
    with open('application.properties', 'rb') as read_prop:
        configs.load(read_prop)
    b_cars, o_cars, test = load_numpy_arrays(configs.get("FOLDER_PATH").data)
    masks = []
    images = []
    for channels in b_cars:
        target_channel = channels[:, :, 3]
        image = channels[:, :, :3]
        masks.append(target_channel)
        images.append(image)

    for channels in o_cars:
        target_channel = channels[:, :, 3]
        image = channels[:, :, :3]
        masks.append(target_channel)
        images.append(image)
    random.Random(SEED).shuffle(images)
    random.Random(SEED).shuffle(masks)
    return images, masks


if __name__ == '__main__':
    configs = Properties()
    with open('application.properties', 'rb') as read_prop:
        configs.load(read_prop)


    # Specify the path to your folder containing NumPy arrays
    # folder_path = 'C:\\Users\\aless\\OneDrive - Danmarks Tekniske Universitet\\Period_13_1\\DL\\project\\carseg_data\\carseg_data\\arrays'
    # landscapes_folder = 'C:\\Users\\aless\\OneDrive - Danmarks Tekniske Universitet\\Period_13_1\\DL\\project\\carseg_data\\carseg_data\\images\\landscapes'

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
            #image_name = os.path.splitext(os.path.basename(configs.get("BLACK_MASK").data))[0][:4]  # Extracting the first 4 digits of the file name without extension
            image, mask = process_image(configs.get("FOLDER_PATH").data, configs.get("BLACK_IMG").data, image_name,'black_5_doors')
            masks.append(mask)
            images.append(image)

    for image_name in os.listdir(configs.get("ORANGE_IMG").data):
        # check if the image ends with png
        if (image_name.endswith(".jpg")):
            #image_name = os.path.splitext(os.path.basename(configs.get("BLACK_MASK").data))[0][:4]  # Extracting the first 4 digits of the file name without extension
            image, mask = process_image(configs.get("FOLDER_PATH").data, configs.get("ORANGE_IMG").data, image_name,'orange_3_doors')
            masks.append(mask)
            images.append(image)


    # for channels in b_cars:
    #     target_channel = channels[:, :, 3]
    #     image = channels[:, :, :3]
    #     target_info = np.vectorize(class_mapping.get)(target_channel)
    #     masks.append(target_channel)
    #     images.append(image)   
    #print(target_channel)
    # plt.figure(figsize=(18, 6))
    # plt.imshow(images[0])
    # plt.imshow(masks[0])
    # im = Image.open(configs.get("BLACK_MASK").data+"\\0001.png")
    # from collections import defaultdict
    # by_color = defaultdict(int)
    # for pixel in im.getdata():
    #     by_color[pixel] += 1

    # print(list(filter(lambda x: by_color[x] > 200, by_color)))

    # amount = 5
    # image_sample = random.choices(b_cars, k=amount)
    # resized_images = []
    # raw_images = []

    # # Define figure size
    # fig = plt.figure(figsize=(18, 6))

    # # Save original images in the figure
    # # ax = plt.subplot(2, amount + 1, 1)
    # # txt = ax.text(0.4, 0.5, 'Original', fontsize=20)
    # # txt.set_clip_on(False)

    # plt.axis('off')
    # for i, image in enumerate(image_sample):
    #     i += len(image_sample) + 3
    #     plt.subplot(2, amount + 1, i)
    #     # image = imread(path, as_gray=True)
    #     # ret, thresh = cv.threshold(image, 0.95, 1, cv.THRESH_BINARY)
    #     #
    #     # cropped = pad2square(image, thresh)  # Make the image square
    #     # image = resize(image, output_shape=image_size, mode='reflect', anti_aliasing=True)  # resizes the image
    #     # resized_images.append(cropped)
    #     # raw_images.append(image)

    #     plt.imshow(image)

    # Show plot
    plt.show()
