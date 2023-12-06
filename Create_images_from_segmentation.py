from PIL import Image, ImageDraw, ImageChops
import random
from PIL import Image
import time
import os
import numpy as np
import cv2
import shutil

from jproperties import Properties


def read_properties_file(file_path):
    """
    Reads a properties file with a simple key-value pair structure.
    """
    properties = {}
    with open(file_path, 'r') as file:
        for line in file:
            if '=' in line:
                key, value = line.strip().split('=', 1)
                properties[key.strip()] = value.strip()
    return properties


def clear_directory(directory):
    if os.path.exists(directory):
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')


def create_combined_mask(hsv_image, hsv_ranges, exclude_black=True):
    # Initialize an empty mask
    combined_mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)

    # Loop over all provided ranges
    for (h_min, s_min, v_min), (h_max, s_max, v_max) in hsv_ranges:
        # Create a mask for the current range
        lower_bound = np.array([h_min, s_min * 255 / 100, v_min * 255 / 100], dtype=np.uint8)
        upper_bound = np.array([h_max, s_max * 255 / 100, v_max * 255 / 100], dtype=np.uint8)
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

        # If exclude_black is True, exclude completely black regions
        if exclude_black and (h_min, s_min, v_min) == (0, 0.0, 0.0):
            mask = cv2.bitwise_not(mask)

        # Combine the current mask with the combined mask
        combined_mask = cv2.bitwise_or(combined_mask, mask)

    return combined_mask


def load_matching_filenames(segmented_folder, non_segmented_folder, extension, count=10):
    segmented_files = {file for file in os.listdir(segmented_folder) if file.endswith(extension)}
    non_segmented_files = {file for file in os.listdir(non_segmented_folder) if file.endswith(extension)}

    matching_files = list(segmented_files.intersection(non_segmented_files))

    return random.sample(matching_files, min(len(matching_files), count))


def load_random_images(image_folder, image_format, max_images=10):
    all_images = [os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith(image_format)]
    return random.sample(all_images, min(len(all_images), max_images))


def load_random_filenames(folder, extension, count=10):
    all_files = [file for file in os.listdir(folder) if file.endswith(extension)]
    return random.sample(all_files, min(len(all_files), count))


def extract_car_using_mask(non_segmented_image_path, mask):
    # Load the non-segmented car image
    car_image = cv2.imread(non_segmented_image_path, cv2.IMREAD_UNCHANGED)

    # Make sure the mask is a single channel image since it's a binary mask
    if len(mask.shape) == 3 and mask.shape[2] == 3:
        # If the mask is a color image, convert it to grayscale
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

    # Threshold the mask to make sure the background is 0
    _, binary_mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

    # Ensure the car image has an alpha channel
    if car_image.shape[2] == 3:
        car_image = cv2.cvtColor(car_image, cv2.COLOR_BGR2BGRA)

    # Set the alpha channel to the binary mask
    car_image[:, :, 3] = binary_mask

    # Convert the BGR image with alpha to an RGBA PIL image for compatibility
    car_image_rgba = cv2.cvtColor(car_image, cv2.COLOR_BGRA2RGBA)
    car_image_pil = Image.fromarray(car_image_rgba)

    return car_image_pil


def composite_car_on_landscape(car, landscape_image_path, target_size=(255, 255)):
    # Load the landscape image
    landscape = Image.open(landscape_image_path).convert("RGBA")

    # Resize the landscape image to the target size using 'BILINEAR' filter
    landscape = landscape.resize(target_size, Image.BILINEAR)

    # Convert the car image to RGBA if it's not already
    if car.mode != "RGBA":
        car = car.convert("RGBA")

    # Resize the car if it's larger than the landscape
    if car.size[0] > target_size[0] or car.size[1] > target_size[1]:
        car.thumbnail(target_size, Image.BILINEAR)

    # Create a new image with the same size as the landscape image
    combined_image = Image.new("RGBA", target_size)

    # Paste the landscape onto the combined image
    combined_image.paste(landscape, (0, 0))

    # # Calculate the position to place the car
    # max_x_position = max(target_size[0] - car.size[0], 0)
    # max_y_position = max(target_size[1] - car.size[1], 0)
    # position = (random.randint(0, max_x_position),
    #             random.randint(0, max_y_position))

    # Paste the car onto the combined image at the random position
    combined_image.paste(car, car)

    # Convert back to RGB to remove the alpha channel for final output
    return combined_image.convert("RGB")


hsv_ranges = [
    ((82, 75, 90), (96, 100, 100)),  # Cyan-like color
    ((112, 85, 85), (128, 100, 100)),  # Blue color
    ((305, 70, 34.0), (325, 100, 44.0)),  # green color
    ((146, 80, 55), (156, 100, 65)),  # purple color
    ((23, 85, 85), (38, 100, 100)),  # Yellow color
    ((14, 65, 75), (25, 100, 100)),  # Orange color
    ((138, 85, 84), (160, 100, 100)),  # Magenta color
    # ((120.0 / 2 - x, 80 - x, 38 - x), (120.0 / 2 + x, 80.0 + x, 38.0 + x)),  # Additional color 1

]


def load_images(image_folder, image_format=('.jpg', '.png'), max_images=3000):
    images = []
    for file_name in sorted(os.listdir(image_folder)):  # Use sorted to ensure consistent order
        if file_name.endswith(image_format):
            image_path = os.path.join(image_folder, file_name)
            with Image.open(image_path) as img:
                images.append(img.copy())  # Copy the image to avoid lazy loading
            if len(images) >= max_images:
                break

    print(f"Loaded {len(images)} images from {image_folder}")
    return images


def process_images(segmented_folder, non_segmented_folder, landscapes_folder, save_dir, total_images=2520,
                   file_extension='.png'):
    print("Starting image processing...")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    landscape_images = load_random_images(landscapes_folder, '.jpg', 255)
    matching_filenames = load_matching_filenames(segmented_folder, non_segmented_folder, file_extension, 50)

    processed_count = 0
    for filename in matching_filenames:
        print(f"Processing: {filename}")
        try:
            segmented_image_path = os.path.join(segmented_folder, filename)
            non_segmented_image_path = os.path.join(non_segmented_folder, filename)

            # Create mask from segmented image
            image_bgr = cv2.imread(segmented_image_path)
            image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
            mask = create_combined_mask(image_hsv, hsv_ranges)

            # Extract car from non-segmented image using mask
            car_extracted = extract_car_using_mask(non_segmented_image_path, np.array(mask))

            # Composite the extracted car onto random landscapes
            for i in range(total_images // len(matching_filenames)):
                landscape_image_path = random.choice(landscape_images)
                combined_image = composite_car_on_landscape(car_extracted, landscape_image_path)
                output_filename = f"{filename[:-4]}_combined_{i}.jpg"
                output_path = os.path.join(save_dir, output_filename)
                combined_image.save(output_path)
                processed_count += 1
                if processed_count >= total_images:
                    break
        except Exception as e:
            print(f"Error processing file {filename}: {e}")

        if processed_count >= total_images:
            break

    print("Image processing completed.")


if __name__ == '__main__':
    # Read configurations
    configs = Properties()
    with open('application.properties', 'rb') as read_prop:
        configs.load(read_prop)

    # Use the properties in your logic
    configurations = [
        {
            "segmented_folder": configs.get("REAL_MASK").data,
            "non_segmented_folder": configs.get('REAL_IMG').data,
            "landscapes_folder": configs.get('LANDSCAPES').data,  # Assuming landscapes are in LANDSCAPES
            "save_directory": configs.get('REAL_CARS_READY').data,
            "file_extension": '.jpg'  # Assuming jpg for real cars
        },
        {
            "segmented_folder": configs.get('BLACK_MASK').data,
            "non_segmented_folder": configs.get('BLACK_IMG').data,
            "landscapes_folder": configs.get('LANDSCAPES').data,  # Assuming the same for all
            "save_directory": configs.get('BLACK_CARS_READY').data,
            "file_extension": '.png'  # Assuming png for black cars
        },
        {
            "segmented_folder": configs.get('ORANGE_MASK').data,  # Check for the typo in 'ORAGNE'
            "non_segmented_folder": configs.get('ORANGE_IMG').data,
            "landscapes_folder": configs.get('LANDSCAPES').data,  # Assuming the same for all
            "save_directory": configs.get('ORANGE_CARS_READY').data,
            "file_extension": '.png'  # Assuming png for orange cars
        }
    ]

    for config in configurations:
        print(f"Clearing directory: {config['save_directory']}")
        clear_directory(config['save_directory'])

        print(f"Processing images for {config['save_directory']}")
        process_images(config["segmented_folder"], config["non_segmented_folder"],
                       config["landscapes_folder"], config["save_directory"], file_extension=config["file_extension"])
