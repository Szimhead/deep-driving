import os
import cv2
import numpy as np
from jproperties import Properties


configs = Properties()
with open('local.properties', 'rb') as read_prop:
    configs.load(read_prop)


def load_numpy_arrays(folder_path):
    cars = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.npy'):
            array = np.load(os.path.join(folder_path, file_name))
            cars.append(array)
    cars = np.array(cars)
    return cars


def fix_masks():
    arrays = load_numpy_arrays(configs.get("FOLDER_PATH").data)
    for ar in arrays:
        ar[:, :, 3][ar[:, :, 3] == 0] = 90
    return arrays


def load_backgrounds():
    landscapes = []
    landscapes_dir = configs.get("LANDSCAPES").data
    for file_name in os.listdir(landscapes_dir)[0::1]:
        landscape_ar = cv2.imread(os.path.join(landscapes_dir, file_name))
        landscape_ar = cv2.resize(landscape_ar, (256, 256))
        landscapes.append(landscape_ar)
    return landscapes


def overlap_car_with_bgr(car, bgr, i):
    result = car.copy()
    result[:, :, :3][car[:, :, 3] == 90] = bgr[car[:, :, 3] == 90]
    cv2.imwrite("new_overlapped/" + str(i) + ".png", result)


if __name__ == '__main__':
    cars = fix_masks()
    landscapes = load_backgrounds()
    i = 0
    for c, l in zip(cars, landscapes):
        overlap_car_with_bgr(c, l, i)
        i += 1


