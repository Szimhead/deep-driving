import os
import numpy as np
from jproperties import Properties
import random
import cv2


SEED = 42


configs = Properties()
with open('local.properties', 'rb') as read_prop:
    configs.load(read_prop)


def get_data():
    cars = []
    masks = []
    data_dir = configs.get("OVERLAPPED").data
    for image_name in os.listdir(data_dir):
        image_path = os.path.join(data_dir, f"{image_name}")
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        car, mask = np.array_split(image, [3], axis=2)
        car = cv2.cvtColor(car, cv2.COLOR_BGR2GRAY)
        mask = mask.reshape((256, 256))
        cars.append(car)
        masks.append(mask)
    random.Random(SEED).shuffle(cars)
    random.Random(SEED).shuffle(masks)
    return cars, masks


if __name__ == '__main__':
    cars, masks = get_data()
    thecar = cars[15]
    adjusted = cv2.convertScaleAbs(thecar, alpha=2)

    cv2.imshow("original", thecar)
    cv2.imshow("adjusted", adjusted)

    cv2.waitKey(0)

