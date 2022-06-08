import cv2
import numpy as np
import matplotlib.image as mpimg
import pandas as pd
import config as c


def load_data(data_dir):
    """ read csv file data to dataframe """
    df = pd.read_csv(data_dir, names=c.columns)
    df['center'] = df['center'].apply(lambda x: c.data_img_dir + "\\" + x.split('\\')[-1])
    df['left'] = df['left'].apply(lambda x: c.data_img_dir + "\\" + x.split('\\')[-1])
    df['right'] = df['right'].apply(lambda x: c.data_img_dir + "\\" + x.split('\\')[-1])
    return df[c.columns_need]


# Load image from path
def load_image(image_path):
    return mpimg.imread(image_path)


def choose_image(center, left, right, steering_angle):
    """
    For each image, choose one: center camera, left camera or right camera
    and adjust steering_angle accordingly.
    """
    choice = np.random.choice(3)
    if choice == 0:
        return load_image(left), steering_angle + 0.2
    elif choice == 1:
        return load_image(right), steering_angle - 0.2
    return load_image(center), steering_angle


def random_translate(image, steering_angle, range_x=100, range_y=10):
    """
    Randomly translate image horizontally and vertically between range (-50, 50) and (-5, 5)
    respectively and adjust steering_angle following horizontally.
    """
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]

    image = cv2.warpAffine(image, trans_m, (width, height))
    steering_angle += trans_x * 0.002
    return image, steering_angle


# Randomly flip image horizontally and adjust steering_angle
def random_flip(image, steering_angle):
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle


# Randomly adjust brightness of image with ratio in range (0.8, 1.2), default 1.0
def random_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:, :, 2] = hsv[:, :, 2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


# Data Augmentation
def augment(center, left, right, steering_angle):
    image, steering_angle = choose_image(center, left, right, steering_angle)
    image, steering_angle = random_flip(image, steering_angle)
    image, steering_angle = random_translate(image, steering_angle)
    image = random_brightness(image)
    return image, steering_angle


def preprocessing(image):
    """
    Drop image from below the horizon and above the steering wheel.
    Resize to (200, 66, channels).
    Split into YUV planes.
    Scale pixel values to the range 0-1.
    """
    image = image[60:135, :, :]
    image = cv2.resize(image, (c.img_width, c.img_height))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    image = image / 255.0
    return image


def batch_generator(df, batch_size, is_training):
    images = df[['center', 'left', 'right']].values
    steers = df['steering'].values

    X = np.empty([batch_size, c.img_height, c.img_width, c.img_channels])
    y = np.empty(batch_size)
    while True:
        i = 0
        for idx in range(len(df)):
            center, left, right = images[idx]
            steering_angle = steers[idx]
            if is_training and np.random.rand() < 0.6:
                image, steering_angle = augment(center, left, right, steering_angle)
            else:
                image = load_image(center)
            X[i] = preprocessing(image)
            y[i] = steering_angle
            i += 1
            if i == batch_size:
                break
        yield X, y
