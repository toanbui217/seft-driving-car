import cv2
import numpy as np
import matplotlib.image as mpimg


def load_image(image_path):
    img = mpimg.imread(image_path)
    return img


def pre_process(img):
    image = img[60:135, :, :]
    image = cv2.resize(image, (200, 66))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    image = image / 255.0
    return image


def choose_image(center, left, right, steering_angle):
    choice = np.random.choice(3)
    if choice == 0:
        return load_image(left), steering_angle + 0.2

    elif choice == 1:

        return load_image(right), steering_angle - 0.2

    return load_image(center), steering_angle


def random_translate(image, steering_angle, range_x, range_y):
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle


def random_flip(image, steering_angle):
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle


def random_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:, :, 2] = hsv[:, :, 2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def augmenter(center, left, right, steering_angle, range_x=100, range_y=10):
    image, steering_angle = choose_image(center, left, right, steering_angle)
    image, steering_angle = random_flip(image, steering_angle)
    image, steering_angle = random_translate(image, steering_angle, range_x, range_y)
    image = random_brightness(image)
    return image, steering_angle


def batch_generator(image_paths, steering_angles, batch_size, is_training):
    images = np.empty([batch_size, 66, 200, 3])
    steers = np.empty(batch_size)
    while True:
        i = 0
        for index in np.random.permutation(image_paths.shape[0]):
            center, left, right = image_paths[index]
            steering_angle = steering_angles[index]
            if is_training and np.random.rand() < 0.6:
                image, steering_angle = augmenter(center, left, right, steering_angle)
            else:
                image = load_image(center)
            images[i] = pre_process(image)
            steers[i] = steering_angle
            i += 1
            if i == batch_size:
                break
        yield images, steers
