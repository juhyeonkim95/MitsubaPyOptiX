import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def load_image(path):
    parent_path = "/home/juhyeon/practical-path-guiding/scenes"
    image = cv.imread(parent_path + path).astype(np.float32)
    # tone_map = cv.createTonemap(gamma=2.2)
    tone_map = cv.createTonemapReinhard(gamma=2.2, intensity=0, light_adapt=0, color_adapt=0)
    image = tone_map.process(image)
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    return image


def main():
    #kitchen = load_image("/kitchen/kitchen-improved.exr")
    kitchen = load_image("/spaceship/spaceship.exr")

    plt.imshow(kitchen)
    plt.show()


if __name__ == '__main__':
    main()