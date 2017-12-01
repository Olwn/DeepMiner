import numpy as np


def multi_flip():
    def func(image):
        x = flip(2, 0.5, 0)(image)
        x = flip(4, 0.5, 0)(x)
        x = flip(8, 0.3, 2)(x)
        return x
    return func


def flip(patch_size=4, p=0.3, option=0):
    def func(image):
        """
        randomly flip image patch
        :param image: 
        :return: 
        """
        for i in range(image.shape[0] / patch_size):
            r1 = patch_size * i
            r2 = patch_size * (i + 1) - 1
            for j in range(image.shape[1] / patch_size):
                c1 = patch_size * j
                c2 = patch_size * (j + 1) - 1
                if np.random.uniform(low=0, high=1) < p:
                    if option == 0:
                        image[r1:r2, c1:c2] = np.transpose(image[r1:r2, c1:c2], axes=[1, 0, 2])
                    elif option == 1:
                        image[r1:r2, c1:c2] = np.fliplr(image[r1:r2, c1:c2])
                    elif option == 2:
                        image[r1:r2, c1:c2] = np.flipud(image[r1:r2, c1:c2])
        return image
    return func


def flip2(patch_size=4, p=0.3, option=0):
    def func(image):
        """
        randomly flip image patch
        :param image: 
        :return: 
        """
        for i in range(image.shape[0] / patch_size):
            r1 = patch_size * i
            r2 = (patch_size + 1) * i - 1
            for j in range(image.shape[1] / patch_size):
                c1 = patch_size * j
                c2 = (patch_size + 1) * j - 1
                if np.random.uniform(low=0, high=1) < p:
                    image[r1:r2, c1:c2] = np.flipud(image[r1:r2, c1:c2])
        return image
    return func

