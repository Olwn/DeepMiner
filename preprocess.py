import numpy as np


def contrast(image):
    m = np.mean(image, axis=(0, 1), keepdims=True, dtype=np.float32)
    f = np.random.uniform(low=0.75, high=1.25, size=(1, 1, 3)).astype(np.float32)
    return (image - m) * f + m


def bright(image):
    v = np.var(image, axis=(0, 1), keepdims=True, dtype=np.float32)
    factor = 0.05
    f = np.random.uniform(low=-1*factor, high=factor, size=(1, 1, 3)).astype(np.float32)
    return image + f * np.sqrt(v)


def noise(mode):
    def gaussian(image):
        image += np.random.normal(scale=0.5, size=image.shape)
        return image

    def mask(image):
        image *= np.random.binomial(n=1, p=1 - 0.25, size=image.shape)
        return image
    out = {'gaussian': gaussian, 'mask': mask}
    return out[mode]


def mask(ratio):
    def func(image):
        h = image.shape[0]
        mask_size = int(ratio * h)
        m = np.mean(image, axis=(0, 1)).reshape((1, 1, 3))
        ty = int(np.random.uniform(low=0, high=mask_size))
        tx = int(np.random.uniform(low=0, high=mask_size))
        image[ty:ty+mask_size, tx:tx+mask_size] = m
        return image
    return func


def multi_flip():
    def func(image):
        x = flip(2, 0.5, 0)(image)
        x = flip(4, 0.5, 0)(x)
        x = flip(8, 0.3, 2)(x)
        return x
    return func


def column_mask(width, spaces):
    def func(image):
        mode = 0 if np.random.uniform(0, 1) < 0.5 else 1
        m = np.mean(image, axis=(0, 1)).reshape((1, 1, 3))
        for i in range(image.shape[0] / (width + spaces)):
            c1 = i * (width + spaces)
            c2 = c1 + width
            if mode == 0:
                image[:, c1:c2] = m
            else:
                image[c1:c2] = m
        return image
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
            r2 = patch_size * (i + 1)
            for j in range(image.shape[1] / patch_size):
                c1 = patch_size * j
                c2 = patch_size * (j + 1)
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

