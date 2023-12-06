import numpy as np
import torchvision.transforms.functional as TF

class MyMirrorTransform:
    """Apply horizontal flip to an image and its keypoints with a certain probability.

    Attributes:
        apply_prob (float): The probability with which the horizontal flip is applied.
    """

    def __init__(self, config):
        self.apply_prob = config['DATA']['APPLY_PROB']

    def apply_image(self, image):
        return TF.hflip(image)

    def apply_keypoints(self, image_width, col):
        mid_point = image_width // 2
        return mid_point - (col - mid_point) if col > mid_point else mid_point + (mid_point - col)

    def __call__(self, image, col):
        if np.random.rand() < self.apply_prob:
            image = self.apply_image(image)
            col = self.apply_keypoints(image.width, col)
        return image, col


