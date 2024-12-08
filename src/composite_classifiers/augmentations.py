
import random
import torchvision.transforms.functional as tf


class BasicAugmentations(object):
    def __init__(self, basic_params):
        self.saturation = basic_params.get("saturation")
        self.hue = basic_params.get("hue")
        self.brightness = basic_params.get("brightness")
        self.gamma = basic_params.get("gamma")
        self.contrast = basic_params.get("contrast")
        self.degrees = basic_params.get("rotate")
        self.shear = basic_params.get("shear")
        self.scale = basic_params.get("scale")
        self.translate = basic_params.get("translate")
        self.hflip = basic_params.get("hflip")

    def __call__(self, img):
        img = tf.adjust_saturation(img, random.uniform(1 - self.saturation, 1 + self.saturation))
        img = tf.adjust_hue(img, random.uniform(-self.hue, self.hue))
        img = tf.adjust_contrast(img, random.uniform(1 - self.contrast, 1 + self.contrast))
        img = tf.adjust_brightness(img, random.uniform(1 - self.brightness, 1 + self.brightness))
        img = tf.adjust_gamma(img, random.uniform(1, 1 + self.gamma))

        rotate_degree = 2 * random.random() * self.degrees - self.degrees
        scale_f = (self.scale[1] - self.scale[0]) * random.random() + self.scale[0]
        tu = 2 * random.random() * self.translate[0] - self.translate[0]
        tv = 2 * random.random() * self.translate[1] - self.translate[1]
        shear_x = 2 * random.random() * self.shear - self.shear
        shear_y = 2 * random.random() * self.shear - self.shear

        do_hflip = False
        if random.random() < self.hflip:
            do_hflip = True

        img = tf.affine(img, translate=[tu, tv], scale=scale_f, angle=rotate_degree, interpolation=tf.InterpolationMode.BILINEAR, shear=[shear_x,shear_y])
        # img = tf.affine(img, translate=[tu, tv], scale=scale_f, angle=rotate_degree, interpolation=tf.InterpolationMode.BILINEAR)
        if do_hflip is True:
            img = tf.hflip(img)

        return img