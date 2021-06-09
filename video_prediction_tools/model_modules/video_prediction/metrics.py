import tensorflow as tf
#import lpips_tf
import math
from skimage.measure import compare_ssim as ssim_ski

def mse(a, b):
    return tf.reduce_mean(tf.squared_difference(a, b), [-3, -2, -1])


def psnr(a, b):
    return tf.image.psnr(a, b, 1.0)


def ssim(a, b):
    return tf.image.ssim(a, b, 1.0)


def psnr_imgs(img1, img2, pixel_max=1.):
    mse_all = mse(img1, img2)
    if mse_all == 0: return 100
    return 20 * math.log10(pixel_max / math.sqrt(mse_all))


def mse_imgs(image1,image2):
    mse = ((image1 - image2)**2).mean(axis=None)
    return mse

# def lpips(input0, input1):
#     if input0.shape[-1].value == 1:
#         input0 = tf.tile(input0, [1] * (input0.shape.ndims - 1) + [3])
#     if input1.shape[-1].value == 1:
#         input1 = tf.tile(input1, [1] * (input1.shape.ndims - 1) + [3])
#
#     distance = lpips_tf.lpips(input0, input1)
#     return -distance

def ssim_images(image1, image2):
    """
    Reference for calculating ssim
    Numpy impelmeentation for ssim https://cvnote.ddlee.cc/2019/09/12/psnr-ssim-python
    https://scikit-image.org/docs/dev/auto_examples/transform/plot_ssim.html
    :param image1 the reference images
    :param image2 the predicte images
    """
    ssim_pred = ssim_ski(image1, image2,
                      data_range = image2.max() - image2.min())
    return ssim_pred
