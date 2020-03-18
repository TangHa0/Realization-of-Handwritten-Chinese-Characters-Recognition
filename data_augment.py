from PIL import Image
from PIL import ImageEnhance
import tensorflow as tf
import cv2
import numpy as np


# image = Image.open('F:\\Python\\data\\train\\00055\\583715.png')
# image.show()
# print(image.format, image.size, image.mode)
image = cv2.imread("F:\\Python\\data\\train\\00055\\583715.png")  # 习
# image = cv2.imread("F:\\Python\\data\\train\\00186\\70808.png")  # 俐
cv2.imshow("resource", image)
# image = image.convert('L')
# image = image.resize((64, 64), Image.ANTIALIAS)
# image = np.asarray(image) / 255.0
# image = image.reshape([-1, 64, 64, 1])


image = tf.image.random_flip_up_down(image)
# image = tf.image.random_contrast(image, 0.8, 2.0)
# image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
# image = tf.image.adjust_brightness(image, delta=-0.1)
# image = tf.image.random_brightness(image, max_delta=0.3)
with tf.Session() as sess:
    result = sess.run(image)
result = np.uint8(result)
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

