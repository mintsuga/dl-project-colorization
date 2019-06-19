import tensorflow as tf
import numpy as np
from skimage.color import lab2rgb, rgb2lab

def convert2rgb(x, predicted):
    assert x.shape[0] == predicted.shape[0], 'number of samples unequal!'

    n_sample = predicted.shape[0]
    ret = []
    for cnt in range(n_sample):
        predicted_val = predicted[cnt]  # get predicted value of a single image

        predicted_img = np.zeros((32, 32, 3))
        predicted_img[:,:,0] = x[cnt][:,:,0]      # light channel stay same
        predicted_img[:,:,1:] = predicted_val[0]    # fill the last channels with predicted values

        predicted_img = (predicted_img * [100, 255, 255]) - [0, 128, 128]   # reverted from norm

        rgb_predicted = lab2rgb(predicted_img)  # convert lab to rgb
        rgb_predicted = rgb_predicted * [255, 255, 255]
        rgb_predicted = rgb_predicted.astype(np.uint8)
        ret.append(rgb_predicted)
    return ret


def write_log(callback, names, logs, batch_cnt):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_cnt)
        callback.writer.flush()
