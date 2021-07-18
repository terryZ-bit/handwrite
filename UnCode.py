import os
import numpy as np
import struct
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

train_data_dir = "HWDB1.1trn_gnt"
test_data_dir = "HWDB1.1tst_gnt"

writer = tf.compat.v1.python_io.TFRecordWriter("data.tfrecord")


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# 读取图像和对应的汉字
def read_from_gnt_dir(gnt_dir=train_data_dir):
    def one_file(f):
        header_size = 10
        while True:
            header = np.fromfile(f, dtype='uint8', count=header_size)
            if not header.size: break
            sample_size = header[0] + (header[1] << 8) + (header[2] << 16) + (header[3] << 24)
            tagcode = header[5] + (header[4] << 8)
            width = header[6] + (header[7] << 8)
            height = header[8] + (header[9] << 8)
            if header_size + width * height != sample_size:
                break
            image = np.fromfile(f, dtype='uint8', count=width * height).reshape((height, width))
            # image_tensor = tf.image.decode_image(image)

            example = tf.train.Example(features=tf.train.Features(
                feature={
                    'label': _int64_feature(tagcode),
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()])),
                    'width': _int64_feature(width),
                    'height': _int64_feature(height),
                }
            ))
            serialized = example.SerializeToString()
            writer.write(serialized)
            yield image, tagcode

    for file_name in os.listdir(gnt_dir):
        if file_name.endswith('.gnt'):
            file_path = os.path.join(gnt_dir, file_name)
            with open(file_path, 'rb') as f:
                for image, tagcode in one_file(f):
                    yield image, tagcode


# 统计样本数
train_counter = 0
test_counter = 0
for image, tagcode in read_from_gnt_dir(gnt_dir=train_data_dir):
    tagcode_unicode = struct.pack('>H', tagcode).decode('gb2312')
    '''plt.figure()
    plt.imshow(image)
    plt.grid(False)
    plt.show()'''
    print("No." + str(train_counter) + str(tagcode_unicode))
    train_counter += 1
for image, tagcode in read_from_gnt_dir(gnt_dir=test_data_dir):
    tagcode_unicode = struct.pack('>H', tagcode).decode('gb2312')
    print("No." + str(test_counter) + str(tagcode_unicode))
    test_counter += 1

# 样本数
print(train_counter, test_counter)
