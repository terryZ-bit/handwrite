import tensorflow as tf
import random
import pathlib
import matplotlib.pyplot as plt

from tensorflow.keras import datasets, layers, models

num_classes = 3754
'''model = models.Sequential([
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPool2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(3755, activation='softmax')

])'''


def create_model():
    model = models.Sequential([
        layers.Conv2D(
            filters=96,
            kernel_size=[11, 11],
            padding='same',
            strides=4,
            activation='relu',
        ),
        layers.MaxPool2D(
            pool_size=[3, 3],
            strides=2,
        ),
        layers.Conv2D(
            filters=256,
            kernel_size=[5, 5],
            activation='relu'
        ),
        layers.MaxPool2D(
            pool_size=[3, 3],
            strides=2
        ),
        layers.Conv2D(
            filters=384,
            kernel_size=[3, 3],
            activation='relu'
        ),
        layers.Conv2D(
            filters=384,
            kernel_size=[3, 3],
            activation='relu'
        ),
        layers.Conv2D(
            filters=256,
            kernel_size=[3, 3],
            activation='relu'
        ),
        layers.MaxPool2D(
            pool_size=[3, 3],
            strides=2
        ),
        layers.Dense(
            4096,
            activation='relu'
        ),
        layers.Dense(
            4096,
            activation='relu'
        ),
        layers.Dense(
            3754,
            activation='softmax'
        )
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


model = create_model()
AUTOTUNE = tf.data.experimental.AUTOTUNE

# train_data_lode
data_root_orig = 'C:\\DeepL\\data\\train'
data_root = pathlib.Path(data_root_orig)
all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)
image_count = len(all_image_paths)
label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
label_to_index = dict((name, index) for index, name in enumerate(label_names))
all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [64, 64])
    image /= 255.0  # normalize to [0,1] range

    return image


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)


# train data
path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))


def load_and_preprocess_from_path_label(path, label):
    return load_and_preprocess_image(path), label


BATCH_SIZE = 32
'''# test data lode
data_test_root_orig = 'E:\\data\\test'
data_test_root = pathlib.Path(data_test_root_orig)
all_test_image_paths = list(data_test_root.glob('*/*'))
all_test_image_paths = [str(path) for path in all_test_image_paths]
random.shuffle(all_test_image_paths)
test_image_count = len(all_test_image_paths)
test_label_names = sorted(item.name for item in data_test_root.glob('*/') if item.is_dir())
test_label_to_index = dict((name, index) for index, name in enumerate(test_label_names))
all_test_image_labels = [test_label_to_index[pathlib.Path(path).parent.name] for path in all_test_image_paths]
# test data
test_path_ds = tf.data.Dataset.from_tensor_slices(all_test_image_paths)
test_image_ds = test_path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

test_label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_test_image_labels, tf.int64))
test_image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
test_ds = tf.data.Dataset.from_tensor_slices((all_test_image_paths, all_test_image_labels))

test_image_label_ds = ds.map(load_and_preprocess_from_path_label)
test_ds = test_image_label_ds.cache(filename='cache_test.tf-data')
test_ds = test_ds.apply(
    tf.data.experimental.shuffle_and_repeat(buffer_size=test_image_count)
)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(1)
test_image_batch, test_label_batch = next(iter(test_ds))'''

image_label_ds = ds.map(load_and_preprocess_from_path_label)

ds = image_label_ds.cache(filename='cache.tf-data')

ds = ds.apply(
    tf.data.experimental.shuffle_and_repeat(buffer_size=image_count)
)

ds = ds.batch(BATCH_SIZE).prefetch(1)

image_batch, label_batch = next(iter(ds))
model.fit(ds, epochs=128, steps_per_epoch=1024)
model.save('E:\\data\\model\\model_easy\\model_1')
