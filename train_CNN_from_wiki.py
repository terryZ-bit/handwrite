import tensorflow as tf
import random
import pathlib
import matplotlib.pyplot as plt

from tensorflow.keras import datasets, layers, models


class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()    # Flatten层将除第一维（batch_size）以外的维度展平
        self.dense1 = tf.keras.layers.Dense(units=4000, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=3754)

    def call(self, inputs):         # [batch_size, 28, 28, 1]
        x = self.flatten(inputs)    # [batch_size, 784]
        x = self.dense1(x)          # [batch_size, 100]
        x = self.dense2(x)          # [batch_size, 10]
        output = tf.nn.softmax(x)
        return output


class CNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=64,  # 卷积层神经元（卷积核）数目
            kernel_size=[5, 5],  # 感受野大小
            padding='same',  # padding策略（vaild 或 same）
            activation=tf.nn.relu  # 激活函数
        )
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.relu
        )
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.flatten = tf.keras.layers.Reshape(target_shape=(16 * 16 * 64,))
        self.dense1 = tf.keras.layers.Dense(units=1024, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=3754)

    def call(self, inputs):
        x = self.conv1(inputs)  # [batch_size, 64, 64, 32]
        x = self.pool1(x)  # [batch_size, 32, 32, 32]
        x = self.conv2(x)  # [batch_size, 32, 32, 64]
        x = self.pool2(x)  # [batch_size, 16, 16, 64]
        x = self.flatten(x)  # [batch_size, 16 * 16 * 64]
        x = self.dense1(x)  # [batch_size, 1024]
        x = self.dense2(x)  # [batch_size, 10]
        output = tf.nn.softmax(x)
        return output

AUTOTUNE = tf.data.experimental.AUTOTUNE

# train_data_lode
data_root_orig = 'E:\\data\\train'
data_root = pathlib.Path(data_root_orig)
all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)
image_count = len(all_image_paths)
label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
label_to_index = dict((name, index) for index, name in enumerate(label_names))
all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]

# test data lode
data_test_root_orig = 'E:\\data\\test'
data_test_root = pathlib.Path(data_test_root_orig)
all_test_image_paths = list(data_test_root.glob('*/*'))
all_test_image_paths = [str(path) for path in all_test_image_paths]
random.shuffle(all_test_image_paths)
test_image_count = len(all_test_image_paths)
test_label_names = sorted(item.name for item in data_test_root.glob('*/') if item.is_dir())
test_label_to_index = dict((name, index) for index, name in enumerate(test_label_names))
all_test_image_labels = [test_label_to_index[pathlib.Path(path).parent.name] for path in all_test_image_paths]


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


# test data
test_path_ds = tf.data.Dataset.from_tensor_slices(all_test_image_paths)
test_image_ds = test_path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

test_label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_test_image_labels, tf.int64))
test_image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
test_ds = tf.data.Dataset.from_tensor_slices((all_test_image_paths, all_test_image_labels))


def load_and_preprocess_from_path_label(path, label):
    return load_and_preprocess_image(path), label


image_label_ds = ds.map(load_and_preprocess_from_path_label)
test_image_label_ds = ds.map(load_and_preprocess_from_path_label)

BATCH_SIZE = 32
ds = image_label_ds.cache(filename='cache.tf-data')
test_ds = test_image_label_ds.cache(filename='cache_test.tf-data')
ds = ds.apply(
    tf.data.experimental.shuffle_and_repeat(buffer_size=image_count)
)
test_ds = test_ds.apply(
    tf.data.experimental.shuffle_and_repeat(buffer_size=test_image_count)
)
ds = ds.batch(BATCH_SIZE).prefetch(1)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(1)
test_image_batch, test_label_batch = next(iter(test_ds))
image_batch, label_batch = next(iter(ds))

num_epochs = 5
learning_rate = 0.001
num_batches = int(image_count // BATCH_SIZE * num_epochs)
model = CNN()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
for batch_index in range(200):
    X = image_batch
    y = label_batch
    with tf.GradientTape() as tape:
        y_pred = model(X)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
        loss = tf.reduce_mean(loss)
        print("batch %d: loss %f" % (batch_index, loss.numpy()))
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
num_batches = int(test_image_count // BATCH_SIZE)
for batch_index in range(200):
    start_index, end_index = batch_index * BATCH_SIZE, (batch_index + 1) * BATCH_SIZE
    y_pred = model.predict(test_image_ds[start_index: end_index])
    sparse_categorical_accuracy.update_state(y_true=test_label_ds[start_index: end_index], y_pred=y_pred)
print("test accuracy: %f" % sparse_categorical_accuracy.result())
tf.saved_model.save(model, "model")