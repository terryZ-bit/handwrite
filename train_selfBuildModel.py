import random
from abc import ABC

import tensorflow as tf
from tensorflow import nn
from tensorflow.keras import layers

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]
train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)


class Model_CNN(tf.keras.Model, ABC):
    def __init__(self):
        super().__init__()
        self.conv_1 = layers.Conv2D(
            96,
            kernel_size=11,
            strides=4,
            activation='relu'
        )
        self.maxPool_1 = layers.MaxPool2D(
            pool_size=3,
            strides=2
        )
        self.conv_2 = layers.Conv2D(
            256, kernel_size=5, activation='relu', padding='same'
        )
        self.maxPool_2 = layers.MaxPool2D(
            pool_size=3, strides=2
        )
        self.conv_3 = layers.Conv2D(
            384, kernel_size=3, activation='relu', padding='same'
        )
        self.conv_4 = layers.Conv2D(
            384, kernel_size=3, activation='relu', padding='same'
        )
        self.conv_5 = layers.Conv2D(
            256, kernel_size=3, activation='relu', padding='same'
        )
        self.maxPool_3 = layers.MaxPool2D(
            pool_size=3, strides=2
        )
        self.dense_1 = layers.Dense(
            4096, activation='relu'
        )
        self.dense_2 = layers.Dense(
            4096, activation='relu'
        )
        self.dense_3 = layers.Dense(
            10
        )

    def call(self, inputs):
        x = self.conv_1(inputs)
        x = self.maxPool_1(x)
        x = self.conv_2(x)
        x = self.maxPool_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.maxPool_3(x)
        x = self.dense_1(x)
        x = nn.dropout(x, 0.5)
        x = self.dense_2(x)
        x = nn.dropout(x, 0.5)
        x = self.dense_3(x)
        output = nn.softmax(x)
        return output


model = Model_CNN()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


@tf.function
def test_step(images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)


EPOCHS = 5

for epoch in range(EPOCHS):
    # 在下一个epoch开始时，重置评估指标
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for images, labels in train_ds:
        train_step(images, labels)

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch + 1,
                          train_loss.result(),
                          train_accuracy.result() * 100,
                          test_loss.result(),
                          test_accuracy.result() * 100))
