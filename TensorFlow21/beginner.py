from __future__ import absolute_import, division, print_function, unicode_literals

# 安装 TensorFlow
import tensorflow as tf

print('tensorflow: ', tf.__version__)

# data class
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
print('x_train: ', x_train.shape)
print('x_test: ', x_test.shape)

# model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# compiler
model.compile(optimizer='adam',\
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# train and evaluate
model.fit(x=x_train, y=y_train, epochs=5)

model.evaluate(x=x_train, y=y_train, verbose=2)

