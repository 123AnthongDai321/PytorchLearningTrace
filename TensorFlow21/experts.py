from __future__ import absolute_import, division, print_function, unicode_literals

# import tensorflow
import tensorflow as tf
print('tensorflow: ', tf.__version__)

# import layers
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

# prepare data
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# tf.data, train data slice into batches and shuffle
train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices(
    (x_test, y_test)).batch(32)

# construct model class
class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.conv1 = Conv2D(32, 3, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(128, activation='relu')
    self.d2 = Dense(10, activation='softmax')

  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)

model = MyModel()

# loss function
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

# train loss
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

# start train
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
  print (template.format(epoch+1,
                         train_loss.result(),
                         train_accuracy.result()*100,
                         test_loss.result(),
                         test_accuracy.result()*100))

"""
tensorflow 2.1
Testing started at 0:55 ...
E:\InstalledSoftware\anaconda37\envs\tensorflow21\python.exe "C:\Program Files\JetBrains\PyCharm Community Edition 2019.3.4\plugins\python-ce\helpers\pycharm\_jb_pytest_runner.py" --path E:/Learning/GIT/PytorchLearningTrace/TensorFlow21/experts.py -- --last-failed
Launching pytest with arguments --last-failed E:/Learning/GIT/PytorchLearningTrace/TensorFlow21/experts.py in E:\Learning\GIT\PytorchLearningTrace\TensorFlow21

============================= test session starts =============================
platform win32 -- Python 3.7.7, pytest-5.4.1, py-1.8.1, pluggy-0.13.1 -- E:\InstalledSoftware\anaconda37\envs\tensorflow21\python.exe
cachedir: .pytest_cache
rootdir: E:\Learning\GIT\PytorchLearningTrace\TensorFlow21
collecting ... collected 1 item
run-last-failure: rerun previous 1 failure

experts.py::test_step ERROR                                              [100%]
test setup failed
file E:\Learning\GIT\PytorchLearningTrace\TensorFlow21\experts.py, line 67
  @tf.function
  def test_step(images, labels):
E       fixture 'images' not found
>       available fixtures: cache, capfd, capfdbinary, caplog, capsys, capsysbinary, doctest_namespace, monkeypatch, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory
>       use 'pytest --fixtures [testpath]' for help on them.


Why does it complain 'images' variable not available?
I think it is Pycharm environment issue because I can not run it
in Pycharm, but correctly by terminal though I also can not explain 
why it can not work well in Pycharm.
"""