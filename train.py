import keras
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Flatten,  MaxPooling2D, Conv2D
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix

class TrainValTensorBoard(TensorBoard):
    '''
    Customize tensorflow plotting to easily compare val and train results
    code from: https://stackoverflow.com/questions/47877475/keras-tensorboard-plot-train-and-validation-scalars-in-a-same-figure/48393723
    '''
    def __init__(self, log_dir='./logs', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()


def load_data(path):
    """Loads the MNIST dataset.
    @:arg
        path: path of the data set
    @:returns
        Tuple of Numpy arrays: `(X_train, y_train), (X_test, y_test)`.
    """
    print("Loading dataset")

    data = np.genfromtxt(path, delimiter=",", skip_header=1)

    y = data[:, 0]
    X = np.delete(data, (0), axis=1)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        break #do 1 split only

    print("Training set count: {}".format(len(X_train)))
    print("Test set count: {}".format(len(X_test)))

    return (X_train, y_train), (X_test, y_test)

def shift(image, d, axis):
    new_img = np.roll(image, d, axis)

    if axis == 0:
        if d >= 1: new_img[0, :, :] = np.zeros((28, 1))
        if d <= -1: new_img[27, :, :] = np.zeros((28, 1))
    if axis == 1:
        if d >= 1: new_img[:, 0, :] = np.zeros((28, 1))
        if d <= -1: new_img[:, 27, :] = np.zeros((28, 1))
    return new_img

def expand_data(images, labels):
    """
    expand training data by augmenting image

    :param images: images to augment
    :param labels: corresponding labels of the images
    :return:
        Arrays of new_images and its respective labels
    """
    print("Expanding data set")
    new_images = np.empty((len(images) * 8, 28, 28, 1))
    new_labels = np.empty((len(images) * 8, len(labels[0])))
    new_image_num = 0
    for i, image in enumerate(images):
        #show_img(image, labels[i])
        for dx in [1, 0, -1]:
            for dy in [1, 0, -1]:
                if dx != 0 or dy != 0: #dx and dy == 0 means the image will not be shifted
                    new_image = shift(shift(image, dx, 0), dy, 1)
                    new_images[new_image_num] = new_image
                    new_labels[new_image_num] = labels[i]
                    new_image_num = new_image_num + 1
                    #show_img(new_image, labels[i])
    return new_images, new_labels

def show_img(image, label):
    image = image.reshape((28, 28))
    plt.title('Label is {label}'.format(label=label.argmax()))
    plt.imshow(image, cmap='gray')
    plt.show()

def train(data_path):
    (X_train, y_train), (X_test, y_test) = load_data(data_path)
    X_train = X_train.reshape(len(X_train), 28, 28, 1).astype('float32')
    X_test = X_test.reshape(len(X_test), 28, 28, 1).astype('float32')

    X_train /= 255
    X_test /= 255

    n_classes = 10
    y_train = keras.utils.to_categorical(y_train, n_classes)
    y_test = keras.utils.to_categorical(y_test, n_classes)

    print("Training Distribution of Data:")
    print(confusion_matrix(y_train.argmax(1), y_train.argmax(1)))
    print("Test Distribution of Data:")
    print(confusion_matrix(y_test.argmax(1), y_test.argmax(1)))

    new_images, new_labels = expand_data(X_train, y_train)
    print("Generated {} new images".format(len(new_images)))
    X_train = np.append(X_train, new_images, axis=0)
    y_train = np.append(y_train, new_labels, axis=0)

    print("Number of training data after augmentation: {}".format(len(X_train)))

    #LeNet-5 neural network architecture
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
    os.makedirs("checkpoint", exist_ok=True)
    checkpoint = ModelCheckpoint("checkpoint/weights.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss', verbose=1, save_best_only=True,
                                    save_weights_only=False, mode='auto', period=1)
    # for visual analysis
    tensorboard = TrainValTensorBoard(write_graph=False)

    model.fit(X_train, y_train, batch_size=4096, epochs=100, verbose=1,
              validation_data=(X_test, y_test), callbacks=[tensorboard, early_stop, checkpoint])

    predictions = model.predict(X_test)
    print("Test Prediction Confusion matrix")
    print(confusion_matrix(y_test.argmax(1), predictions.argmax(1)))

if __name__ == '__main__':
    train(sys.argv[1])