from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
import preprocessing.data_tools as dt
import utils.store_dataset as sd
import utils.metrics as metrics
import utils.store_model as sm
import os
import utils.store_file as sf


np.set_printoptions(threshold=np.inf)

data_path_type='top'
test_data_path_type='norm'

data_path ="" #your own path
labels_path ="" 

test_data_path = ""
test_labels_path =""

num_classes = 2
total_features = 25
act_function='relu'
arrange_vector = [7.0,0.0,13.0,12.0,4.0,16.0,4.0,17.0,4.0,3.0,10.0,0.0,17.0,9.0,4.0,19.0,9.0,3.0,17.0,0.0,2.0,17.0,21.0,0.0,8.0]
save_path = ""
save_path_test = ""


def test_model(model, save_path_test, count, layers):

    print('Testing...', x_test_data.shape, y_test_data.shape)

    result = model.predict(x_test_data, batch_size=None, verbose=0)
    print("Test accuracy: %.1f%%" % metrics.classification_accuracy(result, y_test_data))

    set_metricpath = save_path_test + "/metrics.csv"

    if num_classes > 2:
        conf_matrix = metrics.classification_confusion_matrix(result, y_test_data)
        cohen_kapa = metrics.classification_cohen_kappa(conf_matrix)
        print(conf_matrix)
        print(cohen_kapa)
        try:
            sf.write_to_file(cohen_kapa, set_metricpath)
        except Exception as e:
            print("No directory found, creating...")
            os.system("mkdir -p %s" % save_path_test)
            sf.write_to_file(cohen_kapa, set_metricpath)
    else:
        value, met = metrics.classification_binary_metrics(result, y_test_data)
        for i in range(len(value)):
            print(value[i], ":", met[i])
        try:
            sm.save_metrics_to_file(set_metricpath, count, layers, value)
        except Exception as e:
            print("No directory found, creating...")
            os.system("mkdir -p %s" % save_path_test)
            sm.save_metrics_to_file(set_metricpath, str(count), layers, value)
        print("Tested")

    K.clear_session()


def tcp_cnn(train_data_features, train_labels, test_data_features, test_labels, save_path, layers, count, save_path_test):


    print(dataset.shape, labels.shape)

    batch_size = 128
    epochs = 1

    x_train, x_test = dt.add_zeros_col(train_data_features, test_data_features, total_features)
    y_train, y_test = train_labels, test_labels
    print(x_test.shape, x_train.shape)

    # input image dimensions
    img_rows, img_cols = 5, 5

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation=act_function,
                     input_shape=input_shape,
                     padding='valid'))
    model.add(Conv2D(64, (2, 2), activation=act_function, padding='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation=act_function))
    model.add(Dropout(0.5))
    #model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))

    result = model.predict(x_test, batch_size=None, verbose=0)
    print("Test accuracy: %.1f%%" % metrics.classification_accuracy(result, y_test))

    set_metricpath = save_path + "/metrics.csv"
    modelpath = save_path + "/models/" + str(count)

    if num_classes > 2:
        conf_matrix = metrics.classification_confusion_matrix(result, y_test)
        cohen_kapa = metrics.classification_cohen_kappa(conf_matrix)
        print(conf_matrix)
        print(cohen_kapa)
        try:
            sf.write_to_file(cohen_kapa, set_metricpath)
        except Exception as e:
            print("No directory found, creating...")
            os.system("mkdir -p %s" % save_path)
            sf.write_to_file(cohen_kapa, set_metricpath)
        try:
            model.save(modelpath + 'model.h5')
        except Exception as e:
            print("No directory found, creating...")
            os.system("mkdir -p %s" % save_path + "/models/")
            model.save(modelpath + 'model.h5')
    else:
        value, met = metrics.classification_binary_metrics(result, y_test)
        for i in range(len(value)):
            print(value[i], ":", met[i])
        try:
            sm.save_metrics_to_file(set_metricpath, count, layers, value)
        except Exception as e:
            print("No directory found, creating...")
            os.system("mkdir -p %s" % save_path)
            sm.save_metrics_to_file(set_metricpath, str(count), layers, value)
        try:
            model.save(modelpath + 'model.h5')
        except Exception as e:
            print("No directory found, creating...")
            os.system("mkdir -p %s" % save_path + "/models/")
            model.save(modelpath + 'model.h5')
        print("Saved")

    test_model(model, save_path_test, count, layers)

    K.clear_session()


def cross_validation(dataset, labels):
    k = 10
    print("Aleatorizando")
    dataset, labels = sd.randomize(dataset, labels)
    num_test = np.floor(dataset.shape[0] / k)
    print("\nStarting loop")
    alphasMap = range(k)  # range(k)
    layers = 1
    counter = 0
    for steps in alphasMap:
        print("\nCross Validation Step = " + str(steps))
        counter += 1
        if counter == 1:
            test_dataset = dataset[:int(num_test * counter), :]
            test_labels = labels[:int(num_test * counter), :]
            train_dataset = dataset[int(num_test * counter):, :]
            train_labels = labels[int(num_test * counter):, :]
        elif counter == 10:
            test_dataset = dataset[int(num_test * (counter - 1)):, :]
            test_labels = labels[int(num_test * (counter - 1)):, :]
            train_dataset = dataset[:int(num_test * (counter - 1)), :]
            train_labels = labels[:int(num_test * (counter - 1)), :]
        else:
            test_dataset = dataset[int(num_test * (counter - 1)):int(num_test * counter), :]
            test_labels = labels[int(num_test * (counter - 1)):int(num_test * counter), :]
            train_dataset = dataset[:int(num_test * (counter - 1)), :]
            train_labels = labels[:int(num_test * (counter - 1)), :]
            train_dataset = np.vstack([train_dataset, dataset[int(num_test * counter):, :]])
            train_labels = np.vstack([train_labels, labels[int(num_test * counter):, :]])

        tcp_cnn(train_dataset, train_labels, test_dataset, test_labels, save_path, layers, counter, save_path_test)


def arrange_data(dataset, arrange_vector):
    shuffled_dataset = np.zeros((len(dataset), len(arrange_vector)))
    for index, column in enumerate(arrange_vector):
        shuffled_dataset[:, index] = dataset[:, int(column)]
    return shuffled_dataset

npdata = dt.read_data(data_path)
nplabels = dt.read_data(labels_path)

dataset, labels = sd.randomize(npdata,nplabels)
dataset = arrange_data(dataset, arrange_vector)

npdata_test = dt.read_data(test_data_path)
nplabels_test = dt.read_data(test_labels_path)

test_dataset, test_labels = sd.randomize(npdata_test, nplabels_test)
test_dataset = arrange_data(test_dataset, arrange_vector)


_, x_test_data = dt.add_zeros_col(test_dataset, test_dataset, total_features)
y_test_data = test_labels
print(x_test_data.shape)

# input image dimensions
img_rows, img_cols = 5, 5

if K.image_data_format() == 'channels_first':
    x_test_data = x_test_data.reshape(x_test_data.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_test_data = x_test_data.reshape(x_test_data.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_test_data = x_test_data.astype('float32')
print(x_test_data.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_test_data = keras.utils.to_categorical(y_test_data, num_classes)

cross_validation(dataset, labels)


