from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import getopt
import time
import glob
import sys
import cv2
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #add to main file
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from model import get_model, unfreeze_model

help = 'Usage: train.py [options]\n\
    \t-f, --fine_tune \t fine tune Effecientnet wieghts\n\
    \t-t, --test \t\t test the pretrained model\n\
    \t-r, --reset \t\t resets the weights and initializes a new model'

data_path = '../plant-pathology-2021-fgvc8/'

checkpoint_path = './checkpoint/'
model_path = './model/'
history_path = './history/'

train_dir = 'aug_images/'
test_dir = 'test_images/'
cache_dir = 'cache/'

img_size = (512, 512)
batch_size = 32

lr = 0.0001
initial_epochs = 1

def get_data_gen():
    lower = 0
    upper = 5000
    if os.path.isfile(data_path + cache_dir + 'train-' + str(lower) + '-' + str(upper)):
        with open(data_path + cache_dir + 'train-' + str(lower) + '-' + str(upper), 'rb') as file_pi:
            (data, labels) = pickle.load(file_pi)
        print('training data loaded from cache for: ', lower, ' - ', upper)
        return (data, labels)
    
    print('generating training data...')
    csv_data = pd.read_csv(data_path + 'test.csv')
    csv_data['labels'] = csv_data['labels'].apply(lambda string: string.split(' '))
    s = list(csv_data['labels'])
    mlb = MultiLabelBinarizer()
    labels = pd.DataFrame(mlb.fit_transform(s), columns=mlb.classes_, index=csv_data.index)
    labels = np.array(labels)[lower:upper]

    data = []
    for img_path in sorted(glob.glob(data_path + test_dir + '*.jpg'))[lower:upper]:
        img = cv2.imread(img_path)
        img = cv2.resize(img, img_size)
        data.append(img)
    data = np.array(data)

    with open(data_path + cache_dir + 'train-' + str(lower) + '-' + str(upper), 'wb') as file_pi:
        pickle.dump((data, labels), file_pi)
    
    print('training data generatedfor: ', lower, ' - ', upper)
    return (data, labels)

    # print('preparing training data generator...')
    # csv_data = pd.read_csv(data_path + 'train.csv')
    # csv_data['labels'] = csv_data['labels'].apply(lambda string: string.split(' '))
    # s = list(csv_data['labels'])
    # mlb = MultiLabelBinarizer()
    # labels = pd.DataFrame(mlb.fit_transform(s), columns=mlb.classes_, index=csv_data.index)
    # labels = np.array(labels)

    # labels = np.repeat(labels, 18, 0)
    # imgs = sorted(glob.glob(data_path + train_dir + '*.jpg'))
    # i = 0
    # while (i+1)*batch_size < len(labels):
    #     labels_batch = labels[i*batch_size:i*batch_size+batch_size]
    #     data_batch = []
    #     for img_path in imgs[i*batch_size:i*batch_size+batch_size]:
    #         img = cv2.imread(img_path)
    #         data_batch.append(img)
    #     yield (data_batch, labels_batch)
    #     i += 1

    # train_dataset = image_dataset_from_directory(
    #     data_path + train_dir,
    #     labels=labels,
    #     label_mode='categorical',
    #     shuffle=True,
    #     batch_size=batch_size,
    #     image_size=img_size)
    # AUTOTUNE = tf.data.AUTOTUNE
    # train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    # train_dataset = get_valid_data()

    # return train_dataset

def get_valid_data(test=False):
    if os.path.isfile(data_path + cache_dir + 'val'):
        with open(data_path + cache_dir + 'val', 'rb') as file_pi:
            (data, labels) = pickle.load(file_pi)
        print("validation data loaded from cache")
        return (data, labels)
    
    print('generating validation data...')
    csv_data = pd.read_csv(data_path + 'test.csv')
    csv_data['labels'] = csv_data['labels'].apply(lambda string: string.split(' '))
    s = list(csv_data['labels'])
    mlb = MultiLabelBinarizer()
    labels = pd.DataFrame(mlb.fit_transform(s), columns=mlb.classes_, index=csv_data.index)
    if test:
        labels = np.array(labels)[len(labels)//2:]
    else:
        labels = np.array(labels)[:len(labels)//2]

    data = []
    if test:
        for img_path in sorted(glob.glob(data_path + test_dir + '*.jpg'))[len(labels):]:
            img = cv2.imread(img_path)
            img = cv2.resize(img, img_size)
            data.append(img)
    else:
        for img_path in sorted(glob.glob(data_path + test_dir + '*.jpg'))[:len(labels)]:
            img = cv2.imread(img_path)
            img = cv2.resize(img, img_size)
            data.append(img)
    data = np.array(data)

    with open(data_path + cache_dir + 'val', 'wb') as file_pi:
        pickle.dump((data, labels), file_pi)
    
    print("validation data generated")
    return (data, labels)

def train_classifier(train_dataset, validation_dataset, reset_weights):
    if reset_weights:
        model = get_model(img_size)
        print('model reset and initialized')
    else:
        file_type = '/*h5'
        files = glob.glob(model_path + file_type)
        if len(files) == 0:
            if os.path.isfile(checkpoint_path + 'checkpoint'):
                model = get_model(img_size)
                print(checkpoint_path + 'checkpoint')
                model.load_weights(checkpoint_path + 'checkpoint')
                print('model loaded for training from checkpoint')
            else:
                model = get_model(img_size)
                print('model initialized')
        else:
            recent_model = max(files, key=os.path.getctime)
            print('model: ', recent_model, ' loaded for training from models')
            model = tf.keras.models.load_model(recent_model)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy()])

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        monitor='val_binary_accuracy',
        mode='max',
        save_best_only=True)

    model.summary()

    history = model.fit(
        train_dataset[0],
        train_dataset[1],
        epochs=initial_epochs,
        callbacks=[model_checkpoint_callback],
        validation_data=validation_dataset)

    timestr = time.strftime('%Y%m%d-%H%M%S')
    model.save(model_path + timestr + '.h5')
    print('model: ', timestr, ' saved to models')
    with open(history_path + timestr, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

def fine_tune(train_dataset, validation_dataset):
    file_type = '/*h5'
    files = glob.glob(model_path + file_type)
    if len(files) == 0:
        print('train the classifier first before fine tunning')
        sys.exit(2)
    else:
        recent_model = max(files, key=os.path.getctime)
        print('model: ', recent_model, ' loaded for fine tuning')
        model = tf.keras.models.load_model(recent_model)

    unfreeze_model(model, 100)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr/10.0),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy()])

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        monitor='val_binary_accuracy',
        mode='max',
        save_best_only=True)

    model.summary()

    history = model.fit(
        train_dataset[0],
        train_dataset[1],
        epochs=initial_epochs,
        callbacks=[model_checkpoint_callback],
        validation_data=validation_dataset)

    timestr = time.strftime('%Y%m%d-%H%M%S')
    model.save(model_path + timestr + '.h5')
    print('model: ', timestr, ' saved to models')
    with open(history_path + timestr, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

def test_model(test_dataset):
    file_type = '/*h5'
    files = glob.glob(model_path + file_type)
    if len(files) == 0:
        print('train the classifier first before testing')
        sys.exit(2)
    else:
        recent_model = max(files, key=os.path.getctime)
        print('model: ', recent_model, ' loaded for teseting')
        model = tf.keras.models.load_model(recent_model)

    model.summary()
    model.evaluate(test_dataset[0], test_dataset[1])

def main(argv):
    try:
        opts, args = getopt.getopt(argv, 'hftr', ['help', 'fine_tune', 'test', 'reset'])
    except getopt.GetoptError:
        print(help)
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print(help)
            sys.exit()
        elif opt in ('-t', '--test'):
            # Testing
            test_model(get_valid_data(test=True))
        elif opt in ('-f', '--fine_tune'):
            # Fine tuning
            print('Fine Tuning Model...')
            fine_tune(get_data_gen(), get_valid_data())
        elif opt in ('-r', '--reset'):
            # Classifier training resetting the weights
            print('Training Model Classifier (weights reinitialized)...')
            train_classifier(get_data_gen(), get_valid_data(), reset_weights=True)

    if len(opts) == 0:
        # Classifier training
        print('Training Model Classifier...')
        train_classifier(get_data_gen(), get_valid_data(), reset_weights=False)

if __name__ == '__main__':
    main(sys.argv[1:])