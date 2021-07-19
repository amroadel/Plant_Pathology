import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #add to main file
import tensorflow as tf

def get_model(img_size):
    img_shape = img_size + (3,)
    base_model = tf.keras.applications.EfficientNetB4(
        input_shape=img_shape,
        include_top=False,
        weights='imagenet')
    base_model.trainable = False

    flat = tf.keras.Sequential([
        tf.keras.layers.GlobalMaxPool2D(),
        tf.keras.layers.Dropout(0.2)
    ], name='flatten')

    sig =  tf.keras.activations.sigmoid
    classifier = tf.keras.Sequential([
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(6, activation=sig)
    ], name='classifier')

    inputs = tf.keras.Input(shape=img_shape)
    x = base_model(inputs, training=False)
    x = flat(x)
    outputs = classifier(x)
    model = tf.keras.Model(inputs, outputs)

    return model

def unfreeze_model(model, layers):
    model.layers[1].trainable = True
    if layers != 'all':
        for layer in model.layers[1].layers[:-layers]:
            layer.trainable =  False