import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from utils import batch_generator

data_df = pd.read_csv('data/driving_log.csv',
                      names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])

data_df['center'] = data_df['center'].apply(lambda x: 'data/IMG/' + x.split('\\')[-1])
data_df['left'] = data_df['left'].apply(lambda x: 'data/IMG/' + x.split('\\')[-1])
data_df['right'] = data_df['right'].apply(lambda x: 'data/IMG/' + x.split('\\')[-1])

X = data_df[['center', 'left', 'right']].values
y = data_df['steering'].values

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)


def createModel():
    net = tf.keras.Sequential()

    net.add(tf.keras.layers.Lambda(lambda x: x / 127.5 - 1.0, input_shape=(66, 200, 3)))
    net.add(tf.keras.layers.Conv2D(24, (5, 5), (2, 2), activation='elu'))
    net.add(tf.keras.layers.Conv2D(36, (5, 5), (2, 2), activation='elu'))
    net.add(tf.keras.layers.Conv2D(48, (5, 5), (2, 2), activation='elu'))
    net.add(tf.keras.layers.Conv2D(64, (3, 3), activation='elu'))
    net.add(tf.keras.layers.Conv2D(64, (3, 3), activation='elu'))

    net.add(tf.keras.layers.Flatten())
    net.add(tf.keras.layers.Dense(100, activation='elu'))
    net.add(tf.keras.layers.Dense(50, activation='elu'))
    net.add(tf.keras.layers.Dense(10, activation='elu'))
    net.add(tf.keras.layers.Dense(1, activation='tanh'))

    net.compile(tf.keras.optimizers.Adam(lr=1.0e-4), loss='mse')
    return net

best_model = tf.keras.callbacks.ModelCheckpoint('best_model.h5',
                                                monitor='val_loss',
                                                verbose=1,
                                                save_best_only=True,
                                                mode='auto')


tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs/", histogram_freq=1)

model = createModel()
model.fit_generator(batch_generator(X_train, y_train, 40, True),
                    steps_per_epoch=2000,
                    validation_data=batch_generator(X_valid, y_valid, 40, False),
                    validation_steps=10,
                    epochs=50,
                    callbacks=[best_model, tensorboard_callback]
                    )
