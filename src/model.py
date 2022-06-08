from utils import *
import os
import tensorflow as tf
from tensorflow import keras


class AutoPilot:
    def __init__(self, params):
        self.params = params

        self.model = tf.keras.Sequential()

        self.model.add(tf.keras.layers.Lambda(lambda x: x / 127.5 - 1.0, input_shape=c.input_shape))
        self.model.add(tf.keras.layers.Conv2D(24, (5, 5), (2, 2), activation='elu'))
        self.model.add(tf.keras.layers.Conv2D(36, (5, 5), (2, 2), activation='elu'))
        self.model.add(tf.keras.layers.Conv2D(48, (5, 5), (2, 2), activation='elu'))
        self.model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='elu'))
        self.model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='elu'))

        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(100, activation='elu'))
        self.model.add(tf.keras.layers.Dense(50, activation='elu'))
        self.model.add(tf.keras.layers.Dense(10, activation='elu'))
        self.model.add(tf.keras.layers.Dense(1, activation='tanh'))

        self.model.compile(tf.keras.optimizers.Adam(learning_rate=self.params["learning_rate"]), loss='mse')

        print(self.model.summary())

    def scheduler(self, epoch, lr):
        """ decrease `learning_rate` after each `num_epochs` = `epochs_drop`,
         at a rate of `drop_rate` """

        epochs_drop = self.params['epochs_drop']
        drop_rate = self.params['drop_rate']
        print(f"epoch: {epoch:03d} - learning_rate: {lr}")
        if epoch % epochs_drop == 0 and epoch > 0:
            return lr * drop_rate
        else:
            return lr

    def fit(self, df, split_point):
        """ learning_rate scheduler """
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(self.scheduler)

        # save tensorboard
        log_dir = os.path.join(c.log_dir, c.unique_name)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        # save the best model to .h5 file
        model_path = os.path.join(c.model_dir, c.unique_name + '.h5')
        best_model = keras.callbacks.ModelCheckpoint(filepath=model_path, monitor='val_loss',
                                                     save_best_only=True, verbose=self.params['verbose'])

        self.model.fit(batch_generator(df[:split_point], c.model_params["batch_size"], True),
                       steps_per_epoch=c.model_params["steps_per_epoch"],
                       validation_data=batch_generator(df[split_point:], c.model_params["batch_size"], False),
                       validation_steps=c.model_params["validation_steps"],
                       epochs=self.params['epochs'],
                       callbacks=[best_model, tensorboard_callback, lr_scheduler])

    def predict(self, xte):
        return self.model.predict(xte)
