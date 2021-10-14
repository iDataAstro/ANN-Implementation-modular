import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

import os


def get_prepared_model(no_classes, input_shape, loss, optimizer, metrics, logger):
    """Function creates ANN model and compile.
    Args:
        no_classes ([INT]): No of classes for classificaiton
        input_shape ([int, int]): Input shape for model's input layer
        loss ([str]): Loss function for model
        optimizer ([str]): Optimizer for model
        metrics ([str]): Metrics to watch while training
        logger ([object]): Logger object for logging
    Returns:
        model: ANN demo model
    """
    # Define layers
    LAYERS = [
          tf.keras.layers.Flatten(input_shape=input_shape, name='input_layer'),
          tf.keras.layers.Dense(units=392, activation='relu', name='hidden1'),
          tf.keras.layers.Dense(units=196, activation='relu', name='hidden2'),
          tf.keras.layers.Dense(units=no_classes, activation='softmax', name='output_layer')
    ]

    logger.info("Creating Model..")
    model_ann = tf.keras.models.Sequential(LAYERS)

    logger.info("Compiling Model..")
    model_ann.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    return model_ann


def save_model(model_dir, model, model_suffix, logger):
    """
    args:
        model_dir: directory to save the model
        model: model object to save
        model_suffix: Suffix to save the model
        logger: Logger object for logging
    """
    os.makedirs(model_dir, exist_ok=True)
    model_file = os.path.join(model_dir, f"model_{model_suffix}.h5")
    model.save(model_file)
    logger.info(f"Saved model: {model_file}")


def save_history_plot(history, plot_dir):
    """
    Args:
        history: History object for plotting loss/accuracy curves
        plot_dir: Directory to save plot files
    """
    pd.DataFrame(history.history).plot(figsize=(10,8))
    plt.grid(True)
    os.makedirs(plot_dir, exist_ok=True)
    plot_file = os.path.join(plot_dir, "loss_accuracy.png")
    plt.savefig(plot_file)


def get_callbacks(checkpoint_dir, tensorboard_logs):
    """
    Args:
        checkpoint_dir: Directory to save the model at checkpoint
        tensorboard_logs: Directory to save tensorboard logs
    Returns:
        callback_list: List of created callbacks
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(tensorboard_logs, exist_ok=True)
    tensorboard_cb = tf.keras.callbacks.TensorBoard(tensorboard_logs)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
    ckpt_file_path = os.path.join(checkpoint_dir, "ckpt_model.h5")
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_file_path, save_best_only=True)

    callback_list = [tensorboard_cb, early_stopping_cb, checkpoint_cb]

    return callback_list
