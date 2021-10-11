import tensorflow as tf

import logging
import os

def get_prepared_model(no_classes, input_shape, loss, optimizer, metrics, logger):
    """Function creates ANN model and compile.
    Args:
        no_classes ([INT]): No of classes for classificaiton
        input_shape ([int, int]): Input shape for model's input layer
        loss ([str]): Loss function for model
        optimizer ([str]): Optimizer for model
        metrics ([str]): Metrics to watch while training
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
    os.makedirs(model_dir, exist_ok=True)
    model_file = os.path.join(model_dir, f"model_{model_suffix}.h5")
    model.save(model_file)
    logger.info(f"Saved model: {model_file}")