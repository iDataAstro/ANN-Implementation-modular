import tensorflow as tf


def get_prepared_data(validation_datasize, logger):
    """Function will return MNIST data
    Args:
        validation_datasize: validation datasize for split data
        logger: logger object for logging
    Returns:
        (X_train, y_train), (X_valid, y_valid), (X_test, y_test)
    """

    logger.info("Downloading data..")
    mnist = tf.keras.datasets.mnist
    (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
    logger.info("Normalize and split data..")
    X_valid, X_train = X_train_full[:validation_datasize]/255., X_train_full[validation_datasize:]/255.
    y_valid, y_train = y_train_full[:validation_datasize], y_train_full[validation_datasize:]
    X_test = X_test/255.
    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)