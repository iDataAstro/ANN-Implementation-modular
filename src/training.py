from src.utils.common import read_config, get_logger, get_unique_name
from src.utils.data_mgmt import get_prepared_data
from src.utils.model import get_prepared_model, save_model, save_history_plot, get_callbacks

import argparse


def training(config_path):
    """
    Function create, compile, train and save ANN model.
    Args:
        config_path: Path for config file.
    """
    configs = read_config(config_path)

    logs_dir = configs["logs"]["logs_dir"]
    logger = get_logger(logs_dir, "training.py")
    logger.info("\n\nTraining.py Started ..")

    validation_datasize = configs["params"]["validation_datasize"]
    logger.info("Getting data..")
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = get_prepared_data(validation_datasize, logger)

    no_classes = configs["params"]["no_classes"]
    input_shape = configs["params"]["input_shape"]
    loss = configs["params"]["loss_function"]
    optimizer = configs["params"]["optimizer"]
    metrics = configs["params"]["metrics"]
    checkpoint_dir = configs["artifacts"]["checkpoint_dir"]
    tensorboard_logs = configs["logs"]["tensorboard_logs"]
    logger.info("Getting compiled ann model..")
    model_ann = get_prepared_model(no_classes, input_shape, loss, optimizer, metrics, logger)

    callback_list = get_callbacks(checkpoint_dir, tensorboard_logs)

    EPOCHS = configs["params"]["epochs"]
    logger.info("Model training start..")
    history = model_ann.fit(X_train, y_train, epochs=EPOCHS, validation_data=(X_valid, y_valid),
                            callbacks=callback_list)
    logger.info("Model training ends..")

    plot_dir = configs["artifacts"]["plot_dir"]
    logger.info("Plot Loss/Accuracy curves..")
    save_history_plot(history, plot_dir)

    model_dir = configs["artifacts"]["model_dir"]
    model_suffix = get_unique_name()
    save_model(model_dir, model_ann, model_suffix, logger)
    logger.info("Model saved successfully..")


if __name__ == '__main__':
    args = argparse.ArgumentParser(prog="training.py",
                                   usage='%(prog)s --config|-c config_file_path',
                                   description="To train model provide configuration file.")

    args.add_argument("--config", "-c", default="config.yaml")

    parsed_args = args.parse_args()

    training(parsed_args.config)
