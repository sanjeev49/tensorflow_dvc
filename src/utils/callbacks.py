import tensorflow as tf 
import os
import joblib
import logging
from src.utils.common import get_timestamp


def create_and_save_tensorboard_callback(callback_dir, tensorboard_log_dir):
    unique_name = get_timestamp("tb_logs")

    tb_running_log_dir = os.path.join(tensorboard_log_dir, unique_name)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = tb_running_log_dir)
    tb_callback_filepath = os.path.join(callback_dir, "tensorboard_cb.cb")
    joblib.dump(tensorboard_callback, open(tb_callback_filepath))
    logging.info(f"Tensorboard callback is being saved at {tb_callback_filepath}")

def create_and_save_checkpoint_callback(callback_dir, tensorboard_log_dir):
    pass