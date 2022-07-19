import argparse
import os
import logging
from src.utils.common import read_yaml, create_directories
from src.utils.models import get_vgg16_model, prepare_model
from src.utils.callbacks import create_and_save_tensorboard_callback, create_and_save_checkpoint_callback

STAGE = "prepare_callbacks"

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
)

def prepare_callbacks(config_path, params_path):
    config = read_yaml(config_path)
    
    artifacts = config["artifacts"]
    artifacts_dir = artifacts["ARTIFACTS_DIR"]
    tensorboard_log_dir = os.path.join(artifacts_dir, artifacts["TENSORBOARD_ROOT_LOG_DIR"])
    checkpoint_dir = os.path.join(artifacts_dir, artifacts["CHECKPOINT_DIR"])
    callback_dir = os.path.join(artifacts_dir, artifacts["CALLBACKS_DIR"])

    # creating all the necessary directories 
    create_directories([
        tensorboard_log_dir,
        checkpoint_dir,
        callback_dir
    ])

    create_and_save_tensorboard_callback(callback_dir, tensorboard_log_dir)
    create_and_save_checkpoint_callback(callback_dir, tensorboard_log_dir)





if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        prepare_callbacks(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info(f">>>>> stage {STAGE} completed!<<<<< and saved as binary\n")
    except Exception as e:
        logging.exception(e)
        raise e
