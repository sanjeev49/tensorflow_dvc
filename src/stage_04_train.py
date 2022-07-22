import argparse
import os
import logging
from src.utils.common import read_yaml, create_directories
from src.utils.models import load_full_model
from src.utils.callbacks import get_callbacks
from src.utils.data_mgnt import  train_valid_generator


STAGE = "train_model" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def train_model(config_path, params_path):
    # read config files
    config = read_yaml(config_path)
    params = read_yaml(params_path)
    artifacts =  config["artifacts"]
    artifacts_dir = artifacts["ARTIFACTS_DIR"]
    
    train_model_dir_path = os.path.join(artifacts_dir, artifacts["TRAINED_MODEL_DIR"])
    create_directories([train_model_dir_path])

    untrained_full_model_path = os.path.join(artifacts_dir, artifacts["BASE_MODEL_DIR"],  artifacts["UPDATED_BASE_MODEL_NAME"])

    model = load_full_model(untrained_full_model_path)

    callback_dir_path = os.path.join(artifacts_dir, artifacts["CALLBACKS_DIR"])

    callbacks = get_callbacks(callback_dir_path)
    train_generator, valid_generator = train_valid_generator(
        data_dir=artifacts["DATA_DIR"],
        image_size=tuple(params["IMAGE_SIZE"][:-1]),
        batch_size=params["BATCH_SIZE"],
        do_data_augumentation=params["AUGMENTATION"]
    )




if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        train_model(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e