import argparse
import os
import logging
from src.utils.common import read_yaml, create_directories
from src.utils.models import load_full_model
from src.utils.callbacks import get_callbacks


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
    artirfacts =  config["artifacts"]
    artifacts_dir = artirfacts["ARTIFACTS_DIR"]
    
    train_model_dir_path = os.path.join(artifacts_dir, artirfacts["TRAINED_MODEL_DIR"])
    create_directories([train_model_dir_path])

    untrainned_full_model_path = os.path.join(artifacts_dir, artirfacts["BASE_MODEL_DIR"],  artirfacts["UPDATED_BASE_MODEL_NAME"])

    model = load_full_model(untrainned_full_model_path)

    callback_dir_path = os.path.join(artifacts_dir, artirfacts["CALLBACKS_DIR"])

    callbacks = get_callbacks(callback_dir_path)


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