import logging

import tensorflow as tf


def train_valid_generator(data_dir, image_size, batch_size, do_data_augumentation):
    data_generator_kwargs = dict(
        rescale=1./255,
        validation_split=0.20
    )
    dataflow_kwargs = dict(
        target_size=image_size,
        batch_size=batch_size,
        interpolation="bilinear"
    )
    valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(**data_generator_kwargs)

    valid_generator = valid_datagenerator.flow_from_directory(
        directory=data_dir,
        subset="validation",
        shuffle=False,
        **dataflow_kwargs
    )

    if do_data_augumentation:
        train_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=40,
            horizontal_flip=True,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=2,
            zoom_range=0.2,
            **data_generator_kwargs
        )
        logging.info("Data augumentation is used for traiining")
    else:
        train_data_generator = valid_datagenerator
        logging.info("Data augumentation is not used for trainning")
    train_generator = train_data_generator.flow_from_directory(
        directory=data_dir,
        subset="training",
        shuffle=True,
        **dataflow_kwargs
    )
    logging.info("Trainning and Validation data Generated")
    return train_generator, valid_generator
