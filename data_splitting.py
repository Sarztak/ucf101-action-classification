from pathlib import Path 
import shutil
import pandas as pd 
import numpy as np 
import tensorflow as tf 
from sklearn.model_selection import train_test_split
import cv2
import random
from loguru import logger


def create_train_test_val_split(ucf_dir, save_dir):
    train_dfs, test_dfs, val_dfs = [], [], []
    class_names = []
    for path in ucf_dir.iterdir():
        if path.is_dir():
            folder_name = path.name
            class_names.append(folder_name)

            # find the .avi files and convert path to a string
            avi_paths = [str(avi_path) for avi_path in path.glob("*.avi")]

            # create train, val, test split 60, 20, 20
            train_r, val_r, test_r = 0.6, 0.2, 0.2
            X_train, X_val = train_test_split(avi_paths, train_size=train_r)
            X_val, X_test = train_test_split(X_val, train_size= val_r / (val_r + test_r))

            df_train = pd.DataFrame(X_train, columns=['path'])
            df_train['action'] = folder_name

            df_val = pd.DataFrame(X_val, columns=['path'])
            df_val['action'] = folder_name

            df_test = pd.DataFrame(X_test, columns=['path'])
            df_test['action'] = folder_name

            train_dfs.append(df_train)
            val_dfs.append(df_val)
            test_dfs.append(df_test)

    # merge all the frames across all the classes
    train_df = pd.concat(train_dfs)
    val_df = pd.concat(val_dfs)
    test_df = pd.concat(test_dfs)

    # sort the class names and then assign an index and after that map them
    # also store the mapping 
    class_names = sorted(class_names)
    class_names_idx_map = {_cls: i for i, _cls, in enumerate(class_names)}
    train_df['class_idx'] = train_df['action'].map(class_names_idx_map)
    val_df['class_idx'] = val_df['action'].map(class_names_idx_map)
    test_df['class_idx'] = test_df['action'].map(class_names_idx_map)

    # save to folder
    train_df.to_csv(save_dir / "train_paths.csv", index=False)
    val_df.to_csv(save_dir / "val_paths.csv", index=False)
    test_df.to_csv(save_dir / "test_paths.csv", index=False)


def build_cnn_backbone():
    input_layer = layers.Input(shape=(224, 224, 3))

    x = input_layer # setting the input to a variable that changes is a common pattern
    
    # define the number of filters and where to add pooling
    filters = [2**(5 + i) for i in range(5) for _ in range(2)]

    for i, filter_size in enumerate(filters):
        x = layers.Conv2D(filters=filter_size, kernel_size=(3, 3), padding='same')(x)
        x = layers.ReLU()(x)
        x = layers.BatchNormalization()(x)

        if i % 2 != 0: # apply max pooling at 1, 3, 5, 7, 9 index; after 2nd layer of filters having the same size
            x = layers.MaxPooling2D((2, 2))(x)
    
    # finally to reduce the feature maps to flattened dimension or 1 x 1 
    # we use globalaveragepooling 2d which will take average across the 
    # entire feature may and then the output will be number of channels
    x = layers.GlobalAveragePooling2D()(x)

    model = tf.keras.Model(inputs=input_layer, outputs=x) 

    return model