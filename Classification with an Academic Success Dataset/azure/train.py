import os
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.constraints import MaxNorm
from sklearn.model_selection import GridSearchCV
from azureml.core import Run
from scikeras.wrappers import KerasClassifier

def create_model(init_mode='uniform', activation='relu', dropout_rate=0.0, weight_constraint=1.0, neurons=128):
    inputs = keras.Input(shape=(36,))  # Assuming 36 features based on your dataset

    # Define the hidden layers
    x = keras.layers.Dense(neurons, activation=activation, kernel_initializer=init_mode,
                           kernel_constraint=MaxNorm(weight_constraint))(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(dropout_rate)(x)

    concatenated = keras.layers.Concatenate()([x, inputs])

    y = keras.layers.Dense(neurons, activation=activation, kernel_initializer=init_mode,
                           kernel_constraint=MaxNorm(weight_constraint))(concatenated)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Dropout(dropout_rate)(y)

    concatenated = keras.layers.Concatenate()([y, concatenated])

    z = keras.layers.Dense(neurons, activation=activation, kernel_initializer=init_mode,
                           kernel_constraint=MaxNorm(weight_constraint))(concatenated)
    z = keras.layers.BatchNormalization()(z)
    z = keras.layers.Dropout(dropout_rate)(z)

    concatenated = keras.layers.Concatenate()([z, concatenated])

    t = keras.layers.Dense(neurons, activation=activation, kernel_initializer=init_mode,
                           kernel_constraint=MaxNorm(weight_constraint))(concatenated)
    t = keras.layers.BatchNormalization()(t)
    t = keras.layers.Dropout(dropout_rate)(t)

    # Define the output layer
    outputs = keras.layers.Dense(units=3, kernel_initializer=init_mode, activation='softmax')(t)

    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# Parsing script arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, help='Directory where the dataset is stored')
args = parser.parse_args()

# Load your data
train_path = os.path.join(args.data_dir, 'train.csv')
test_path = os.path.join(args.data_dir, 'test.csv')

train_df = pd.read_csv(train_path).drop(columns=['ID'])
test_df = pd.read_csv(test_path).drop(columns=['ID'])

# Prepare the input and output features
input_features = train_df.columns[:-1]  # All columns except the last one
output_feature = train_df.columns[-1]   # The last column

inputs = train_df[input_features].values
outputs = train_df[output_feature].values

# Initialize the AzureML run context
run = Run.get_context()

# Set up the KerasClassifier with the model and parameters
nn_classifier = KerasClassifier(
    model=create_model,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
    validation_split=0.2,
    verbose=0,
    callbacks=[keras.callbacks.ReduceLROnPlateau, keras.callbacks.EarlyStopping],
    callbacks__0__monitor="val_loss",
    callbacks__0__factor=0.5,
    callbacks__0__verbose=0,
    callbacks__0__min_delta=0.0001,
    callbacks__0__patience=5,
    callbacks__0__min_lr=0.0000001,
    callbacks__1__min_delta=0.0001,
    callbacks__1__patience=8,
    callbacks__1__restore_best_weights=True
)

# Define the parameter grid
param_grid = dict(
    batch_size=[128, 256, 512, 1024, 2048],
    epochs=[100, 128, 150],
    optimizer=['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'],
    model__init_mode=['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform',
                      'he_normal', 'he_uniform'],
    model__activation=['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear'],
    model__dropout_rate=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    model__weight_constraint=[1.0, 2.0, 3.0, 4.0, 5.0],
    model__neurons=[64, 128, 256, 512, 1024]
)

# Perform GridSearchCV
grid = GridSearchCV(estimator=nn_classifier, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(inputs, outputs)

# Log results
run.log("Best Accuracy", grid_result.best_score_)
run.log("Best Parameters", grid_result.best_params_)

# Save the best model
os.makedirs('outputs', exist_ok=True)
grid_result.best_estimator_.model_.save('outputs/best_model.h5')

run.complete()
