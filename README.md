# README

# Time-Series-Forecasting

### Project Description

This project involves building and training a neural network to predict time-indexed variables of a multivariate household electric power consumption time series dataset. The model uses a window of past 24 observations of the 7 variables to predict the next 24 observations of the same variables.

### Dataset

The dataset used is a subset of the Individual Household Electric Power Consumption Dataset, which has measurements of electric power consumption in one household with a one-minute sampling rate over a period of almost 4 years. The dataset has been cleaned and reduced to the first 60 days for this project. It contains 7 features ordered by time.

- **Source:** [Individual Household Electric Power Consumption Dataset](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption)

### Model Requirements

1. **Input Shape:** (BATCH_SIZE, 24, 7)
2. **Output Shape:** (BATCH_SIZE, 24, 7)
3. The last layer must be a Dense layer with 7 neurons.
4. Constants such as `SPLIT_TIME`, `N_FEATURES`, `BATCH_SIZE`, `N_PAST`, `N_FUTURE`, and `SHIFT` must not be changed.
5. The data normalization code must not be changed.
6. The dataset windowing code must not be changed.
7. The seed setting code must not be changed.

### Usage Instructions

1. **Download and Extract Dataset**

   The dataset is downloaded and extracted automatically using the `download_and_extract_data()` function.

2. **Data Normalization**

   The dataset is normalized using min-max scaling in the `normalize_series()` function.

3. **Windowed Dataset**

   The `windowed_dataset()` function maps the time series dataset into windows of features and respective targets for training and validation.

4. **Model Definition**

   The model is defined using a bidirectional LSTM architecture. It includes:
   - Input layer with shape `(24, 7)`
   - Bidirectional LSTM layers
   - RepeatVector layer to match the future prediction steps
   - TimeDistributed Dense layer to output the 7 features for each future step

5. **Model Compilation and Training**

   The model is compiled with the Adam optimizer and MSE loss. It is then trained for 10 epochs.

6. **Model Saving**

   The trained model is saved as `mymodel.h5`.

### Google Colab Usage

To train the model using Google Colab, ensure that you have the necessary libraries installed:

```python
!pip install pandas tensorflow
```

You can then copy the code into a Colab notebook and run the cells to train and save the model.

### Code

```python
import urllib
import zipfile
import pandas as pd
import tensorflow as tf

# This function downloads and extracts the dataset to the directory that
# contains this file.
def download_and_extract_data():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/certificate/household_power.zip'
    urllib.request.urlretrieve(url, 'household_power.zip')
    with zipfile.ZipFile('household_power.zip', 'r') as zip_ref:
        zip_ref.extractall()

# This function normalizes the dataset using min max scaling.
def normalize_series(data, min, max):
    data = data - min
    data = data / max
    return data

# This function is used to map the time series dataset into windows of
# features and respective targets, to prepare it for training and
# validation. 
def windowed_dataset(series, batch_size, n_past=24, n_future=24, shift=1):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(size=n_past + n_future, shift=shift, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(n_past + n_future))
    ds = ds.map(lambda w: (w[:n_past], w[n_past:]))
    return ds.batch(batch_size).prefetch(1)

# This function loads the data from CSV file, normalizes the data and
# splits the dataset into train and validation data. It also uses
# windowed_dataset() to split the data into windows of observations and
# targets. Finally, it defines, compiles and trains a neural network. This
# function returns the final trained model.
def solution_model():
    # Downloads and extracts the dataset to the directory that
    # contains this file.
    download_and_extract_data()
    # Reads the dataset from the CSV.
    df = pd.read_csv('household_power_consumption.csv', sep=',', infer_datetime_format=True, index_col='datetime', header=0)

    # Number of features in the dataset. We use all features as predictors to
    # predict all features at future time steps.
    N_FEATURES = len(df.columns)

    # Normalizes the data
    data = df.values
    data = normalize_series(data, data.min(axis=0), data.max(axis=0))

    # Splits the data into training and validation sets.
    SPLIT_TIME = int(len(data) * 0.5)
    x_train = data[:SPLIT_TIME]
    x_valid = data[SPLIT_TIME:]

    # Clear any previous models from memory.
    tf.keras.backend.clear_session()
    tf.random.set_seed(42)

    # DO NOT CHANGE BATCH_SIZE IF YOU ARE USING STATEFUL LSTM/RNN/GRU.
    BATCH_SIZE = 32

    # Number of past time steps based on which future observations should be predicted.
    N_PAST = 24

    # Number of future time steps which are to be predicted.
    N_FUTURE = 24

    # By how many positions the window slides to create a new window of observations.
    SHIFT = 1

    # Code to create windowed train and validation datasets.
    train_set = windowed_dataset(series=x_train, batch_size=BATCH_SIZE, n_past=N_PAST, n_future=N_FUTURE, shift=SHIFT)
    valid_set = windowed_dataset(series=x_valid, batch_size=BATCH_SIZE, n_past=N_PAST, n_future=N_FUTURE, shift=SHIFT)

    # Code to define the model.
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(N_PAST, N_FEATURES)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.RepeatVector(N_FUTURE),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(N_FEATURES))
    ])

    # Code to compile the model
    model.compile(optimizer='adam', loss='mse')

    # Code to train the model
    model.fit(train_set, validation_data=valid_set, epochs=10)

    return model

# Note that you'll need to save your model as a .h5 like this.
if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")
```

### Author

This project was created by lokesh kollareddy. If you have any questions or need further assistance, feel free to contact me at lokeshkollareddy@gmail.com.
