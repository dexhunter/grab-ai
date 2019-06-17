import pandas as pd
import numpy as np
import Geohash
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, LSTM, BatchNormalization

def read_data():
    df = pd.read_csv("data/training.csv")

    labelencoder = LabelEncoder()
    df['geo_encoded'] = labelencoder.fit_transform(df['geohash6'])

    # extract hour and minute from timestamp column
    df[['h','m']] = df['timestamp'].str.split(':',expand=True)
    df['h'] = df['h'].astype('int64')
    df['m'] = df['m'].astype('int64')

    # extract day of week (DoW) from day
    df['dow'] = df['day'] % 7
    df.drop(columns=['geohash6', 'timestamp'], inplace=True)

    # We'll prototype using 100k sample since executing full 4 millions dataset will slow down our step
    # once we have good model we can gradually increase the size and observe the accuracy.
    df_sample = df.sample(100000, random_state=1)
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_sample[['day','h','m','dow','geo_encoded']] = scaler.fit_transform(df_sample[['day','h','m','dow','geo_encoded']])

    # print(df_sample.head())
    X = []
    for i in range(20):
        X.append(df_sample.shift(-1-i).fillna(-1).values)
    X = np.array(X)

    X = X.reshape(X.shape[1],X.shape[0],X.shape[2])

    # split into train and test
    n_train = int(0.8*len(X))
    X_train = X[:n_train,:,:]
    X_test = X[n_train:,:,:]

    y = df_sample['demand'].values
    y_train = y[:n_train]
    y_test = y[n_train:]

    return X_train, X_test, y_train, y_test

def build_model(X_train, X_test, y_train, y_test):

    model = Sequential()
    model.add(LSTM(units=32, return_sequences=True, input_shape=(X_train.shape[1],X_train.shape[2])))
    model.add(BatchNormalization())
    model.add(LSTM(units=32, return_sequences=True))
    model.add(BatchNormalization())
    model.add(LSTM(units=32))
    model.add(BatchNormalization())
    model.add(Dense(units=1))

    # print(model.summary())

    model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])

    model.fit(X_train, y_train, epochs=2, batch_size=2, verbose=2)

    model.save('test_model.h5')

    predicted_value = model.predict(X_test)

    # print(mean_squared_error(y_test,predicted_value))

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = read_data()
    build_model(X_train, X_test, y_train, y_test)

    # for testing
    # model = load_model('test_model.h5')
    # predicted = model.predict(X_test)
    # print(predicted)
    # print(y_test)