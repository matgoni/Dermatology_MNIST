import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class Propuesta_1:
    def __init__(self, images, label):
        aplanado = images.reshape(images.shape[0], -1)
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(aplanado)

        self.y_onehot = to_categorical(label)

    def CrearModelo(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_scaled, self.y_onehot, test_size=0.2, random_state=42)

        self.model = Sequential()
        self.model.add(Dense(128, input_dim=self.X_train.shape[1], activation='relu'))
        self.model.add(Dense(64, activation='relu'))    
        self.model.add(Dense(7, activation='softmax'))

        self.best_model = ModelCheckpoint(filepath='model.keras', monitor='val_accuracy', save_best_only=True)

        self.optimi = SGD(learning_rate=0.03)
        self.model.compile(optimizer=self.optimi, loss='mse', metrics=['accuracy'])

        self.history = self.model.fit(self.X_train, self.y_train, epochs=100,
                            batch_size=10, validation_split=0.1, callbacks=[self.best_model])

        self.model = tf.keras.models.load_model('model.keras')

        return self.model
    
    def EvaluarModelo(self):
        if self.model is not None:
            self.loss, self.accuracy = self.model.evaluate(self.X_test, self.y_test)
            print(f'Precisión en el conjunto de prueba: {self.accuracy*100:.2f}%')
        else:
            print("El modelo no está cargado.")

class Propuesta_2:
    def __init__(self, images, label):
        aplanado = images.reshape(images.shape[0], -1)
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(aplanado)

        self.y_onehot = to_categorical(label)

    def CrearModelo(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_scaled, self.y_onehot, test_size=0.2, random_state=42)

        self.model = Sequential()
        self.model.add(Dense(128, input_dim=self.X_train.shape[1], activation='sigmoid'))
        self.model.add(Dense(64, activation='sigmoid'))    
        self.model.add(Dense(7, activation='softmax'))

        self.best_model = ModelCheckpoint(filepath='model.keras', monitor='val_accuracy', save_best_only=True)

        self.optimi = SGD(learning_rate=0.03)
        self.model.compile(optimizer=self.optimi, loss='mse', metrics=['accuracy'])

        self.history = self.model.fit(self.X_train, self.y_train, epochs=100,
                            batch_size=10, validation_split=0.1, callbacks=[self.best_model])

        self.model = tf.keras.models.load_model('model.keras')

        return self.model
    
    def EvaluarModelo(self):
        if self.model is not None:
            self.loss, self.accuracy = self.model.evaluate(self.X_test, self.y_test)
            print(f'Precisión en el conjunto de prueba: {self.accuracy*100:.2f}%')
        else:
            print("El modelo no está cargado.")

class Propuesta_3:
    def __init__(self, images, label):
        aplanado = images.reshape(images.shape[0], -1)
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(aplanado)

        self.y_onehot = to_categorical(label)

    def CrearModelo(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_scaled, self.y_onehot, test_size=0.2, random_state=42)

        self.model = Sequential()
        self.model.add(Dense(128, input_dim=self.X_train.shape[1], activation='elu'))
        self.model.add(Dense(64, activation='elu'))    
        self.model.add(Dense(7, activation='softmax'))

        self.best_model = ModelCheckpoint(filepath='model.keras', monitor='val_accuracy', save_best_only=True)

        self.optimi = SGD(learning_rate=0.03)
        self.model.compile(optimizer=self.optimi, loss='mse', metrics=['accuracy'])

        self.history = self.model.fit(self.X_train, self.y_train, epochs=100,
                            batch_size=10, validation_split=0.1, callbacks=[self.best_model])

        self.model = tf.keras.models.load_model('model.keras')

        return self.model
    
    def EvaluarModelo(self):
        if self.model is not None:
            self.loss, self.accuracy = self.model.evaluate(self.X_test, self.y_test)
            print(f'Precisión en el conjunto de prueba: {self.accuracy*100:.2f}%')
        else:
            print("El modelo no está cargado.")

class Propuesta_4:
    def __init__(self, images, label):
        aplanado = images.reshape(images.shape[0], -1)
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(aplanado)

        self.y_onehot = to_categorical(label)

    def CrearModelo(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_scaled, self.y_onehot, test_size=0.2, random_state=42)

        self.model = Sequential()
        self.model.add(Dense(128, input_dim=self.X_train.shape[1], activation='tanh'))
        self.model.add(Dense(64, activation='tanh'))    
        self.model.add(Dense(7, activation='softmax'))

        self.best_model = ModelCheckpoint(filepath='model.keras', monitor='val_accuracy', save_best_only=True)

        self.optimi = SGD(learning_rate=0.03)
        self.model.compile(optimizer=self.optimi, loss='mse', metrics=['accuracy'])

        self.history = self.model.fit(self.X_train, self.y_train, epochs=100,
                            batch_size=10, validation_split=0.1, callbacks=[self.best_model])

        self.model = tf.keras.models.load_model('model.keras')

        return self.model
    
    def EvaluarModelo(self):
        if self.model is not None:
            self.loss, self.accuracy = self.model.evaluate(self.X_test, self.y_test)
            print(f'Precisión en el conjunto de prueba: {self.accuracy*100:.2f}%')
        else:
            print("El modelo no está cargado.")

class Propuesta_5:
    def __init__(self, images, label):
        aplanado = images.reshape(images.shape[0], -1)
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(aplanado)

        self.y_onehot = to_categorical(label)

    def CrearModelo(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_scaled, self.y_onehot, test_size=0.2, random_state=42)

        self.model = Sequential()
        self.model.add(Dense(128, input_dim=self.X_train.shape[1], activation='tanh'))
        self.model.add(Dense(64, activation='relu'))    
        self.model.add(Dense(7, activation='softmax'))

        self.best_model = ModelCheckpoint(filepath='model.keras', monitor='val_accuracy', save_best_only=True)

        self.optimi = SGD(learning_rate=0.03)
        self.model.compile(optimizer=self.optimi, loss='mse', metrics=['accuracy'])

        self.history = self.model.fit(self.X_train, self.y_train, epochs=100,
                            batch_size=10, validation_split=0.1, callbacks=[self.best_model])

        self.model = tf.keras.models.load_model('model.keras')

        return self.model
    
    def EvaluarModelo(self):
        if self.model is not None:
            self.loss, self.accuracy = self.model.evaluate(self.X_test, self.y_test)
            print(f'Precisión en el conjunto de prueba: {self.accuracy*100:.2f}%')
        else:
            print("El modelo no está cargado.")
