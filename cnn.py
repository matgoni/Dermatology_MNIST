from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np

class Propuesta_1:
    def __init__(self, images, label):
        # Normalizamos las imágenes al rango [0, 1]
        self.X_scaled = images / 255.0
        self.y_onehot = to_categorical(label)

    def CrearModelo(self):
        # División de los datos de entrenamiento y prueba
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_scaled, self.y_onehot, test_size=0.2, random_state=42)

        # Creación del modelo
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Dropout(0.2))

        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Dropout(0.3))

        self.model.add(Conv2D(128, (3, 3), activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Dropout(0.4))

        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.001)))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
        self.model.add(Dropout(0.3)) 
        self.model.add(Dense(7, activation='softmax'))

        # Configuración del guardado del mejor modelo
        self.best_model = ModelCheckpoint(filepath='model.keras', monitor='val_accuracy', save_best_only=True)

        # Compilación del modelo
        self.optimi = Adam(learning_rate=0.0001)
        self.model.compile(optimizer=self.optimi, loss='categorical_crossentropy', metrics=['accuracy'])

        # Entrenamiento del modelo
        self.history = self.model.fit(self.X_train, self.y_train, epochs=200,
                                    batch_size=16, validation_split=0.1, callbacks=[self.best_model])

        # Carga del mejor modelo después del entrenamiento
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
        # Normalizamos las imágenes al rango [0, 1]
        self.X_scaled = images / 255.0
        self.y_onehot = to_categorical(label)

    def CrearModelo(self):
        # División de los datos de entrenamiento y prueba
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_scaled, self.y_onehot, test_size=0.2, random_state=42)

        # Creación del modelo
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), activation='sigmoid', input_shape=(28, 28, 3)))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Dropout(0.2))

        self.model.add(Conv2D(64, (3, 3), activation='sigmoid'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Dropout(0.3))

        self.model.add(Conv2D(128, (3, 3), activation='sigmoid'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Dropout(0.4))

        self.model.add(Flatten())
        self.model.add(Dense(256, activation='sigmoid', kernel_regularizer=l2(0.001)))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(64, activation='sigmoid', kernel_regularizer=l2(0.001)))
        self.model.add(Dropout(0.3)) 
        self.model.add(Dense(7, activation='softmax'))

        # Configuración del guardado del mejor modelo
        self.best_model = ModelCheckpoint(filepath='model.keras', monitor='val_accuracy', save_best_only=True)

        # Compilación del modelo
        self.optimi = Adam(learning_rate=0.0001)
        self.model.compile(optimizer=self.optimi, loss='categorical_crossentropy', metrics=['accuracy'])

        # Entrenamiento del modelo
        self.history = self.model.fit(self.X_train, self.y_train, epochs=200,
                                    batch_size=16, validation_split=0.1, callbacks=[self.best_model])

        # Carga del mejor modelo después del entrenamiento
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
        # Normalizamos las imágenes al rango [0, 1]
        self.X_scaled = images / 255.0
        self.y_onehot = to_categorical(label)

    def CrearModelo(self):
        # División de los datos de entrenamiento y prueba
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_scaled, self.y_onehot, test_size=0.2, random_state=42)

        # Creación del modelo
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), activation='elu', input_shape=(28, 28, 3)))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Dropout(0.2))

        self.model.add(Conv2D(64, (3, 3), activation='elu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Dropout(0.3))

        self.model.add(Conv2D(128, (3, 3), activation='elu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Dropout(0.4))

        self.model.add(Flatten())
        self.model.add(Dense(256, activation='elu', kernel_regularizer=l2(0.001)))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(64, activation='elu', kernel_regularizer=l2(0.001)))
        self.model.add(Dropout(0.3)) 
        self.model.add(Dense(7, activation='softmax'))

        # Configuración del guardado del mejor modelo
        self.best_model = ModelCheckpoint(filepath='model.keras', monitor='val_accuracy', save_best_only=True)

        # Compilación del modelo
        self.optimi = Adam(learning_rate=0.0001)
        self.model.compile(optimizer=self.optimi, loss='categorical_crossentropy', metrics=['accuracy'])

        # Entrenamiento del modelo
        self.history = self.model.fit(self.X_train, self.y_train, epochs=200,
                                    batch_size=16, validation_split=0.1, callbacks=[self.best_model])

        # Carga del mejor modelo después del entrenamiento
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
        # Normalizamos las imágenes al rango [0, 1]
        self.X_scaled = images / 255.0
        self.y_onehot = to_categorical(label)

    def CrearModelo(self):
        # División de los datos de entrenamiento y prueba
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_scaled, self.y_onehot, test_size=0.2, random_state=42)

        # Creación del modelo
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), activation='tanh', input_shape=(28, 28, 3)))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Dropout(0.2))

        self.model.add(Conv2D(64, (3, 3), activation='tanh'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Dropout(0.3))

        self.model.add(Conv2D(128, (3, 3), activation='tanh'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Dropout(0.4))

        self.model.add(Flatten())
        self.model.add(Dense(256, activation='tanh', kernel_regularizer=l2(0.001)))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(64, activation='tanh', kernel_regularizer=l2(0.001)))
        self.model.add(Dropout(0.3)) 
        self.model.add(Dense(7, activation='softmax'))

        # Configuración del guardado del mejor modelo
        self.best_model = ModelCheckpoint(filepath='model.keras', monitor='val_accuracy', save_best_only=True)

        # Compilación del modelo
        self.optimi = Adam(learning_rate=0.0001)
        self.model.compile(optimizer=self.optimi, loss='categorical_crossentropy', metrics=['accuracy'])

        # Entrenamiento del modelo
        self.history = self.model.fit(self.X_train, self.y_train, epochs=200,
                                    batch_size=16, validation_split=0.1, callbacks=[self.best_model])

        # Carga del mejor modelo después del entrenamiento
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
        # Normalizamos las imágenes al rango [0, 1]
        self.X_scaled = images / 255.0
        self.y_onehot = to_categorical(label)

    def CrearModelo(self):
        # División de los datos de entrenamiento y prueba
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_scaled, self.y_onehot, test_size=0.2, random_state=42)

        # Creación del modelo
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), activation='sigmoid', input_shape=(28, 28, 3)))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Dropout(0.2))

        self.model.add(Conv2D(64, (3, 3), activation='tanh'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Dropout(0.3))

        self.model.add(Conv2D(128, (3, 3), activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Dropout(0.4))

        self.model.add(Flatten())
        self.model.add(Dense(256, activation='elu', kernel_regularizer=l2(0.001)))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(64, activation='sigmoid', kernel_regularizer=l2(0.001)))
        self.model.add(Dropout(0.3)) 
        self.model.add(Dense(7, activation='softmax'))

        # Configuración del guardado del mejor modelo
        self.best_model = ModelCheckpoint(filepath='model.keras', monitor='val_accuracy', save_best_only=True)

        # Compilación del modelo
        self.optimi = Adam(learning_rate=0.0001)
        self.model.compile(optimizer=self.optimi, loss='categorical_crossentropy', metrics=['accuracy'])

        # Entrenamiento del modelo
        self.history = self.model.fit(self.X_train, self.y_train, epochs=200,
                                    batch_size=16, validation_split=0.1, callbacks=[self.best_model])

        # Carga del mejor modelo después del entrenamiento
        self.model = tf.keras.models.load_model('model.keras')

        return self.model
    
    def EvaluarModelo(self):
        if self.model is not None:
            self.loss, self.accuracy = self.model.evaluate(self.X_test, self.y_test)
            print(f'Precisión en el conjunto de prueba: {self.accuracy*100:.2f}%')
        else:
            print("El modelo no está cargado.")


class RedNeuronal:
    def __init__(self, image, label):
        #Iniciar las redes por separado
        self.red_1 = Propuesta_1(image, label)
        self.red_2 = Propuesta_2(image, label)
        self.red_3 = Propuesta_3(image, label)
        self.red_4 = Propuesta_4(image, label)
        self.red_5 = Propuesta_5(image, label)

        #Crear los modelos de cada red
        try:
            self.red_1.CrearModelo()
            self.red_2.CrearModelo()
            self.red_3.CrearModelo()
            self.red_4.CrearModelo()
            self.red_5.CrearModelo()
        except:
            print("Ocurrio un error al crear el modelo, favor de verificarlo")
    
    def EvaluarModelos(self):
        try:
            self.red_1.EvaluarModelo()
            self.red_2.EvaluarModelo()
            self.red_3.EvaluarModelo()
            self.red_4.EvaluarModelo()
            self.red_5.EvaluarModelo()
        except:
            print("Ocurrio un error, favor de verificarlo \n Es posible que el modelo no se haya creado correctamente")

    def Predecir(self, nuevasImgs):
        try:
            prediccion_temp_1 = self.red_1.model.predict(nuevasImgs)
            prediccion_temp_2 = self.red_2.model.predict(nuevasImgs)
            prediccion_temp_3 = self.red_3.model.predict(nuevasImgs)
            prediccion_temp_4 = self.red_4.model.predict(nuevasImgs)
            prediccion_temp_5 = self.red_5.model.predict(nuevasImgs)
        except:
            print("Ocurrio un error en la predicción, favor de verificarlo")

        try:
            prop_1 = np.argmax(prediccion_temp_1, axis=1)
            prop_2 = np.argmax(prediccion_temp_2, axis=1)
            prop_3 = np.argmax(prediccion_temp_3, axis=1)
            prop_4 = np.argmax(prediccion_temp_4, axis=1)
            prop_5 = np.argmax(prediccion_temp_5, axis=1)
        except:
            print("Ocurrio un error en la selección final por modelo, favor de verificarlo")

        posibilidades_totales = []
        for p1, p2, p3, p4, p5 in zip(prop_1, prop_2, prop_3, prop_4, prop_5):
            valores, cuentas = np.unique([p1, p2, p3, p4, p5], return_counts=True)
            temp_mayorFrec = valores[np.argmax(cuentas)]
            posibilidades_totales.append(temp_mayorFrec)

        return np.array(posibilidades_totales)
        