import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


# Dados de treino
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([0, 0, 0, 1, 1])


# Modelo (com a definição do formato de entrada)
model = keras.Sequential([
    layers.Input(shape=(1,)), # Garante que o modelo saiba o formato desde o início
    layers.Dense(4, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])


model.compile(optimizer='adam', loss='binary_crossentropy')


model.fit(X, y, epochs=200, verbose=0)


# Correção aqui: passando um Array NumPy em vez de uma lista Python
print(model.predict(np.array([[4]])))