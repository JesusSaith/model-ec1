import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt

print(tf.__version__)

# Generar datos de entrada X y calcular y = 10x - 5
X = np.linspace(-10.0, 10.0, 100)
np.random.shuffle(X)
y = 10 * X - 5

# Dividir los datos en conjuntos de entrenamiento, validación y prueba
train_end = int(0.6 * len(X))
test_start = int(0.8 * len(X))

X_train, y_train = X[:train_end], y[:train_end]
X_test, y_test = X[test_start:], y[test_start:]
X_val, y_val = X[train_end:test_start], y[train_end:test_start]

tf.keras.backend.clear_session()
linear_model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1], name='Single')
])

# Compilar el modelo
linear_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), loss='mse', metrics=['mae'])
print(linear_model.summary())

# Entrenar el modelo
history = linear_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=500)

# 10 valores asignados
input_values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Predicciones
predictions = linear_model.predict(input_values)

print("Predicciones para los 10 valores asignados:")
for i in range(10):
    print(f"Entrada: {input_values[i]:.2f}, Predicción: {predictions[i][0]:.2f}")

# Obtener los pesos (W) y el sesgo (b)
for layer in linear_model.layers:
    weights, biases = layer.get_weights()
    print(f"\nCapa: {layer.name}")
    print(f"Pesos (W): {weights}")
    print(f"Sesgo (b): {biases}")

# Guardar el modelo
model_name = 'model-ec1'
export_path = f'./{model_name}/1/'
tf.saved_model.save(linear_model, os.path.join(export_path))
