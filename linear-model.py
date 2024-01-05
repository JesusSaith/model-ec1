import numpy as np
import os
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

print(tf.__version__)

#  X con 100 valores
X = np.linspace(-10.0, 10.0, 100)
np.random.shuffle(X)

# Calcular  fórmula y = 10x - 5
y = 10 * X - 5

# Dividir los datos en conjuntos de entrenamiento, validación y prueba
train_end = int(0.6 * len(X))
test_start = int(0.8 * len(X))

X_train, y_train = X[:train_end], y[:train_end]
X_test, y_test = X[test_start:], y[test_start:]
X_val, y_val = X[train_end:test_start], y[train_end:test_start]

# Normalizar los datos
scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train.reshape(-1, 1))
X_val_normalized = scaler.transform(X_val.reshape(-1, 1))
X_test_normalized = scaler.transform(X_test.reshape(-1, 1))

tf.keras.backend.clear_session()
linear_model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# Compilar el modelo
linear_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), loss='mse', metrics=['mae'])
print(linear_model.summary())

# Epochs
history = linear_model.fit(X_train_normalized, y_train, validation_data=(X_val_normalized, y_val), epochs=500)

# Visualizar la pérdida y la métrica
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# 10 valores asignados
input_values = np.array([-8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
input_values_normalized = scaler.transform(input_values.reshape(-1, 1))

predictions = linear_model.predict(input_values_normalized)

# Predicciones
print("Predicciones para los 10 valores asignados:")
for i in range(10):
    print(f"Entrada: {input_values[i]:.2f}, Predicción: {predictions[i][0]:.2f}")

# (W y b)
for layer in linear_model.layers:
    weights, biases = layer.get_weights()
    print(f"\nCapa: {layer.name}")
    print(f"Pesos (W): {weights}")
    print(f"Bias (b): {biases}")

model_name = 'model-ec1'
export_path = f'./{model_name}/1/'
tf.saved_model.save(linear_model, os.path.join(export_path))
