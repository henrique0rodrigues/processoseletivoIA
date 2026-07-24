import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# ---------------------------------------------------------------------------
# Projeto 1 — Classificação MNIST
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o dataset MNIST via tf.keras.datasets.mnist
#   2. Normalizar as imagens para [0, 1] e ajustar o shape para (28, 28, 1)
#   3. Separar um conjunto de validação (ex: validation_split ou split manual)
#   4. Construir uma CNN com 3-4 blocos Conv2D + BatchNormalization + MaxPooling2D,
#      seguida de Dropout antes da camada de saída (10 classes, softmax)
#   5. Treinar com EarlyStopping monitorando a perda de validação
#   6. Exibir a acurácia de validação final no terminal
#   7. Salvar o modelo treinado como "model.h5"
# ---------------------------------------------------------------------------

# insira seu código aqui

# Carregando e preprocessando o dataset
print("Carregando o dataset MNIST...")
(x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalizando as imagens para [0, 1] e ajustando o shape para (28, 28, 1)
x_train_full = x_train_full.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train_full = np.expand_dims(x_train_full, -1)
x_test = np.expand_dims(x_test, -1)

print(
    f"Shape do conjunto de treino: {x_train_full.shape}, Shape do conjunto de teste: {x_test.shape}")

# Construção da arquitetura CNN

print("Construindo a arquitetura da CNN...")

model = models.Sequential([

    # bloco 1
    layers.Conv2D(32, (3, 3), activation='relu',
                  padding='same', input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # bloco 2
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # bloco 3
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2), padding='same'),

    # camada densa e dropout
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),  # Dropout exigido antes da camada de saída
    # 10 classes para os dígitos de 0 a 9
    layers.Dense(10, activation='softmax')
])

# Compilando o modelo
print("Compilando o modelo...")
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Configurando EarlyStopping e validation_split
print("Treinando o modelo com EarlyStopping...")
early_stop = callbacks.EarlyStopping(monitor='val_loss',
                                     patience=3,
                                     restore_best_weights=True,
                                     verbose=1)

# treinando usando validation_split de 10% para validação
history = model.fit(x_train_full, y_train_full,
                    epochs=15,
                    batch_size=64,
                    validation_split=0.1,
                    callbacks=[early_stop],
                    verbose=1)

# avaliação final e salvamento do artefato
print("Avaliando e salvando o modelo treinado...")

# obtendo acuracia final de validação
val_acc = max(history.history['val_accuracy'])
print(f"Acurácia final de validação: {val_acc:.4f}")


# salvando o modelo treinado
model_filename = "model.h5"
model.save(model_filename)
print(f"Modelo treinado salvo como '{model_filename}'")
