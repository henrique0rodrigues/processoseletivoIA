import os
import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------------
# Projeto 1 — Inferência com o Modelo Otimizado (model.tflite)
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar especificamente o "model.tflite" (o artefato de edge, não o
#      model.h5) usando tf.lite.Interpreter
#   2. Rodar inferência em pelo menos 5 amostras do conjunto de teste do MNIST
#   3. Imprimir no terminal, para cada amostra: classe predita vs. classe real
# ---------------------------------------------------------------------------

tflite_model_path = "model.tflite"

# Carregando o modelo otimizado
print(" Carregando o modelo otimizado")
if not os.path.exists(tflite_model_path):
    raise FileNotFoundError(
        f"Erro: O arquivo '{tflite_model_path}' não foi encontrado.")

# Carregando o interpretador TFLite
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Obtendo detalhes das entradas e saídas
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("interpreter carregado com sucesso.")

# Carregando apenas o conjunto de teste do MNIST para validação real
print("Carregando o conjunto de teste do MNIST...")
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalizando e ajustando o shapes iguais ao treino
x_test = x_test.astype("float32") / 255.0
x_test = np.expand_dims(x_test, axis=-1)

# Executando inferência em 10 amostras

n_samples = 10
print(f"Rodando inferência em {n_samples} amostras do conjunto de teste:\n")
print("=" * 62)
print(f"{'Amostra':<12} | {'Classe Real':<14} | {'Classe Predita':<14} | {'Resultado'}")
print("=" * 62)

acertos = 0

np.random.seed(42)  # Para reprodutibilidade
indices_aleatorios = np.random.choice(len(x_test), n_samples, replace=False)

for idx, i in enumerate(indices_aleatorios, 1):
    # preparando a amostra individual
    input_data = np.expand_dims(x_test[i], axis=0).astype(np.float32)

    # enviando a amostra para o interpretador
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # rodando a inferência
    interpreter.invoke()

    # obtendo a saída
    output_data = interpreter.get_tensor(output_details[0]['index'])
    classe_predita = np.argmax(output_data[0])
    classe_real = y_test[i]

    if classe_predita == classe_real:
        resultado = "Acerto"
        acertos += 1
    else:
        resultado = "Erro"

    print(
        f"Amostra #{idx:<3} | Digito: {classe_real:<6} | Predito: {classe_predita:<5} | {resultado}")

    print("-" * 62)
print(
    f"Total de acertos: {acertos}/{n_samples} | Acurácia: {acertos/n_samples*100:.2f}%")
