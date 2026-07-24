import tensorflow as tf
import os

# ---------------------------------------------------------------------------
# Projeto 1 — Otimização do Modelo (MNIST)
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o modelo treinado em "model.h5"
#   2. Converter para TensorFlow Lite usando tf.lite.TFLiteConverter
#   3. Aplicar uma técnica de otimização (ex: Dynamic Range Quantization,
#      via converter.optimizations = [tf.lite.Optimize.DEFAULT])
#   4. Salvar o resultado como "model.tflite"
# ---------------------------------------------------------------------------

# insira seu código aqui

# Configurando os arquivos de entrada e saída
input_model_path = "model.h5"
output_tflite_path = "model.tflite"

print(f"Carregando o modelo treinado de {input_model_path}...")
if not os.path.exists(input_model_path):
    raise FileNotFoundError(
        f"O arquivo {input_model_path} não foi encontrado.")

# Carregando o modelo Keras
model = tf.keras.models.load_model(input_model_path)
print("Modelo carregado com sucesso.")

# Criando o conversor TFLite
print("Convertendo o modelo para TensorFlow Lite com otimização...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Aplicando otimização (Dynamic Range Quantization)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_quant_model = converter.convert()
print("Conversão para TensorFlow Lite com otimização concluída.")

# Salvando o modelo otimizado
with open(output_tflite_path, "wb") as f:
    f.write(tflite_quant_model)

# comparação numerica dos tamanhos dos arquivos
size_h5 = os.path.getsize(input_model_path)/1024
size_tflite = os.path.getsize(output_tflite_path)/1024
reduction = ((size_h5 - size_tflite) / size_h5) * 100
print(f"[Sucesso] Arquivo salvo: {output_tflite_path}")
print(f"Tamanho do modelo original (model.h5): {size_h5:.2f} KB")
print(f"Tamanho do modelo otimizado (model.tflite): {size_tflite:.2f} KB")
print(f"Redução de tamanho: {reduction:.2f}%")
