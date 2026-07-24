# Projeto 1 — Classificação MNIST

## 💻 O Desafio Técnico

## 📝 Relatório do Candidato

👤 **Nome Completo:** Henrique Oliveira Rodrigues

### 1️⃣ Resumo da Arquitetura do Modelo

A arquitetura implementada em `train_model.py` foi projetada utilizando uma Rede Neural Convolucional sequencial focada em eficiência para Edge AI. A rede é composta por:
* 3 Blocos Convolucionais: Cada bloco utiliza camadas `Conv2D` com ativação ReLU (com progressão de 32, 64 e 128 filtros, kernel 3x3), seguidas por `BatchNormalization` para estabilizar a distribuição das ativações durante o treino e `MaxPooling2D` (pool 2x2) para redução espacial e invariância a translações.
* Camada de Regularização: Aplicação explícita de `Dropout(0.5)` logo após a camada densa intermediária (64 neurônios) e antes da camada de saída, prevenindo overfitting.
* Camada de Saída: Camada `Dense` com 10 neurônios e ativação `softmax`, representando a distribuição de probabilidade das 10 classes numéricas.
* Estratégia de Validação: Utilização de split explícito de 10% dos dados para validação em conjunto com callback de `EarlyStopping` monitorando a perda de validação (`val_loss`, paciência = 3), garantindo a restauração automática dos melhores pesos do treinamento.

### 2️⃣ Bibliotecas Utilizadas

* TensorFlow / Keras: Utilizada como framework principal para processamento do dataset MNIST, modelagem da CNN, compilação, treinamento e exportação/conversão para Edge AI.
* NumPy: Utilizada para manipulação matricial de tensores, normalização em ponto flutuante e dimensionamento de canais, além da amostragem aleatória controlada na inferência.
* OS: Utilizada para verificação de existência de arquivos locais e cálculo métrico comparativo de tamanho de arquivos em disco.

### 3️⃣ Técnica de Otimização do Modelo

A técnica aplicada em `optimize_model.py` foi a Quantização de Alcance Dinâmico (Dynamic Range Quantization), implementada via `tf.lite.Optimize.DEFAULT` no conversor do TensorFlow Lite. 
Essa abordagem analisa os pesos da rede neural e os quantiza para números inteiros de 8 bits durante o armazenamento do arquivo. Durante a execução da inferência na borda, os pesos são desquantizados dinamicamente, permitindo uma drástica redução na pegada de memória sem necessitar de um dataset de calibração extra e preservando a precisão matemática da rede.

### 4️⃣ Resultados Obtidos

* Acurácia de Validaçao Final: ~99.33%.
* Tamanho do arquivo original (`model.h5`): 2.709,56 KB.
* Tamanho do modelo otimizado (`model.tflite`): 234,52 KB.
* Taxa de Redução de Tamanho: 91,34% de economia de memória.

### 5️⃣ Comentários Adicionais (Opcional)

### 6️⃣ Exemplo de Inferência

Abaixo, a saída gerada pela execução de 10 amostras aleatórias via `run_inference.py` utilizando o interpretador do arquivo de borda:

```text
==============================================================
Amostra      | Classe Real    | Classe Predita | Resultado
==============================================================
Amostra #1   | Digito: 6      | Predito: 6     | Acerto
--------------------------------------------------------------
Amostra #2   | Digito: 2      | Predito: 2     | Acerto
--------------------------------------------------------------
Amostra #3   | Digito: 3      | Predito: 3     | Acerto
--------------------------------------------------------------
Amostra #4   | Digito: 7      | Predito: 7     | Acerto
--------------------------------------------------------------
Amostra #5   | Digito: 2      | Predito: 2     | Acerto
--------------------------------------------------------------
Amostra #6   | Digito: 2      | Predito: 2     | Acerto
--------------------------------------------------------------
Amostra #7   | Digito: 3      | Predito: 3     | Acerto
--------------------------------------------------------------
Amostra #8   | Digito: 4      | Predito: 4     | Acerto
--------------------------------------------------------------
Amostra #9   | Digito: 7      | Predito: 7     | Acerto
--------------------------------------------------------------
Amostra #10  | Digito: 6      | Predito: 6     | Acerto
--------------------------------------------------------------
Total de acertos: 10/10 | Acurácia: 100.00%
```
Em todas as 10 amostras testadas aleatoriamente, o modelo quantizado obteve assertividade de 100%. Isso comprova que o artefato `.tflite` gerado manteve a capacidade de generalização da rede original, sendo ideal para implantação em microcontroladores e sistemas embarcados.