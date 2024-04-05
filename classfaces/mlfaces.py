import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Caminho para o diretório que contém as imagens
data_dir = './faces/lfw_funneled'

# Carregar o conjunto de imagens
dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2, # Por exemplo, 20% das imagens serão usadas para validação
    subset="training", # Use "validation" para carregar as imagens de validação
    seed=123, # Semente para garantir que a divisão de treino/validação seja consistente
    image_size=(180, 180), # Tamanho das imagens, ajuste conforme necessário
    batch_size=32) # Tamanho do lote

# Exemplo de como visualizar uma imagem do conjunto de dados
for images, labels in dataset.take(1):
    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(labels[i].numpy())
    plt.show()

# Ajustar as imagens para o modelo
dataset = dataset.map(lambda x, y: (x / 255.0, y))

# Configurar o modelo
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu), 
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Treinar o modelo
model.fit(dataset, epochs=10)

# Avaliar o modelo
test_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(180, 180),
    batch_size=32)

test_dataset = test_dataset.map(lambda x, y: (x / 255.0, y))
model.evaluate(test_dataset)

# Fazer previsões
classifications = model.predict(test_dataset)
print(classifications[0])