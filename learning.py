from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Flatten, Conv2D, Dense, MaxPooling2D
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
import glob as gb
import cv2
import os

# PASTAS
TRAIN_DIR = 'Violence_Datasets/train'
TEST_DIR = 'Violence_Datasets/test'
MODEL_PATH = 'model1.h5'
BATCH_SIZE = 64

# Funções de visualização
def view_random_image(target_dir, target_class):
    target_folder = os.path.join(target_dir, target_class)
    random_image = np.random.choice(os.listdir(target_folder))
    img = plt.imread(os.path.join(target_folder, random_image))
    plt.imshow(img)
    plt.title(target_class)
    plt.axis('off')
    print(f'Image Shape: {img.shape}')

def count_images(directory):
    for folder in os.listdir(directory):
        files = gb.glob(pathname=os.path.join(directory, folder, "*jpg"))
        print(f"For {directory} data, found {len(files)} in folder {folder}")

count_images(TRAIN_DIR)
count_images(TEST_DIR)


# Classificação

class_names = ['violence', 'non_violence']

plt.figure(figsize=(10, 10))
for i in range(18):
    plt.subplot(3, 6, i + 1)
    class_name = np.random.choice(class_names)
    img = view_random_image(target_dir=TRAIN_DIR, target_class=class_name)

# Preparação do treinamento
train_datagen = ImageDataGenerator(
    rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=(128, 128), batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_set = test_datagen.flow_from_directory(
    TRAIN_DIR, target_size=(128, 128), batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Classificação
classifier = Sequential()
# Adiciona uma camada de convolução 2D com 16 filtros, janela de 3x3, ativação ReLU e especifica a forma de entrada
classifier.add(Conv2D(16, (3,3), input_shape=(128,128,3), activation='relu'))
# Adiciona uma camada de pooling máxima 2D com janela de pooling 2x2
classifier.add(MaxPooling2D(pool_size=(2,2)))
# Adiciona uma segunda camada de convolução 2D com 32 filtros e ativação ReLU
classifier.add(Conv2D(32, (3,3), activation='relu'))
# Adiciona uma camada de achatamento para converter a saída em um vetor unidimensional
classifier.add(Flatten())
# Adiciona uma camada densa totalmente conectada com 128 unidades e ativação ReLU
classifier.add(Dense(units=128, activation='relu'))
# Adiciona a camada de saída com 2 unidades (para classificação binária) e ativação ReLU (corrigir para 'softmax')
classifier.add(Dense(units=2, activation='softmax'))
# Compila o modelo, configurando otimizador, função de perda e métricas de avaliação
classifier.compile(optimizer='adam', loss='categorical_crossentropy', 
                   metrics=['accuracy'])
# Imprime um resumo do modelo, mostrando a arquitetura da rede e o número de parâmetros
classifier.summary()

# Carregar ou treinar o modelo
if os.path.exists(MODEL_PATH):
    loaded_model = tf.keras.models.load_model(MODEL_PATH)
    print("Modelo carregado com sucesso")
    loaded_model.evaluate(test_set)
else:
    history = classifier.fit(training_set, epochs=10, validation_data=test_set)
    classifier.save(MODEL_PATH)
    classifier.evaluate(test_set)

    pd.DataFrame(history.history)[['loss','val_loss']].plot()
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    pd.DataFrame(history.history)[['accuracy','val_accuracy']].plot()
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    loaded_model = tf.keras.loaded_model(MODEL_PATH)

directory = 'media'

image_paths = [os.path.join(directory, filename) 
        for filename in os.listdir(directory) 
        if filename.lower().endswith(('.webp','.png', '.jpg', '.jfif', '.jpeg'))]

# Loop para cada imagem encontrada

for image_path in image_paths:
    image = cv2.imread(image_path)

    if image is not None:
        image_pil = Image.fromarray(image, 'RGB')
        resized_image = image_pil.resize((128, 128))
        expanded_input = np.expand_dims(resized_image, axis=0)
        input_data = expanded_input / 255.0
        predictions = loaded_model.predict(input_data)
        result = np.argmax(predictions)

        print(f"A classe prevista para {image_path} é: {result}")
    else:
        print(f'Não foi possível ler a imagem {image_path} ou a imagem não existe')