import os
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import shutil

# Caminhos das pastas
base_dir = 'image/'  # Diretório base com subpastas "dermatite", "berne", "saudavel"
classes = ['dermatite', 'berne', 'saudavel']

# Carregar as imagens e rótulos
def load_images(base_dir, classes):
    images = []
    labels = []
    
    for label, class_name in enumerate(classes):
        class_dir = os.path.join(base_dir, class_name)
        
        # Para cada arquivo de imagem na pasta
        for filename in os.listdir(class_dir):
            img_path = os.path.join(class_dir, filename)
            images.append(img_path)
            labels.append(label)
    
    return np.array(images), np.array(labels)

# Carregar as imagens
images, labels = load_images(base_dir, classes)

# Dividindo as imagens com holdout estratificado 80% treino e 20% teste
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_idx, test_idx in split.split(images, labels):
    train_images, test_images = images[train_idx], images[test_idx]
    train_labels, test_labels = labels[train_idx], labels[test_idx]

# Função para criar diretórios e mover imagens
def create_and_move_files(images, labels, base_dir, set_type):
    for i, img_path in enumerate(images):
        label_name = classes[labels[i]]
        target_dir = os.path.join(base_dir, set_type, label_name)
        
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        
        # Movendo a imagem para o novo diretório (opcionalmente pode copiar com shutil.copy)
        shutil.copy(img_path, target_dir)

# Criar e mover as imagens para as pastas de treino e teste
train_base_dir = 'image/train'
test_base_dir = 'image/test'

create_and_move_files(train_images, train_labels, train_base_dir, 'train')
create_and_move_files(test_images, test_labels, test_base_dir, 'test')

# Gerador de dados para treino e teste
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_base_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_base_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

