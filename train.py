import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Caminhos das pastas de treino e teste
train_base_dir = 'imagetrain'  
test_base_dir = 'image/test'    

# Gerador de dados para as imagens de treino e teste
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_base_dir,
    target_size=(64, 64),  # Assegurar que a imagem seja redimensionada corretamente
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_base_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

# Função para aplicar limiarização (binarização)
def thresholding(x, threshold=0.5):
    return tf.where(x > threshold, 1.0, 0.0)

# Função para aplicar CROP e ROI
def crop_and_roi(x):
    # Corta a imagem para focar na região central (exemplo)
    cropped = tf.image.central_crop(x, central_fraction=0.7)
    return cropped

# CNN Model
def build_cnn(input_shape):
    model = models.Sequential()
    
    # Primeira camada com operação de CROP e limiarização
    model.add(layers.Lambda(crop_and_roi, input_shape=input_shape))  # Aplica CROP e ROI
    model.add(layers.Lambda(thresholding))  # Aplica Limiarização
    
    # Camadas de convolução e pooling
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Flatten())  # Achata para alimentar o MLP
    
    return model

# MLP Model
def build_mlp():
    model = models.Sequential()
    
    # Camada densa para classificação
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))  # 3 classes: Dermatite Nodular, Berne, Saudável
    
    return model

# Função recursiva para calcular a soma dos pesos absolutos da rede
def recursive_weight_sum(layer_weights, idx=0):
    if idx >= len(layer_weights):
        return 0
    return np.sum(np.abs(layer_weights[idx])) + recursive_weight_sum(layer_weights, idx + 1)

# Função de ordenação das probabilidades
def sort_predictions(predictions):
    return sorted(enumerate(predictions), key=lambda x: x[1], reverse=True)

# Combinar CNN e MLP
def combine_models(cnn_model):
    x = layers.Dense(64, activation='relu')(cnn_model.output)
    output = layers.Dense(3, activation='softmax')(x)
    
    final_model = models.Model(inputs=cnn_model.input, outputs=output)
    final_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return final_model

# Construir e treinar o modelo
cnn = build_cnn(input_shape=(64, 64, 3))  # Imagens 64x64 com 3 canais (RGB)
final_model = combine_models(cnn)

# Treinamento
history = final_model.fit(train_generator, epochs=10, validation_data=test_generator)

# Avaliação do modelo
test_loss, test_acc = final_model.evaluate(test_generator)
print(f"Acurácia do Teste: {test_acc * 100:.2f}%")

# Salvar o modelo
model_save_path = 'bovino_classification_model.h5'
final_model.save(model_save_path)
print(f"Modelo salvo em {model_save_path}")

# Exemplo de cálculo recursivo
model_weights = final_model.get_weights()
total_weight_sum = recursive_weight_sum(model_weights)
print(f"Soma total dos pesos absolutos (recursiva): {total_weight_sum}")
