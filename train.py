import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import csv
import os

# Função para salvar as métricas das últimas 5 épocas
def salvar_metricas_ultimas_epocas(history, num_epocas=5):
    """Salva as métricas das últimas 'num_epocas' épocas do histórico de treino."""
    ultimas_epocas = {}
    for metric in history.history:
        ultimas_epocas[metric] = history.history[metric][-num_epocas:]
    return ultimas_epocas

# Caminhos das pastas de treino e teste
train_base_dir = 'image/train/train'
test_base_dir = 'image/test/test'

# Gerador de dados com data augmentation para prevenir overfitting no treino
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Apenas normalização para os dados de teste
test_datagen = ImageDataGenerator(rescale=1./255)

# Gerador de dados para o conjunto de treino
train_generator = train_datagen.flow_from_directory(
    train_base_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

# Gerador de dados para o conjunto de teste (sem aumento de dados)
test_generator = test_datagen.flow_from_directory(
    test_base_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Exibe o mapeamento das classes
class_indices = train_generator.class_indices
print("Mapeamento de classes:", class_indices)

# Função para aplicar uma limiarização simples nas imagens
def thresholding(x, threshold=0.5):
    """Aplica uma função de limiarização nas imagens para torná-las binárias."""
    return tf.where(x > threshold, 1.0, 0.0)

# Função para aplicar um recorte central (CROP) para focar em regiões de interesse
def crop_and_roi(x):
    """Aplica um recorte central nas imagens para focar nas áreas mais importantes."""
    cropped = tf.image.central_crop(x, central_fraction=0.7)
    return cropped

# Função para construir a CNN
def build_cnn(input_shape):
    """Constrói a parte da rede convolucional (CNN) do modelo."""
    inputs = layers.Input(shape=input_shape)
    
    # Primeira camada aplica recorte e limiarização
    x = layers.Lambda(crop_and_roi)(inputs)
    x = layers.Lambda(thresholding)(x)
    
    # Camadas convolucionais com regularização L2 para evitar overfitting
    x = layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Flatten()(x)
    
    return models.Model(inputs=inputs, outputs=x)

# Função para construir a parte MLP do modelo
def build_mlp():
    """Constrói a parte do modelo correspondente ao MLP para classificação."""
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(3, activation='softmax'))
    
    return model

# Função recursiva para calcular a soma dos pesos absolutos da rede
def recursive_weight_sum(layer_weights, idx=0):
    """Calcula recursivamente a soma dos pesos absolutos de todas as camadas do modelo."""
    if idx >= len(layer_weights):
        return 0
    return np.sum(np.abs(layer_weights[idx])) + recursive_weight_sum(layer_weights, idx + 1)

# Função para ordenar as previsões
def sort_predictions(predictions):
    """Ordena as previsões por ordem de confiança."""
    return sorted(enumerate(predictions), key=lambda x: x[1], reverse=True)

# Combinar CNN e MLP em um modelo final
def combine_models(cnn_model):
    """Combina a parte CNN com o MLP em um modelo completo."""
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(cnn_model.output)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(3, activation='softmax')(x)
    
    final_model = models.Model(inputs=cnn_model.input, outputs=output)
    final_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return final_model

# Construção do modelo CNN
cnn = build_cnn(input_shape=(64, 64, 3))

# Combina a CNN com o MLP
final_model = combine_models(cnn)

# Configuração do Early Stopping para interromper o treino caso a validação pare de melhorar
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Redução da taxa de aprendizado caso o modelo pare de melhorar
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

# Configuração para salvar o melhor modelo em um arquivo .h5
checkpoint = ModelCheckpoint('modelo_melhor_val_loss.h5', monitor='val_loss', save_best_only=True, mode='min')

# Treinamento do modelo com Early Stopping, ModelCheckpoint e redução de learning rate
history = final_model.fit(train_generator, 
                          epochs=100, 
                          validation_data=test_generator, 
                          callbacks=[early_stopping, reduce_lr, checkpoint])

# Salvar as métricas das últimas 5 épocas de treino
metricas_finais = salvar_metricas_ultimas_epocas(history)
print("Métricas das últimas 5 épocas:", metricas_finais)

# Avaliação do modelo no conjunto de teste
test_loss, test_acc = final_model.evaluate(test_generator)
print(f"Acurácia no conjunto de teste: {test_acc * 100:.2f}%")

# Seleção de 5 imagens de cada classe para avaliação detalhada
berne_indices = [i for i, label in enumerate(test_generator.labels) if label == class_indices['berne']][:5]
dermatite_indices = [i for i, label in enumerate(test_generator.labels) if label == class_indices['dermatite']][:5]
saudavel_indices = [i for i, label in enumerate(test_generator.labels) if label == class_indices['saudavel']][:5]

selected_indices = berne_indices + dermatite_indices + saudavel_indices

# Salvar previsões e métricas em um CSV
with open('metricas_finais.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    
    # Salva as métricas das últimas épocas
    writer.writerow(metricas_finais.keys())  # Cabeçalhos das métricas
    writer.writerows(zip(*metricas_finais.values()))  # Valores das métricas
    
    # Cabeçalhos para as previsões
    writer.writerow(['Imagem', 'Classe Verdadeira', 'Classe Prevista', 'Acurácia (%)'])
    
    # Para cada imagem selecionada, salvar previsões
    for idx in selected_indices:
        img = test_generator[0][0][idx]  # Obtem uma imagem específica
        true_label_idx = test_generator.labels[idx]  # Classe verdadeira (índice numérico)
        
        # Fazer a previsão
        pred = final_model.predict(np.expand_dims(img, axis=0))
        pred_class_idx = np.argmax(pred)  # Classe prevista (índice numérico)
        pred_confidence = np.max(pred) * 100  # Confiança da previsão
        
        # Converter os índices numéricos para os nomes das classes
        true_class = list(class_indices.keys())[list(class_indices.values()).index(true_label_idx)]
        pred_class = list(class_indices.keys())[list(class_indices.values()).index(pred_class_idx)]
        
        # Nome do arquivo da imagem
        img_name = os.path.basename(test_generator.filenames[idx])
        
        # Escreve os dados no CSV
        writer.writerow([img_name, true_class, pred_class, f"{pred_confidence:.2f}%"])


# Salvar o modelo
model_save_path = 'bovino_classification_model.h5'
final_model.save(model_save_path)
print(f"Modelo salvo em {model_save_path}")

# Exemplo de cálculo recursivo
model_weights = final_model.get_weights()
total_weight_sum = recursive_weight_sum(model_weights)
print(f"Soma total dos pesos absolutos (recursiva): {total_weight_sum}")
