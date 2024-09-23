# README - Classificação de Imagens de Bovinos

Este repositório contém um projeto para classificar imagens de bovinos em três categorias: Dermatite Nodular, Berne e Saudável. O projeto é dividido em três partes principais: processamento de dados, treinamento do modelo e API para predições.

## Estrutura do Projeto

```
image/
  berne/
  dermatite/
  saudavel/  
  train/
    train/
      berne/
      dermatite/
      saudavel/
  test/
    test/
      berne/
      dermatite/
      saudavel/
bovino_classification_model.h5
train.py
app.py
holdout.py
```

## Pré-requisitos

- Python 3.6 ou superior
- TensorFlow 2.x
- Keras
- Flask
- scikit-learn
- NumPy
- PIL (Pillow)

Você pode instalar as dependências necessárias utilizando o pip:

```bash
pip install tensorflow flask scikit-learn numpy pillow
```

## 1. Processamento de Dados

O script `holdout.py` é responsável por organizar as imagens em diretórios de treino e teste usando o método de holdout estratificado. Isso garante que as classes estejam distribuídas igualmente entre os conjuntos de treino e teste.

### Como Executar

1. **Organize suas Imagens:**
   Certifique-se de que suas imagens estejam organizadas na estrutura de diretórios abaixo:

   ```
   image/
     dermatite/
     berne/
     saudavel/
   ```

2. **Execute o Script:**
   Execute o script `holdout.py` para dividir as imagens em conjuntos de treino e teste:

   ```bash
   python holdout_split.py
   ```

   Isso criará as pastas `train` e `test` dentro da pasta `image` com as subpastas para cada classe.

## 2. Treinamento do Modelo

O script `train.py` é responsável por treinar o modelo de classificação de imagens usando uma CNN (Rede Neural Convolucional).

### Como Executar

1. **Certifique-se de que as pastas `image/train` e `image/test` estão organizadas corretamente.** O código assume a seguinte estrutura:

   ```
   image/
     train/
       train/
         berne/
         dermatite/
         saudavel/
     test/
       test/
         berne/
         dermatite/
         saudavel/
   ```

2. **Execute o Script de Treinamento:**
   Para treinar o modelo, execute o seguinte comando:

   ```bash
   python train.py
   ```

   - **O código vai:**
     - Carregar as imagens a partir das pastas.
     - Redimensionar as imagens para 64x64 pixels e normalizá-las.
     - Definir e compilar um modelo de CNN.
     - Treinar o modelo por 100 épocas.
     - Avaliar o modelo usando o conjunto de teste.
     - Salvar o modelo treinado em um arquivo `bovino_classification_model.h5`.

## 3. API para Predições

O script `app.py` cria uma API utilizando Flask para permitir a predição de novas imagens.

### Como Executar

1. **Certifique-se de que o modelo treinado está salvo:**
   Após treinar o modelo com o `train.py`, o arquivo `bovino_classification_model.h5` deve estar presente no diretório.

2. **Execute o Script da API:**
   Para iniciar a API, execute o seguinte comando:

   ```bash
   python api.py
   ```

   - **A API estará disponível em** `http://127.0.0.1:5000/predict`.

3. **Fazendo uma Predição:**
   Para fazer uma predição, você pode usar ferramentas como Postman ou cURL para enviar uma requisição POST com uma imagem.

   **Exemplo de uso com cURL:**

   ```bash
   curl -X POST -F 'file=@/caminho/para/sua/imagem.jpg' http://127.0.0.1:5000/predict
   ```

   **A resposta será um JSON contendo:**
   - `predicted_class`: a classe prevista (Dermatite Nodular, Berne, Saudável).
   - `accuracy`: a probabilidade associada à classe prevista.

## Conclusão

Esse projeto fornece uma abordagem completa para classificar imagens de bovinos. Certifique-se de seguir os passos na ordem correta e de que suas imagens estão organizadas corretamente para evitar erros. Se você tiver dúvidas ou encontrar problemas, sinta-se à vontade para entrar em contato!