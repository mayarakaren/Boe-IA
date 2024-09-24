### Instruções de Treinamento da IA: Explicação do Processo

Aqui está uma explicação detalhada sobre como o processo de treinamento da IA ocorrerá ao longo de 3 dias, ajustando a quantidade de épocas e salvando as informações automaticamente.

---

### **Objetivo**
Treinar um modelo de rede neural para classificação de doenças dermatológicas bovinas em três categorias: **Dermatite Nodular**, **Berne** e **Saudável**. O treinamento será realizado em três fases, onde cada fase corresponde a um dia com uma quantidade de épocas ajustada.

---

### **Dia 1: Treinamento com 100 Épocas**

#### **Processo de Treinamento**
1. **Início do Treinamento**:
   - O treinamento começará com **100 épocas**. Durante essas épocas, o modelo processará as imagens de treinamento, ajustando seus pesos para melhorar a capacidade de prever corretamente as classes.
   - O conjunto de validação (imagens de teste) será utilizado para verificar a performance do modelo enquanto ele é treinado, fornecendo métricas de **acurácia** (accuracy) e **perda** (loss), tanto para o treino quanto para o teste.

2. **Coleta de Métricas**:
   - O código automaticamente coleta as métricas do treinamento. Após o final das 100 épocas, as **métricas das últimas 5 épocas** são extraídas.
   - As principais métricas salvas incluem:
     - **Acurácia no treinamento e validação**: Indica a porcentagem de classificações corretas.
     - **Perda no treinamento e validação**: Mostra o quão bem o modelo está se ajustando aos dados.
   - Essas métricas são salvas diretamente em um arquivo **CSV**, então não há necessidade de fazer isso manualmente. Você terá um registro das métricas prontamente disponível.

3. **Testes com Imagens Específicas**:
   - O modelo será testado usando **5 imagens de cada classe** que não foram usadas no treinamento. Esse processo também é automatizado pelo código.
   - O modelo fará previsões para essas imagens, e a **acurácia para cada classe** será registrada no arquivo CSV. Isso é importante para observar como o modelo se comporta ao classificar imagens que não viu durante o treinamento.

---

### **Dia 2: Treinamento com 250 Épocas**

`history = final_model.fit(train_generator, epochs=250, validation_data=test_generator)`

#### **Processo de Ajuste**
1. **Alteração no Número de Épocas**:
   - No segundo dia, o número de épocas será ajustado para **250**. A mudança no número de épocas visa melhorar a performance do modelo, permitindo que ele treine por mais tempo e aprenda melhor os padrões presentes nas imagens.
   - O restante do processo segue o mesmo fluxo: o modelo treina, ajusta seus pesos, e valida com as imagens de teste durante o processo.

2. **Coleta de Métricas**:
   - Como no Dia 1, as métricas das **últimas 5 épocas** serão salvas automaticamente no CSV. Esse histórico será importante para comparar o desempenho do modelo entre os diferentes dias de treinamento e diferentes quantidades de épocas.
   - Além das métricas de acurácia e perda, o código também salva a acurácia por classe com as 5 imagens de teste específicas.

3. **Teste Final**:
   - Assim como no primeiro dia, após o término do treinamento, o modelo será testado com as mesmas 5 imagens de cada classe, permitindo que você observe como o desempenho pode ter mudado após mais épocas de treinamento.

---

### **Dia 3: Treinamento com 500 Épocas**

`history = final_model.fit(train_generator, epochs=500, validation_data=test_generator)`

#### **Última Fase do Treinamento**
1. **Ajuste Final nas Épocas**:
   - No terceiro e último dia, o número de épocas será ajustado para **500**. Esse número mais alto de épocas dará ao modelo mais oportunidades de refinar seus pesos e, idealmente, melhorar sua capacidade de classificação.
   - O treinamento segue o mesmo processo dos dias anteriores, porém, com um maior número de iterações sobre os dados.

2. **Coleta de Métricas**:
   - Novamente, as **métricas das últimas 5 épocas** serão salvas, assim como as acurácias por classe para as imagens de teste. Isso permitirá a comparação final de como o desempenho do modelo se altera conforme ele treina por mais tempo.

3. **Teste com Imagens**:
   - No final do terceiro dia, o modelo será novamente testado com as 5 imagens de cada classe, e a acurácia individual por classe será registrada, permitindo a análise detalhada de seu desempenho.

---

### **Observações Finais**

- **Salvamento Automático das Informações**: Você não precisa realizar manualmente o salvamento das métricas ou das acurácias por classe. O código já está configurado para coletar todas essas informações e armazená-las automaticamente em um arquivo CSV ao final de cada fase de treinamento.
  
- **Comparação ao Longo dos Dias**: Com os dados armazenados em CSV, você poderá facilmente comparar o desempenho do modelo entre os diferentes dias de treinamento, visualizando a evolução das métricas e da acurácia conforme o número de épocas aumenta.

- **Análise Pós-Treinamento**: Ao final dos três dias, você terá uma visão clara de como o modelo performa para cada classe, além de poder utilizar as métricas armazenadas para criar gráficos que mostram a evolução da acurácia e perda ao longo do tempo.

---