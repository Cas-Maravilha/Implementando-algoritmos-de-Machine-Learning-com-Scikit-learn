# Comparação de Algoritmos de Classificação

Este script implementa e compara diferentes algoritmos de classificação seguindo exatamente os passos especificados na tarefa.

## 🎯 Objetivo

Implementar e comparar diferentes algoritmos de classificação para o dataset de sementes de trigo, seguindo os passos:

1. **Separação dos dados** (70% treino, 30% teste)
2. **Implementação de 5 algoritmos** diferentes
3. **Treinamento dos modelos**
4. **Avaliação com múltiplas métricas**
5. **Comparação de desempenho**

## 🔧 Algoritmos Implementados

### 1. K-Nearest Neighbors (KNN)
- **Parâmetros**: n_neighbors=5, metric='euclidean'
- **Características**: Classificação baseada em vizinhança
- **Dados**: Normalizados

### 2. Support Vector Machine (SVM)
- **Parâmetros**: kernel='rbf', C=1.0, gamma='scale'
- **Características**: Classificação baseada em margens
- **Dados**: Normalizados

### 3. Random Forest
- **Parâmetros**: n_estimators=100
- **Características**: Ensemble de árvores de decisão
- **Dados**: Originais (não normalizados)

### 4. Naive Bayes
- **Parâmetros**: GaussianNB (padrão)
- **Características**: Classificação probabilística
- **Dados**: Originais (não normalizados)

### 5. Logistic Regression
- **Parâmetros**: max_iter=1000
- **Características**: Classificação linear
- **Dados**: Normalizados

## 📊 Métricas de Avaliação

Para cada algoritmo, são calculadas as seguintes métricas:

- **Acurácia (Accuracy)**: Proporção de predições corretas
- **Precisão (Precision)**: Proporção de predições positivas corretas
- **Recall**: Proporção de casos positivos identificados corretamente
- **F1-Score**: Média harmônica entre precisão e recall
- **Matriz de Confusão**: Visualização detalhada dos erros

## 🚀 Como Executar

### 1. Configuração do Ambiente

```bash
# Ativar o ambiente virtual
.\venv_ml_graos\Scripts\activate
```

### 2. Executar a Comparação

```bash
# Executar comparação de algoritmos
python comparacao_algoritmos.py
```

## 📈 Resultados Esperados

### 📊 Saídas do Script

1. **Exploração dos Dados**
   - Informações do dataset
   - Estatísticas descritivas
   - Distribuição das classes (gráfico de pizza)

2. **Preparação dos Dados**
   - Separação 70% treino / 30% teste
   - Distribuição das classes em cada conjunto
   - Normalização dos dados

3. **Treinamento dos Modelos**
   - Tempo de treinamento para cada algoritmo
   - Confirmação de treinamento bem-sucedido

4. **Avaliação dos Modelos**
   - Métricas de desempenho para cada algoritmo
   - Comparação lado a lado

5. **Visualizações**
   - Matrizes de confusão para todos os modelos
   - Gráficos de comparação de métricas
   - Identificação do melhor modelo

6. **Relatórios Detalhados**
   - Classification report para cada algoritmo
   - Análise de erros de classificação

## 📋 Estrutura do Código

### Classe `ComparacaoAlgoritmos`

```python
class ComparacaoAlgoritmos:
    def __init__(self, random_state=42)
    def carregar_dados(self)
    def explorar_dados(self)
    def preparar_dados(self)
    def definir_algoritmos(self)
    def treinar_modelos(self)
    def avaliar_modelos(self)
    def criar_matrizes_confusao(self)
    def comparar_desempenho(self)
    def relatorios_detalhados(self)
    def analise_erros(self)
    def executar_analise_completa(self)
```

### Métodos Principais

1. **`carregar_dados()`**: Carrega o dataset de sementes da UCI
2. **`preparar_dados()`**: Separa dados (70/30) e normaliza
3. **`definir_algoritmos()`**: Instancia os 5 algoritmos
4. **`treinar_modelos()`**: Treina todos os modelos
5. **`avaliar_modelos()`**: Calcula métricas de desempenho
6. **`comparar_desempenho()`**: Compara e visualiza resultados

## 🎯 Características Especiais

### ✅ Separação Estratificada
- **test_size=0.3**: 30% para teste, 70% para treino
- **stratify=y**: Mantém proporção das classes
- **random_state=42**: Reproduzibilidade

### ✅ Normalização Inteligente
- **Algoritmos que precisam**: KNN, SVM, Logistic Regression
- **Algoritmos que não precisam**: Random Forest, Naive Bayes
- **StandardScaler**: Normalização z-score

### ✅ Avaliação Robusta
- **Múltiplas métricas**: Accuracy, Precision, Recall, F1-Score
- **Matrizes de confusão**: Visualização detalhada
- **Relatórios completos**: Classification report por classe

### ✅ Análise de Erros
- **Identificação de erros**: Amostras mal classificadas
- **Detalhamento**: Classe real vs. classe predita
- **Insights**: Padrões nos erros

## 📊 Interpretação dos Resultados

### 🏆 Identificação do Melhor Modelo
- **Critério principal**: F1-Score (média harmônica)
- **Critério secundário**: Acurácia
- **Visualização**: Gráficos de barras comparativos

### 📈 Análise de Performance
- **Overfitting**: Comparação treino vs. teste
- **Bias-Variance**: Análise dos erros
- **Robustez**: Consistência entre métricas

### 🔍 Insights dos Dados
- **Features importantes**: Análise de importância (Random Forest)
- **Correlações**: Relações entre variáveis
- **Distribuições**: Padrões por classe

## 🚀 Próximos Passos

### 🔬 Melhorias Técnicas
1. **Otimização de Hiperparâmetros**: GridSearchCV para cada algoritmo
2. **Validação Cruzada**: K-fold cross-validation
3. **Ensemble Methods**: Voting, Stacking, Bagging
4. **Feature Selection**: Seleção de features mais relevantes

### 📊 Análises Avançadas
1. **Curvas ROC**: Análise de threshold
2. **Curvas de Aprendizado**: Bias vs. Variance
3. **Análise de Outliers**: Detecção de amostras anômalas
4. **Interpretabilidade**: SHAP, LIME

### 🎯 Aplicações Práticas
1. **Deploy do Modelo**: API para predições
2. **Monitoramento**: Acompanhamento de performance
3. **Retreinamento**: Pipeline automático
4. **Documentação**: Guias de uso

## 📞 Suporte

Se encontrar problemas:

1. **Verificar ambiente**: Execute `teste_basico.py` primeiro
2. **Dependências**: Certifique-se de que todas estão instaladas
3. **Conexão**: Verifique sua conexão com a internet
4. **Logs**: Verifique as mensagens de output
5. **Documentação**: Consulte este README

## 📚 Referências

- [Scikit-learn Classification](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.neighbors)
- [UCI Seeds Dataset](https://archive.ics.uci.edu/ml/datasets/seeds)
- [Machine Learning Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)

---

**🎉 Script pronto para execução! Execute `python comparacao_algoritmos.py` para começar a análise.** 