# Análise de Classificação de Grãos - CRISP-DM

Este projeto implementa uma análise completa de classificação de grãos seguindo a metodologia CRISP-DM (Cross-Industry Standard Process for Data Mining).

## 📋 Estrutura do Projeto

```
├── analise_graos.py          # Script principal da análise
├── teste_basico.py           # Script de teste das bibliotecas
├── requirements.txt          # Dependências do projeto
├── seeds_dataset.md         # Descrição do dataset
└── README.md                # Este arquivo
```

## 🎯 Objetivo

Desenvolver um modelo de machine learning capaz de classificar sementes de trigo em três variedades diferentes:
- **Kama** (classe 1)
- **Rosa** (classe 2) 
- **Canadian** (classe 3)

## 📊 Dataset

- **Fonte**: UCI Machine Learning Repository
- **Atributos**: 7 características físicas das sementes
  - Área
  - Perímetro
  - Compacidade
  - Comprimento do Kernel
  - Largura do Kernel
  - Coeficiente de Assimetria
  - Comprimento do Sulco

## 🚀 Como Executar

### 1. Configuração do Ambiente

```bash
# Ativar o ambiente virtual
.\venv_ml_graos\Scripts\activate

# Verificar se está ativo (deve aparecer (venv_ml_graos) no início da linha)
```

### 2. Teste Básico

```bash
# Executar teste básico para verificar se tudo está funcionando
python teste_basico.py
```

### 3. Análise Completa

```bash
# Executar análise completa com machine learning
python analise_graos.py
```

## 📈 Metodologia CRISP-DM

### 1. Entendimento do Negócio
- Definição do objetivo
- Contexto do problema
- Métricas de sucesso

### 2. Entendimento dos Dados
- Carregamento e exploração dos dados
- Análise estatística descritiva
- Identificação de padrões

### 3. Preparação dos Dados
- Tratamento de valores ausentes
- Normalização dos dados
- Divisão treino/teste

### 4. Modelagem
- Teste de diferentes algoritmos:
  - K-Nearest Neighbors (KNN)
  - Support Vector Machine (SVM)
  - Random Forest
- Otimização de hiperparâmetros

### 5. Avaliação
- Métricas de desempenho
- Matriz de confusão
- Comparação entre modelos

### 6. Implantação
- Documentação do modelo
- Recomendações para uso

## 🔧 Dependências

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- jupyter (opcional)

## 📝 Resultados Esperados

A análise irá gerar:
- Estatísticas descritivas do dataset
- Distribuição das classes
- Comparação de performance entre modelos
- Relatórios de classificação
- Identificação do melhor modelo

## 🎯 Próximos Passos

1. **Coleta de Dados**: Expandir o dataset com mais amostras
2. **Feature Engineering**: Criar novas características relevantes
3. **Otimização**: Testar mais algoritmos e hiperparâmetros
4. **Produção**: Implementar o modelo em ambiente de produção
5. **Monitoramento**: Acompanhar o desempenho ao longo do tempo

## 📞 Suporte

Se encontrar problemas:
1. Verifique se o ambiente virtual está ativo
2. Execute o `teste_basico.py` primeiro
3. Verifique sua conexão com a internet (para carregar o dataset)
4. Certifique-se de que todas as dependências estão instaladas

## 📚 Referências

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/seeds)
- [CRISP-DM Methodology](https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining)
- [Scikit-learn Documentation](https://scikit-learn.org/) 