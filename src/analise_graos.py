#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análise de Classificação de Grãos
=================================

Este script implementa uma análise completa de classificação de grãos 
seguindo a metodologia CRISP-DM.

Autor: Assistente IA
Data: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

# Configurações de visualização
plt.style.use('seaborn')
sns.set_palette('husl')

# Configurações do pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

def main():
    print("=" * 60)
    print("ANÁLISE DE CLASSIFICAÇÃO DE GRÃOS - CRISP-DM")
    print("=" * 60)
    
    # 1. ENTENDIMENTO DO NEGÓCIO
    print("\n1. ENTENDIMENTO DO NEGÓCIO")
    print("-" * 30)
    print("Objetivo: Classificar sementes de trigo em 3 variedades")
    print("Dataset: Seeds Dataset (UCI Machine Learning Repository)")
    print("Classes: Kama (1), Rosa (2), Canadian (3)")
    print("Atributos: 7 características físicas das sementes")
    
    # 2. ENTENDIMENTO DOS DADOS
    print("\n2. ENTENDIMENTO DOS DADOS")
    print("-" * 30)
    
    # Carregando os dados
    print("Carregando dataset...")
    colunas = ['area', 'perimetro', 'compacidade', 'comprimento_kernel', 
               'largura_kernel', 'coef_assimetria', 'comprimento_sulco', 'classe']
    
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt'
    df = pd.read_csv(url, names=colunas, sep=r'\s+')
    
    print(f"Dataset carregado com {df.shape[0]} amostras e {df.shape[1]} atributos")
    print("\nPrimeiras 5 linhas:")
    print(df.head())
    
    print("\nInformações do dataset:")
    print(df.info())
    
    print("\nEstatísticas descritivas:")
    print(df.describe())
    
    print("\nValores nulos por coluna:")
    print(df.isnull().sum())
    
    # Distribuição das classes
    print("\nDistribuição das classes:")
    print(df['classe'].value_counts().sort_index())
    
    # 3. PREPARAÇÃO DOS DADOS
    print("\n3. PREPARAÇÃO DOS DADOS")
    print("-" * 30)
    
    # Separando features e target
    X = df.drop('classe', axis=1)
    y = df['classe']
    
    # Dividindo em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Conjunto de treino: {X_train.shape[0]} amostras")
    print(f"Conjunto de teste: {X_test.shape[0]} amostras")
    
    # Normalizando os dados
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. MODELAGEM
    print("\n4. MODELAGEM")
    print("-" * 30)
    
    # Definindo os modelos
    models = {
        'KNN': KNeighborsClassifier(),
        'SVM': SVC(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42)
    }
    
    # Treinando e avaliando cada modelo
    results = {}
    for name, model in models.items():
        print(f"\nTreinando {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        
        print(f"Acurácia: {accuracy:.3f}")
        print("\nRelatório de Classificação:")
        print(classification_report(y_test, y_pred))
    
    # 5. AVALIAÇÃO
    print("\n5. AVALIAÇÃO")
    print("-" * 30)
    
    # Identificando o melhor modelo
    best_model_name = max(results, key=results.get)
    print(f"Melhor modelo: {best_model_name}")
    print(f"Acurácia: {results[best_model_name]:.3f}")
    
    # Comparação dos modelos
    print("\nComparação dos modelos:")
    for name, accuracy in results.items():
        print(f"{name}: {accuracy:.3f}")
    
    # 6. CONCLUSÕES
    print("\n6. CONCLUSÕES")
    print("-" * 30)
    print("✓ Análise concluída com sucesso!")
    print("✓ Dataset carregado e explorado")
    print("✓ Modelos treinados e avaliados")
    print(f"✓ Melhor modelo: {best_model_name}")
    print("\nPróximos passos:")
    print("- Coletar mais dados para melhorar a robustez")
    print("- Explorar outras características relevantes")
    print("- Implementar o modelo em produção")
    print("- Monitorar o desempenho ao longo do tempo")

if __name__ == "__main__":
    main() 