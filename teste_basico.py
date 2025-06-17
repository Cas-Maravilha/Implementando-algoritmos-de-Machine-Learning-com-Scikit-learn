#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teste Básico - Análise de Grãos
===============================

Script simples para testar se as bibliotecas estão funcionando.
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 50)
print("TESTE BÁSICO - ANÁLISE DE GRÃOS")
print("=" * 50)

# Testando importações
print("\n✓ Pandas importado com sucesso!")
print(f"Versão do Pandas: {pd.__version__}")

print("\n✓ NumPy importado com sucesso!")
print(f"Versão do NumPy: {np.__version__}")

print("\n✓ Matplotlib importado com sucesso!")
print(f"Versão do Matplotlib: {matplotlib.__version__}")

print("\n✓ Seaborn importado com sucesso!")
print(f"Versão do Seaborn: {sns.__version__}")

# Testando carregamento de dados
print("\n" + "=" * 30)
print("CARREGANDO DATASET")
print("=" * 30)

try:
    # Carregando os dados
    colunas = ['area', 'perimetro', 'compacidade', 'comprimento_kernel', 
               'largura_kernel', 'coef_assimetria', 'comprimento_sulco', 'classe']
    
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt'
    df = pd.read_csv(url, names=colunas, sep=r'\s+')
    
    print(f"✓ Dataset carregado com sucesso!")
    print(f"  - Amostras: {df.shape[0]}")
    print(f"  - Atributos: {df.shape[1]}")
    
    print("\nPrimeiras 5 linhas:")
    print(df.head())
    
    print("\nDistribuição das classes:")
    print(df['classe'].value_counts().sort_index())
    
    print("\nEstatísticas básicas:")
    print(df.describe())
    
    print("\n" + "=" * 30)
    print("TESTE CONCLUÍDO COM SUCESSO!")
    print("=" * 30)
    
except Exception as e:
    print(f"❌ Erro ao carregar dados: {e}")
    print("Verifique sua conexão com a internet.")

print("\nPróximo passo: Executar análise completa com machine learning!") 