# -*- coding: utf-8 -*-
"""
Comparação de Algoritmos de Classificação
=========================================

Este script implementa e compara diferentes algoritmos de classificação
seguindo os passos especificados:

1. Separação dos dados (70% treino, 30% teste)
2. Implementação de 5 algoritmos diferentes
3. Treinamento dos modelos
4. Avaliação com múltiplas métricas
5. Comparação de desempenho

Algoritmos implementados:
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Random Forest
- Naive Bayes
- Logistic Regression

Autor: Assistente IA
Data: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import warnings
import time

# Configurações
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette('husl')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Configurações do pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

class ComparacaoAlgoritmos:
    """
    Classe para comparar diferentes algoritmos de classificação
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.models = {}
        self.results = {}
        self.class_names = ['Kama', 'Rosa', 'Canadian']
        
        # Definir nomes das features
        self.feature_names = [
            'area', 'perimetro', 'compacidade', 'comprimento_kernel',
            'largura_kernel', 'coef_assimetria', 'comprimento_sulco'
        ]
    
    def carregar_dados(self):
        """Carrega o dataset de sementes"""
        try:
            print("📊 Carregando dataset...")
            
            colunas = self.feature_names + ['classe']
            url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt'
            self.df = pd.read_csv(url, names=colunas, sep=r'\s+')
            
            print(f"✅ Dataset carregado com {self.df.shape[0]} amostras e {self.df.shape[1]} atributos")
            return True
            
        except Exception as e:
            print(f"❌ Erro ao carregar dados: {str(e)}")
            return False
    
    def explorar_dados(self):
        """Exploração básica dos dados"""
        print("\n" + "="*60)
        print("EXPLORAÇÃO DOS DADOS")
        print("="*60)
        
        print(f"\n📊 Shape do dataset: {self.df.shape}")
        print(f"📋 Colunas: {list(self.df.columns)}")
        
        print("\n📈 Primeiras 5 linhas:")
        print(self.df.head())
        
        print("\n📊 Informações do dataset:")
        print(self.df.info())
        
        print("\n📈 Estatísticas descritivas:")
        print(self.df.describe())
        
        print("\n🔍 Valores nulos por coluna:")
        print(self.df.isnull().sum())
        
        # Distribuição das classes
        print("\n🎯 Distribuição das classes:")
        class_counts = self.df['classe'].value_counts().sort_index()
        for i, count in class_counts.items():
            print(f"  Classe {i} ({self.class_names[i-1]}): {count} amostras ({count/len(self.df)*100:.1f}%)")
        
        # Visualizar distribuição das classes
        plt.figure(figsize=(10, 6))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        plt.pie(class_counts.values, labels=[f'{self.class_names[i-1]}\n({count})' 
                                           for i, count in class_counts.items()], 
               colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('Distribuição das Classes', fontsize=14, fontweight='bold')
        plt.show()
    
    def preparar_dados(self):
        """Prepara os dados para modelagem"""
        print("\n" + "="*60)
        print("PREPARAÇÃO DOS DADOS")
        print("="*60)
        
        # Separando features e target
        self.X = self.df.drop('classe', axis=1)
        self.y = self.df['classe']
        
        print(f"📊 Features (X): {self.X.shape}")
        print(f"🎯 Target (y): {self.y.shape}")
        
        # SEPARAÇÃO DOS DADOS: 70% treino, 30% teste
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=self.random_state, stratify=self.y
        )
        
        print(f"\n📈 Conjunto de treino: {self.X_train.shape[0]} amostras (70%)")
        print(f"📊 Conjunto de teste: {self.X_test.shape[0]} amostras (30%)")
        
        # Verificar distribuição das classes nos conjuntos
        print(f"\n🎯 Distribuição das classes no treino:")
        train_counts = self.y_train.value_counts().sort_index()
        for i, count in train_counts.items():
            print(f"  Classe {i} ({self.class_names[i-1]}): {count} amostras")
        
        print(f"\n🎯 Distribuição das classes no teste:")
        test_counts = self.y_test.value_counts().sort_index()
        for i, count in test_counts.items():
            print(f"  Classe {i} ({self.class_names[i-1]}): {count} amostras")
        
        # Normalização dos dados
        print(f"\n🔧 Normalizando dados...")
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("✅ Dados normalizados com sucesso!")
    
    def definir_algoritmos(self):
        """Define os algoritmos de classificação"""
        print("\n" + "="*60)
        print("DEFININDO ALGORITMOS DE CLASSIFICAÇÃO")
        print("="*60)
        
        # 1. K-Nearest Neighbors (KNN)
        knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
        
        # 2. Support Vector Machine (SVM)
        svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=self.random_state)
        
        # 3. Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        
        # 4. Naive Bayes
        nb = GaussianNB()
        
        # 5. Logistic Regression
        lr = LogisticRegression(random_state=self.random_state, max_iter=1000)
        
        self.models = {
            'K-Nearest Neighbors (KNN)': knn,
            'Support Vector Machine (SVM)': svm,
            'Random Forest': rf,
            'Naive Bayes': nb,
            'Logistic Regression': lr
        }
        
        print("✅ Algoritmos definidos:")
        for name in self.models.keys():
            print(f"  - {name}")
    
    def treinar_modelos(self):
        """Treina todos os modelos"""
        print("\n" + "="*60)
        print("TREINAMENTO DOS MODELOS")
        print("="*60)
        
        for name, model in self.models.items():
            print(f"\n🔧 Treinando {name}...")
            start_time = time.time()
            
            # Alguns modelos precisam de dados normalizados, outros não
            if name in ['K-Nearest Neighbors (KNN)', 'Support Vector Machine (SVM)', 'Logistic Regression']:
                model.fit(self.X_train_scaled, self.y_train)
            else:
                model.fit(self.X_train, self.y_train)
            
            training_time = time.time() - start_time
            print(f"  ⏱️ Tempo de treinamento: {training_time:.3f} segundos")
            print(f"  ✅ Modelo treinado com sucesso!")
    
    def avaliar_modelos(self):
        """Avalia todos os modelos no conjunto de teste"""
        print("\n" + "="*60)
        print("AVALIAÇÃO DOS MODELOS")
        print("="*60)
        
        for name, model in self.models.items():
            print(f"\n📊 Avaliando {name}...")
            
            # Fazer predições
            if name in ['K-Nearest Neighbors (KNN)', 'Support Vector Machine (SVM)', 'Logistic Regression']:
                y_pred = model.predict(self.X_test_scaled)
            else:
                y_pred = model.predict(self.X_test)
            
            # Calcular métricas
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted')
            recall = recall_score(self.y_test, y_pred, average='weighted')
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            
            # Armazenar resultados
            self.results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'y_pred': y_pred
            }
            
            # Imprimir resultados
            print(f"  📈 Acurácia: {accuracy:.3f}")
            print(f"  📊 Precisão: {precision:.3f}")
            print(f"  🔄 Recall: {recall:.3f}")
            print(f"  ⚖️ F1-Score: {f1:.3f}")
    
    def criar_matrizes_confusao(self):
        """Cria matrizes de confusão para todos os modelos"""
        print("\n" + "="*60)
        print("MATRIZES DE CONFUSÃO")
        print("="*60)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for i, (name, model) in enumerate(self.models.items()):
            y_pred = self.results[name]['y_pred']
            cm = confusion_matrix(self.y_test, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=self.class_names, yticklabels=self.class_names, ax=axes[i])
            axes[i].set_title(f'Matriz de Confusão - {name}', fontweight='bold')
            axes[i].set_xlabel('Predição')
            axes[i].set_ylabel('Real')
        
        # Remover subplot extra
        axes[-1].remove()
        
        plt.tight_layout()
        plt.show()
    
    def comparar_desempenho(self):
        """Compara o desempenho dos diferentes algoritmos"""
        print("\n" + "="*60)
        print("COMPARAÇÃO DE DESEMPENHO")
        print("="*60)
        
        # Criar DataFrame com resultados
        results_df = pd.DataFrame(self.results).T
        print("\n📊 Tabela de Resultados:")
        print(results_df.round(3))
        
        # Visualizar comparação
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            values = [self.results[name][metric] for name in self.models.keys()]
            axes[i].bar(self.models.keys(), values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
            axes[i].set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')
            axes[i].set_ylabel('Score')
            axes[i].tick_params(axis='x', rotation=45)
            
            # Adicionar valores nas barras
            for j, v in enumerate(values):
                axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        # Identificar melhor modelo
        best_model = max(self.results.keys(), key=lambda x: self.results[x]['f1_score'])
        print(f"\n🏆 MELHOR MODELO: {best_model}")
        print(f"📈 F1-Score: {self.results[best_model]['f1_score']:.3f}")
        print(f"🎯 Acurácia: {self.results[best_model]['accuracy']:.3f}")
    
    def relatorios_detalhados(self):
        """Gera relatórios detalhados de classificação"""
        print("\n" + "="*60)
        print("RELATÓRIOS DETALHADOS DE CLASSIFICAÇÃO")
        print("="*60)
        
        for name, model in self.models.items():
            y_pred = self.results[name]['y_pred']
            
            print(f"\n📋 Relatório de Classificação - {name}")
            print("=" * 50)
            print(classification_report(self.y_test, y_pred, target_names=self.class_names))
    
    def analise_erros(self):
        """Analisa os erros de classificação"""
        print("\n" + "="*60)
        print("ANÁLISE DE ERROS DE CLASSIFICAÇÃO")
        print("="*60)
        
        # Identificar amostras mal classificadas
        for name, model in self.models.items():
            y_pred = self.results[name]['y_pred']
            errors = self.y_test != y_pred
            
            if errors.sum() > 0:
                print(f"\n❌ {name}: {errors.sum()} erros de classificação")
                print("Amostras mal classificadas:")
                
                error_indices = self.X_test.index[errors]
                for idx in error_indices[:5]:  # Mostrar apenas os primeiros 5 erros
                    true_class = self.y_test.loc[idx]
                    pred_class = y_pred[errors][list(error_indices).index(idx)]
                    print(f"  Amostra {idx}: Real={true_class}({self.class_names[true_class-1]}) -> Pred={pred_class}({self.class_names[pred_class-1]})")
            else:
                print(f"\n✅ {name}: 0 erros de classificação!")
    
    def executar_analise_completa(self):
        """Executa a análise completa de comparação de algoritmos"""
        print("=" * 60)
        print("COMPARAÇÃO DE ALGORITMOS DE CLASSIFICAÇÃO")
        print("=" * 60)
        
        # 1. Carregar dados
        if not self.carregar_dados():
            return False
        
        # 2. Explorar dados
        self.explorar_dados()
        
        # 3. Preparar dados (70% treino, 30% teste)
        self.preparar_dados()
        
        # 4. Definir algoritmos
        self.definir_algoritmos()
        
        # 5. Treinar modelos
        self.treinar_modelos()
        
        # 6. Avaliar modelos
        self.avaliar_modelos()
        
        # 7. Criar matrizes de confusão
        self.criar_matrizes_confusao()
        
        # 8. Comparar desempenho
        self.comparar_desempenho()
        
        # 9. Relatórios detalhados
        self.relatorios_detalhados()
        
        # 10. Análise de erros
        self.analise_erros()
        
        print(f"\n{'='*60}")
        print("ANÁLISE CONCLUÍDA COM SUCESSO!")
        print(f"{'='*60}")
        print("✅ Dados carregados e explorados")
        print("✅ Dados separados (70% treino, 30% teste)")
        print("✅ 5 algoritmos implementados e treinados")
        print("✅ Modelos avaliados com múltiplas métricas")
        print("✅ Matrizes de confusão geradas")
        print("✅ Comparação de desempenho realizada")
        print("✅ Relatórios detalhados criados")
        print("✅ Análise de erros concluída")
        
        return True

def main():
    """Função principal"""
    try:
        # Criar instância da comparação
        comparacao = ComparacaoAlgoritmos(random_state=42)
        
        # Executar análise completa
        success = comparacao.executar_analise_completa()
        
        if success:
            print("\n🎉 Comparação de algoritmos concluída com sucesso!")
        else:
            print("\n❌ Erro na execução da análise.")
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Análise interrompida pelo usuário.")
    except Exception as e:
        print(f"\n❌ Erro inesperado: {str(e)}")

if __name__ == "__main__":
    main() 