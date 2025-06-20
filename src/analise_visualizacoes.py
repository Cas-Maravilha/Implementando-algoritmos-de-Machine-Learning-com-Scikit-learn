# -*- coding: utf-8 -*-
"""
Análise Visual dos Resultados - Classificação de Grãos
======================================================

Este script gera visualizações específicas para complementar a análise
dos resultados da comparação de algoritmos de classificação.

Autor: Assistente IA
Data: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import warnings

# Configurações
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette('husl')
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 12

class AnaliseVisual:
    """
    Classe para gerar visualizações da análise de classificação
    """
    
    def __init__(self):
        self.df = None
        self.X = None
        self.y = None
        self.class_names = ['Kama', 'Rosa', 'Canadian']
        self.feature_names = [
            'area', 'perimetro', 'compacidade', 'comprimento_kernel',
            'largura_kernel', 'coef_assimetria', 'comprimento_sulco'
        ]
        
    def carregar_dados(self):
        """Carrega o dataset"""
        try:
            colunas = self.feature_names + ['classe']
            url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt'
            self.df = pd.read_csv(url, names=colunas, sep=r'\s+')
            
            self.X = self.df.drop('classe', axis=1)
            self.y = self.df['classe']
            
            print("✅ Dados carregados com sucesso!")
            return True
        except Exception as e:
            print(f"❌ Erro ao carregar dados: {str(e)}")
            return False
    
    def analise_exploratoria_completa(self):
        """Análise exploratória completa com visualizações"""
        print("📊 Gerando análise exploratória completa...")
        
        # 1. Distribuição das features por classe
        self.plot_distribuicao_features()
        
        # 2. Matriz de correlação
        self.plot_correlacao()
        
        # 3. Boxplots por classe
        self.plot_boxplots()
        
        # 4. Scatter plots das features mais importantes
        self.plot_scatter_importantes()
        
        # 5. Análise de importância de features
        self.plot_importancia_features()
        
        # 6. Análise de outliers
        self.plot_outliers()
        
        print("✅ Análise exploratória concluída!")
    
    def plot_distribuicao_features(self):
        """Plota distribuição das features por classe"""
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.ravel()
        
        for i, feature in enumerate(self.feature_names):
            for j, class_name in enumerate(self.class_names):
                class_data = self.df[self.df['classe'] == j + 1][feature]
                axes[i].hist(class_data, alpha=0.7, label=class_name, bins=15)
            
            axes[i].set_title(f'Distribuição de {feature}', fontweight='bold')
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Frequência')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        # Remover subplot extra
        axes[-1].remove()
        
        plt.tight_layout()
        plt.suptitle('Distribuição das Features por Classe', fontsize=16, fontweight='bold', y=1.02)
        plt.show()
    
    def plot_correlacao(self):
        """Plota matriz de correlação"""
        plt.figure(figsize=(12, 10))
        
        # Calcular correlação
        corr_matrix = self.df.corr()
        
        # Mascarar diagonal superior
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Plotar heatmap
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        
        plt.title('Matriz de Correlação - Features e Classe', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_boxplots(self):
        """Plota boxplots das features por classe"""
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.ravel()
        
        for i, feature in enumerate(self.feature_names):
            # Criar boxplot
            data_to_plot = [self.df[self.df['classe'] == j + 1][feature] for j in range(3)]
            
            bp = axes[i].boxplot(data_to_plot, labels=self.class_names, patch_artist=True)
            
            # Cores diferentes para cada classe
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            axes[i].set_title(f'Boxplot de {feature}', fontweight='bold')
            axes[i].set_ylabel(feature)
            axes[i].grid(True, alpha=0.3)
        
        # Remover subplot extra
        axes[-1].remove()
        
        plt.tight_layout()
        plt.suptitle('Boxplots das Features por Classe', fontsize=16, fontweight='bold', y=1.02)
        plt.show()
    
    def plot_scatter_importantes(self):
        """Plota scatter plots das features mais importantes"""
        # Features mais importantes baseado na análise anterior
        important_features = ['compacidade', 'comprimento_kernel', 'coef_assimetria']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, feature1 in enumerate(important_features):
            for j, feature2 in enumerate(important_features[i+1:], i+1):
                if j < 3:  # Limitar a 3 plots
                    ax_idx = i
                    
                    for k, class_name in enumerate(self.class_names):
                        class_data = self.df[self.df['classe'] == k + 1]
                        axes[ax_idx].scatter(class_data[feature1], class_data[feature2], 
                                           label=class_name, alpha=0.7, s=50)
                    
                    axes[ax_idx].set_xlabel(feature1)
                    axes[ax_idx].set_ylabel(feature2)
                    axes[ax_idx].set_title(f'{feature1} vs {feature2}', fontweight='bold')
                    axes[ax_idx].legend()
                    axes[ax_idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle('Scatter Plots das Features Mais Importantes', fontsize=16, fontweight='bold', y=1.02)
        plt.show()
    
    def plot_importancia_features(self):
        """Plota importância das features usando Random Forest"""
        # Treinar Random Forest para obter importância das features
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42, stratify=self.y
        )
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        # Obter importância das features
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Plotar
        plt.figure(figsize=(12, 8))
        
        plt.bar(range(len(importances)), importances[indices], 
               color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8'])
        
        plt.xticks(range(len(importances)), [self.feature_names[i] for i in indices], rotation=45)
        plt.xlabel('Features')
        plt.ylabel('Importância')
        plt.title('Importância das Features - Random Forest', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Adicionar valores nas barras
        for i, v in enumerate(importances[indices]):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def plot_outliers(self):
        """Plota análise de outliers"""
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.ravel()
        
        for i, feature in enumerate(self.feature_names):
            # Criar boxplot para detectar outliers
            data_to_plot = [self.df[self.df['classe'] == j + 1][feature] for j in range(3)]
            
            bp = axes[i].boxplot(data_to_plot, labels=self.class_names, patch_artist=True)
            
            # Cores diferentes para cada classe
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            axes[i].set_title(f'Outliers - {feature}', fontweight='bold')
            axes[i].set_ylabel(feature)
            axes[i].grid(True, alpha=0.3)
        
        # Remover subplot extra
        axes[-1].remove()
        
        plt.tight_layout()
        plt.suptitle('Análise de Outliers por Feature e Classe', fontsize=16, fontweight='bold', y=1.02)
        plt.show()
    
    def analise_erros_detalhada(self):
        """Análise detalhada dos erros de classificação"""
        print("🔍 Gerando análise detalhada de erros...")
        
        # Treinar modelo para análise de erros
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42, stratify=self.y
        )
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        # Identificar erros
        errors_mask = y_test != y_pred
        error_indices = X_test.index[errors_mask]
        
        # Plotar erros em scatter plot
        self.plot_erros_scatter(X_test, y_test, y_pred, error_indices, errors_mask)
        
        # Análise de confusão detalhada
        self.plot_confusao_detalhada(y_test, y_pred)
        
        print("✅ Análise de erros concluída!")
    
    def plot_erros_scatter(self, X_test, y_test, y_pred, error_indices, errors_mask):
        """Plota scatter plot destacando os erros"""
        # Features mais importantes
        feature1, feature2 = 'compacidade', 'comprimento_kernel'
        
        plt.figure(figsize=(12, 8))
        
        # Plotar todos os pontos
        for i, class_name in enumerate(self.class_names):
            class_mask = y_test == i + 1
            class_data = X_test[class_mask]
            
            plt.scatter(class_data[feature1], class_data[feature2], 
                       label=f'{class_name} (Correto)', alpha=0.6, s=60)
        
        # Destacar erros
        error_data = X_test.loc[error_indices]
        error_true = y_test.loc[error_indices]
        error_pred = y_pred[errors_mask]
        
        for i, (idx, true_class, pred_class) in enumerate(zip(error_indices, error_true, error_pred)):
            plt.scatter(error_data.loc[idx, feature1], error_data.loc[idx, feature2], 
                       color='red', s=200, marker='x', linewidth=3,
                       label=f'Erro: {self.class_names[true_class-1]} → {self.class_names[pred_class-1]}' if i == 0 else "")
        
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.title('Análise de Erros de Classificação', fontsize=16, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_confusao_detalhada(self, y_test, y_pred):
        """Plota matriz de confusão detalhada"""
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(10, 8))
        
        # Criar heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        
        plt.title('Matriz de Confusão Detalhada - Random Forest', fontsize=16, fontweight='bold')
        plt.xlabel('Predição')
        plt.ylabel('Real')
        
        # Adicionar estatísticas
        total = cm.sum()
        correct = cm.diagonal().sum()
        accuracy = correct / total
        
        plt.text(0.5, -0.15, f'Acurácia: {accuracy:.3f} ({correct}/{total})', 
                ha='center', va='center', transform=plt.gca().transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    def resumo_estatistico(self):
        """Gera resumo estatístico das features por classe"""
        print("📊 Gerando resumo estatístico...")
        
        # Estatísticas por classe
        stats_by_class = {}
        
        for i, class_name in enumerate(self.class_names):
            class_data = self.df[self.df['classe'] == i + 1].drop('classe', axis=1)
            stats_by_class[class_name] = class_data.describe()
        
        # Criar figura com múltiplos subplots
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        for i, (class_name, stats) in enumerate(stats_by_class.items()):
            # Plotar estatísticas
            means = stats.loc['mean']
            stds = stats.loc['std']
            
            x_pos = np.arange(len(self.feature_names))
            
            axes[i].bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, 
                       label=f'Média ± Desvio Padrão', color=['#FF6B6B', '#4ECDC4', '#45B7D1'][i])
            
            axes[i].set_title(f'Estatísticas - Classe {class_name}', fontweight='bold')
            axes[i].set_ylabel('Valor')
            axes[i].set_xticks(x_pos)
            axes[i].set_xticklabels(self.feature_names, rotation=45)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle('Resumo Estatístico por Classe', fontsize=16, fontweight='bold', y=1.02)
        plt.show()
        
        print("✅ Resumo estatístico concluído!")

def main():
    """Função principal"""
    try:
        print("=" * 60)
        print("ANÁLISE VISUAL DOS RESULTADOS - CLASSIFICAÇÃO DE GRÃOS")
        print("=" * 60)
        
        # Criar instância da análise visual
        analise = AnaliseVisual()
        
        # Carregar dados
        if not analise.carregar_dados():
            return False
        
        # Executar análises
        analise.analise_exploratoria_completa()
        analise.analise_erros_detalhada()
        analise.resumo_estatistico()
        
        print("\n" + "=" * 60)
        print("ANÁLISE VISUAL CONCLUÍDA COM SUCESSO!")
        print("=" * 60)
        print("✅ Distribuição das features por classe")
        print("✅ Matriz de correlação")
        print("✅ Boxplots e análise de outliers")
        print("✅ Scatter plots das features importantes")
        print("✅ Importância das features")
        print("✅ Análise detalhada de erros")
        print("✅ Resumo estatístico")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Erro inesperado: {str(e)}")
        return False

if __name__ == "__main__":
    main() 