#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análise Aprimorada de Classificação de Grãos
============================================

Este script implementa uma análise completa de classificação de grãos 
com aprimoramentos seguindo a metodologia CRISP-DM.

Melhorias implementadas:
- Visualizações avançadas
- Otimização de hiperparâmetros com GridSearchCV
- Validação cruzada robusta
- Análise de importância de features
- Tratamento de exceções
- Uso de pipelines do scikit-learn

Autor: Assistente IA
Data: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, 
    StratifiedKFold, learning_curve, validation_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score, roc_curve
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
import logging
from datetime import datetime
import os

# Configurações
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configurações de visualização
plt.style.use('seaborn-v0_8')
sns.set_palette('husl')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Configurações do pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

class AnaliseGraosAprimorada:
    """
    Classe para análise aprimorada de classificação de grãos
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
        self.models = {}
        self.results = {}
        self.best_model = None
        self.feature_names = [
            'area', 'perimetro', 'compacidade', 'comprimento_kernel',
            'largura_kernel', 'coef_assimetria', 'comprimento_sulco'
        ]
        self.class_names = ['Kama', 'Rosa', 'Canadian']
        
        # Criar diretório para salvar gráficos
        self.output_dir = 'output_graficos'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def carregar_dados(self):
        """Carrega e prepara os dados com tratamento de exceções"""
        try:
            logger.info("Carregando dataset...")
            colunas = self.feature_names + ['classe']
            
            url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt'
            self.df = pd.read_csv(url, names=colunas, sep=r'\s+')
            
            logger.info(f"Dataset carregado com {self.df.shape[0]} amostras e {self.df.shape[1]} atributos")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao carregar dados: {str(e)}")
            return False
    
    def explorar_dados(self):
        """Exploração detalhada dos dados com visualizações"""
        try:
            logger.info("Iniciando exploração dos dados...")
            
            # Informações básicas
            print("\n" + "="*60)
            print("EXPLORAÇÃO DOS DADOS")
            print("="*60)
            
            print(f"\nDataset shape: {self.df.shape}")
            print(f"Colunas: {list(self.df.columns)}")
            
            print("\nPrimeiras 5 linhas:")
            print(self.df.head())
            
            print("\nInformações do dataset:")
            print(self.df.info())
            
            print("\nEstatísticas descritivas:")
            print(self.df.describe())
            
            print("\nValores nulos por coluna:")
            print(self.df.isnull().sum())
            
            # Distribuição das classes
            print("\nDistribuição das classes:")
            class_counts = self.df['classe'].value_counts().sort_index()
            for i, count in class_counts.items():
                print(f"Classe {i} ({self.class_names[i-1]}): {count} amostras ({count/len(self.df)*100:.1f}%)")
            
            # Criar visualizações
            self._criar_visualizacoes_exploracao()
            
        except Exception as e:
            logger.error(f"Erro na exploração dos dados: {str(e)}")
    
    def _criar_visualizacoes_exploracao(self):
        """Cria visualizações para exploração dos dados"""
        try:
            # 1. Distribuição das classes
            plt.figure(figsize=(15, 10))
            
            plt.subplot(2, 3, 1)
            class_counts = self.df['classe'].value_counts().sort_index()
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            plt.pie(class_counts.values, labels=[f'{self.class_names[i-1]}\n({count})' 
                                               for i, count in class_counts.items()], 
                   colors=colors, autopct='%1.1f%%', startangle=90)
            plt.title('Distribuição das Classes', fontsize=14, fontweight='bold')
            
            # 2. Histogramas das features
            for i, feature in enumerate(self.feature_names):
                plt.subplot(2, 3, i+2)
                for classe in [1, 2, 3]:
                    data = self.df[self.df['classe'] == classe][feature]
                    plt.hist(data, alpha=0.7, label=self.class_names[classe-1], 
                            bins=15, color=colors[classe-1])
                plt.xlabel(feature.replace('_', ' ').title())
                plt.ylabel('Frequência')
                plt.title(f'Distribuição de {feature.replace("_", " ").title()}')
                plt.legend()
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/distribuicao_classes_features.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # 3. Matriz de correlação
            plt.figure(figsize=(10, 8))
            correlation_matrix = self.df.corr()
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                       center=0, square=True, linewidths=0.5)
            plt.title('Matriz de Correlação', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/matriz_correlacao.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # 4. Boxplots das features por classe
            plt.figure(figsize=(15, 10))
            for i, feature in enumerate(self.feature_names):
                plt.subplot(2, 4, i+1)
                data_to_plot = [self.df[self.df['classe'] == classe][feature] for classe in [1, 2, 3]]
                plt.boxplot(data_to_plot, labels=self.class_names, patch_artist=True)
                plt.title(f'{feature.replace("_", " ").title()}')
                plt.ylabel('Valor')
                plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/boxplots_features.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # 5. Pairplot para visualizar relações entre features
            plt.figure(figsize=(12, 8))
            pair_data = self.df.copy()
            pair_data['classe_nome'] = pair_data['classe'].map({1: 'Kama', 2: 'Rosa', 3: 'Canadian'})
            
            # Selecionar apenas algumas features para o pairplot
            selected_features = ['area', 'perimetro', 'compacidade', 'comprimento_kernel']
            sns.pairplot(pair_data[selected_features + ['classe_nome']], 
                        hue='classe_nome', diag_kind='kde')
            plt.savefig(f'{self.output_dir}/pairplot_features.png', dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            logger.error(f"Erro ao criar visualizações: {str(e)}")
    
    def preparar_dados(self):
        """Prepara os dados para modelagem"""
        try:
            logger.info("Preparando dados para modelagem...")
            
            # Separando features e target
            self.X = self.df.drop('classe', axis=1)
            self.y = self.df['classe']
            
            # Dividindo em conjuntos de treino e teste
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=self.random_state, stratify=self.y
            )
            
            logger.info(f"Conjunto de treino: {self.X_train.shape[0]} amostras")
            logger.info(f"Conjunto de teste: {self.X_test.shape[0]} amostras")
            
            return True
            
        except Exception as e:
            logger.error(f"Erro na preparação dos dados: {str(e)}")
            return False
    
    def criar_pipelines(self):
        """Cria pipelines para os diferentes modelos"""
        try:
            logger.info("Criando pipelines dos modelos...")
            
            # Pipeline para KNN
            knn_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', KNeighborsClassifier())
            ])
            
            # Pipeline para SVM
            svm_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', SVC(random_state=self.random_state, probability=True))
            ])
            
            # Pipeline para Random Forest
            rf_pipeline = Pipeline([
                ('classifier', RandomForestClassifier(random_state=self.random_state))
            ])
            
            # Pipeline para Logistic Regression
            lr_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(random_state=self.random_state, max_iter=1000))
            ])
            
            self.models = {
                'KNN': knn_pipeline,
                'SVM': svm_pipeline,
                'Random Forest': rf_pipeline,
                'Logistic Regression': lr_pipeline
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Erro ao criar pipelines: {str(e)}")
            return False
    
    def otimizar_hiperparametros(self):
        """Otimiza hiperparâmetros usando GridSearchCV"""
        try:
            logger.info("Iniciando otimização de hiperparâmetros...")
            
            # Definir parâmetros para cada modelo
            param_grids = {
                'KNN': {
                    'classifier__n_neighbors': [3, 5, 7, 9, 11],
                    'classifier__weights': ['uniform', 'distance'],
                    'classifier__metric': ['euclidean', 'manhattan']
                },
                'SVM': {
                    'classifier__C': [0.1, 1, 10, 100],
                    'classifier__kernel': ['rbf', 'linear'],
                    'classifier__gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
                },
                'Random Forest': {
                    'classifier__n_estimators': [50, 100, 200],
                    'classifier__max_depth': [None, 10, 20, 30],
                    'classifier__min_samples_split': [2, 5, 10],
                    'classifier__min_samples_leaf': [1, 2, 4]
                },
                'Logistic Regression': {
                    'classifier__C': [0.1, 1, 10, 100],
                    'classifier__penalty': ['l1', 'l2'],
                    'classifier__solver': ['liblinear', 'saga']
                }
            }
            
            # Configurar validação cruzada
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            
            # Otimizar cada modelo
            for name, model in self.models.items():
                logger.info(f"Otimizando {name}...")
                
                grid_search = GridSearchCV(
                    model, param_grids[name], cv=cv, scoring='accuracy',
                    n_jobs=-1, verbose=1
                )
                
                grid_search.fit(self.X_train, self.y_train)
                
                # Atualizar modelo com os melhores parâmetros
                self.models[name] = grid_search.best_estimator_
                
                logger.info(f"Melhores parâmetros para {name}: {grid_search.best_params_}")
                logger.info(f"Melhor score CV: {grid_search.best_score_:.3f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Erro na otimização de hiperparâmetros: {str(e)}")
            return False
    
    def avaliar_modelos(self):
        """Avalia todos os modelos com múltiplas métricas"""
        try:
            logger.info("Avaliando modelos...")
            
            # Configurar validação cruzada
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            
            for name, model in self.models.items():
                logger.info(f"Avaliando {name}...")
                
                # Validação cruzada
                cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring='accuracy')
                
                # Predições no conjunto de teste
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test) if hasattr(model, 'predict_proba') else None
                
                # Métricas
                accuracy = accuracy_score(self.y_test, y_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(self.y_test, y_pred, average='weighted')
                
                # AUC-ROC (se disponível)
                auc_score = None
                if y_pred_proba is not None:
                    try:
                        auc_score = roc_auc_score(self.y_test, y_pred_proba, multi_class='ovr')
                    except:
                        pass
                
                # Armazenar resultados
                self.results[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'auc_score': auc_score,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba
                }
                
                # Imprimir resultados
                print(f"\n{name}:")
                print(f"  Acurácia: {accuracy:.3f}")
                print(f"  Precisão: {precision:.3f}")
                print(f"  Recall: {recall:.3f}")
                print(f"  F1-Score: {f1:.3f}")
                if auc_score:
                    print(f"  AUC-ROC: {auc_score:.3f}")
                print(f"  CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            
            return True
            
        except Exception as e:
            logger.error(f"Erro na avaliação dos modelos: {str(e)}")
            return False
    
    def analisar_importancia_features(self):
        """Analisa a importância das features"""
        try:
            logger.info("Analisando importância das features...")
            
            # Para Random Forest
            if 'Random Forest' in self.models:
                rf_model = self.models['Random Forest']
                feature_importance = rf_model.named_steps['classifier'].feature_importances_
                
                # Criar DataFrame com importância das features
                importance_df = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': feature_importance
                }).sort_values('importance', ascending=False)
                
                print("\nImportância das Features (Random Forest):")
                print(importance_df)
                
                # Visualizar importância das features
                plt.figure(figsize=(10, 6))
                sns.barplot(data=importance_df, x='importance', y='feature')
                plt.title('Importância das Features - Random Forest', fontsize=14, fontweight='bold')
                plt.xlabel('Importância')
                plt.ylabel('Feature')
                plt.tight_layout()
                plt.savefig(f'{self.output_dir}/importancia_features.png', dpi=300, bbox_inches='tight')
                plt.show()
            
            # Para outros modelos que suportam coeficientes
            for name, model in self.models.items():
                if name != 'Random Forest' and hasattr(model.named_steps['classifier'], 'coef_'):
                    coef = model.named_steps['classifier'].coef_
                    if coef.ndim > 1:
                        coef = np.mean(np.abs(coef), axis=0)
                    else:
                        coef = np.abs(coef)
                    
                    importance_df = pd.DataFrame({
                        'feature': self.feature_names,
                        'importance': coef
                    }).sort_values('importance', ascending=False)
                    
                    print(f"\nImportância das Features ({name}):")
                    print(importance_df)
            
            return True
            
        except Exception as e:
            logger.error(f"Erro na análise de importância das features: {str(e)}")
            return False
    
    def criar_visualizacoes_resultados(self):
        """Cria visualizações dos resultados"""
        try:
            logger.info("Criando visualizações dos resultados...")
            
            # 1. Comparação de performance dos modelos
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.ravel()
            
            for i, metric in enumerate(metrics):
                values = [self.results[name][metric] for name in self.models.keys()]
                axes[i].bar(self.models.keys(), values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
                axes[i].set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')
                axes[i].set_ylabel('Score')
                axes[i].tick_params(axis='x', rotation=45)
                
                # Adicionar valores nas barras
                for j, v in enumerate(values):
                    axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/comparacao_modelos.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # 2. Matrizes de confusão
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.ravel()
            
            for i, (name, model) in enumerate(self.models.items()):
                y_pred = self.results[name]['y_pred']
                cm = confusion_matrix(self.y_test, y_pred)
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=self.class_names, yticklabels=self.class_names, ax=axes[i])
                axes[i].set_title(f'Matriz de Confusão - {name}', fontweight='bold')
                axes[i].set_xlabel('Predição')
                axes[i].set_ylabel('Real')
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/matrizes_confusao.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # 3. Curvas de aprendizado
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.ravel()
            
            for i, (name, model) in enumerate(self.models.items()):
                train_sizes, train_scores, val_scores = learning_curve(
                    model, self.X_train, self.y_train, cv=5, 
                    train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy'
                )
                
                train_mean = np.mean(train_scores, axis=1)
                train_std = np.std(train_scores, axis=1)
                val_mean = np.mean(val_scores, axis=1)
                val_std = np.std(val_scores, axis=1)
                
                axes[i].plot(train_sizes, train_mean, label='Treino', color='blue')
                axes[i].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
                axes[i].plot(train_sizes, val_mean, label='Validação', color='red')
                axes[i].fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
                axes[i].set_title(f'Curva de Aprendizado - {name}', fontweight='bold')
                axes[i].set_xlabel('Tamanho do Conjunto de Treino')
                axes[i].set_ylabel('Score')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/curvas_aprendizado.png', dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            logger.error(f"Erro ao criar visualizações dos resultados: {str(e)}")
    
    def identificar_melhor_modelo(self):
        """Identifica o melhor modelo baseado em múltiplas métricas"""
        try:
            logger.info("Identificando melhor modelo...")
            
            # Calcular score composto (média ponderada das métricas)
            for name in self.results:
                results = self.results[name]
                composite_score = (
                    results['accuracy'] * 0.3 +
                    results['precision'] * 0.25 +
                    results['recall'] * 0.25 +
                    results['f1_score'] * 0.2
                )
                self.results[name]['composite_score'] = composite_score
            
            # Encontrar melhor modelo
            best_model_name = max(self.results.keys(), 
                                key=lambda x: self.results[x]['composite_score'])
            self.best_model = self.models[best_model_name]
            
            print(f"\n{'='*60}")
            print("RESULTADOS FINAIS")
            print(f"{'='*60}")
            
            # Tabela de resultados
            results_df = pd.DataFrame(self.results).T
            print("\nComparação dos Modelos:")
            print(results_df[['accuracy', 'precision', 'recall', 'f1_score', 'cv_mean', 'composite_score']].round(3))
            
            print(f"\n🎯 MELHOR MODELO: {best_model_name}")
            print(f"Score Composto: {self.results[best_model_name]['composite_score']:.3f}")
            print(f"Acurácia: {self.results[best_model_name]['accuracy']:.3f}")
            print(f"F1-Score: {self.results[best_model_name]['f1_score']:.3f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Erro ao identificar melhor modelo: {str(e)}")
            return False
    
    def executar_analise_completa(self):
        """Executa a análise completa"""
        try:
            print("=" * 60)
            print("ANÁLISE APRIMORADA DE CLASSIFICAÇÃO DE GRÃOS")
            print("=" * 60)
            
            # 1. Carregar dados
            if not self.carregar_dados():
                return False
            
            # 2. Explorar dados
            self.explorar_dados()
            
            # 3. Preparar dados
            if not self.preparar_dados():
                return False
            
            # 4. Criar pipelines
            if not self.criar_pipelines():
                return False
            
            # 5. Otimizar hiperparâmetros
            if not self.otimizar_hiperparametros():
                return False
            
            # 6. Avaliar modelos
            if not self.avaliar_modelos():
                return False
            
            # 7. Analisar importância das features
            self.analisar_importancia_features()
            
            # 8. Criar visualizações dos resultados
            self.criar_visualizacoes_resultados()
            
            # 9. Identificar melhor modelo
            self.identificar_melhor_modelo()
            
            print(f"\n{'='*60}")
            print("ANÁLISE CONCLUÍDA COM SUCESSO!")
            print(f"{'='*60}")
            print("✓ Dados carregados e explorados")
            print("✓ Pipelines criados e otimizados")
            print("✓ Modelos treinados e avaliados")
            print("✓ Visualizações geradas")
            print("✓ Importância das features analisada")
            print(f"✓ Melhor modelo identificado")
            print(f"\n📁 Gráficos salvos em: {self.output_dir}/")
            
            return True
            
        except Exception as e:
            logger.error(f"Erro na execução da análise: {str(e)}")
            return False

def main():
    """Função principal"""
    try:
        # Criar instância da análise
        analise = AnaliseGraosAprimorada(random_state=42)
        
        # Executar análise completa
        success = analise.executar_analise_completa()
        
        if success:
            print("\n🎉 Análise concluída com sucesso!")
        else:
            print("\n❌ Erro na execução da análise.")
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Análise interrompida pelo usuário.")
    except Exception as e:
        print(f"\n❌ Erro inesperado: {str(e)}")

if __name__ == "__main__":
    main()
