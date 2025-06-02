"""
Análise de Letras de Músicas Populares (1959-2023)
Detecção de Conteúdo Inapropriado: Racismo, Homofobia, Discurso de Ódio e Preconceito

Este módulo implementa algoritmos de machine learning para identificar
conteúdo potencialmente ofensivo em letras de música.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# NLP Libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from textblob import TextBlob
import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Utilities
import re
import warnings
import os
from datetime import datetime
import json

warnings.filterwarnings('ignore')

class MusicLyricsAnalyzer:
    """
    Classe principal para análise de letras de música e detecção de conteúdo inapropriado
    """
    
    def __init__(self):
        """Inicializa o analisador com configurações padrão"""
        self.setup_nltk()
        self.setup_models()
        self.offensive_keywords = self.load_offensive_keywords()
        self.vectorizer = None
        self.models = {}
        self.results = {}
        
    def setup_nltk(self):
        """Baixa recursos necessários do NLTK"""
        nltk_downloads = [
            'punkt', 'stopwords', 'wordnet', 'vader_lexicon', 
            'omw-1.4', 'averaged_perceptron_tagger'
        ]
        
        for resource in nltk_downloads:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                print(f"Baixando {resource}...")
                nltk.download(resource, quiet=True)
    
    def setup_models(self):
        """Inicializa modelos de transformers para análise de sentimento"""
        try:
            # Modelo para detecção de discurso de ódio
            self.hate_speech_classifier = pipeline(
                "text-classification",
                model="unitary/toxic-bert"
            )
            
            # Modelo para análise de sentimento
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest"
            )
            
        except Exception as e:
            print(f"Erro ao carregar modelos de transformers: {e}")
            self.hate_speech_classifier = None
            self.sentiment_analyzer = None
    
    def load_offensive_keywords(self):
        """
        Carrega palavras-chave relacionadas a conteúdo ofensivo
        Em um projeto real, isso seria carregado de um arquivo ou base de dados
        """
        return {
            'racism': [
                'racist', 'racial', 'prejudice', 'discrimination', 'stereotype',
                'bigot', 'supremacy', 'segregation', 'apartheid'
            ],
            'homophobia': [
                'homophobic', 'gay', 'lesbian', 'queer', 'transgender', 'lgbtq',
                'heterosexual', 'orientation', 'pride', 'closet'
            ],
            'hate_speech': [
                'hate', 'violence', 'threat', 'kill', 'death', 'murder',
                'assault', 'attack', 'destroy', 'eliminate'
            ],
            'general_offensive': [
                'stupid', 'idiot', 'moron', 'loser', 'worthless', 'pathetic',
                'disgusting', 'revolting', 'repulsive', 'abhorrent'
            ]
        }
    
    def preprocess_text(self, text):
        """
        Pré-processa o texto para análise
        
        Args:
            text (str): Texto original
            
        Returns:
            str: Texto processado
        """
        if not isinstance(text, str):
            return ""
        
        # Converter para minúsculas
        text = text.lower()
        
        # Remover caracteres especiais e números
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remover espaços extras
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenização
        tokens = word_tokenize(text)
        
        # Remover stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        
        # Lemmatização
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(tokens)
    
    def extract_features(self, texts):
        """
        Extrai características dos textos para machine learning
        
        Args:
            texts (list): Lista de textos
            
        Returns:
            dict: Dicionário com diferentes tipos de features
        """
        features = {}
        
        # TF-IDF Features
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
            tfidf_features = self.vectorizer.fit_transform(texts)
        else:
            tfidf_features = self.vectorizer.transform(texts)
        
        features['tfidf'] = tfidf_features
        
        # Features baseadas em palavras-chave ofensivas
        keyword_features = []
        for text in texts:
            text_lower = text.lower()
            feature_vector = []
            
            for category, keywords in self.offensive_keywords.items():
                count = sum(1 for keyword in keywords if keyword in text_lower)
                feature_vector.append(count)
            
            keyword_features.append(feature_vector)
        
        features['keywords'] = np.array(keyword_features)
        
        # Features de sentimento usando TextBlob
        sentiment_features = []
        for text in texts:
            blob = TextBlob(text)
            sentiment_features.append([
                blob.sentiment.polarity,
                blob.sentiment.subjectivity
            ])
        
        features['sentiment'] = np.array(sentiment_features)
        
        return features
    
    def analyze_with_transformers(self, texts, batch_size=16):
        """
        Analisa textos usando modelos de transformers
        
        Args:
            texts (list): Lista de textos
            batch_size (int): Tamanho do batch para processamento
            
        Returns:
            dict: Resultados da análise
        """
        results = {
            'hate_speech': [],
            'sentiment': []
        }
        
        # Processar em batches para evitar problemas de memória
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            # Análise de discurso de ódio
            if self.hate_speech_classifier:
                try:
                    hate_results = self.hate_speech_classifier(batch)
                    results['hate_speech'].extend(hate_results)
                except Exception as e:
                    print(f"Erro na análise de discurso de ódio: {e}")
                    results['hate_speech'].extend([{'label': 'UNKNOWN', 'score': 0.0}] * len(batch))
            
            # Análise de sentimento
            if self.sentiment_analyzer:
                try:
                    sentiment_results = self.sentiment_analyzer(batch)
                    results['sentiment'].extend(sentiment_results)
                except Exception as e:
                    print(f"Erro na análise de sentimento: {e}")
                    results['sentiment'].extend([{'label': 'UNKNOWN', 'score': 0.0}] * len(batch))
        
        return results
    
    def train_models(self, X_train, y_train):
        """
        Treina múltiplos modelos de machine learning
        
        Args:
            X_train: Features de treino
            y_train: Labels de treino
        """
        # Definir modelos para testar (removendo MultinomialNB para evitar problema com valores negativos)
        models_to_train = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'svm': SVC(random_state=42, probability=True),
            'random_forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'gradient_boosting': GradientBoostingClassifier(random_state=42)
        }
        
        print("Treinando modelos...")
        
        for name, model in models_to_train.items():
            print(f"Treinando {name}...")
            
            # Criar pipeline com escalonamento para modelos que precisam
            if name in ['logistic_regression', 'svm']:
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', model)
                ])
            else:
                pipeline = Pipeline([
                    ('classifier', model)
                ])
            
            # Treinar modelo
            pipeline.fit(X_train, y_train)
            
            # Validação cruzada
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
            
            self.models[name] = {
                'model': pipeline,
                'cv_scores': cv_scores,
                'mean_cv_score': cv_scores.mean(),
                'std_cv_score': cv_scores.std()
            }
            
            print(f"{name} - CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    def evaluate_models(self, X_test, y_test):
        """
        Avalia os modelos treinados no conjunto de teste
        
        Args:
            X_test: Features de teste
            y_test: Labels de teste
        """
        evaluation_results = {}
        
        for name, model_info in self.models.items():
            model = model_info['model']
            
            # Predições
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Métricas
            accuracy = accuracy_score(y_test, y_pred)
            
            evaluation_results[name] = {
                'accuracy': accuracy,
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
            }
            
            if y_pred_proba is not None:
                evaluation_results[name]['auc_roc'] = roc_auc_score(y_test, y_pred_proba)
            
            print(f"\n{name.upper()} - Accuracy: {accuracy:.4f}")
            print(classification_report(y_test, y_pred))
        
        self.results['evaluation'] = evaluation_results
        return evaluation_results
    
    def visualize_results(self, data):
        """
        Cria visualizações dos resultados da análise
        
        Args:
            data (pd.DataFrame): DataFrame com os dados analisados
        """
        # Configurar style
        plt.style.use('seaborn-v0_8')
        
        # 1. Distribuição de conteúdo ofensivo por década
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Subplot 1: Distribuição temporal
        if 'year' in data.columns and 'is_offensive' in data.columns:
            data['decade'] = (data['year'] // 10) * 10
            decade_stats = data.groupby('decade')['is_offensive'].agg(['count', 'sum', 'mean']).reset_index()
            
            axes[0, 0].bar(decade_stats['decade'], decade_stats['mean'] * 100)
            axes[0, 0].set_title('Porcentagem de Conteúdo Ofensivo por Década')
            axes[0, 0].set_xlabel('Década')
            axes[0, 0].set_ylabel('% Conteúdo Ofensivo')
        
        # Subplot 2: Tipos de conteúdo ofensivo
        if 'offense_type' in data.columns:
            offense_counts = data['offense_type'].value_counts()
            axes[0, 1].pie(offense_counts.values, labels=offense_counts.index, autopct='%1.1f%%')
            axes[0, 1].set_title('Distribuição por Tipo de Ofensa')
        
        # Subplot 3: Sentimento vs Conteúdo Ofensivo
        if 'sentiment_score' in data.columns and 'is_offensive' in data.columns:
            offensive_sentiment = data[data['is_offensive'] == 1]['sentiment_score']
            non_offensive_sentiment = data[data['is_offensive'] == 0]['sentiment_score']
            
            axes[1, 0].hist([non_offensive_sentiment, offensive_sentiment], 
                           bins=30, alpha=0.7, label=['Não Ofensivo', 'Ofensivo'])
            axes[1, 0].set_title('Distribuição de Sentimento')
            axes[1, 0].set_xlabel('Score de Sentimento')
            axes[1, 0].set_ylabel('Frequência')
            axes[1, 0].legend()
        
        # Subplot 4: Performance dos modelos
        if self.results and 'evaluation' in self.results:
            model_names = list(self.results['evaluation'].keys())
            accuracies = [self.results['evaluation'][name]['accuracy'] for name in model_names]
            
            axes[1, 1].bar(model_names, accuracies)
            axes[1, 1].set_title('Accuracy dos Modelos')
            axes[1, 1].set_xlabel('Modelo')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('analysis_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Visualização interativa com Plotly
        self.create_interactive_visualizations(data)
    
    def create_interactive_visualizations(self, data):
        """
        Cria visualizações interativas com Plotly
        
        Args:
            data (pd.DataFrame): DataFrame com os dados analisados
        """
        # Timeline de conteúdo ofensivo
        if 'year' in data.columns and 'is_offensive' in data.columns:
            yearly_stats = data.groupby('year')['is_offensive'].agg(['count', 'sum', 'mean']).reset_index()
            yearly_stats['percentage'] = yearly_stats['mean'] * 100
            
            fig = px.line(yearly_stats, x='year', y='percentage', 
                         title='Evolução Temporal do Conteúdo Ofensivo em Músicas')
            fig.update_layout(xaxis_title='Ano', yaxis_title='% Conteúdo Ofensivo')
            fig.write_html('timeline_offensive_content.html')
            fig.show()
        
        # Matriz de confusão interativa
        if self.results and 'evaluation' in self.results:
            best_model = max(self.results['evaluation'].items(), 
                           key=lambda x: x[1]['accuracy'])
            
            cm = np.array(best_model[1]['confusion_matrix'])
            
            fig = px.imshow(cm, text_auto=True, aspect="auto",
                           title=f'Matriz de Confusão - {best_model[0]}')
            fig.update_layout(xaxis_title='Predito', yaxis_title='Real')
            fig.write_html('confusion_matrix.html')
            fig.show()
    
    def generate_report(self, data, output_file='analysis_report.json'):
        """
        Gera relatório detalhado da análise
        
        Args:
            data (pd.DataFrame): DataFrame com os dados analisados
            output_file (str): Nome do arquivo de saída
        """
        report = {
            'metadata': {
                'analysis_date': datetime.now().isoformat(),
                'total_songs': len(data),
                'analysis_period': f"{data['year'].min()}-{data['year'].max()}" if 'year' in data.columns else 'Unknown'
            },
            'summary_statistics': {},
            'model_performance': self.results.get('evaluation', {}),
            'recommendations': []
        }
        
        # Estatísticas descritivas
        if 'is_offensive' in data.columns:
            total_offensive = data['is_offensive'].sum()
            percentage_offensive = (total_offensive / len(data)) * 100
            
            report['summary_statistics'] = {
                'total_offensive_content': int(total_offensive),
                'percentage_offensive': round(percentage_offensive, 2),
                'total_clean_content': int(len(data) - total_offensive),
                'percentage_clean': round(100 - percentage_offensive, 2)
            }
        
        # Análise por década
        if 'year' in data.columns and 'is_offensive' in data.columns:
            data['decade'] = (data['year'] // 10) * 10
            decade_analysis = data.groupby('decade')['is_offensive'].agg(['count', 'sum', 'mean']).to_dict()
            report['decade_analysis'] = decade_analysis
        
        # Recomendações baseadas nos resultados
        if self.results and 'evaluation' in self.results:
            best_model = max(self.results['evaluation'].items(), 
                           key=lambda x: x[1]['accuracy'])
            
            report['recommendations'] = [
                f"Modelo recomendado: {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.4f})",
                "Implementar sistema de moderação baseado nos modelos treinados",
                "Monitorar tendências temporais de conteúdo ofensivo",
                "Estabelecer thresholds de confiança para classificação automática"
            ]
        
        # Salvar relatório
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"Relatório salvo em: {output_file}")
        return report

def load_sample_data():
    """
    Carrega dados de exemplo para demonstração
    Em um projeto real, isso carregaria dados do Kaggle ou outras fontes
    """
    # Dados simulados para demonstração
    np.random.seed(42)
    
    sample_data = {
        'title': [
            'Love Song', 'Rock Anthem', 'Pop Hit', 'Ballad', 'Rap Song',
            'Country Tune', 'Jazz Standard', 'Blues Track', 'Folk Song', 'Electronic Beat'
        ] * 100,
        'artist': [
            'Artist A', 'Artist B', 'Artist C', 'Artist D', 'Artist E'
        ] * 200,
        'year': np.random.randint(1959, 2024, 1000),
        'lyrics': [
            'This is a beautiful love song about happiness and joy',
            'Aggressive lyrics with potential offensive content and hate',
            'Peaceful melody with positive vibes and good feelings',
            'Dark lyrics with violent themes and discrimination',
            'Uplifting song about unity and love for everyone'
        ] * 200,
        'genre': np.random.choice(['Pop', 'Rock', 'Hip-Hop', 'Country', 'Jazz'], 1000)
    }
    
    # Simular labels de conteúdo ofensivo
    offensive_indicators = ['hate', 'violent', 'discrimination', 'aggressive', 'offensive']
    is_offensive = []
    offense_type = []
    
    for lyric in sample_data['lyrics']:
        if any(indicator in lyric.lower() for indicator in offensive_indicators):
            is_offensive.append(1)
            offense_type.append(np.random.choice(['hate_speech', 'violence', 'discrimination']))
        else:
            is_offensive.append(0)
            offense_type.append('clean')
    
    sample_data['is_offensive'] = is_offensive
    sample_data['offense_type'] = offense_type
    
    return pd.DataFrame(sample_data)

def main():
    """Função principal para executar a análise completa"""
    print("=== Análise de Letras de Músicas Populares (1959-2023) ===")
    print("Detecção de Conteúdo Inapropriado: Racismo, Homofobia, Discurso de Ódio")
    print()
    
    # Inicializar analisador
    analyzer = MusicLyricsAnalyzer()
    
    # Carregar dados (substitua pela sua fonte de dados real)
    print("Carregando dados...")
    data = load_sample_data()
    print(f"Dados carregados: {len(data)} músicas")
    
    # Pré-processar letras
    print("Pré-processando textos...")
    data['processed_lyrics'] = data['lyrics'].apply(analyzer.preprocess_text)
    
    # Extrair features
    print("Extraindo features...")
    features = analyzer.extract_features(data['processed_lyrics'].tolist())
    
    # Combinar diferentes tipos de features
    X = np.hstack([
        features['tfidf'].toarray(),
        features['keywords'],
        features['sentiment']
    ])
    y = data['is_offensive'].values
    
    # Dividir dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Treinar modelos
    analyzer.train_models(X_train, y_train)
    
    # Avaliar modelos
    print("\nAvaliando modelos...")
    analyzer.evaluate_models(X_test, y_test)
    
    # Análise com transformers (para algumas amostras)
    print("\nAnalisando com modelos de transformers...")
    sample_lyrics = data['lyrics'].head(20).tolist()
    transformer_results = analyzer.analyze_with_transformers(sample_lyrics)
    
    # Adicionar resultados de sentimento ao dataframe
    sentiment_scores = []
    for lyric in data['lyrics']:
        blob = TextBlob(lyric)
        sentiment_scores.append(blob.sentiment.polarity)
    
    data['sentiment_score'] = sentiment_scores
    
    # Criar visualizações
    print("\nGerando visualizações...")
    analyzer.visualize_results(data)
    
    # Gerar relatório
    print("\nGerando relatório...")
    report = analyzer.generate_report(data)
    
    print("\n=== Análise Concluída ===")
    print(f"Relatório salvo em: analysis_report.json")
    print(f"Visualizações salvas em: analysis_results.png")
    
    return analyzer, data, report

if __name__ == "__main__":
    analyzer, data, report = main() 