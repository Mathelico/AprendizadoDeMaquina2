"""
Script para carregar dados de letras de música do Kaggle
Dataset: Top 100 Songs and Lyrics from 1959 to 2019
"""

import pandas as pd
import requests
import os
import zipfile
from pathlib import Path
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MusicDataLoader:
    """Classe para carregar dados de letras de música"""
    
    def __init__(self, data_dir="data"):
        """
        Inicializa o carregador de dados
        
        Args:
            data_dir (str): Diretório para armazenar os dados
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def load_kaggle_dataset(self, dataset_path=None):
        """
        Carrega dataset do Kaggle
        
        Args:
            dataset_path (str): Caminho para o arquivo CSV baixado do Kaggle
            
        Returns:
            pd.DataFrame: DataFrame com os dados carregados
        """
        if dataset_path is None:
            # Procurar por arquivos CSV na pasta data
            csv_files = list(self.data_dir.glob("*.csv"))
            if not csv_files:
                logger.warning("Nenhum arquivo CSV encontrado na pasta data/")
                logger.info("Por favor, baixe o dataset do Kaggle e coloque na pasta 'data/'")
                logger.info("Link: https://www.kaggle.com/datasets/brianblakely/top-100-songs-and-lyrics-from-1959-to-2019")
                return None
            dataset_path = csv_files[0]
            logger.info(f"Usando arquivo: {dataset_path}")
        
        try:
            # Carregar dados
            df = pd.read_csv(dataset_path, encoding='utf-8')
            logger.info(f"Dataset carregado: {len(df)} linhas, {len(df.columns)} colunas")
            
            # Processar dados
            df = self.preprocess_dataset(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Erro ao carregar dataset: {e}")
            return None
    
    def preprocess_dataset(self, df):
        """
        Pré-processa o dataset carregado
        
        Args:
            df (pd.DataFrame): DataFrame original
            
        Returns:
            pd.DataFrame: DataFrame processado
        """
        logger.info("Pré-processando dataset...")
        
        # Padronizar nomes das colunas
        column_mapping = {
            'Song': 'title',
            'Artist': 'artist', 
            'Year': 'year',
            'Lyrics': 'lyrics',
            'Genre': 'genre'
        }
        
        # Renomear colunas se existirem
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})
        
        # Remover linhas com letras vazias
        if 'lyrics' in df.columns:
            df = df.dropna(subset=['lyrics'])
            df = df[df['lyrics'].str.strip() != '']
        
        # Converter ano para inteiro
        if 'year' in df.columns:
            df['year'] = pd.to_numeric(df['year'], errors='coerce')
            df = df.dropna(subset=['year'])
            df['year'] = df['year'].astype(int)
        
        # Limpar texto das letras
        if 'lyrics' in df.columns:
            df['lyrics'] = df['lyrics'].astype(str)
            df['lyrics'] = df['lyrics'].str.replace('\n', ' ')
            df['lyrics'] = df['lyrics'].str.replace('\r', ' ')
            
        logger.info(f"Dataset processado: {len(df)} linhas válidas")
        return df
    
    def create_labels(self, df, manual_labels_file=None):
        """
        Cria labels para conteúdo ofensivo
        NOTA: Esta é uma versão simplificada. Em um projeto real,
        seria necessário um processo de anotação manual por especialistas.
        
        Args:
            df (pd.DataFrame): DataFrame com os dados
            manual_labels_file (str): Arquivo com labels manuais (opcional)
            
        Returns:
            pd.DataFrame: DataFrame com labels adicionados
        """
        logger.info("Criando labels para conteúdo ofensivo...")
        
        if manual_labels_file and os.path.exists(manual_labels_file):
            # Carregar labels manuais se disponível
            manual_labels = pd.read_csv(manual_labels_file)
            df = df.merge(manual_labels, on=['title', 'artist'], how='left')
        else:
            # Criar labels baseadas em palavras-chave (método simplificado)
            # IMPORTANTE: Este é um método básico e pode gerar falsos positivos
            offensive_keywords = {
                'hate_speech': [
                    'hate', 'kill', 'murder', 'violence', 'destroy', 
                    'eliminate', 'annihilate', 'exterminate'
                ],
                'discrimination': [
                    'racist', 'discrimination', 'prejudice', 'bigot', 
                    'supremacy', 'segregation', 'apartheid'
                ],
                'homophobia': [
                    'homophobic', 'faggot', 'dyke', 'queer' # Nota: incluindo termos pejorativos apenas para detecção
                ],
                'general_offensive': [
                    'bitch', 'whore', 'slut', 'cunt', 'fuck', 'shit',
                    'damn', 'hell', 'bastard', 'asshole'
                ]
            }
            
            # Inicializar colunas
            df['is_offensive'] = 0
            df['offense_type'] = 'clean'
            df['offense_keywords'] = ''
            
            # Verificar cada música
            for idx, row in df.iterrows():
                lyrics_lower = row['lyrics'].lower()
                found_keywords = []
                offense_types = []
                
                for category, keywords in offensive_keywords.items():
                    for keyword in keywords:
                        if keyword in lyrics_lower:
                            found_keywords.append(keyword)
                            if category not in offense_types:
                                offense_types.append(category)
                
                if found_keywords:
                    df.at[idx, 'is_offensive'] = 1
                    df.at[idx, 'offense_type'] = ', '.join(offense_types)
                    df.at[idx, 'offense_keywords'] = ', '.join(found_keywords)
        
        # Estatísticas
        total_offensive = df['is_offensive'].sum()
        percentage = (total_offensive / len(df)) * 100
        
        logger.info(f"Labels criados: {total_offensive} músicas ofensivas ({percentage:.1f}%)")
        
        return df
    
    def save_processed_data(self, df, filename="processed_music_data.csv"):
        """
        Salva dados processados
        
        Args:
            df (pd.DataFrame): DataFrame processado
            filename (str): Nome do arquivo de saída
        """
        output_path = self.data_dir / filename
        df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"Dados salvos em: {output_path}")
        
    def load_processed_data(self, filename="processed_music_data.csv"):
        """
        Carrega dados já processados
        
        Args:
            filename (str): Nome do arquivo processado
            
        Returns:
            pd.DataFrame: DataFrame carregado
        """
        file_path = self.data_dir / filename
        if file_path.exists():
            df = pd.read_csv(file_path, encoding='utf-8')
            logger.info(f"Dados processados carregados: {len(df)} linhas")
            return df
        else:
            logger.warning(f"Arquivo não encontrado: {file_path}")
            return None

def download_sample_data():
    """
    Baixa dados de exemplo se não houver dados reais disponíveis
    """
    logger.info("Tentando baixar dados de exemplo...")
    
    # URL de exemplo (substitua por fonte real se disponível)
    sample_data = {
        'title': [
            'Love Me Tender', 'Hound Dog', 'Jailhouse Rock', 'Heartbreak Hotel',
            'Can\'t Help Myself', 'My Girl', 'I Want to Hold Your Hand', 
            'Help!', 'Yesterday', 'Hey Jude', 'Stairway to Heaven',
            'Hotel California', 'Bohemian Rhapsody', 'Billie Jean',
            'Like a Prayer', 'Smells Like Teen Spirit', 'Wonderwall',
            'Baby One More Time', 'Crazy in Love', 'Umbrella'
        ],
        'artist': [
            'Elvis Presley', 'Elvis Presley', 'Elvis Presley', 'Elvis Presley',
            'Four Tops', 'The Temptations', 'The Beatles', 'The Beatles',
            'The Beatles', 'The Beatles', 'Led Zeppelin', 'Eagles',
            'Queen', 'Michael Jackson', 'Madonna', 'Nirvana', 'Oasis',
            'Britney Spears', 'Beyoncé', 'Rihanna'
        ],
        'year': [
            1956, 1956, 1957, 1956, 1965, 1965, 1963, 1965, 1965, 1968,
            1971, 1976, 1975, 1983, 1989, 1991, 1995, 1999, 2003, 2007
        ],
        'lyrics': [
            'Love me tender, love me sweet, never let me go',
            'You ain\'t nothin\' but a hound dog, cryin\' all the time',
            'Everybody in the whole cell block was dancin\' to the jailhouse rock',
            'Well since my baby left me, I found a new place to dwell',
            'Sugar pie honey bunch, you know that I love you',
            'I\'ve got sunshine on a cloudy day',
            'I want to hold your hand, I want to hold your hand',
            'Help, I need somebody, help, not just anybody',
            'Yesterday, all my troubles seemed so far away',
            'Hey Jude, don\'t be afraid, you were made to go out and get her',
            'There\'s a lady who\'s sure all that glitters is gold',
            'Welcome to the Hotel California, such a lovely place',
            'Is this the real life, is this just fantasy',
            'Billie Jean is not my lover, she\'s just a girl',
            'Life is a mystery, everyone must stand alone',
            'Load up on guns, bring your friends, it\'s fun to lose and to pretend',
            'Today is gonna be the day that they\'re gonna throw it back to you',
            'Oh baby baby, how was I supposed to know',
            'Crazy in love, got me looking so crazy right now',
            'You can stand under my umbrella, ella, ella'
        ]
    }
    
    df = pd.DataFrame(sample_data)
    
    # Criar pasta data se não existir
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Salvar dados de exemplo
    output_path = data_dir / "sample_music_data.csv"
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    logger.info(f"Dados de exemplo salvos em: {output_path}")
    return df

def main():
    """Função principal para testar o carregador de dados"""
    loader = MusicDataLoader()
    
    # Tentar carregar dados do Kaggle
    df = loader.load_kaggle_dataset()
    
    if df is None:
        logger.info("Carregando dados de exemplo...")
        df = download_sample_data()
        df = loader.preprocess_dataset(df)
    
    # Criar labels
    df = loader.create_labels(df)
    
    # Salvar dados processados
    loader.save_processed_data(df)
    
    # Mostrar estatísticas
    print("\n=== ESTATÍSTICAS DOS DADOS ===")
    print(f"Total de músicas: {len(df)}")
    print(f"Período: {df['year'].min()} - {df['year'].max()}")
    print(f"Conteúdo ofensivo: {df['is_offensive'].sum()} ({df['is_offensive'].mean()*100:.1f}%)")
    print(f"Artistas únicos: {df['artist'].nunique()}")
    
    return df

if __name__ == "__main__":
    data = main() 