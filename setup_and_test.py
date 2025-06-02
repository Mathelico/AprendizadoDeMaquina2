#!/usr/bin/env python3
"""
Script de configuração inicial e teste rápido do projeto
Análise de Letras de Músicas Populares (1959-2023)
"""

import subprocess
import sys
import os
from pathlib import Path
import time

def install_requirements():
    """Instala as dependências necessárias"""
    print("🔧 Instalando dependências...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependências instaladas com sucesso!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Erro ao instalar dependências")
        return False

def download_nltk_data():
    """Baixa dados necessários do NLTK"""
    print("📥 Baixando dados do NLTK...")
    try:
        import nltk
        nltk_downloads = ['punkt', 'stopwords', 'wordnet', 'vader_lexicon', 'omw-1.4']
        for resource in nltk_downloads:
            print(f"   Baixando {resource}...")
            nltk.download(resource, quiet=True)
        print("✅ Dados do NLTK baixados com sucesso!")
        return True
    except Exception as e:
        print(f"❌ Erro ao baixar dados do NLTK: {e}")
        return False

def create_directories():
    """Cria diretórios necessários"""
    print("📁 Criando diretórios...")
    directories = ['data', 'results', 'results/models', 'docs']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"   ✓ {directory}/")
    
    print("✅ Diretórios criados!")
    return True

def test_imports():
    """Testa se todas as importações funcionam"""
    print("🧪 Testando importações...")
    
    imports_to_test = [
        ('pandas', 'pd'),
        ('numpy', 'np'),
        ('sklearn', None),
        ('nltk', None),
        ('textblob', None),
        ('matplotlib.pyplot', 'plt'),
        ('seaborn', 'sns'),
        ('plotly.express', 'px')
    ]
    
    failed_imports = []
    
    for package, alias in imports_to_test:
        try:
            if alias:
                exec(f"import {package} as {alias}")
            else:
                exec(f"import {package}")
            print(f"   ✓ {package}")
        except ImportError as e:
            print(f"   ❌ {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"❌ Falha ao importar: {', '.join(failed_imports)}")
        return False
    else:
        print("✅ Todas as importações funcionaram!")
        return True

def run_quick_test():
    """Executa um teste rápido do analisador"""
    print("🏃 Executando teste rápido...")
    
    try:
        # Importar módulos locais
        from data_loader import MusicDataLoader, download_sample_data
        from music_lyrics_analyzer import MusicLyricsAnalyzer
        
        print("   ✓ Módulos locais importados com sucesso")
        
        # Criar dados de exemplo
        print("   📊 Criando dados de exemplo...")
        data = download_sample_data()
        
        # Inicializar analisador
        print("   🔍 Inicializando analisador...")
        analyzer = MusicLyricsAnalyzer()
        
        # Processar uma amostra pequena
        print("   ⚙️ Processando amostra...")
        sample_lyrics = data['lyrics'].head(5).tolist()
        processed = [analyzer.preprocess_text(lyric) for lyric in sample_lyrics]
        
        print("   ✓ Pré-processamento funcionou")
        
        # Testar extração de features
        print("   🔎 Testando extração de features...")
        features = analyzer.extract_features(processed)
        
        print("   ✓ Extração de features funcionou")
        print(f"   📊 Shape das features TF-IDF: {features['tfidf'].shape}")
        
        print("✅ Teste rápido concluído com sucesso!")
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste rápido: {e}")
        return False

def display_next_steps():
    """Mostra os próximos passos para o usuário"""
    print("\n" + "="*60)
    print("🎉 CONFIGURAÇÃO CONCLUÍDA!")
    print("="*60)
    
    print("\n📋 PRÓXIMOS PASSOS:")
    print("1. 📥 Baixar dados reais do Kaggle:")
    print("   https://www.kaggle.com/datasets/brianblakely/top-100-songs-and-lyrics-from-1959-to-2019")
    print("   Coloque o arquivo CSV na pasta 'data/'")
    
    print("\n2. 🚀 Executar análise:")
    print("   python data_loader.py          # Carregar e processar dados")
    print("   python music_lyrics_analyzer.py  # Análise completa")
    print("   jupyter notebook analysis_notebook.ipynb  # Análise interativa")
    
    print("\n3. 📊 Arquivos gerados:")
    print("   data/processed_music_data.csv  # Dados processados")
    print("   analysis_report.json           # Relatório detalhado")
    print("   analysis_results.png           # Visualizações")
    
    print("\n4. 🔧 Personalização:")
    print("   Edite as palavras-chave ofensivas em music_lyrics_analyzer.py")
    print("   Ajuste parâmetros dos modelos conforme necessário")
    
    print("\n📚 DOCUMENTAÇÃO:")
    print("   README.md - Guia completo do projeto")
    print("   analysis_notebook.ipynb - Tutorial interativo")

def main():
    """Função principal do script de configuração"""
    print("🎵 CONFIGURAÇÃO DO PROJETO")
    print("Análise de Letras de Músicas Populares (1959-2023)")
    print("Detecção de Conteúdo Inapropriado")
    print("="*60)
    
    start_time = time.time()
    
    # Verificar Python
    if sys.version_info < (3, 7):
        print("❌ Python 3.7+ é necessário")
        return False
    
    print(f"✅ Python {sys.version.split()[0]} detectado")
    
    # Etapas de configuração
    steps = [
        ("Criando diretórios", create_directories),
        ("Instalando dependências", install_requirements),
        ("Baixando dados do NLTK", download_nltk_data),
        ("Testando importações", test_imports),
        ("Executando teste rápido", run_quick_test)
    ]
    
    for step_name, step_function in steps:
        print(f"\n🔄 {step_name}...")
        success = step_function()
        if not success:
            print(f"❌ Falha em: {step_name}")
            print("⚠️  Verifique os erros acima e tente novamente")
            return False
    
    # Calcular tempo total
    total_time = time.time() - start_time
    print(f"\n⏱️  Configuração concluída em {total_time:.1f} segundos")
    
    # Mostrar próximos passos
    display_next_steps()
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎊 Projeto configurado e pronto para uso!")
    else:
        print("\n❌ Configuração falhou. Verifique os erros acima.")
        sys.exit(1) 