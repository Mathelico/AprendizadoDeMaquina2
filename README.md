# Análise de Letras de Músicas Populares (1959-2023)
## Detecção de Conteúdo Inapropriado: Racismo, Homofobia, Discurso de Ódio e Preconceito

Este projeto implementa algoritmos de machine learning e processamento de linguagem natural (NLP) para identificar e analisar conteúdo potencialmente ofensivo em letras de música ao longo de mais de 6 décadas.

## 🎯 Objetivos

- **Análise Temporal**: Investigar a evolução do conteúdo inapropriado em músicas populares de 1959 a 2023
- **Detecção Automática**: Desenvolver modelos de classificação para identificar automaticamente conteúdo ofensivo
- **Classificação Multiclasse**: Distinguir entre diferentes tipos de conteúdo inapropriado (racismo, homofobia, discurso de ódio)
- **Análise Comparativa**: Comparar diferentes abordagens de NLP e machine learning
- **Insights Culturais**: Gerar insights sobre mudanças sociais refletidas na música popular

## 🛠 Tecnologias Utilizadas

### Machine Learning & NLP
- **Scikit-learn**: Modelos tradicionais de ML (SVM, Random Forest, Naive Bayes, etc.)
- **Transformers**: Modelos de deep learning (BERT, RoBERTa) para análise de sentimento
- **NLTK & TextBlob**: Processamento de linguagem natural
- **SpaCy**: NLP avançado e análise linguística

### Análise de Dados
- **Pandas & NumPy**: Manipulação e análise de dados
- **Matplotlib & Seaborn**: Visualizações estáticas
- **Plotly**: Visualizações interativas
- **WordCloud**: Nuvens de palavras

### Desenvolvimento
- **Jupyter Notebooks**: Análise interativa
- **Python 3.8+**: Linguagem principal

## 📁 Estrutura do Projeto

```
AprendizadoDeMaquina2/
├── README.md                     # Este arquivo
├── requirements.txt              # Dependências do projeto
├── music_lyrics_analyzer.py      # Classe principal do analisador
├── data_loader.py               # Script para carregar dados
├── analysis_notebook.ipynb     # Notebook para análise interativa
├── data/                        # Diretório para dados
│   ├── processed_music_data.csv # Dados processados
│   └── sample_music_data.csv    # Dados de exemplo
├── results/                     # Resultados e relatórios
│   ├── analysis_report.json    # Relatório detalhado
│   ├── analysis_results.png    # Visualizações
│   └── models/                 # Modelos treinados
└── docs/                       # Documentação adicional
```

## 🚀 Como Usar

### 1. Configuração do Ambiente

```bash
# Clonar o repositório
git clone https://github.com/seu-usuario/AprendizadoDeMaquina2.git
cd AprendizadoDeMaquina2

# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Instalar dependências
pip install -r requirements.txt
```

### 2. Preparação dos Dados

#### Opção A: Usar dados do Kaggle (Recomendado)
1. Baixe o dataset do Kaggle: [Top 100 Songs and Lyrics from 1959 to 2019](https://www.kaggle.com/datasets/brianblakely/top-100-songs-and-lyrics-from-1959-to-2019)
2. Coloque o arquivo CSV na pasta `data/`
3. Execute o carregador de dados:

```bash
python data_loader.py
```

#### Opção B: Usar dados de exemplo
```bash
python data_loader.py
# Os dados de exemplo serão gerados automaticamente
```

### 3. Análise Completa

#### Via Script Principal
```bash
python music_lyrics_analyzer.py
```

#### Via Jupyter Notebook (Recomendado)
```bash
jupyter notebook analysis_notebook.ipynb
```

## 📊 Funcionalidades

### 1. Pré-processamento de Texto
- Limpeza e normalização de letras
- Remoção de stopwords
- Lemmatização e stemming
- Tokenização avançada

### 2. Extração de Features
- **TF-IDF**: Representação vetorial de documentos
- **Keywords**: Features baseadas em palavras-chave ofensivas
- **Sentiment**: Análise de polaridade emocional
- **N-gramas**: Padrões linguísticos

### 3. Modelos de Machine Learning
- **Naive Bayes**: Baseline probabilístico
- **SVM**: Support Vector Machines
- **Random Forest**: Ensemble de árvores
- **Gradient Boosting**: Boosting avançado
- **Logistic Regression**: Regressão logística

### 4. Modelos de Deep Learning
- **BERT**: Representações contextuais
- **RoBERTa**: Análise de sentimento avançada
- **Toxic-BERT**: Detecção específica de discurso de ódio

### 5. Visualizações e Relatórios
- Timeline de evolução temporal
- Distribuição por gêneros musicais
- Matrizes de confusão interativas
- Nuvens de palavras
- Heatmaps de correlação
- Relatórios em JSON

## 🎵 Tipos de Conteúdo Analisado

### Categorias de Detecção
1. **Racismo**: Linguagem discriminatória racial
2. **Homofobia**: Conteúdo anti-LGBTQ+
3. **Discurso de Ódio**: Linguagem violenta e ameaçadora
4. **Preconceito Geral**: Outros tipos de discriminação

### Métricas de Avaliação
- **Accuracy**: Precisão geral
- **Precision**: Precisão por classe
- **Recall**: Sensibilidade
- **F1-Score**: Harmônica entre precision e recall
- **AUC-ROC**: Área sob a curva ROC

## 📈 Resultados Esperados

### Análises Temporais
- Identificação de décadas com maior incidência de conteúdo ofensivo
- Correlação com eventos históricos e mudanças sociais
- Evolução da linguagem musical

### Performance dos Modelos
- Comparação entre abordagens tradicionais e deep learning
- Identificação do modelo mais eficaz
- Análise de falsos positivos e negativos

### Insights Culturais
- Gêneros musicais com maior propensão a conteúdo ofensivo
- Mudanças na aceitação social de certas linguagens
- Impacto de movimentos sociais na música

## ⚠️ Considerações Éticas

### Responsabilidade na Classificação
- **Contexto Cultural**: Consideração de mudanças históricas na linguagem
- **Liberdade Artística**: Respeito à expressão criativa
- **Viés Algorítmico**: Monitoramento contínuo de preconceitos nos modelos
- **Transparência**: Documentação clara dos critérios de classificação

### Limitações
- Dependência da qualidade dos dados de treinamento
- Possibilidade de falsos positivos em contextos artísticos
- Variação cultural na percepção de conteúdo ofensivo
- Evolução da linguagem ao longo do tempo

## 🔧 Configurações Avançadas

### Personalização de Keywords
Edite o arquivo `music_lyrics_analyzer.py` na função `load_offensive_keywords()` para adicionar ou remover palavras-chave específicas.

### Ajuste de Modelos
```python
# Exemplo de configuração personalizada
analyzer = MusicLyricsAnalyzer()
analyzer.vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 3),
    min_df=5
)
```

### Modelos Customizados
```python
# Adicionar modelo personalizado
from sklearn.ensemble import ExtraTreesClassifier
custom_model = ExtraTreesClassifier(n_estimators=200)
analyzer.models['extra_trees'] = custom_model
```

## 📚 Referências e Inspirações

- Dataset Kaggle: [Top 100 Songs and Lyrics](https://www.kaggle.com/datasets/brianblakely/top-100-songs-and-lyrics-from-1959-to-2019)
- Pesquisas em Hate Speech Detection
- Análises de sentimento em música
- Estudos culturais sobre evolução musical

## 🤝 Contribuições

Contribuições são bem-vindas! Para contribuir:

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

### Áreas para Contribuição
- Melhoria dos algoritmos de detecção
- Adição de novos tipos de análise
- Otimização de performance
- Documentação e tutoriais
- Testes unitários
- Interface web/API

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para detalhes.

## 📞 Contato

- **Autor**: [Seu Nome]
- **Email**: [seu.email@example.com]
- **LinkedIn**: [seu-linkedin]
- **GitHub**: [seu-github]

## 🙏 Agradecimentos

- Comunidade científica de NLP
- Mantenedores das bibliotecas utilizadas
- Contribuidores do dataset
- Comunidade open source

---

**Nota**: Este projeto é desenvolvido para fins educacionais e de pesquisa. A detecção de conteúdo ofensivo é uma tarefa complexa que requer consideração cuidadosa do contexto cultural e histórico.