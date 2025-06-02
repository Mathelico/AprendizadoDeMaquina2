# 🚀 Guia de Início Rápido

## Análise de Letras de Músicas Populares (1959-2023)
### Detecção de Conteúdo Inapropriado

---

## ⚡ Execução Imediata (3 comandos)

### 1️⃣ Configurar Projeto
```bash
python setup_and_test.py
```

### 2️⃣ Carregar Dados
```bash
python data_loader.py
```

### 3️⃣ Executar Análise
```bash
python music_lyrics_analyzer.py
```

**OU** usar o notebook interativo:
```bash
jupyter notebook analysis_notebook.ipynb
```

---

## 📋 O que o Projeto Faz

✅ **Analisa letras de música** de 1959 a 2023  
✅ **Detecta conteúdo ofensivo** (racismo, homofobia, discurso de ódio)  
✅ **Treina múltiplos modelos** de machine learning  
✅ **Gera visualizações** e relatórios detalhados  
✅ **Compara performance** de diferentes algoritmos  

---

## 📊 Arquivos Gerados

Após a execução, você terá:

- `analysis_report.json` - Relatório completo com estatísticas
- `analysis_results.png` - Gráficos e visualizações
- `data/processed_music_data.csv` - Dataset processado
- `timeline_offensive_content.html` - Timeline interativo
- `confusion_matrix.html` - Matriz de confusão interativa

---

## 🎯 Principais Funcionalidades

### 🔍 Detecção Automática
- **Palavras-chave ofensivas**: Sistema baseado em dicionários
- **Análise de sentimento**: TextBlob + modelos transformers
- **Machine Learning**: 5 algoritmos diferentes (SVM, Random Forest, etc.)
- **Deep Learning**: BERT, RoBERTa para análise avançada

### 📈 Análise Temporal
- **Evolução por décadas**: Como mudou o conteúdo ao longo do tempo
- **Tendências por gênero**: Pop, Rock, Hip-Hop, etc.
- **Correlações históricas**: Eventos sociais vs conteúdo musical

### 🎨 Visualizações
- **Gráficos temporais**: Timeline de evolução
- **Distribuições**: Por gênero, década, tipo de ofensa
- **Nuvens de palavras**: Palavras mais frequentes
- **Heatmaps**: Correlações entre variáveis

---

## 📝 Dados

### 🔄 Dados de Exemplo (Automático)
O projeto vem com dados simulados para demonstração imediata.

### 📥 Dados Reais (Opcional)
Para usar dados reais do Kaggle:
1. Baixe: [Top 100 Songs and Lyrics (1959-2019)](https://www.kaggle.com/datasets/brianblakely/top-100-songs-and-lyrics-from-1959-to-2019)
2. Coloque o CSV na pasta `data/`
3. Execute novamente os scripts

---

## 🛠 Personalização Rápida

### Adicionar Palavras-chave
Edite `music_lyrics_analyzer.py`, função `load_offensive_keywords()`:

```python
'nova_categoria': [
    'palavra1', 'palavra2', 'palavra3'
]
```

### Ajustar Modelos
```python
# Mais features TF-IDF
analyzer.vectorizer = TfidfVectorizer(max_features=10000)

# Modelo personalizado
from sklearn.ensemble import ExtraTreesClassifier
custom_model = ExtraTreesClassifier(n_estimators=200)
```

---

## 🎓 Contexto Acadêmico

### 📚 Ideal para:
- **Trabalhos de Machine Learning**
- **Projetos de NLP**
- **Análise de Sentimentos**
- **Estudos Culturais**
- **Ciência de Dados**

### 🔬 Métodos Implementados:
- Pré-processamento de texto (NLTK)
- Vetorização TF-IDF
- Modelos supervisionados
- Validação cruzada
- Métricas de avaliação
- Análise exploratória

---

## ⚠️ Considerações Importantes

### 🎯 Objetivo Educacional
Este projeto é desenvolvido para fins **educacionais e de pesquisa**.

### 🤖 Limitações dos Modelos
- Falsos positivos podem ocorrer
- Contexto artístico nem sempre é considerado
- Dependente da qualidade dos dados de treino

### 🎨 Liberdade Artística
- Respeita a expressão criativa
- Considera contexto histórico
- Transparente nos critérios

---

## 🆘 Troubleshooting

### ❌ Erro de Importação
```bash
pip install -r requirements.txt
```

### ❌ Erro NLTK
```python
import nltk
nltk.download('all')
```

### ❌ Sem Dados
```bash
python data_loader.py  # Cria dados de exemplo
```

---

## 📞 Precisa de Ajuda?

1. **README.md** - Documentação completa
2. **analysis_notebook.ipynb** - Tutorial passo-a-passo
3. **Código comentado** - Todas as funções explicadas

---

**🎵 Aproveite a análise musical!** 🎵 