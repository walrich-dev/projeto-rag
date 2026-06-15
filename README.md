# 🤖 Assistente RAG — Python & Inteligência Artificial

Assistente inteligente de perguntas e respostas construído com **RAG** (Retrieval-Augmented Generation). Responde perguntas sobre Python, Machine Learning, Deep Learning e LangChain com base em documentos reais, citando as fontes.

## 🏗️ Arquitetura

```
Usuário → Pergunta → [Embedding da Query]
                          ↓
                   [Vectorstore ChromaDB]
                    busca por similaridade
                          ↓
                   [Top-4 Chunks Relevantes]
                          ↓
              [Prompt: contexto + pergunta] → LLM → Resposta + Fontes
```

## 📁 Estrutura do Projeto

```
projeto_rag/
├── docs/                              ← base de conhecimento
│   ├── python_fundamentos.txt
│   ├── machine_learning_conceitos.txt
│   ├── deep_learning_transformers.txt
│   └── langchain_e_rag.txt
├── src/
│   ├── loader.py       ← carrega e chunkeia documentos
│   ├── embedder.py     ← gera embeddings e gerencia vectorstore
│   ├── retriever.py    ← busca por similaridade
│   ├── chain.py        ← monta o pipeline RAG
│   └── app.py          ← interface Streamlit
├── eval/
│   ├── perguntas_teste.json   ← 10 perguntas de avaliação
│   ├── avaliar.py             ← script de métricas
│   └── resultados.json        ← gerado após avaliação
├── indexar.py          ← indexa os documentos (rodar 1x)
├── .env.example        ← template de variáveis de ambiente
├── .gitignore
└── requirements.txt
```

## 🚀 Como Usar

### 1. Pré-requisitos

```bash
# Python 3.11+
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

### 2. Configurar o LLM

**Opção A — Ollama (gratuito, local):**
```bash
# Instale em ollama.com
ollama pull llama3.1
ollama serve
```

**Opção B — Gemini (Google, gratuito):**
```bash
cp .env.example .env
# Edite .env: LLM_PROVIDER=gemini, MODEL_NAME=gemini-1.5-flash
# Adicione sua GOOGLE_API_KEY (obtenha em aistudio.google.com)
```

### 3. Indexar os documentos

```bash
python indexar.py
```

### 4. Rodar o app

```bash
streamlit run src/app.py
```

## 📊 Avaliação

```bash
python eval/avaliar.py
```

### Resultados (exemplo após avaliação):

| # | Pergunta | Acerto | Fonte OK | Tempo |
|---|----------|--------|----------|-------|
| 1 | Diferença entre lista e tupla? | 1.0 | sim | 2.1s |
| 2 | O que é overfitting? | 1.0 | sim | 2.4s |
| 3 | Como funciona o attention? | 0.8 | sim | 3.1s |
| 4 | O que é RAG? | 1.0 | sim | 2.8s |
| 5 | Tipos de aprendizado em ML? | 1.0 | sim | 2.2s |
| 6 | O que é LoRA? | 0.9 | sim | 3.4s |
| 7 | chunk_size vs chunk_overlap? | 0.9 | sim | 2.5s |
| 8 | O que é temperature em LLMs? | 1.0 | sim | 2.3s |
| 9 | Como funciona K-Fold? | 0.8 | sim | 2.9s |
| 10 | Capital da Austrália? (fora do escopo) | 1.0 | — | 1.2s |
| **Média** | | **0.94** | **90%** | **2.5s** |

### Métricas-alvo:

| Métrica | Meta | Status |
|---------|------|--------|
| Precisão | > 70% | ✅ |
| Citação de fontes | 100% existentes | ✅ |
| Latência média | < 8s | ✅ |
| Recusa fora do escopo | 100% | ✅ |

## 🛠️ Decisões Técnicas

| Componente | Escolha | Motivo |
|-----------|---------|--------|
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` | Gratuito, bom suporte ao português |
| Vectorstore | ChromaDB | Local, sem custo, fácil setup |
| Chunk size | 600 chars + 80 overlap | Balanço entre precisão e contexto |
| LLM padrão | Ollama (llama3.1) | Gratuito, sem API key |
| Interface | Streamlit | Rápido, Python nativo |

## 🔧 Stack

- Python 3.11+
- LangChain
- ChromaDB
- sentence-transformers
- Streamlit
- Ollama (LLM local gratuito)
