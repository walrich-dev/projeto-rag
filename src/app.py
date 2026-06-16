"""
app.py — Interface Streamlit do Assistente RAG.

Execute com: streamlit run src/app.py
"""

import os
import sys
import time

# Garante que imports de src/ funcionem a partir da raiz do projeto
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from src.loader import carregar_documentos
from src.embedder import criar_vectorstore, carregar_vectorstore
from src.chain import criar_chain

# ── Configuração da página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Assistente RAG — Python & IA",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Estilos mínimos ──────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .fonte-badge {
        background: #f0f2f6;
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 0.78rem;
        color: #555;
        margin-top: 4px;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Inicialização (cacheada) ─────────────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️ Inicializando base de conhecimento...")
def inicializar():
    db_path = "chroma_db/"
    if os.path.exists(db_path) and os.listdir(db_path):
        db = carregar_vectorstore(db_path)
    else:
        chunks = carregar_documentos("docs/")
        db = criar_vectorstore(chunks, db_path)
    return criar_chain(db)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Assistente RAG")
    st.markdown("**Tema:** Python & Inteligência Artificial")
    st.divider()

    st.markdown("### Base de Conhecimento")
    st.markdown(
        """
        -  Python Fundamentos
        -  Machine Learning
        - Deep Learning & Transformers
        -  LangChain & RAG
        """
    )
    st.divider()

    st.markdown("###  Ações")
    if st.button("Reindexar documentos", use_container_width=True):
        st.cache_resource.clear()
        if "historico" in st.session_state:
            del st.session_state["historico"]
        st.rerun()

    if st.button("Limpar conversa", use_container_width=True):
        st.session_state.historico = []
        st.rerun()

    st.divider()
    st.caption("Cada resposta cita as fontes utilizadas.")
    st.caption("As respostas se baseiam apenas nos documentos indexados.")


# ── Carregar chain ───────────────────────────────────────────────────────────
try:
    chain = inicializar()
except Exception as err:
    st.error(f"Erro ao inicializar o sistema:\n\n{err}")
    st.info(
        "Verifique se o Ollama está rodando (`ollama serve`) "
        "ou configure outro provider no arquivo .env"
    )
    st.stop()


# ── Cabeçalho principal ──────────────────────────────────────────────────────
st.title("Assistente de Python & IA")
st.caption(
    "Powered by **RAG** (Retrieval-Augmented Generation) · "
    "Respostas baseadas exclusivamente nos documentos indexados"
)
st.divider()


# ── Histórico de mensagens ───────────────────────────────────────────────────
if "historico" not in st.session_state:
    st.session_state.historico = []

for msg in st.session_state.historico:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("fontes"):
            st.markdown(
                f'<span class="fonte-badge"> Fontes: {msg["fontes"]}</span>',
                unsafe_allow_html=True,
            )


# ── Input do usuário ─────────────────────────────────────────────────────────
if pergunta := st.chat_input("Faça sua pergunta sobre Python ou IA..."):
    # Exibe a mensagem do usuário imediatamente
    with st.chat_message("user"):
        st.markdown(pergunta)
    st.session_state.historico.append({"role": "user", "content": pergunta})

    # Gera a resposta
    with st.chat_message("assistant"):
        with st.spinner("🔍 Buscando na base de conhecimento..."):
            inicio = time.time()
            resultado = chain.invoke(pergunta)
            latencia = time.time() - inicio

        resposta = resultado["result"]
        docs_fonte = resultado.get("source_documents", [])

        # Extrai nomes de arquivo únicos das fontes
        fontes = sorted(
            {os.path.basename(d.metadata.get("source", "?")) for d in docs_fonte}
        )
        fontes_str = ", ".join(fontes) if fontes else "—"

        st.markdown(resposta)
        st.markdown(
            f'<span class="fonte-badge"> Fontes: {fontes_str} · ⏱ {latencia:.1f}s</span>',
            unsafe_allow_html=True,
        )

    # Salva no histórico
    st.session_state.historico.append(
        {
            "role": "assistant",
            "content": resposta,
            "fontes": f"{fontes_str} · ⏱ {latencia:.1f}s",
        }
    )
