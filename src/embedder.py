"""
embedder.py — Geração de embeddings e gerenciamento do vectorstore.

Usa sentence-transformers com modelo multilíngue para suporte nativo
ao português sem custo de API.
"""

import os
import shutil
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def _get_embeddings() -> HuggingFaceEmbeddings:
    print(f"🔄 Carregando modelo de embeddings: {EMBEDDING_MODEL}")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def criar_vectorstore(chunks: list, pasta_db: str = "chroma_db/") -> Chroma:
    """Cria um vectorstore do zero a partir dos chunks fornecidos."""
    if os.path.exists(pasta_db):
        shutil.rmtree(pasta_db)
        print(f"🗑️  Vectorstore anterior removido: {pasta_db}")

    embeddings = _get_embeddings()
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=pasta_db,
    )

    total = db._collection.count()
    print(f"✅ Vectorstore criado em '{pasta_db}' com {total} vetores")
    return db


def carregar_vectorstore(pasta_db: str = "chroma_db/") -> Chroma:
    """Carrega um vectorstore já existente em disco."""
    if not os.path.exists(pasta_db):
        raise FileNotFoundError(
            f"Vectorstore não encontrado em '{pasta_db}'.\n"
            "Execute 'python indexar.py' primeiro para indexar os documentos."
        )

    embeddings = _get_embeddings()
    db = Chroma(
        persist_directory=pasta_db,
        embedding_function=embeddings,
    )

    total = db._collection.count()
    print(f"✅ Vectorstore carregado: {total} vetores em '{pasta_db}'")
    return db
