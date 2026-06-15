"""
loader.py — Carregamento e divisão de documentos.

Suporta .txt e .pdf. O RecursiveCharacterTextSplitter divide
respeitando separadores naturais da língua (parágrafos, frases)
antes de cortar no meio de uma palavra.
"""

import os
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFDirectoryLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter


def carregar_documentos(pasta: str = "docs/") -> list:
    if not os.path.exists(pasta):
        raise FileNotFoundError(f"Pasta '{pasta}' não encontrada.")

    arquivos = os.listdir(pasta)
    txts = [f for f in arquivos if f.endswith(".txt")]
    pdfs = [f for f in arquivos if f.endswith(".pdf")]

    if not txts and not pdfs:
        raise ValueError(
            f"Nenhum arquivo .txt ou .pdf encontrado em '{pasta}'.\n"
            "Adicione documentos e tente novamente."
        )

    docs = []

    if txts:
        loader_txt = DirectoryLoader(
            pasta,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=True,
        )
        docs.extend(loader_txt.load())
        print(f"📄 {len(txts)} arquivo(s) .txt carregado(s)")

    if pdfs:
        loader_pdf = PyPDFDirectoryLoader(pasta)
        docs.extend(loader_pdf.load())
        print(f"📄 {len(pdfs)} arquivo(s) .pdf carregado(s)")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    print(f"✅ {len(docs)} doc(s) → {len(chunks)} chunks gerados")
    return chunks
