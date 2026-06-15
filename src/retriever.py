"""
retriever.py — Busca de documentos relevantes no vectorstore.

Expõe funções para criar retrievers e inspecionar resultados,
separando a lógica de busca da lógica de geração (chain.py).
"""

from langchain_community.vectorstores import Chroma


def criar_retriever(vectorstore: Chroma, k: int = 4):
    """
    Retorna um retriever de similaridade que busca os k chunks
    mais próximos semanticamente à query.
    """
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


def buscar_documentos(retriever, query: str) -> list:
    """Executa a busca e retorna os documentos relevantes."""
    return retriever.invoke(query)


def formatar_contexto(docs: list) -> str:
    """
    Formata os documentos recuperados em um bloco de contexto
    pronto para ser injetado no prompt do LLM.
    """
    partes = []
    for i, doc in enumerate(docs, 1):
        fonte = doc.metadata.get("source", "desconhecida")
        partes.append(f"[Trecho {i} — {fonte}]\n{doc.page_content}")
    return "\n\n---\n\n".join(partes)


def exibir_resultados(docs: list) -> None:
    """Imprime os chunks recuperados para depuração."""
    print(f"\n🔍 {len(docs)} chunk(s) recuperado(s):\n")
    for i, doc in enumerate(docs, 1):
        fonte = doc.metadata.get("source", "?")
        preview = doc.page_content[:120].replace("\n", " ")
        print(f"  [{i}] {fonte}\n      {preview}...\n")
