"""
indexar.py — Script de indexação dos documentos.

Execute UMA VEZ antes de iniciar o app:
    python indexar.py

Reindexe sempre que adicionar ou remover documentos da pasta docs/.
"""

from src.loader import carregar_documentos
from src.embedder import criar_vectorstore

if __name__ == "__main__":
    print("=" * 50)
    print("🚀 Iniciando indexação dos documentos")
    print("=" * 50)

    chunks = carregar_documentos("docs/")
    db = criar_vectorstore(chunks, "chroma_db/")

    print("\n" + "=" * 50)
    print("✅ Indexação concluída!")
    print(f"   Chunks indexados: {db._collection.count()}")
    print("   Agora execute: streamlit run src/app.py")
    print("=" * 50)
