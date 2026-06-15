"""
avaliar.py — Avalia o pipeline RAG contra as perguntas de teste.

Execute com:
    python eval/avaliar.py

Gera um relatório de métricas no terminal e salva em eval/resultados.json.
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embedder import carregar_vectorstore
from src.chain import criar_chain


def avaliar():
    print("=" * 60)
    print("🧪 Iniciando avaliação do pipeline RAG")
    print("=" * 60)

    db = carregar_vectorstore("chroma_db/")
    chain = criar_chain(db)

    with open("eval/perguntas_teste.json", "r", encoding="utf-8") as f:
        perguntas = json.load(f)

    resultados = []
    total_acerto = 0
    total_fonte_ok = 0
    latencias = []

    print(f"\n📋 Avaliando {len(perguntas)} perguntas...\n")

    for item in perguntas:
        pid = item["id"]
        pergunta = item["pergunta"]
        esperada = item["resposta_esperada"]
        na_base = item["na_base"]

        print(f"[{pid:02d}] {pergunta[:70]}...")

        inicio = time.time()
        resultado = chain.invoke(pergunta)
        latencia = round(time.time() - inicio, 2)

        resposta = resultado["result"]
        docs_fonte = resultado.get("source_documents", [])
        fontes = [os.path.basename(d.metadata.get("source", "?")) for d in docs_fonte]

        # ── Avaliação manual simplificada ────────────────────────────────────
        # Para perguntas fora do escopo: verifica se o modelo recusou
        if not na_base:
            recusou = "não encontrei" in resposta.lower() or "não está" in resposta.lower()
            acerto = 1.0 if recusou else 0.0
            fonte_ok = True  # recusa correta não requer fonte
            print(f"     ⚠️  Fora do escopo | Recusou: {'✅' if recusou else '❌'} | {latencia}s")
        else:
            # Verificação simples por palavras-chave da resposta esperada
            palavras_chave = [
                w.lower() for w in esperada.split()
                if len(w) > 4 and w.lower() not in {"são", "para", "como", "que", "uma", "dos"}
            ]
            acertos_kw = sum(1 for kw in palavras_chave if kw in resposta.lower())
            acerto = round(min(acertos_kw / max(len(palavras_chave), 1), 1.0), 2)
            fonte_ok = bool(fontes)
            print(f"     Acerto KW: {acerto:.0%} | Fontes: {', '.join(set(fontes)) or '—'} | {latencia}s")

        total_acerto += acerto
        if fonte_ok:
            total_fonte_ok += 1
        latencias.append(latencia)

        resultados.append({
            "id": pid,
            "pergunta": pergunta,
            "resposta": resposta,
            "resposta_esperada": esperada,
            "acerto": acerto,
            "fonte_ok": fonte_ok,
            "fontes": list(set(fontes)),
            "latencia_s": latencia,
            "dificuldade": item["dificuldade"],
            "categoria": item["categoria"],
        })

    # ── Métricas Finais ──────────────────────────────────────────────────────
    n = len(perguntas)
    media_acerto = round(total_acerto / n, 2)
    pct_fonte = round(total_fonte_ok / n * 100, 1)
    lat_media = round(sum(latencias) / n, 2)
    lat_max = round(max(latencias), 2)

    print("\n" + "=" * 60)
    print("📊 RESULTADOS FINAIS")
    print("=" * 60)
    print(f"  Precisão média (keyword):  {media_acerto:.0%}")
    print(f"  Citação de fontes:         {pct_fonte}%")
    print(f"  Latência média:            {lat_media}s")
    print(f"  Latência máxima:           {lat_max}s")
    print(f"  Meta precisão (>70%):      {'✅' if media_acerto >= 0.7 else '❌'}")
    print(f"  Meta latência (<8s):       {'✅' if lat_media < 8 else '❌'}")
    print("=" * 60)

    # ── Tabela para o README ─────────────────────────────────────────────────
    print("\n📋 Tabela para o README:\n")
    print(f"{'#':<4} {'Pergunta':<50} {'Acerto':<8} {'Fonte':<7} {'Tempo'}")
    print("-" * 80)
    for r in resultados:
        q = r["pergunta"][:47] + "..." if len(r["pergunta"]) > 47 else r["pergunta"]
        print(f"{r['id']:<4} {q:<50} {r['acerto']:<8.1f} {'sim' if r['fonte_ok'] else 'não':<7} {r['latencia_s']}s")
    print("-" * 80)
    print(f"{'Média':<55} {media_acerto:<8.2f} {pct_fonte}%   {lat_media}s")

    # ── Salvar resultados ────────────────────────────────────────────────────
    saida = {
        "metricas": {
            "precisao_media": media_acerto,
            "pct_fonte_ok": pct_fonte,
            "latencia_media_s": lat_media,
            "latencia_max_s": lat_max,
        },
        "resultados": resultados,
    }
    with open("eval/resultados.json", "w", encoding="utf-8") as f:
        json.dump(saida, f, ensure_ascii=False, indent=2)
    print("\n✅ Resultados salvos em eval/resultados.json")


if __name__ == "__main__":
    avaliar()
