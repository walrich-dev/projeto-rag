"""
chain.py — Coração do pipeline RAG (LangChain Expression Language).

Usa LCEL (LangChain Expression Language), o padrão do LangChain 1.x,
substituindo o RetrievalQA deprecated. Retorna resposta + docs de origem.
"""

import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["context", "question"],
    template="""Você é um assistente especialista em Python e Inteligência Artificial.
Use o contexto abaixo para responder à pergunta de forma clara e objetiva em português.
Se o contexto não contiver nenhuma informação relevante sobre o assunto perguntado,
responda: "Não encontrei essa informação na base de conhecimento."

Contexto:
{context}

Pergunta: {question}
Resposta:""",
)


def _criar_llm():
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    model = os.getenv("MODEL_NAME", "llama3.1")

    if provider == "ollama":
        from langchain_ollama import OllamaLLM
        print(f"🤖 LLM: Ollama ({model})")
        return OllamaLLM(model=model, temperature=0)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY não definida no .env")
        model_name = model if model != "llama3.1" else "gemini-1.5-flash"
        print(f"🤖 LLM: Google Gemini ({model_name})")
        return ChatGoogleGenerativeAI(model=model_name, temperature=0, google_api_key=api_key)

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY não definida no .env")
        model_name = model if model != "llama3.1" else "gpt-4o-mini"
        print(f"🤖 LLM: OpenAI ({model_name})")
        return ChatOpenAI(model=model_name, temperature=0, openai_api_key=api_key)

    raise ValueError(
        f"Provider '{provider}' não suportado. Use 'ollama', 'gemini' ou 'openai'."
    )


def _format_docs(docs: list) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def criar_chain(vectorstore):
    """
    Retorna um chain LCEL. Ao invocar com {"query": "..."} retorna:
      {
        "question": "...",
        "source_documents": [...],
        "result": "..."
      }
    """
    llm = _criar_llm()
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # Subchain: recebe contexto já formatado + pergunta → gera resposta
    answer_chain = (
        RunnablePassthrough.assign(context=lambda x: _format_docs(x["context"]))
        | PROMPT_TEMPLATE
        | llm
        | StrOutputParser()
    )

    # Chain completo: recupera docs, gera resposta, devolve os dois
    full_chain = RunnableParallel(
        question=RunnablePassthrough(),
        source_documents=retriever,
    ).assign(
        result=lambda x: answer_chain.invoke(
            {"context": x["source_documents"], "question": x["question"]}
        )
    )

    print("✅ Chain RAG pronta (LCEL)")
    return full_chain
