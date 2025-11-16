from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from app.services import rag_llm
from app.core.config import settings
import logging
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    file_handler = logging.FileHandler("app_logs.txt", mode='w', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('[%(levelname)s] %(asctime)s - %(name)s - %(message)s'))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('[%(levelname)s] %(asctime)s - %(name)s - %(message)s'))
    logger.addHandler(console_handler)

# ---------------------------------------------------------------------------
# CONSTANTES
# ---------------------------------------------------------------------------
SIMILARITY_THRESHOLD = 0.80

# ---------------------------------------------------------------------------
# BUILD RAG CHAIN
# ---------------------------------------------------------------------------
def build_chain():
    logger.info("Carregando modelo de embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/e5-base-v2",
        model_kwargs={"device": "cpu"}
    )

    logger.info("Carregando FAISS em: %s", settings.vector_db_path)
    db = FAISS.load_local(
        settings.vector_db_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    logger.info("FAISS carregado com sucesso.")

    # -----------------------------------------------------------------------
    # Função interna de pergunta ao RAG
    # -----------------------------------------------------------------------
    def perguntar_ao_rag(pergunta: str):
        logger.info(f"📥 Pergunta: {pergunta}")

        docs = db.similarity_search(pergunta, k=3)
        if not docs:
            return "Desculpe, não encontrei informações relevantes na base."

        pergunta_emb = embeddings.embed_query(pergunta)

        doc_embs = np.array([
            embeddings.embed_documents([doc.page_content])[0]
            for doc in docs
        ])

        sims = cosine_similarity([pergunta_emb], doc_embs)[0]

        relevant_docs = [
            doc for doc, sim in zip(docs, sims)
            if sim >= SIMILARITY_THRESHOLD
        ]

        if not relevant_docs:
            return "Desculpe, não encontrei a informação necessária nos meus documentos."

        contexto = "\n\n".join(doc.page_content.strip() for doc in relevant_docs)

        logger.info(f"📚 Contexto selecionado: {contexto}")

        try:
            start = time.time()
            resposta = rag_llm.responder_pergunta(pergunta, contexto)
            elapsed = time.time() - start

            logger.info(f"⏱️ Tempo total: {elapsed:.2f}s")
            logger.info(f"💬 Resposta final: {resposta}")

            return resposta

        except Exception as e:
            logger.error(f"Erro ao processar resposta: {e}")
            return "Erro interno ao gerar resposta."

    return perguntar_ao_rag
